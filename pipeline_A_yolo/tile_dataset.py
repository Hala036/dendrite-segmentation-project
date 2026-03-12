"""
tile_dataset.py - Pipeline A: YOLO Instance Segmentation
===============================================================
PURPOSE OF THIS FILE:
    Prepares the final dataset for YOLO training by:
        1. Tiling ONLY the training set (12 images → ~60-80 tiles)
        2. Copying val and test sets as-is (full images, untouched)

    WHY TILE ONLY TRAIN?
        - Train:  tiling creates more samples + makes thin dendrites larger
                  relative to tile size → model learns better
        - Val:    full images → honest feedback during training
        - Test:   full images → honest final metrics for the report
                  (tiled test metrics would not represent real-world performance)

    INPUT STRUCTURE (exported from Roboflow, no tiling applied):
        data/annotated/
            train/
                images/   ← 12 full SEM images
                labels/   ← 12 .txt files (YOLO polygon format)
            val/
                images/   ← 3 full images
                labels/   ← 3 .txt files
            test/
                images/   ← 5 full images
                labels/   ← 5 .txt files
            data.yaml

    OUTPUT STRUCTURE (ready for train.py):
        data/tiled/
            train/
                images/   ← ~60-80 tiles (640x640)
                labels/   ← corresponding .txt files with adjusted coordinates
            val/
                images/   ← 3 full images (copied as-is)
                labels/   ← 3 .txt files (copied as-is)
            test/
                images/   ← 5 full images (copied as-is)
                labels/   ← 5 .txt files (copied as-is)
            data.yaml     ← updated paths pointing to tiled directory
"""

import cv2
import numpy as np
import shutil
import yaml
from pathlib import Path


# ==============================================================================
# CORE TILING LOGIC
# ==============================================================================

def tile_image_and_labels(image_path: Path,
                           label_path: Path,
                           output_images_dir: Path,
                           output_labels_dir: Path,
                           tile_size: int = 640,
                           overlap: float = 0.2) -> int:
    """
    Cuts one image into overlapping tiles and adjusts YOLO polygon labels.

    HOW COORDINATE ADJUSTMENT WORKS:
        YOLO labels store polygon coordinates normalized to [0,1] relative
        to the FULL image. When we cut a tile, we need to:
            1. Convert normalized coords → absolute pixel coords
            2. Check if the polygon falls within this tile
            3. Clip the polygon to the tile boundary
            4. Re-normalize coords relative to the TILE size

        A polygon is included in a tile if its bounding box center
        falls within the tile region. This avoids duplicating the same
        dendrite across many tiles.

    Args:
        image_path:         Path to full SEM image
        label_path:         Path to corresponding YOLO .txt label file
        output_images_dir:  Where to save tile images
        output_labels_dir:  Where to save tile label files
        tile_size:          Width and height of each tile in pixels
        overlap:            Fraction of tile_size to overlap between tiles (0.2 = 20%)

    Returns:
        Number of tiles generated from this image
    """
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  WARNING: Could not read {image_path}, skipping")
        return 0

    img_h, img_w = image.shape[:2]
    stride = int(tile_size * (1 - overlap))   # step size between tile starts

    # Read all polygon annotations for this image
    annotations = []
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = list(map(float, line.split()))
                    class_id = int(parts[0])
                    # coords are alternating x,y pairs normalized to [0,1]
                    coords = parts[1:]
                    annotations.append((class_id, coords))

    tile_count = 0
    stem = image_path.stem   # filename without extension

    # Slide the tile window across the image
    y_start = 0
    while y_start < img_h:
        y_end = min(y_start + tile_size, img_h)

        x_start = 0
        while x_start < img_w:
            x_end = min(x_start + tile_size, img_w)

            # Crop the tile from the image
            tile_img = image[y_start:y_end, x_start:x_end]

            # Pad to exact tile_size if we're at the image boundary
            pad_h = tile_size - tile_img.shape[0]
            pad_w = tile_size - tile_img.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile_img = cv2.copyMakeBorder(
                    tile_img, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=0
                )

            # Adjust annotations for this tile
            tile_annotations = []
            for class_id, coords in annotations:
                # Convert normalized coords → absolute pixel coords
                abs_coords = []
                for i in range(0, len(coords), 2):
                    px = coords[i]   * img_w    # x in full image pixels
                    py = coords[i+1] * img_h    # y in full image pixels
                    abs_coords.append((px, py))

                # Find bounding box center of this polygon
                xs = [p[0] for p in abs_coords]
                ys = [p[1] for p in abs_coords]
                center_x = sum(xs) / len(xs)
                center_y = sum(ys) / len(ys)

                # Only include polygon if its center is within this tile
                # This prevents the same dendrite appearing in multiple tiles
                if not (x_start <= center_x < x_end and
                        y_start <= center_y < y_end):
                    continue

                # Shift coords to tile-local coordinates and clip to tile bounds
                tile_coords = []
                for px, py in abs_coords:
                    local_x = np.clip(px - x_start, 0, tile_size) / tile_size
                    local_y = np.clip(py - y_start, 0, tile_size) / tile_size
                    tile_coords.extend([local_x, local_y])

                tile_annotations.append((class_id, tile_coords))

            # Save tile image
            tile_name = f"{stem}_tile_{y_start}_{x_start}"
            tile_img_path = output_images_dir / f"{tile_name}.png"
            cv2.imwrite(str(tile_img_path), tile_img)

            # Save tile label (even if empty — YOLO expects a label file per image)
            tile_lbl_path = output_labels_dir / f"{tile_name}.txt"
            with open(tile_lbl_path, 'w') as f:
                for class_id, coords in tile_annotations:
                    coord_str = ' '.join(f'{c:.6f}' for c in coords)
                    f.write(f"{class_id} {coord_str}\n")

            tile_count += 1

            if x_end >= img_w:
                break
            x_start += stride

        if y_end >= img_h:
            break
        y_start += stride

    return tile_count


# ==============================================================================
# DATASET BUILDER
# ==============================================================================

def tile_dataset(input_dir: str,
                        output_dir: str,
                        tile_size: int = 640,
                        overlap: float = 0.2) -> None:
    """
    Builds the final training-ready dataset:
        - Tiles the train split
        - Copies val and test splits as-is

    Args:
        input_dir:  Root of Roboflow export (contains train/, val/, test/, data.yaml)
        output_dir: Where to write the prepared dataset
        tile_size:  Tile dimensions in pixels (default 640 — standard for YOLO)
        overlap:    Overlap fraction between adjacent tiles (default 0.2 = 20%)
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)

    # Verify input structure
    assert (input_path / 'data.yaml').exists(), \
        f"data.yaml not found in {input_dir} — is this a valid Roboflow export?"

    # ── TRAIN: tile images ────────────────────────────────────────────────────
    print("=" * 55)
    print("STEP 1: Tiling train set")
    print("=" * 55)

    train_img_out = output_path / 'train' / 'images'
    train_lbl_out = output_path / 'train' / 'labels'
    train_img_out.mkdir(parents=True, exist_ok=True)
    train_lbl_out.mkdir(parents=True, exist_ok=True)

    train_images = sorted((input_path / 'train' / 'images').glob('*'))
    total_tiles  = 0

    for img_path in train_images:
        if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            continue
        lbl_path = input_path / 'train' / 'labels' / (img_path.stem + '.txt')
        n = tile_image_and_labels(
            img_path, lbl_path,
            train_img_out, train_lbl_out,
            tile_size, overlap
        )
        print(f"  {img_path.name:40s} → {n} tiles")
        total_tiles += n

    print(f"\n  Total train tiles: {total_tiles}  (from {len(train_images)} images)")

    # ── VAL + TEST: copy as-is ────────────────────────────────────────────────
    # Roboflow exports often use 'valid' not 'val' — resolve from data.yaml
    def resolve_split_folder(path_value: str, fallback_key: str) -> str:
        """
        Extracts the split folder name from yaml paths like:
            ../valid/images
            valid/images
            /abs/path/to/valid/images
            valid
        """
        parts = [p for p in Path(path_value).parts if p not in ('', '.', '/')]

        if 'images' in parts:
            idx = parts.index('images')
            if idx > 0:
                return parts[idx - 1]

        for p in reversed(parts):
            if p not in ('..', 'images'):
                return p

        return fallback_key

    with open(input_path / 'data.yaml', 'r') as f:
        yaml_data_in = yaml.safe_load(f)
    split_folders = []
    for key in ['val', 'test']:
        path_val = yaml_data_in.get(key, f"../{key}/images")
        folder = resolve_split_folder(path_val, key)
        split_folders.append((key, folder))

    for split_key, folder in split_folders:
        split_input = input_path / folder
        if not split_input.exists():
            print(f"\n  WARNING: {folder}/ not found in input (for {split_key}), skipping")
            continue

        print(f"\nSTEP {'2' if split_key == 'val' else '3'}: Copying {split_key} set (no tiling)")
        split_output = output_path / split_key
        if split_output.exists():
            shutil.rmtree(split_output)
        shutil.copytree(str(split_input), str(split_output))
        n_imgs = len(list((split_output / 'images').glob('*')))
        print(f"  Copied {n_imgs} full images → {split_output}")

    # ── data.yaml: update paths ───────────────────────────────────────────────
    print("\nSTEP 4: Writing data.yaml")

    with open(input_path / 'data.yaml', 'r') as f:
        yaml_data = yaml.safe_load(f)

    # Update paths to point to the new tiled directory
    yaml_data['path']  = str(output_path.resolve())
    yaml_data['train'] = 'train/images'
    yaml_data['val']   = 'val/images'
    if (output_path / 'test').exists():
        yaml_data['test'] = 'test/images'

    output_yaml = output_path / 'data.yaml'
    with open(output_yaml, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"  Saved: {output_yaml}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("DATASET READY")
    print("=" * 55)
    print(f"  Train tiles : {total_tiles}")
    print(f"  Val images  : {len(list((output_path / 'val' / 'images').glob('*')))}")
    print(f"  Test images : {len(list((output_path / 'test' / 'images').glob('*'))) if (output_path / 'test').exists() else 0}")
    print(f"  Output dir  : {output_path.resolve()}")
    print(f"\nNext step: python pipeline_A_yolo/train.py --data {output_yaml}")


# ==============================================================================
# QUICK TEST / ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Prepare tiled YOLO dataset')
    parser.add_argument('--input',     default='data/annotated',
                        help='Roboflow export directory (default: data/annotated)')
    parser.add_argument('--output',    default='data/tiled',
                        help='Output directory (default: data/tiled)')
    parser.add_argument('--tile-size', type=int, default=640,
                        help='Tile size in pixels (default: 640)')
    parser.add_argument('--overlap',   type=float, default=0.2,
                        help='Overlap fraction between tiles (default: 0.2)')
    args = parser.parse_args()

    tile_dataset(
        input_dir=args.input,
        output_dir=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap
    )
