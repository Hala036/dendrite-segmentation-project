"""
dl_pipeline/coco_to_yolo.py

Convert Roboflow COCO Segmentation JSON export to YOLOv8 segmentation format.

Roboflow exports COCO JSON with this structure:
    dataset/
        train/  _annotations.coco.json + images
        valid/  _annotations.coco.json + images
        test/   _annotations.coco.json + images

This script converts to YOLO format:
    data/annotated/
        images/
            train/  *.jpg / *.png
            valid/
            test/
        labels/
            train/  *.txt  (one per image, polygon format)
            valid/
            test/

Each YOLO label line:
    <class_id> x1 y1 x2 y2 ... xn yn
    (all coordinates normalized to [0,1])

Usage:
    python dl_pipeline/coco_to_yolo.py \\
        --coco-dir path/to/roboflow_coco_download/ \\
        --output-dir data/annotated/
"""

import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_coco_json(json_path: str) -> dict:
    """Load and validate a COCO annotation JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    required = ['images', 'annotations', 'categories']
    for key in required:
        if key not in data:
            raise ValueError(f"Missing key '{key}' in COCO JSON: {json_path}")
    return data


def build_image_map(coco_data: dict) -> Dict[int, dict]:
    """Build a dict mapping image_id → image info."""
    return {img['id']: img for img in coco_data['images']}


def build_category_map(coco_data: dict) -> Dict[int, int]:
    """
    Map COCO category_id → YOLO class index (0-based).
    
    For single-class projects (dendrite only), all categories map to 0.
    """
    categories = sorted(coco_data['categories'], key=lambda c: c['id'])
    return {cat['id']: idx for idx, cat in enumerate(categories)}


def segmentation_to_yolo(
    segmentation: List[float],
    img_width: int,
    img_height: int
) -> List[float]:
    """
    Convert a COCO flat polygon [x1,y1,x2,y2,...] to YOLO normalized format.
    
    Args:
        segmentation: Flat list of absolute pixel coordinates [x1,y1,x2,y2,...]
        img_width: Image width in pixels.
        img_height: Image height in pixels.
    
    Returns:
        Flat list of normalized coordinates [x1,y1,x2,y2,...] in [0,1].
    """
    normalized = []
    for i, val in enumerate(segmentation):
        if i % 2 == 0:  # x coordinate
            normalized.append(round(val / img_width, 6))
        else:            # y coordinate
            normalized.append(round(val / img_height, 6))
    return normalized


def convert_split(
    coco_json_path: str,
    images_src_dir: str,
    output_images_dir: str,
    output_labels_dir: str,
    split_name: str
) -> Tuple[int, int, int]:
    """
    Convert one split (train/valid/test) from COCO to YOLO format.
    
    Args:
        coco_json_path: Path to _annotations.coco.json
        images_src_dir: Directory containing source images for this split.
        output_images_dir: Destination for images.
        output_labels_dir: Destination for .txt label files.
        split_name: 'train', 'valid', or 'test' (for logging).
    
    Returns:
        Tuple of (n_images_processed, n_annotations, n_empty_images).
    """
    Path(output_images_dir).mkdir(parents=True, exist_ok=True)
    Path(output_labels_dir).mkdir(parents=True, exist_ok=True)

    coco_data = load_coco_json(coco_json_path)
    image_map = build_image_map(coco_data)
    cat_map = build_category_map(coco_data)

    # Group annotations by image_id
    annotations_by_image: Dict[int, List[dict]] = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    n_images = 0
    n_annotations = 0
    n_empty = 0

    for img_id, img_info in image_map.items():
        filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        stem = Path(filename).stem

        # Copy image to output
        src_img = Path(images_src_dir) / filename
        if not src_img.exists():
            # Try without subdirectory
            src_img = Path(images_src_dir) / Path(filename).name
        
        if src_img.exists():
            dst_img = Path(output_images_dir) / src_img.name
            shutil.copy2(str(src_img), str(dst_img))
        else:
            print(f"  [Warning] Image not found: {filename}")

        # Write YOLO label file
        label_path = Path(output_labels_dir) / f"{stem}.txt"
        anns = annotations_by_image.get(img_id, [])

        if not anns:
            # Negative sample: write empty label file
            label_path.write_text("")
            n_empty += 1
        else:
            lines = []
            for ann in anns:
                if not ann.get('segmentation'):
                    continue

                class_idx = cat_map[ann['category_id']]

                # COCO segmentation can be a list of polygons (for complex shapes)
                # Take the largest polygon if multiple exist
                polygons = ann['segmentation']
                if isinstance(polygons[0], list):
                    # Multiple polygons: use the largest
                    polygon = max(polygons, key=len)
                else:
                    polygon = polygons

                if len(polygon) < 6:
                    # Need at least 3 points (6 values) for a valid polygon
                    continue

                normalized = segmentation_to_yolo(polygon, img_width, img_height)
                coord_str = ' '.join(map(str, normalized))
                lines.append(f"{class_idx} {coord_str}")
                n_annotations += 1

            label_path.write_text('\n'.join(lines))

        n_images += 1

    print(f"  [{split_name}] {n_images} images, "
          f"{n_annotations} annotations, "
          f"{n_empty} negative (empty) samples")

    return n_images, n_annotations, n_empty


def write_dataset_yaml(
    output_dir: str,
    class_names: List[str],
    yaml_path: str
) -> None:
    """Write the dataset.yaml file for YOLOv8 training."""
    content = f"""# YOLOv8 Segmentation Dataset — Dendrites
# Auto-generated by coco_to_yolo.py

path: {Path(output_dir).resolve()}
train: images/train
val: images/valid
test: images/test

nc: {len(class_names)}
names:
"""
    for i, name in enumerate(class_names):
        content += f"  {i}: {name}\n"

    Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
    Path(yaml_path).write_text(content)
    print(f"\n[OK] dataset.yaml written: {yaml_path}")


def convert_roboflow_coco(
    coco_dir: str,
    output_dir: str,
    yaml_path: str = None
) -> None:
    """
    Convert a full Roboflow COCO export to YOLOv8 segmentation format.

    Expected input structure (Roboflow COCO download):
        coco_dir/
            train/
                _annotations.coco.json
                image1.jpg
                image2.jpg
                ...
            valid/
                _annotations.coco.json
                ...
            test/
                _annotations.coco.json
                ...

    Args:
        coco_dir: Path to the downloaded Roboflow COCO zip extraction.
        output_dir: Where to write the converted YOLO dataset.
        yaml_path: Where to write dataset.yaml (default: output_dir/dataset.yaml).
    """
    coco_dir = Path(coco_dir)
    output_dir = Path(output_dir)
    yaml_path = yaml_path or str(output_dir / 'dataset.yaml')

    print(f"\n{'='*60}")
    print(f"  COCO -> YOLO Segmentation Converter")
    print(f"  Input:  {coco_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # Detect splits present
    splits = []
    for split in ['train', 'valid', 'test']:
        json_path = coco_dir / split / '_annotations.coco.json'
        if json_path.exists():
            splits.append(split)
        else:
            print(f"[Skip] No annotations found for split: {split}")

    if not splits:
        raise FileNotFoundError(
            f"No _annotations.coco.json found in {coco_dir}/train|valid|test/\n"
            f"Make sure you extracted the Roboflow zip correctly."
        )

    # Get class names from first available split
    first_json = coco_dir / splits[0] / '_annotations.coco.json'
    coco_data = load_coco_json(str(first_json))
    class_names = [cat['name'] for cat in
                   sorted(coco_data['categories'], key=lambda c: c['id'])]
    print(f"[Info] Classes found: {class_names}")

    # Convert each split
    total_images = 0
    for split in splits:
        json_path = str(coco_dir / split / '_annotations.coco.json')
        images_src = str(coco_dir / split)
        out_images = str(output_dir / 'images' / split)
        out_labels = str(output_dir / 'labels' / split)

        n_imgs, n_anns, n_empty = convert_split(
            json_path, images_src, out_images, out_labels, split
        )
        total_images += n_imgs

    # Write dataset.yaml
    write_dataset_yaml(str(output_dir), class_names, yaml_path)

    print(f"\n[OK] Conversion complete!")
    print(f"    Total images converted: {total_images}")
    print(f"\nNext step - run training:")
    print(f"    python pipeline_A_DL/train.py --data {yaml_path}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert Roboflow COCO export to YOLOv8 segmentation format"
    )
    p.add_argument('--coco-dir', required=True,
                   help="Path to extracted Roboflow COCO zip")
    p.add_argument('--output-dir', default='data/annotated/',
                   help="Output directory for YOLO dataset")
    p.add_argument('--yaml', default='dl_pipeline/dataset.yaml',
                   help="Path to write dataset.yaml")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    convert_roboflow_coco(
        coco_dir=args.coco_dir,
        output_dir=args.output_dir,
        yaml_path=args.yaml
    )