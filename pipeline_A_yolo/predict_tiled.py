"""
Run tiled YOLO inference on full SEM images.

This file cuts a large image into tiles, predicts each tile, and then joins the
tile masks back into one full-size mask. It uses the same tile idea as the
training dataset, so the object scale stays similar.

Examples:
    python predict_tiled.py --weights outputs/weights/best.pt \
        --folder data/annotated/test/images --output outputs/masks_tiled

    python predict_tiled.py --weights outputs/weights/best.pt \
        --image data/annotated/test/images/img001.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics")


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_prediction(image_path: str,
                         mask: np.ndarray,
                         save_path: str | None = None) -> None:
    """
    Save a simple overlay image of the predicted mask.
    """
    original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise ValueError(f"Could not read image: {image_path}")

    rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    overlay = rgb.copy()
    overlay[mask > 0] = [0, 200, 0]
    blended = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {save_path}")


# ==============================================================================
# TILING HELPERS
# ==============================================================================

def tile_coordinates(img_h: int,
                     img_w: int,
                     tile_size: int,
                     overlap: float) -> list[tuple[int, int, int, int]]:
    """
    Return tile boxes as `(x_start, y_start, x_end, y_end)`.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0,1). Got {overlap}")

    stride = int(tile_size * (1.0 - overlap))
    if stride <= 0:
        raise ValueError(
            f"Invalid stride={stride}. Reduce overlap or increase tile_size."
        )

    boxes: list[tuple[int, int, int, int]] = []

    y_start = 0
    while y_start < img_h:
        y_end = min(y_start + tile_size, img_h)

        x_start = 0
        while x_start < img_w:
            x_end = min(x_start + tile_size, img_w)
            boxes.append((x_start, y_start, x_end, y_end))

            if x_end >= img_w:
                break
            x_start += stride

        if y_end >= img_h:
            break
        y_start += stride

    return boxes


def predict_tile_mask(model,
                      tile_bgr: np.ndarray,
                      conf_threshold: float,
                      iou_threshold: float,
                      tile_size: int) -> np.ndarray:
    """
    Predict one tile and return a mask in the range `[0, 1]`.
    """
    results = model.predict(
        source=tile_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )

    result = results[0]
    tile_prob = np.zeros((tile_size, tile_size), dtype=np.float32)

    if result.masks is None or len(result.masks) == 0:
        return tile_prob

    masks_data = result.masks.data.cpu().numpy()  # (N, H, W)

    for inst_mask in masks_data:
        if inst_mask.shape != (tile_size, tile_size):
            inst_mask = cv2.resize(
                inst_mask,
                (tile_size, tile_size),
                interpolation=cv2.INTER_LINEAR
            )
        tile_prob = np.maximum(tile_prob, inst_mask.astype(np.float32))

    return tile_prob


# ==============================================================================
# TILED INFERENCE
# ==============================================================================

def predict_single_image_tiled(model,
                               image_path: str,
                               conf_threshold: float = 0.25,
                               iou_threshold: float = 0.45,
                               tile_size: int = 640,
                               overlap: float = 0.2,
                               mask_threshold: float = 0.5) -> np.ndarray:
    """
    Run tiled inference on one image and return the final binary mask.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_h, img_w = image.shape[:2]
    boxes = tile_coordinates(img_h, img_w, tile_size, overlap)

    full_prob = np.zeros((img_h, img_w), dtype=np.float32)

    for x0, y0, x1, y1 in boxes:
        tile = image[y0:y1, x0:x1]

        # Pad border tiles to exact tile_size (same strategy as tile_dataset.py)
        pad_h = tile_size - tile.shape[0]
        pad_w = tile_size - tile.shape[1]
        if pad_h > 0 or pad_w > 0:
            tile = cv2.copyMakeBorder(
                tile,
                0, pad_h,
                0, pad_w,
                cv2.BORDER_CONSTANT,
                value=0
            )

        tile_prob = predict_tile_mask(
            model=model,
            tile_bgr=tile,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            tile_size=tile_size,
        )

        # Remove padded border before stitching
        h_eff = y1 - y0
        w_eff = x1 - x0
        tile_prob = tile_prob[:h_eff, :w_eff]

        # Overlap handling: keep strongest probability at each pixel
        full_prob[y0:y1, x0:x1] = np.maximum(full_prob[y0:y1, x0:x1], tile_prob)

    binary_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    binary_mask[full_prob > mask_threshold] = 255
    return binary_mask


def predict_folder_tiled(model,
                         folder_path: str,
                         output_dir: str,
                         conf_threshold: float = 0.25,
                         iou_threshold: float = 0.45,
                         tile_size: int = 640,
                         overlap: float = 0.2,
                         mask_threshold: float = 0.5,
                         visualize: bool = False) -> dict:
    """
    Run tiled prediction for every image in a folder.
    """
    folder = Path(folder_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if visualize:
        viz_dir = output.parent / 'overlays'
        viz_dir.mkdir(parents=True, exist_ok=True)

    extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    image_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in extensions])

    if not image_paths:
        print(f"  WARNING: No images found in {folder_path}")
        return {}

    print(f"\nRunning TILED prediction on {len(image_paths)} images...")
    print(f"  Confidence threshold : {conf_threshold}")
    print(f"  Tile size            : {tile_size}")
    print(f"  Overlap              : {overlap}")
    print(f"  Output directory     : {output}\n")

    results_map = {}

    for i, img_path in enumerate(image_paths, 1):
        print(f"  [{i:2d}/{len(image_paths)}] {img_path.name}...", end=' ')
        try:
            mask = predict_single_image_tiled(
                model=model,
                image_path=str(img_path),
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                tile_size=tile_size,
                overlap=overlap,
                mask_threshold=mask_threshold,
            )

            mask_path = output / f"{img_path.stem}_yolo_tiled_mask.png"
            cv2.imwrite(str(mask_path), mask)

            coverage = (mask > 0).mean() * 100
            print(f"done  ({coverage:.1f}% dendrite coverage)")

            results_map[img_path.name] = str(mask_path)

            if visualize:
                viz_path = viz_dir / f"{img_path.stem}_tiled_overlay.png"
                visualize_prediction(str(img_path), mask, str(viz_path))

        except Exception as e:
            print(f"FAILED - {e}")
            results_map[img_path.name] = None

    successful = sum(1 for v in results_map.values() if v is not None)
    print(f"\nDone: {successful}/{len(image_paths)} images")
    print(f"Masks saved to: {output}")
    return results_map


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main(args):
    print("=" * 55)
    print("PIPELINE A - YOLO26 TILED INFERENCE")
    print("=" * 55)
    print(f"  Weights    : {args.weights}")
    print(f"  Confidence : {args.conf}")
    print(f"  Tile size  : {args.tile_size}")
    print(f"  Overlap    : {args.overlap}")

    weights_path = Path(args.weights)
    assert weights_path.exists(), \
        f"Weights not found: {args.weights}\n" \
        f"Did you run pipeline_A_yolo/train.py first?"

    print("\nLoading model...")
    model = YOLO(str(weights_path))
    print("  ✓ Model loaded\n")

    if args.image:
        mask = predict_single_image_tiled(
            model=model,
            image_path=args.image,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            tile_size=args.tile_size,
            overlap=args.overlap,
            mask_threshold=args.mask_threshold,
        )

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path = output_dir / f"{Path(args.image).stem}_yolo_tiled_mask.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"Mask saved: {mask_path}")

        if args.visualize:
            viz_path = output_dir / f"{Path(args.image).stem}_tiled_overlay.png"
            visualize_prediction(args.image, mask, str(viz_path))

    elif args.folder:
        predict_folder_tiled(
            model=model,
            folder_path=args.folder,
            output_dir=args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            tile_size=args.tile_size,
            overlap=args.overlap,
            mask_threshold=args.mask_threshold,
            visualize=args.visualize,
        )

    else:
        print("ERROR: Provide either --image or --folder")
        print("Examples:")
        print("  python predict_tiled.py --weights outputs/weights/best.pt "
              "--folder data/annotated/test/images --tile-size 640 --overlap 0.2")
        print("  python predict_tiled.py --weights outputs/weights/best.pt "
              "--image data/annotated/test/images/img001.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='YOLO26-seg tiled inference on full SEM test images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--weights', type=str,
                        default='outputs/weights/best.pt',
                        help='Path to trained weights from pipeline_A_yolo/train.py')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image path')
    parser.add_argument('--folder', type=str, default=None,
                        help='Folder of test images')
    parser.add_argument('--output', type=str, default='comparison_results/yolo_tiled/masks',
                        help='Where to save output masks')
    parser.add_argument('--conf', type=float, default=0.05,
                        help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='NMS IoU threshold (0-1)')
    parser.add_argument('--tile-size', type=int, default=640,
                        help='Tile size used for tiled inference')
    parser.add_argument('--overlap', type=float, default=0.2,
                        help='Overlap fraction between adjacent tiles')
    parser.add_argument('--mask-threshold', type=float, default=0.5,
                        help='Threshold applied to merged tile probability map')
    parser.add_argument('--visualize', action='store_true',
                        help='Save overlay visualizations')

    main(parser.parse_args())
