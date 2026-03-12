"""
predict_tiled.py - Pipeline A: YOLO Instance Segmentation Inference
====================================================================
PURPOSE OF THIS FILE:
    Runs the trained YOLO26-seg model on full SEM images and produces
    binary segmentation masks comparable to Pipeline B's output.

    THE TILING PROBLEM AT INFERENCE TIME:
        The model was trained on 640x640 tiles. If you feed it a full
        1280x960 image directly, two things go wrong:
            1. YOLO resizes it to 640x640 internally — dendrites become
               tiny and the model misses them (wrong scale)
            2. The model has never seen full images during training —
               it expects the tile-scale appearance of dendrites

        Solution: tile the input image the SAME WAY as training,
        run the model on each tile, then stitch all tile masks back
        into one full-image mask.

    STITCHING CHALLENGE:
        Tiles overlap by 20%. A dendrite near a tile edge appears in
        two tiles and gets predicted twice. We handle this with a
        confidence-weighted average in the overlap zones — the higher
        confidence prediction wins.

USAGE:
    # Single image:
    python predict_tiled.py --weights outputs/weights/best.pt
                            --image data/annotated/test/images/img001.png
                            --output outputs/masks

    # Entire test set:
    python predict_tiled.py --weights outputs/weights/best.pt
                            --folder data/annotated/test/images
                            --output outputs/masks

    # With visualization:
    python predict_tiled.py --weights outputs/weights/best.pt
                            --folder data/annotated/test/images
                            --output outputs/masks
                            --visualize
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics")


# ==============================================================================
# TILE COORDINATE GENERATOR
# ==============================================================================

def get_tile_coords(img_h: int, img_w: int,
                    tile_size: int = 640,
                    overlap: float = 0.2) -> list:
    """
    Generates (y_start, y_end, x_start, x_end) for every tile position.

    MUST match the tiling logic in tile_dataset.py exactly —
    same tile_size, same overlap, same stride calculation.
    If these differ, the model sees images at a different scale than
    it was trained on and performance degrades.

    Args:
        img_h:     Full image height in pixels
        img_w:     Full image width in pixels
        tile_size: Tile size used during training (default 640)
        overlap:   Overlap fraction used during training (default 0.2)

    Returns:
        List of (y_start, y_end, x_start, x_end) tuples
    """
    stride = int(tile_size * (1 - overlap))
    coords = []

    y_start = 0
    while y_start < img_h:
        y_end = min(y_start + tile_size, img_h)

        x_start = 0
        while x_start < img_w:
            x_end = min(x_start + tile_size, img_w)
            coords.append((y_start, y_end, x_start, x_end))

            if x_end >= img_w:
                break
            x_start += stride

        if y_end >= img_h:
            break
        y_start += stride

    return coords


# ==============================================================================
# SINGLE IMAGE PREDICTION WITH TILING
# ==============================================================================

def predict_single_image(model,
                          image_path: str,
                          tile_size: int = 640,
                          overlap: float = 0.2,
                          conf_threshold: float = 0.25,
                          iou_threshold: float = 0.45) -> np.ndarray:
    """
    Runs tiled inference on one full SEM image and returns a binary mask.

    STITCHING STRATEGY — CONFIDENCE ACCUMULATION:
        Instead of hard OR/AND decisions in overlap zones, we use a
        floating-point accumulation approach:

            score_map:  accumulates raw confidence scores per pixel
            count_map:  counts how many tiles covered each pixel

        For each tile:
            - Run YOLO prediction → get instance masks + confidence scores
            - For each detected dendrite instance:
                Add (mask * confidence) to score_map in the tile region
            - Add 1 to count_map for every pixel in the tile region

        Final mask:
            average_score = score_map / count_map
            binary_mask   = average_score > final_threshold

        In overlap zones where a dendrite appears in multiple tiles:
            - High-confidence detections from both tiles reinforce each other
            - A false positive in one tile gets diluted by the zero-score
              from the neighboring tile that correctly rejected it

    Args:
        model:           Loaded YOLO model
        image_path:      Path to full SEM image
        tile_size:       Must match training tile size
        overlap:         Must match training overlap
        conf_threshold:  Minimum confidence to keep a detection (0-1)
        iou_threshold:   NMS IoU threshold for removing duplicate detections

    Returns:
        Binary mask as uint8 numpy array (0 or 255), same size as input image
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_h, img_w = image.shape[:2]

    # Accumulation maps — float32 for precise averaging
    score_map = np.zeros((img_h, img_w), dtype=np.float32)
    count_map = np.zeros((img_h, img_w), dtype=np.float32)

    tile_coords = get_tile_coords(img_h, img_w, tile_size, overlap)

    for (y_start, y_end, x_start, x_end) in tile_coords:
        # Crop tile from full image
        tile = image[y_start:y_end, x_start:x_end]

        # Pad to exact tile_size if at image boundary
        pad_h = tile_size - tile.shape[0]
        pad_w = tile_size - tile.shape[1]
        if pad_h > 0 or pad_w > 0:
            tile = cv2.copyMakeBorder(
                tile, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=0
            )

        # Run YOLO prediction on this tile
        results = model.predict(
            source=tile,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,      # suppress per-tile output
            stream=False
        )

        result = results[0]

        # Get actual tile dimensions (before padding)
        actual_h = y_end - y_start
        actual_w = x_end - x_start

        # Mark this tile region as covered (for averaging)
        count_map[y_start:y_end, x_start:x_end] += 1.0

        # If no detections in this tile, move on
        if result.masks is None or len(result.masks) == 0:
            continue

        # Process each detected dendrite instance
        masks_data = result.masks.data.cpu().numpy()   # shape: (N, H, W)
        confidences = result.boxes.conf.cpu().numpy()  # shape: (N,)

        for inst_mask, conf in zip(masks_data, confidences):
            # inst_mask is normalized [0,1] float, same size as tile input
            # Resize to tile_size x tile_size if needed
            if inst_mask.shape != (tile_size, tile_size):
                inst_mask = cv2.resize(
                    inst_mask,
                    (tile_size, tile_size),
                    interpolation=cv2.INTER_LINEAR
                )

            # Crop back to actual tile dimensions (remove padding)
            inst_mask_cropped = inst_mask[:actual_h, :actual_w]

            # Accumulate: confidence-weighted mask score
            score_map[y_start:y_end, x_start:x_end] += \
                inst_mask_cropped * conf

    # Average scores across all tiles that covered each pixel
    # Avoid division by zero (pixels not covered by any tile → 0)
    avg_score = np.where(count_map > 0, score_map / count_map, 0.0)

    # Threshold the averaged score map to get binary mask
    # 0.15 is lower than conf_threshold because averaged scores are diluted
    # by neighboring tiles that may have returned zero for background pixels
    final_threshold = conf_threshold * 0.6
    binary_mask = (avg_score > final_threshold).astype(np.uint8) * 255

    return binary_mask


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_prediction(image_path: str,
                          mask: np.ndarray,
                          save_path: str = None) -> None:
    """
    Creates a side-by-side visualization: original image + mask overlay.

    Args:
        image_path: Path to original SEM image
        mask:       Binary mask from predict_single_image()
        save_path:  If provided, saves the figure here
    """
    import matplotlib.pyplot as plt

    original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # Create green overlay
    rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    overlay = rgb.copy()
    overlay[mask > 0] = [0, 200, 0]   # green dendrites
    blended = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original SEM Image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('YOLO Prediction Mask', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Overlay (green = dendrite)', fontsize=12)
    axes[2].axis('off')

    stem = Path(image_path).stem
    plt.suptitle(f'YOLO Prediction — {stem}', fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved visualization: {save_path}")

    plt.show()
    plt.close()


# ==============================================================================
# BATCH PREDICTION
# ==============================================================================

def predict_folder(model,
                   folder_path: str,
                   output_dir: str,
                   tile_size: int = 640,
                   overlap: float = 0.2,
                   conf_threshold: float = 0.25,
                   iou_threshold: float = 0.45,
                   visualize: bool = False) -> dict:
    """
    Runs tiled prediction on all images in a folder.

    Args:
        model:          Loaded YOLO model
        folder_path:    Directory containing SEM images
        output_dir:     Where to save binary masks
        tile_size:      Must match training tile size
        overlap:        Must match training overlap
        conf_threshold: Minimum confidence threshold
        iou_threshold:  NMS IoU threshold
        visualize:      If True, saves overlay visualizations too

    Returns:
        Dictionary mapping image filename → output mask path
    """
    folder = Path(folder_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if visualize:
        viz_dir = output.parent / 'visuals' / 'yolo_predictions'
        viz_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    image_paths = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    if len(image_paths) == 0:
        print(f"  WARNING: No images found in {folder_path}")
        return {}

    print(f"\nRunning tiled prediction on {len(image_paths)} images...")
    print(f"  Tile size: {tile_size}x{tile_size}, overlap: {overlap:.0%}")
    print(f"  Confidence threshold: {conf_threshold}")
    print(f"  Output directory: {output}\n")

    results_map = {}

    for i, img_path in enumerate(image_paths, 1):
        print(f"  [{i:2d}/{len(image_paths)}] {img_path.name}...", end=' ')

        try:
            mask = predict_single_image(
                model, img_path,
                tile_size, overlap,
                conf_threshold, iou_threshold
            )

            # Save binary mask
            mask_path = output / f"{img_path.stem}_yolo_mask.png"
            cv2.imwrite(str(mask_path), mask)

            # Count detected dendrite pixels as rough quality indicator
            coverage = (mask > 0).mean() * 100
            print(f"done  ({coverage:.1f}% dendrite coverage)")

            results_map[img_path.name] = str(mask_path)

            # Save visualization if requested
            if visualize:
                viz_path = viz_dir / f"{img_path.stem}_overlay.png"
                visualize_prediction(str(img_path), mask, str(viz_path))

        except Exception as e:
            print(f"FAILED — {e}")
            results_map[img_path.name] = None

    successful = sum(1 for v in results_map.values() if v is not None)
    print(f"\nCompleted: {successful}/{len(image_paths)} images processed")
    print(f"Masks saved to: {output}")

    return results_map


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main(args):
    print("=" * 55)
    print("PIPELINE A — YOLO26 TILED INFERENCE")
    print("=" * 55)
    print(f"  Weights:    {args.weights}")
    print(f"  Tile size:  {args.tile_size}")
    print(f"  Overlap:    {args.overlap:.0%}")
    print(f"  Confidence: {args.conf}")

    # Load model
    weights_path = Path(args.weights)
    assert weights_path.exists(), \
        f"Weights not found: {args.weights}\n" \
        f"Did you run train_yolo.py first?"

    print(f"\nLoading model from {weights_path}...")
    model = YOLO(str(weights_path))
    print("  ✓ Model loaded")

    # Single image mode
    if args.image:
        print(f"\nPredicting single image: {args.image}")
        mask = predict_single_image(
            model, args.image,
            args.tile_size, args.overlap,
            args.conf, args.iou
        )

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.image).stem
        mask_path = output_dir / f"{stem}_yolo_mask.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"  Mask saved: {mask_path}")

        if args.visualize:
            viz_path = output_dir / f"{stem}_overlay.png"
            visualize_prediction(args.image, mask, str(viz_path))

    # Folder mode
    elif args.folder:
        predict_folder(
            model=model,
            folder_path=args.folder,
            output_dir=args.output,
            tile_size=args.tile_size,
            overlap=args.overlap,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            visualize=args.visualize
        )

    else:
        print("\nERROR: Provide either --image or --folder")
        print("Examples:")
        print("  python predict_tiled.py --weights outputs/weights/best.pt "
              "--image data/annotated/test/images/img001.png")
        print("  python predict_tiled.py --weights outputs/weights/best.pt "
              "--folder data/annotated/test/images --visualize")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Tiled YOLO26-seg inference on SEM images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model
    parser.add_argument(
        '--weights', type=str,
        default='outputs/weights/best.pt',
        help='Path to trained model weights (best.pt from train_yolo.py)'
    )

    # Input — one of these two required
    parser.add_argument(
        '--image', type=str,
        default=None,
        help='Path to a single SEM image for prediction'
    )
    parser.add_argument(
        '--folder', type=str,
        default=None,
        help='Path to folder of SEM images (processes all images in folder)'
    )

    # Output
    parser.add_argument(
        '--output', type=str,
        default='outputs/masks',
        help='Directory to save binary mask outputs'
    )

    # Tiling — MUST match tile_dataset.py settings
    parser.add_argument(
        '--tile-size', type=int,
        default=640,
        help='Tile size (must match tile_dataset.py --tile-size)'
    )
    parser.add_argument(
        '--overlap', type=float,
        default=0.2,
        help='Tile overlap fraction (must match tile_dataset.py --overlap)'
    )

    # Detection thresholds
    parser.add_argument(
        '--conf', type=float,
        default=0.25,
        help='Minimum confidence threshold for detections (0-1). '
             'Lower = more detections but more false positives. '
             'Higher = fewer but more precise detections.'
    )
    parser.add_argument(
        '--iou', type=float,
        default=0.45,
        help='NMS IoU threshold for removing duplicate detections. '
             'Lower = more aggressive duplicate removal.'
    )

    # Visualization
    parser.add_argument(
        '--visualize', action='store_true',
        help='Save overlay visualizations alongside masks'
    )

    args = parser.parse_args()
    main(args)