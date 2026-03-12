"""
predict.py - Pipeline A: YOLO Instance Segmentation Inference
=============================================================
PURPOSE OF THIS FILE:
    Runs the trained YOLO26-seg model on full SEM images and produces
    binary segmentation masks comparable to Pipeline B's classic CV output.

    The test set images are full, untiled images — YOLO handles resizing
    internally and scales the output masks back to the original image size.

USAGE:
    # Entire test set (recommended):
    python predict.py --weights outputs/weights/best.pt
                      --folder data/annotated/test/images
                      --output outputs/masks

    # Single image:
    python predict.py --weights outputs/weights/best.pt
                      --image data/annotated/test/images/img001.png

    # With visualization:
    python predict.py --weights outputs/weights/best.pt
                      --folder data/annotated/test/images
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
# SINGLE IMAGE PREDICTION
# ==============================================================================

def predict_single_image(model,
                          image_path: str,
                          conf_threshold: float = 0.25,
                          iou_threshold: float = 0.45) -> np.ndarray:
    """
    Runs YOLO inference on one full SEM image and returns a binary mask.

    YOLO handles everything internally:
        - Resizes input to 640x640 for the model
        - Runs inference
        - Scales output masks back to original image dimensions

    For instance segmentation, YOLO returns one mask per detected dendrite
    instance. We merge all instance masks into a single binary mask to match
    the format produced by Pipeline B (classic CV).

    Args:
        model:          Loaded YOLO model
        image_path:     Path to full SEM image
        conf_threshold: Minimum confidence to keep a detection (0-1)
        iou_threshold:  NMS IoU threshold for removing duplicate detections

    Returns:
        Binary mask as uint8 numpy array (0 or 255), same size as input image
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_h, img_w = image.shape[:2]

    # Run inference — YOLO resizes internally, output is at original resolution
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )

    result = results[0]

    # Start with empty mask
    binary_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # No detections → return empty mask
    if result.masks is None or len(result.masks) == 0:
        return binary_mask

    # Merge all instance masks into one binary mask
    # Each instance mask is a float [0,1] array at original image resolution
    masks_data = result.masks.data.cpu().numpy()   # shape: (N, H, W)

    for inst_mask in masks_data:
        # Resize to original image size if YOLO returned different dimensions
        if inst_mask.shape != (img_h, img_w):
            inst_mask = cv2.resize(
                inst_mask,
                (img_w, img_h),
                interpolation=cv2.INTER_LINEAR
            )
        # Threshold and merge into combined mask
        binary_mask[inst_mask > 0.5] = 255

    return binary_mask


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_prediction(image_path: str,
                          mask: np.ndarray,
                          save_path: str = None) -> None:
    """
    Saves a 3-panel figure: original | binary mask | green overlay.

    Args:
        image_path: Path to original SEM image
        mask:       Binary mask from predict_single_image()
        save_path:  If provided, saves the figure here
    """
    import matplotlib.pyplot as plt

    original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # Green overlay
    rgb     = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    overlay = rgb.copy()
    overlay[mask > 0] = [0, 200, 0]
    blended = cv2.addWeighted(rgb, 0.6, overlay, 0.4, 0)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original SEM Image')
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('YOLO Mask')
    axes[2].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Overlay (green = dendrite)')

    for ax in axes:
        ax.axis('off')

    plt.suptitle(f"YOLO Prediction — {Path(image_path).stem}", fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")

    plt.show()
    plt.close()


# ==============================================================================
# BATCH PREDICTION
# ==============================================================================

def predict_folder(model,
                   folder_path: str,
                   output_dir: str,
                   conf_threshold: float = 0.25,
                   iou_threshold: float = 0.45,
                   visualize: bool = False) -> dict:
    """
    Runs prediction on all images in a folder.

    Args:
        model:          Loaded YOLO model
        folder_path:    Directory containing full SEM images
        output_dir:     Where to save binary masks
        conf_threshold: Minimum confidence threshold
        iou_threshold:  NMS IoU threshold
        visualize:      If True, saves overlay visualizations too

    Returns:
        Dictionary mapping image filename → saved mask path
    """
    folder = Path(folder_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if visualize:
        viz_dir = Path('outputs/visuals/yolo_predictions')
        viz_dir.mkdir(parents=True, exist_ok=True)

    extensions  = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    image_paths = sorted([p for p in folder.iterdir()
                          if p.suffix.lower() in extensions])

    if not image_paths:
        print(f"  WARNING: No images found in {folder_path}")
        return {}

    print(f"\nRunning prediction on {len(image_paths)} images...")
    print(f"  Confidence threshold : {conf_threshold}")
    print(f"  Output directory     : {output}\n")

    results_map = {}

    for i, img_path in enumerate(image_paths, 1):
        print(f"  [{i:2d}/{len(image_paths)}] {img_path.name}...", end=' ')

        try:
            mask = predict_single_image(
                model, img_path, conf_threshold, iou_threshold
            )

            mask_path = output / f"{img_path.stem}_yolo_mask.png"
            cv2.imwrite(str(mask_path), mask)

            coverage = (mask > 0).mean() * 100
            print(f"done  ({coverage:.1f}% dendrite coverage)")

            results_map[img_path.name] = str(mask_path)

            if visualize:
                viz_path = viz_dir / f"{img_path.stem}_overlay.png"
                visualize_prediction(str(img_path), mask, str(viz_path))

        except Exception as e:
            print(f"FAILED — {e}")
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
    print("PIPELINE A — YOLO26 INFERENCE")
    print("=" * 55)
    print(f"  Weights    : {args.weights}")
    print(f"  Confidence : {args.conf}")

    weights_path = Path(args.weights)
    assert weights_path.exists(), \
        f"Weights not found: {args.weights}\n" \
        f"Did you run pipeline_A_yolo/train.py first?"

    print(f"\nLoading model...")
    model = YOLO(str(weights_path))
    print("  ✓ Model loaded\n")

    if args.image:
        mask = predict_single_image(model, args.image, args.conf, args.iou)

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        mask_path  = output_dir / f"{Path(args.image).stem}_yolo_mask.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"Mask saved: {mask_path}")

        if args.visualize:
            visualize_prediction(
                args.image, mask,
                str(output_dir / f"{Path(args.image).stem}_overlay.png")
            )

    elif args.folder:
        predict_folder(
            model=model,
            folder_path=args.folder,
            output_dir=args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            visualize=args.visualize
        )

    else:
        print("ERROR: Provide either --image or --folder")
        print("Examples:")
        print("  python predict.py --weights outputs/weights/best.pt "
              "--folder data/annotated/test/images --visualize")
        print("  python predict.py --weights outputs/weights/best.pt "
              "--image data/annotated/test/images/img001.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='YOLO26-seg inference on full SEM test images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--weights', type=str,
                        default='outputs/weights/best.pt',
                        help='Path to trained weights from pipeline_A_yolo/train.py')
    parser.add_argument('--image',   type=str, default=None,
                        help='Single image path')
    parser.add_argument('--folder',  type=str, default=None,
                        help='Folder of test images')
    parser.add_argument('--output',  type=str, default='outputs/masks',
                        help='Where to save output masks')
    parser.add_argument('--conf',    type=float, default=0.25,
                        help='Confidence threshold (0-1)')
    parser.add_argument('--iou',     type=float, default=0.45,
                        help='NMS IoU threshold (0-1)')
    parser.add_argument('--visualize', action='store_true',
                        help='Save overlay visualizations')

    main(parser.parse_args())
