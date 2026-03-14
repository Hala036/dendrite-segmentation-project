"""
evaluation/metrics.py
Computes IoU, Dice, Precision, and Recall for both pipelines
against Roboflow YOLO-format ground truth annotations.

Usage:
    python evaluation/metrics.py

Expected directory structure:
    data/annotated/test/images/   ← test images (.jpg or .png)
    data/annotated/test/labels/   ← YOLO seg .txt files (ground truth)
    outputs/masks/                ← binary masks from Pipeline B
    outputs/masks_tiled/                ← binary masks from Pipeline A (tiled)
"""

import numpy as np
import cv2
import csv
from pathlib import Path


# ─── FILENAME MAPPING ─────────────────────────────────────────────────────────
# Maps test image stem → (classic_mask_filename, yolo_mask_filename)
MASK_MAP = {
    '2e-9_100s_019_png_jpg.rf.b4efef9d266371373c001e722b3a6cdb': (
        '019_mask.png',
        '2e-9_100s_019_png_jpg.rf.b4efef9d266371373c001e722b3a6cdb_yolo_tiled_mask.png'
    ),
    '2e-9_100s_020_png_jpg.rf.28e6c8afce1ee097aed6472ab4c5fe32': (
        '020_mask.png',
        '2e-9_100s_020_png_jpg.rf.28e6c8afce1ee097aed6472ab4c5fe32_yolo_tiled_mask.png'
    ),
    '70nm_diameter_100nm_pitch_018_png_jpg.rf.14ff5a098b7670efd72a3c414831e6ae': (
        '70nm_diameter_018_mask.png',
        '70nm_diameter_100nm_pitch_018_png_jpg.rf.14ff5a098b7670efd72a3c414831e6ae_yolo_tiled_mask.png'
    ),
    '70nm_diameter_100nm_pitch_020_png_jpg.rf.c0cd080921e33287c758b1ea9d8d9368': (
        '70nm_diameter_020_mask.png',
        '70nm_diameter_100nm_pitch_020_png_jpg.rf.c0cd080921e33287c758b1ea9d8d9368_yolo_tiled_mask.png'
    ),
    '70nm_diameter_100nm_pitch_021_png_jpg.rf.8b0cda18515211137f806f7eae5d6361': (
        '70nm_diameter_021_mask.png',
        '70nm_diameter_100nm_pitch_021_png_jpg.rf.8b0cda18515211137f806f7eae5d6361_yolo_tiled_mask.png'
    ),
}


# ─── CONFIG ───────────────────────────────────────────────────────────────────
TEST_IMAGES_DIR  = Path('data/annotated/test/images')
TEST_LABELS_DIR  = Path('data/annotated/test/labels')
CLASSIC_MASK_DIR = Path('outputs/masks')
YOLO_MASK_DIR    = Path('outputs/masks_tiled')
OUTPUT_CSV       = Path('outputs/metrics_results.csv')


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def polygon_to_mask(label_path: Path, img_h: int, img_w: int) -> np.ndarray:
    """
    Convert a YOLO segmentation label file to a binary mask.
    YOLO seg format per line: class x1 y1 x2 y2 ... (all normalized 0-1)
    Multiple lines = multiple instances, all merged into one binary mask.
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not label_path.exists():
        print(f"  WARNING: label file not found: {label_path}")
        return mask

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # skip malformed lines
            coords = list(map(float, parts[1:]))  # drop class id
            if len(coords) % 2 != 0:
                coords = coords[:-1]  # drop trailing value if odd
            points = np.array([
                [int(coords[i] * img_w), int(coords[i + 1] * img_h)]
                for i in range(0, len(coords), 2)
            ], dtype=np.int32)
            if len(points) >= 3:  # need at least a triangle
                cv2.fillPoly(mask, [points], 255)
    return mask


def compute_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """
    Compute pixel-level IoU, Dice, Precision, Recall.
    Both inputs are grayscale numpy arrays — binarized at threshold 127.
    Returns a dict with all four metrics rounded to 4 decimal places.
    """
    pred = pred_mask > 127
    gt   = gt_mask   > 127

    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()

    iou       = intersection / union        if union > 0             else 0.0
    dice      = (2 * intersection) / (pred.sum() + gt.sum()) \
                                            if (pred.sum() + gt.sum()) > 0 else 0.0
    precision = intersection / pred.sum()   if pred.sum() > 0        else 0.0
    recall    = intersection / gt.sum()     if gt.sum() > 0          else 0.0

    return {
        'iou':       round(float(iou),       4),
        'dice':      round(float(dice),      4),
        'precision': round(float(precision), 4),
        'recall':    round(float(recall),    4),
    }


def load_mask(path: Path) -> np.ndarray | None:
    """Load a binary mask PNG. Returns None with a warning if file missing."""
    if not path.exists():
        print(f"  WARNING: mask not found: {path}")
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  WARNING: could not read mask: {path}")
    return mask


def resolve_mask_path(image_stem: str, pipeline_name: str) -> Path | None:
    """
    Resolve the predicted mask filename for one test image and one pipeline.

    The test images use long Roboflow names, but the saved classic masks use
    shorter custom names like `019_mask.png`. YOLO tiled masks keep the long
    image stem and add `_yolo_tiled_mask.png`.
    """
    mapped = MASK_MAP.get(image_stem)

    if pipeline_name == 'Classic':
        mask_dir = CLASSIC_MASK_DIR
        if mapped:
            return mask_dir / mapped[0]

        # Fallback for cases where the classic mask follows the image stem.
        return mask_dir / f'{image_stem}.png'

    if pipeline_name == 'YOLO':
        mask_dir = YOLO_MASK_DIR
        if mapped:
            return mask_dir / mapped[1]

        # Fallback for direct output naming.
        tiled_name = mask_dir / f'{image_stem}_yolo_tiled_mask.png'
        if tiled_name.exists():
            return tiled_name
        return mask_dir / f'{image_stem}.png'

    raise ValueError(f"Unknown pipeline: {pipeline_name}")



# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # Collect test images (.jpg and .png)
    test_images = sorted([
        p for ext in ('*.jpg', '*.jpeg', '*.png')
        for p in TEST_IMAGES_DIR.glob(ext)
    ])

    if not test_images:
        print(f"No test images found in {TEST_IMAGES_DIR}")
        return

    results = []

    print(f"\n{'='*80}")
    print(f"{'IMAGE':<55} {'PIPELINE':<10} {'IoU':>6} {'Dice':>6} {'Prec':>6} {'Rec':>6}")
    print(f"{'='*80}")

    for img_path in test_images:
        if img_path.stem not in MASK_MAP:
            print(f"No mask mapping for {img_path.name}, skipping")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        h, w = img.shape[:2]

        label_path = TEST_LABELS_DIR / img_path.with_suffix('.txt').name
        gt_mask = polygon_to_mask(label_path, h, w)

        short_name = img_path.stem[:52]

        for pipeline_name in ('Classic', 'YOLO'):
            mask_path = resolve_mask_path(img_path.stem, pipeline_name)
            pred_mask = load_mask(mask_path)

            if pred_mask is None:
                print(f"  Skipping {pipeline_name} for {img_path.name}")
                continue

            if pred_mask.shape != (h, w):
                pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            m = compute_metrics(pred_mask, gt_mask)

            print(f"{short_name:<55} {pipeline_name:<10} "
                  f"{m['iou']:>6.4f} {m['dice']:>6.4f} "
                  f"{m['precision']:>6.4f} {m['recall']:>6.4f}")

            results.append({
                'image':     img_path.name,
                'pipeline':  pipeline_name,
                **m
            })

    if not results:
        print("No results computed.")
        return

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*80}")
    for pipeline_name in ('Classic', 'YOLO'):
        subset = [r for r in results if r['pipeline'] == pipeline_name]
        if not subset:
            continue
        mean_iou  = np.mean([r['iou']  for r in subset])
        mean_dice = np.mean([r['dice'] for r in subset])
        mean_prec = np.mean([r['precision'] for r in subset])
        mean_rec  = np.mean([r['recall']    for r in subset])
        print(f"{'MEAN — ' + pipeline_name:<55} {'':>10} "
              f"{mean_iou:>6.4f} {mean_dice:>6.4f} "
              f"{mean_prec:>6.4f} {mean_rec:>6.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'pipeline', 'iou', 'dice', 'precision', 'recall'])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {OUTPUT_CSV}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
