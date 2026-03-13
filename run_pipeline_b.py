"""
run_pipeline_b.py - Pipeline B comparison runner
================================================
Runs the classic pipeline on dataset test images and saves comparison artifacts:
  1) Cropped original image (metadata bar removed)
  2) Postprocessed-mask overlay on original
  3) Skeleton (tips/forks) overlay on original

USAGE:
  .venv/bin/python run_pipeline_b.py
"""

from pathlib import Path
import cv2

from pipeline_B_classic.preprocessing import preprocess
from pipeline_B_classic.segmentation import segment, save_mask_overlay
from pipeline_B_classic.postprocessing import postprocess
from pipeline_B_classic.skeletonization import skeletonize_mask, save_skeleton_overlay


# ==============================================================================
# CONFIG
# ==============================================================================

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'data' / 'annotated' / 'test' / 'images'  # dataset images, not raw
OUT_ROOT = ROOT / 'comparison_results' / 'classic'

ORIG_DIR = OUT_ROOT / 'original1'
MASK_OVERLAY_DIR = OUT_ROOT / 'mask_overlay1'
SKEL_OVERLAY_DIR = OUT_ROOT / 'skeleton_overlay1'

for d in [ORIG_DIR, MASK_OVERLAY_DIR, SKEL_OVERLAY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

PARAMS = {
    # preprocessing
    'crop_fraction': 0.0,
    'clahe_clip': 2.0,
    'clahe_tile': 8,
    'bilateral_d': 9,
    'bilateral_sigma': 75,

    # segmentation
    'method': 'otsu',
    'block_size': 11,
    'C': 2,

    # postprocessing
    'min_area': 300,
    'closing_kernel': -1,
    'erosion_size': 5,
    'iters': 2,
    'apply_shape_filter': True,
    'bottom_fraction': 0.16,
    'bottom_min_width_fraction': 0.45,
    'large_area_threshold': 5000,
    'solidity_threshold': 0.85,
    'max_compact_aspect_ratio': 1.8,

    # skeletonization
    'peak_min_distance': 5,
}


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    image_paths = sorted([
        p for p in DATA_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_paths:
        raise FileNotFoundError(f"No dataset test images found at: {DATA_DIR}")

    print(f"Found {len(image_paths)} dataset test images")
    print(f"Saving to: {OUT_ROOT}\n")

    for i, image_path in enumerate(image_paths, 1):
        stem = image_path.stem
        print(f"[{i:02d}/{len(image_paths)}] {image_path.name} ...", end=' ')

        # a) preprocess includes metadata-bar crop; this is our "original"
        pre = preprocess(
            str(image_path),
            crop_fraction=PARAMS['crop_fraction'],
            clahe_clip=PARAMS['clahe_clip'],
            clahe_tile=PARAMS['clahe_tile'],
            bilateral_d=PARAMS['bilateral_d'],
            bilateral_sigma=PARAMS['bilateral_sigma'],
        )
        original = pre['raw']
        cv2.imwrite(str(ORIG_DIR / f"{stem}_original.png"), original)

        # b) segmentation + postprocessing, then overlay postprocessed mask on original
        seg = segment(
            pre['denoised'],
            method=PARAMS['method'],
            block_size=PARAMS['block_size'],
            C=PARAMS['C'],
        )
        post = postprocess(
            seg['mask'],
            min_area=PARAMS['min_area'],
            closing_kernel=PARAMS['closing_kernel'],
            erosion_size=PARAMS['erosion_size'],
            iters=PARAMS['iters'],
            apply_shape_filter=PARAMS['apply_shape_filter'],
            bottom_fraction=PARAMS['bottom_fraction'],
            bottom_min_width_fraction=PARAMS['bottom_min_width_fraction'],
            large_area_threshold=PARAMS['large_area_threshold'],
            solidity_threshold=PARAMS['solidity_threshold'],
            max_compact_aspect_ratio=PARAMS['max_compact_aspect_ratio'],
        )
        save_mask_overlay(
            preprocessed_image=original,
            mask=post['filtered'],
            save_path=str(MASK_OVERLAY_DIR / f"{stem}_mask_overlay.png"),
            alpha=0.4,
        )

        # c) skeletonize and overlay thicker skeleton on original
        skel = skeletonize_mask(
            post['filtered'],
            peak_min_distance=PARAMS['peak_min_distance'],
        )
        save_skeleton_overlay(
            original_image=original,
            results=skel,
            save_path=str(SKEL_OVERLAY_DIR / f"{stem}_skeleton_overlay.png"),
            alpha=0.5,
            thickness=3,
        )

        print("OK")

    print("\nDone.")
    print(f"  Original (cropped): {ORIG_DIR}")
    print(f"  Mask overlays:      {MASK_OVERLAY_DIR}")
    print(f"  Skeleton overlays:  {SKEL_OVERLAY_DIR}")


if __name__ == '__main__':
    main()
