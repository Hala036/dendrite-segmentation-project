"""
run_pipeline_a.py - Pipeline A: Main Runner
============================================
PURPOSE OF THIS FILE:
    Runs preprocessing and segmentation in sequence on all images in data/raw/.
    Saves all outputs to the outputs/ directory.

USAGE:
    # Run on all images in data/raw/
    python run_pipeline_a.py

    # Run on a single image
    python run_pipeline_a.py --image data/raw/sample_01.png

    # Run with custom parameters
    python run_pipeline_a.py --block_size 51 --C 6
"""

import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from preprocessing import preprocess, visualize_steps
from segmentation import segment, visualize_segmentation

# ==============================================================================
# CONFIGURATION — change these defaults if needed
# ==============================================================================

DEFAULT_INPUT_DIR  = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("outputs")
SUPPORTED_FORMATS  = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ==============================================================================
# SINGLE IMAGE RUNNER
# ==============================================================================

def run_single(image_path: Path,
               block_size: int,
               C: float,
               electrode_fraction: float,
               save_visuals: bool = True) -> dict:
    """
    Runs the full pipeline (so far) on a single image.

    Args:
        image_path:          Path to raw SEM image
        block_size:          Adaptive threshold block size
        C:                   Adaptive threshold constant
        electrode_fraction:  Bottom fraction to exclude (electrode base)
        save_visuals:        Whether to save step-by-step visual outputs

    Returns:
        Dictionary with all intermediate and final results
    """
    print(f"\n  Processing: {image_path.name}")

    # ── Step 1: Preprocessing ──────────────────────────────────────────────
    print("    [1/2] Preprocessing...")
    prep = preprocess(str(image_path))
    denoised = prep['denoised']

    # ── Step 2: Segmentation ───────────────────────────────────────────────
    print("    [2/2] Segmentation...")
    seg = segment(
        denoised,
        method='otsu',
        block_size=block_size,
        C=C,
        electrode_fraction=electrode_fraction
    )

    # ── Save outputs ───────────────────────────────────────────────────────
    stem = image_path.stem  # filename without extension

    # Save final binary mask
    mask_path = DEFAULT_OUTPUT_DIR / "masks" / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), seg['mask'])

    # Save step-by-step visuals for report
    if save_visuals:
        prep_visual_path = DEFAULT_OUTPUT_DIR / "visuals" / f"{stem}_preprocessing.png"
        seg_visual_path  = DEFAULT_OUTPUT_DIR / "visuals" / f"{stem}_segmentation.png"

        visualize_steps(prep, save_path=str(prep_visual_path))
        visualize_segmentation(denoised, seg, save_path=str(seg_visual_path))

    print(f"    ✓ Mask saved  → {mask_path}")

    return {
        'image_path': image_path,
        'prep':       prep,
        'seg':        seg
    }


# ==============================================================================
# BATCH RUNNER
# ==============================================================================

def run_batch(input_dir: Path,
              block_size: int,
              C: float,
              electrode_fraction: float) -> None:
    """
    Runs the pipeline on every image in input_dir.
    Only saves visuals for the first 5 images to avoid cluttering outputs/.

    Args:
        input_dir:           Directory containing raw SEM images
        block_size:          Adaptive threshold block size
        C:                   Adaptive threshold constant
        electrode_fraction:  Bottom fraction to exclude
    """
    # Collect all supported image files
    image_paths = [
        p for p in sorted(input_dir.iterdir())
        if p.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not image_paths:
        print(f"No images found in {input_dir}")
        print(f"Supported formats: {SUPPORTED_FORMATS}")
        return

    print(f"Found {len(image_paths)} images in {input_dir}")
    print(f"Parameters: block_size={block_size}, C={C}, electrode_fraction={electrode_fraction}")

    for i, image_path in enumerate(tqdm(image_paths, desc="Pipeline A")):
        # Only save step visuals for first 5 images
        save_visuals = (i < 5)
        run_single(image_path, block_size, C, electrode_fraction, save_visuals)

    print(f"\n✓ Done. Processed {len(image_paths)} images.")
    print(f"  Masks   → {DEFAULT_OUTPUT_DIR / 'masks'}/")
    print(f"  Visuals → {DEFAULT_OUTPUT_DIR / 'visuals'}/")


# ==============================================================================
# SETUP — create output directories if they don't exist
# ==============================================================================

def setup_output_dirs() -> None:
    for subdir in ["masks", "skeletons", "visuals"]:
        (DEFAULT_OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline A: Preprocessing + Segmentation for SEM dendrite images"
    )

    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a single image. If omitted, processes all images in data/raw/"
    )
    parser.add_argument(
        "--block_size", type=int, default=35,
        help="Adaptive threshold block size (must be odd, default: 35)"
    )
    parser.add_argument(
        "--C", type=float, default=4,
        help="Adaptive threshold constant offset (default: 4)"
    )
    parser.add_argument(
        "--electrode_fraction", type=float, default=0.35,
        help="Fraction of bottom image height to exclude as electrode base (default: 0.35)"
    )
    parser.add_argument(
        "--no_visuals", action="store_true",
        help="Skip saving visual outputs (faster for large datasets)"
    )

    return parser.parse_args()


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    args = parse_args()
    setup_output_dirs()

    if args.image:
        # Single image mode
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: image not found at {args.image}")
        else:
            run_single(
                image_path,
                block_size=args.block_size,
                C=args.C,
                electrode_fraction=args.electrode_fraction,
                save_visuals=not args.no_visuals
            )
    else:
        # Batch mode
        if not DEFAULT_INPUT_DIR.exists():
            print(f"Error: input directory not found at {DEFAULT_INPUT_DIR}")
            print("Create the directory and place your SEM images inside it.")
        else:
            run_batch(
                DEFAULT_INPUT_DIR,
                block_size=args.block_size,
                C=args.C,
                electrode_fraction=args.electrode_fraction
            )