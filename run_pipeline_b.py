"""
run_pipeline_b.py - Pipeline B: Classic Computer Vision
========================================================
PURPOSE OF THIS FILE:
    Runs the full classic CV pipeline on every image in data/raw/
    and saves all outputs to the appropriate output directories.

    This is the file you run ONCE after all parameters are tuned
    in the notebook. It processes the entire dataset automatically
    without needing to know any image names in advance.

USAGE:
    python run_pipeline_b.py

OUTPUTS:
    outputs/masks/       → binary masks     (<stem>_mask.png)
    outputs/skeletons/   → skeleton images  (<stem>_skeleton.png)
    outputs/visuals/     → full 5-panel pipeline figures (<stem>_pipeline.png)

REQUIREMENTS:
    Run from the project root directory (dendrite-segmentation/)
    All images in data/raw/ must be readable by OpenCV (.png, .tif, .jpg, .bmp)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import traceback

# Add project root to path so imports work regardless of where you run from
sys.path.append(str(Path(__file__).parent))

from pipeline_B_classic.preprocessing  import preprocess
from pipeline_B_classic.segmentation   import segment
from pipeline_B_classic.postprocessing import postprocess
from pipeline_B_classic.skeletonization import skeletonize_mask


# ==============================================================================
# PARAMETERS — copy final values from notebook Section 7
# ==============================================================================

PARAMS = {
    # preprocessing
    'clahe_clip':           2.0,
    'clahe_tile':           8,
    'bilateral_d':          9,

    # segmentation
    'otsu_offset':          0,
    'block_size':           11,
    'C':                    2,
    'method':               'otsu',

    # postprocessing
    'min_area':             300,
    'closing_kernel':       -1,    # -1 = adaptive
    'erosion_size':         5,
    'iterations':           3,

    # skeletonization
    'peak_min_distance':    5,
}

# Supported image extensions — add more if needed
IMAGE_EXTENSIONS = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp'}


# ==============================================================================
# OUTPUT DIRECTORIES
# ==============================================================================

ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / 'data' / 'raw'
MASK_DIR    = ROOT / 'outputs' / 'masks'
SKEL_DIR    = ROOT / 'outputs' / 'skeletons'
VISUAL_DIR  = ROOT / 'outputs' / 'visuals'

for d in [MASK_DIR, SKEL_DIR, VISUAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# SINGLE IMAGE PROCESSOR
# ==============================================================================

def process_image(image_path: Path, params: dict) -> dict:
    """
    Runs the full pipeline on a single image.

    Args:
        image_path: Path object pointing to the raw SEM image
        params:     Dictionary of all pipeline parameters

    Returns:
        Dictionary with all pipeline outputs and analysis results.
        Returns None if processing failed.
    """
    # ── Preprocessing ─────────────────────────────────────────────────────────
    pre = preprocess(
        str(image_path),
        clahe_clip      = params['clahe_clip'],
        clahe_tile      = params['clahe_tile'],
        bilateral_d     = params['bilateral_d'],
    )

    # ── Segmentation ──────────────────────────────────────────────────────────
    seg = segment(
        pre['denoised'],
        method      = params['method'],
        block_size  = params['block_size'],
        C           = params['C'],
    )

    # ── Postprocessing ────────────────────────────────────────────────────────
    post = postprocess(
        seg['mask'],
        min_area        = params['min_area'],
        closing_kernel  = params['closing_kernel'],
        erosion_size    = params['erosion_size'],
        iters           = params['iterations'],
    )

    # ── Skeletonization ───────────────────────────────────────────────────────
    skel = skeletonize_mask(
        post['reconstructed'],
        peak_min_distance = params['peak_min_distance'],
    )

    return {
        'raw':          cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE),
        'preprocessed': pre['denoised'],
        'mask':         seg['mask'],
        'postprocessed':post['reconstructed'],
        'skeleton':     skel['skeleton'],
        'analysis':     skel['analysis'],
    }


# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================

def save_outputs(stem: str, results: dict) -> None:
    """
    Saves mask, skeleton, and 5-panel visual for one image.

    Args:
        stem:    Image filename without extension (used as output prefix)
        results: Dictionary returned by process_image()
    """
    # Binary mask
    cv2.imwrite(str(MASK_DIR / f'{stem}_mask.png'), results['postprocessed'])

    # Skeleton
    cv2.imwrite(str(SKEL_DIR / f'{stem}_skeleton.png'), results['skeleton'])

    # 5-panel figure: Raw → Pre → Seg → Post → Skeleton overlay
    analysis = results['analysis']

    skeleton_overlay = cv2.cvtColor(results['postprocessed'], cv2.COLOR_GRAY2RGB)
    skeleton_overlay[results['skeleton'] > 0]          = [0,   255, 0  ]  # green
    skeleton_overlay[analysis['tips'] > 0]             = [255, 0,   0  ]  # red
    skeleton_overlay[analysis['forks'] > 0]            = [0,   0,   255]  # blue

    stages = [
        (results['raw'],           'Raw SEM'),
        (results['preprocessed'],  'Preprocessed'),
        (results['mask'],          'Segmentation'),
        (results['postprocessed'], 'Postprocessed'),
        (skeleton_overlay,         f'Skeleton\nTips={analysis["tip_count"]}  Forks={analysis["fork_count"]}'),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(26, 6))
    fig.suptitle(stem, fontsize=13)

    for ax, (img, title) in zip(axes, stages):
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(str(VISUAL_DIR / f'{stem}_pipeline.png'), bbox_inches='tight', dpi=150)
    plt.close()   # close instead of show — avoids popups when running in batch


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    # Collect all image files in data/raw/
    image_paths = sorted([
        p for p in DATA_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_paths:
        print(f'No images found in {DATA_DIR}')
        print(f'Supported extensions: {IMAGE_EXTENSIONS}')
        sys.exit(1)

    print(f'Found {len(image_paths)} images in {DATA_DIR}')
    print(f'Output directories:')
    print(f'  Masks:    {MASK_DIR}')
    print(f'  Skeleton: {SKEL_DIR}')
    print(f'  Visuals:  {VISUAL_DIR}')
    print()

    # Results table — collected for summary at the end
    summary = []
    failed  = []

    for i, image_path in enumerate(image_paths, 1):
        stem = image_path.stem
        print(f'[{i:02d}/{len(image_paths)}] Processing: {image_path.name} ...', end=' ', flush=True)

        try:
            results = process_image(image_path, PARAMS)
            save_outputs(stem, results)

            a = results['analysis']
            summary.append({
                'name':   image_path.name,
                'tips':   a['tip_count'],
                'forks':  a['fork_count'],
                'length': a['total_length'],
            })

            print(f'OK  (tips={a["tip_count"]}, forks={a["fork_count"]}, length={a["total_length"]}px)')

        except Exception as e:
            print(f'FAILED')
            print(f'  Error: {e}')
            traceback.print_exc()
            failed.append(image_path.name)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print(f'DONE — {len(summary)}/{len(image_paths)} images processed successfully')
    print('=' * 60)

    if summary:
        print(f'\n{"Image":<40} {"Tips":>6} {"Forks":>6} {"Length":>8}')
        print('-' * 62)
        for row in summary:
            print(f'{row["name"]:<40} {row["tips"]:>6} {row["forks"]:>6} {row["length"]:>8}px')

        tips_all   = [r['tips']   for r in summary]
        forks_all  = [r['forks']  for r in summary]
        length_all = [r['length'] for r in summary]

        print('-' * 62)
        print(f'{"Average":<40} {sum(tips_all)/len(tips_all):>6.1f} '
              f'{sum(forks_all)/len(forks_all):>6.1f} '
              f'{sum(length_all)/len(length_all):>8.0f}px')

    if failed:
        print(f'\nFailed images ({len(failed)}):')
        for name in failed:
            print(f'  - {name}')


if __name__ == '__main__':
    main()