"""
segmentation.py - Pipeline B: Classic Computer Vision
======================================================
PURPOSE OF THIS FILE:
    This is the core decision-making step of the entire pipeline.
    We take the clean, denoised grayscale image from preprocessing.py
    and convert it into a BINARY MASK — a black and white image where:
        WHITE (255) = dendrite pixel
        BLACK (0)   = background pixel

    Every pixel gets a label. No gray areas. This is called binarization.

WHERE WE ARE IN THE PIPELINE:
    preprocessing.py  →  [segmentation.py]  →  postprocessing.py
    (clean grayscale)     (binary mask)         (refined mask)

THE CORE CHALLENGE:
    We need to find a "threshold" brightness value T such that:
        pixel > T  →  dendrite (white)
        pixel ≤ T  →  background (black)

    The problem: SEM images have UNEVEN lighting across the image.
    The electrode base at the bottom is very bright, the background at
    the top is very dark, and dendrites sit somewhere in between.

    A single global threshold T will always fail:
        - If T is low  → dark background regions get classified as dendrite
        - If T is high → thin dendrite tips disappear entirely

    Solution: use a threshold that ADAPTS to local brightness at each pixel.

STEPS IN THIS FILE:
    1. Adaptive Thresholding  → initial binary mask (fast, local)
    2. Otsu's Method          → alternative for comparison/reference
    3. Combined strategy      → pick the best approach per image quality
"""

import cv2
import numpy as np
from pathlib import Path


# ==============================================================================
# STEP 1: ADAPTIVE THRESHOLDING (PRIMARY METHOD)
# ==============================================================================

def adaptive_threshold(image: np.ndarray,
                        block_size: int = 35,
                        C: float = 4) -> np.ndarray:
    """
    NOTE: delete this method if not used later, right now otsu is better and is the default.
    Applies Adaptive Thresholding to produce a binary mask.

    WHY ADAPTIVE AND NOT GLOBAL?
        Imagine the SEM image split into regions:
            - Top region:    very dark background (low brightness)
            - Middle region: dendrites growing upward (medium brightness)
            - Bottom region: bright electrode base (high brightness)

        A global threshold T picks ONE value for the whole image.
        If T=127 (midpoint), it correctly finds dendrites in the middle,
        but the bright electrode base gets entirely classified as dendrite,
        and thin dark-region branches disappear.

        Adaptive thresholding computes a DIFFERENT threshold for every pixel,
        based on the average brightness of its local neighborhood.

        Think of it as: "is this pixel brighter than its surroundings?"
        rather than: "is this pixel brighter than some fixed value?"

    HOW IT WORKS (step by step):
        For each pixel P at position (x, y):
            1. Take a square neighborhood of size block_size × block_size
               centered on P
            2. Compute the mean brightness of that neighborhood
            3. Set threshold T_local = mean - C
            4. If P > T_local → white (dendrite)
               If P ≤ T_local → black (background)

    THE block_size PARAMETER:
        Controls how large the local neighborhood is.

        - Too small (e.g., 5-11):  threshold reacts to individual noise pixels.
          Every tiny bright speck becomes "dendrite." Mask is extremely noisy.

        - Too large (e.g., 100+):  the neighborhood is so big it approaches
          global thresholding. Loses the adaptive advantage in uneven regions.

        - Sweet spot (25-51):      large enough to average over real structure,
          small enough to adapt to local lighting changes.

        RULE: block_size MUST be an odd number (OpenCV requirement).

    THE C PARAMETER (constant subtraction):
        After computing the local mean, we subtract C before comparing.
        This acts as a fine-tuning offset:

        - C = 0:  threshold equals local mean exactly. Very sensitive —
                  half the pixels in any region will become white.

        - C > 0:  pixel must be BRIGHTER than the local mean by C units
                  to be classified as dendrite. Reduces false positives
                  in noisy flat regions.

        - C < 0:  even pixels DARKER than their neighborhood get classified
                  as dendrite. Rarely useful for this application.

        Typical range for SEM images: C between 2 and 10.
        Start with 4 and adjust based on visual inspection.

    Args:
        image:      2D grayscale array from preprocessing (denoised output)
        block_size: Size of local neighborhood (must be odd, default 35)
        C:          Constant subtracted from local mean (default 4)

    Returns:
        Binary 2D array: 255 = dendrite, 0 = background
    """
    # Safety check: block_size must be odd
    if block_size % 2 == 0:
        block_size += 1

    binary_mask = cv2.adaptiveThreshold(
        image,
        maxValue=255,                           # value assigned to dendrite pixels
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,  # use mean of neighborhood
        thresholdType=cv2.THRESH_BINARY,        # bright pixels → white
        blockSize=block_size,
        C=C
    )

    return binary_mask


# ==============================================================================
# STEP 2: OTSU'S METHOD (REFERENCE / FALLBACK)
# ==============================================================================

def otsu_threshold(image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Applies Otsu's automatic global thresholding.

    WHAT IS OTSU'S METHOD?
        Otsu's algorithm automatically finds the single best global threshold
        by looking at the image histogram and finding the value T that
        MAXIMIZES the separation between two classes (background vs dendrite).

        Mathematically, it minimizes the weighted sum of within-class variances:
            - Class 1: all pixels with value ≤ T (background)
            - Class 2: all pixels with value > T  (dendrite)

        It finds T such that pixels within each class are as similar as
        possible to each other, and as different as possible from the other class.

    WHEN TO USE OTSU vs ADAPTIVE:
        Your project document says Otsu is only suitable when lighting is
        "completely uniform (without shadows)."

        USE OTSU when:
            - The image has very even, flat lighting
            - There's a clear bimodal histogram (two visible peaks)
            - You want a quick sanity check or baseline comparison

        USE ADAPTIVE when:
            - Typical SEM images with uneven lighting (most cases)
            - There's a bright electrode base + dark upper background
            - Thin dendrite tips exist in shadowed regions

        In practice: run BOTH and compare visually. Otsu gives you a useful
        baseline to see how much improvement adaptive thresholding provides.

    Args:
        image: 2D grayscale array from preprocessing

    Returns:
        Tuple of:
            - binary_mask: 2D array, 255 = dendrite, 0 = background
            - threshold_value: the T value Otsu automatically selected
              (useful to log for your report)
    """
    threshold_value, binary_mask = cv2.threshold(
        image,
        thresh=0,               # ignored when using OTSU flag
        maxval=255,
        type=cv2.THRESH_BINARY + cv2.THRESH_OTSU   # OTSU finds T automatically
    )

    return binary_mask, threshold_value


# ==============================================================================
# MASTER FUNCTION
# ==============================================================================

def segment(preprocessed_image: np.ndarray,
            method: str = 'otsu',
            block_size: int = 35,
            C: float = 4,
            electrode_fraction: float = 0.35) -> dict:
    """
    Runs the full segmentation step on a preprocessed SEM image.

    Args:
        preprocessed_image:  The 'denoised' output from preprocessing.preprocess()
        method:              'adaptive' (recommended) or 'otsu' (for comparison)
        block_size:          Adaptive threshold neighborhood size (odd number)
        C:                   Adaptive threshold constant offset
        electrode_fraction:  Fraction of bottom image to exclude (electrode base)

    Returns:
        Dictionary with keys:
            'adaptive_raw'   → adaptive threshold result before electrode masking
            'otsu_raw'       → otsu result before electrode masking
            'otsu_value'     → the T value Otsu selected (log this in your report)
            'mask'           → FINAL clean mask ready for postprocessing.py
            'method_used'    → which method was used for the final mask
    """
    # Always compute both for comparison purposes
    adaptive_raw = adaptive_threshold(preprocessed_image, block_size, C)
    otsu_raw, otsu_value = otsu_threshold(preprocessed_image)

    # Select which method drives the final mask
    if method == 'adaptive':
        raw_mask = adaptive_raw
    elif method == 'otsu':
        raw_mask = otsu_raw
    else:
        raise ValueError(f"method must be 'adaptive' or 'otsu', got: '{method}'")

    return {
        'adaptive_raw': adaptive_raw,
        'otsu_raw':     otsu_raw,
        'otsu_value':   otsu_value,
        'mask':         raw_mask,     # <-- this is what postprocessing.py receives
        'method_used':  method
    }


# ==============================================================================
# VISUALIZATION HELPER
# ==============================================================================

def visualize_segmentation(preprocessed_image: np.ndarray,
                            seg_results: dict,
                            save_path = None) -> None:
    """
    Plots the segmentation results for inspection.

    Shows: preprocessed input | adaptive result | otsu result | final mask

    Args:
        preprocessed_image: The denoised image that was fed into segment()
        seg_results:        Dictionary returned by segment()
        save_path:          If provided, saves the figure to this path
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    images = [
        (preprocessed_image,            'Preprocessed Input'),
        (seg_results['adaptive_raw'],   'Adaptive Threshold'),
        (seg_results['otsu_raw'],       f"Otsu (T={seg_results['otsu_value']:.1f})"),
        (seg_results['mask'],           f"Final Mask ({seg_results['method_used']})"),
    ]

    for ax, (img, title) in zip(axes, images):
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=13)
        ax.axis('off')

    plt.suptitle('Segmentation Pipeline Steps', fontsize=15, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to {save_path}")

    plt.show()


def save_mask_overlay(preprocessed_image: np.ndarray,
                      mask: np.ndarray,
                      save_path: str,
                      alpha: float = 0.4) -> None:
    """
    Saves a standalone overlay image (no multi-panel figure).

    Green pixels indicate predicted dendrite mask over the preprocessed input.
    """
    gray = preprocessed_image
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay = rgb.copy()
    overlay[mask > 0] = [0, 200, 0]
    blended = cv2.addWeighted(rgb, 1.0 - alpha, overlay, alpha, 0)

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    print(f"Saved overlay to {out_path}")


# ==============================================================================
# PARAMETER TUNING HELPER
# ==============================================================================

def tune_adaptive_parameters(image: np.ndarray) -> None:
    """
    Interactive grid search over block_size and C values.
    Run this in your notebook to find the best parameters for your dataset.

    Shows a grid of results so you can visually pick the best combination.

    Args:
        image: Preprocessed (denoised) SEM image
    """
    import matplotlib.pyplot as plt

    block_sizes = [15, 25, 35, 51]
    C_values    = [2, 4, 8, 12]

    fig, axes = plt.subplots(
        len(C_values), len(block_sizes),
        figsize=(4 * len(block_sizes), 4 * len(C_values))
    )

    for i, C in enumerate(C_values):
        for j, bs in enumerate(block_sizes):
            mask = adaptive_threshold(image, block_size=bs, C=C)
            axes[i][j].imshow(mask, cmap='gray')
            axes[i][j].set_title(f'block={bs}, C={C}', fontsize=9)
            axes[i][j].axis('off')

    plt.suptitle('Adaptive Threshold Parameter Grid\n(pick the clearest dendrite separation)',
                 fontsize=13)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# QUICK TEST
# ==============================================================================

if __name__ == "__main__":
    import sys
    from preprocessing import preprocess

    if len(sys.argv) < 2:
        print("Usage: python segmentation.py <path_to_image>")
        print("Example: python segmentation.py data/raw/sample_01.png")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Processing: {image_path}")

    # Run preprocessing first
    prep_results = preprocess(image_path)
    denoised = prep_results['denoised']

    # Run segmentation
    seg_results = segment(denoised)

    print(f"  Otsu auto-selected threshold: {seg_results['otsu_value']:.1f}")
    print(f"  Method used: {seg_results['method_used']}")
    print(f"  Dendrite pixels in final mask: {np.sum(seg_results['mask'] == 255)}")
    print(f"  Background pixels:             {np.sum(seg_results['mask'] == 0)}")

    visualize_segmentation(
        denoised,
        seg_results,
        save_path="outputs/visuals/segmentation_check.png"
    )
    save_mask_overlay(
        preprocessed_image=denoised,
        mask=seg_results['mask'],
        save_path="outputs/visuals/segmentation_overlay.png"
    )

    # Uncomment to explore parameter tuning:
    # tune_adaptive_parameters(denoised)
