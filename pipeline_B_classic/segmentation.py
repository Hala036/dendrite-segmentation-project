"""
Convert the preprocessed grayscale image into a binary dendrite mask.

This file tries thresholding methods and returns the mask that will be used by
the next pipeline step.
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
    Apply adaptive thresholding to get a binary mask.

    Args:
        image: Preprocessed grayscale image.
        block_size: Local window size. Must be odd.
        C: Small constant subtracted from the local mean.

    Returns:
        Binary mask with 255 for dendrites and 0 for background.
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
    Apply Otsu thresholding and return the chosen threshold value.

    Args:
        image: Preprocessed grayscale image.

    Returns:
        Binary mask and the threshold value picked by Otsu.
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
    Run segmentation on a preprocessed image.

    Args:
        preprocessed_image: Output image from preprocessing.
        method: `'adaptive'` or `'otsu'`.
        block_size: Window size for adaptive thresholding.
        C: Constant used in adaptive thresholding.
        electrode_fraction: Unused parameter kept for compatibility.

    Returns:
        Dictionary with both raw threshold masks and the final chosen mask.
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
    Show the segmentation results side by side.

    Args:
        preprocessed_image: Image passed into `segment()`.
        seg_results: Output from `segment()`.
        save_path: Optional path to save the figure.
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
    Save a simple mask overlay image.
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
    Show a grid of adaptive threshold settings for quick comparison.

    Args:
        image: Preprocessed image.
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
