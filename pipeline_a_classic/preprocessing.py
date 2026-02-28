"""
preprocessing.py - Pipeline A: Classic Computer Vision
=======================================================

STEPS IN ORDER:
    1. Load image as grayscale
    2. Histogram Normalization  → gives all images the same brightness range
    3. CLAHE for local contrast
    4. Bilateral Filter to remove noise while keeping dendrite edges sharp
"""

import cv2
import numpy as np
from pathlib import Path


# ==============================================================================
# STEP 1: IMAGE LOADING
# ==============================================================================

def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from disk and converts it to grayscale.

    Args:
        image_path: Path to the image file (jpg, png, tif, etc.)

    Returns:
        2D numpy array of shape (H, W) with pixel values in range [0, 255]

    Raises:
        FileNotFoundError: if the image path doesn't exist
        ValueError: if the image can't be read by OpenCV
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # cv2.IMREAD_GRAYSCALE forces single-channel loading, which is what we want for SEM images
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"OpenCV could not read the image: {image_path}")

    return image

def crop_metadata_bar(image: np.ndarray, crop_fraction: float = 0.09) -> np.ndarray:
    """
    Removes the SEM metadata bar at the bottom of the image.
    crop_fraction: what fraction of image height to remove (default 9%)
    """
    h = image.shape[0]
    crop_pixels = int(h * crop_fraction)
    return image[:h - crop_pixels, :]


# ==============================================================================
# STEP 2: HISTOGRAM NORMALIZATION
# ==============================================================================

def normalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Stretches pixel values to use the full range [0, 255].
    so that we can set consistent thresholds and filters later.
    new_value = (old_value - min) / (max - min) * 255

    Args:
        image: 2D grayscale numpy array

    Returns:
        Normalized 2D array with values in [0, 255], dtype uint8
    """
    # cv2.normalize with NORM_MINMAX does exactly the linear stretch described above
    normalized = np.empty_like(image, dtype=np.uint8)
    cv2.normalize(
        image,
        normalized,
        alpha=0,            # minimum output value
        beta=255,           # maximum output value
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U     # output : unsigned 8-bit integer (0-255)
    )

    return normalized


# ==============================================================================
# STEP 3: CLAHE - LOCAL CONTRAST ENHANCEMENT
# ==============================================================================

def apply_clahe(image: np.ndarray,
                clip_limit: float = 2.0,
                tile_size: int = 8) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).

    clip limit:
        - Low clip (1.0-2.0): gentle enhancement, safer for noisy images
        - High clip (3.0-4.0): aggressive enhancement, risks noise amplification

    Args:
        image:      2D grayscale numpy array (after normalization)
        clip_limit: Maximum contrast amplification per tile (default 2.0)
        tile_size:  Size of each local tile in pixels (default 8x8)

    Returns:
        Contrast-enhanced 2D array, same shape and dtype as input
    """
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_size, tile_size)
    )

    enhanced = clahe.apply(image)

    return enhanced


# ==============================================================================
# STEP 4: BILATERAL FILTER - EDGE-PRESERVING DENOISING
# ==============================================================================

def apply_bilateral_filter(image: np.ndarray,
                            diameter: int = 9,
                            sigma_color: float = 75,
                            sigma_space: float = 75) -> np.ndarray:
    """
    Applies a Bilateral Filter to remove noise while preserving edges.

        NOTE: may need to tune these in notebook.

    Args:
        image:       2D grayscale array (after CLAHE)
        diameter:    Pixel neighborhood diameter
        sigma_color: Filter sigma in color/intensity space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Denoised 2D array with preserved edges, same shape as input
    """
    denoised = cv2.bilateralFilter(
        image,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )

    return denoised


# ==============================================================================
# MASTER FUNCTION - run all steps in order
# ==============================================================================

def preprocess(image_path: str,
               clahe_clip: float = 2.0,
               clahe_tile: int = 8,
               bilateral_d: int = 9,
               bilateral_sigma: float = 75) -> dict:
    """
    Runs the full preprocessing pipeline on a single SEM image.

    Returns a dictionary with intermediate results.

    Args:
        image_path:       Path to raw SEM image
        clahe_clip:       CLAHE clip limit
        clahe_tile:       CLAHE tile grid size
        bilateral_d:      Bilateral filter diameter
        bilateral_sigma:  Bilateral filter sigma (color + space)

    Returns:
        Dictionary with keys:
            'raw'        → original loaded image
            'normalized' → after histogram normalization
            'clahe'      → after contrast enhancement
            'denoised'   → after bilateral filter
    """
    raw = load_image(image_path)
    raw = crop_metadata_bar(raw)  # Remove SEM metadata bar if present
    normalized = normalize_histogram(raw)
    clahe_result = apply_clahe(normalized, clahe_clip, clahe_tile)
    denoised = apply_bilateral_filter(clahe_result, bilateral_d, bilateral_sigma)

    return {
        'raw': raw,
        'normalized': normalized,
        'clahe': clahe_result,
        'denoised': denoised
    }


# ==============================================================================
# VISUALIZATION HELPER
# ==============================================================================

def visualize_steps(results: dict, save_path = None) -> None:
    """
    Plots all preprocessing steps side by side for inspection.

    Args:
        results:   Dictionary returned by preprocess()
        save_path: saves the figure to this path (outputs/visuals/preprocessing_check.png)
    """
    import matplotlib.pyplot as plt

    titles = ['Raw', 'Normalized', 'CLAHE', 'Bilateral Filter']
    keys   = ['raw', 'normalized', 'clahe', 'denoised']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, key, title in zip(axes, keys, titles):
        ax.imshow(results[key], cmap='gray')
        ax.set_title(title, fontsize=13)
        ax.axis('off')

    plt.suptitle('Preprocessing Pipeline Steps', fontsize=15, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to {save_path}")

    plt.show()


# ==============================================================================
# TEST - main to test on a single image
# ==============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <path_to_image>")
        print("Example: python preprocessing.py data/raw/sample_01.png")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Processing: {image_path}")

    results = preprocess(image_path)

    print(f"  Raw shape:       {results['raw'].shape}")
    print(f"  Raw value range: [{results['raw'].min()}, {results['raw'].max()}]")
    print(f"  After norm:      [{results['normalized'].min()}, {results['normalized'].max()}]")
    print(f"  Ready for segmentation: {results['denoised'].shape}")

    visualize_steps(results, save_path="outputs/visuals/preprocessing_check.png")
