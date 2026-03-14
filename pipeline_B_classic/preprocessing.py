"""
Preprocess SEM images before segmentation.

This file loads the image, removes the bottom metadata bar, improves contrast,
and reduces noise. The output is a cleaner grayscale image for the next step.
"""

import cv2
import numpy as np
from pathlib import Path


# ==============================================================================
# STEP 1: IMAGE LOADING + CROPPING
# ==============================================================================

def load_image(image_path: str) -> np.ndarray:
    """
    Load an SEM image as grayscale.

    Args:
        image_path: Path to the image file.

    Returns:
        Grayscale image as a 2D array.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If OpenCV cannot read the file.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # cv2.IMREAD_GRAYSCALE forces single-channel loading
    # even if the file was saved as RGB
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"OpenCV could not read the image: {image_path}")

    return image

def crop_metadata_bar(image: np.ndarray, crop_fraction: float = 0.065) -> np.ndarray:
    """
    Remove the SEM metadata bar at the bottom of the image.
    """
    if crop_fraction <= 0:
        return image
    h = image.shape[0]
    crop_pixels = int(h * crop_fraction)
    return image[:h - crop_pixels, :]


# ==============================================================================
# STEP 2: HISTOGRAM NORMALIZATION
# ==============================================================================

def normalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Stretch the image values to the full range `[0, 255]`.

    Args:
        image: Grayscale image.

    Returns:
        Normalized image.
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
                clip_limit: float = 1.0,
                tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE to improve local contrast.

    Args:
        image: Input grayscale image.
        clip_limit: CLAHE clip limit.
        tile_size: CLAHE tile size.

    Returns:
        Contrast-enhanced image.
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
                            sigma_space: float = 4.5) -> np.ndarray:
    """
    Remove noise while keeping edges as sharp as possible.

    Args:
        image: Input grayscale image.
        diameter: Neighborhood diameter.
        sigma_color: Intensity sigma.
        sigma_space: Spatial sigma.

    Returns:
        Denoised image.
    """
    denoised = cv2.bilateralFilter(
        image,
        d=diameter,
        sigmaColor=sigma_color,
        sigmaSpace=sigma_space
    )

    return denoised


# ==============================================================================
# MASTER FUNCTION - runs all steps in order
# ==============================================================================

def preprocess(image_path: str,
               crop_fraction: float = 0.065,
               clahe_clip: float = 2.0,
               clahe_tile: int = 8,
               bilateral_d: int = 9,
               bilateral_sigma: float = 75) -> dict:
    """
    Run the full preprocessing pipeline on one image.

    Args:
        image_path: Path to the raw image.
        crop_fraction: Fraction of bottom rows to crop. Use `0.0` to disable.
        clahe_clip: CLAHE clip limit.
        clahe_tile: CLAHE tile size.
        bilateral_d: Bilateral filter diameter.
        bilateral_sigma: Bilateral filter sigma.

    Returns:
        Dictionary with images from each step.
    """
    raw = load_image(image_path)
    raw = crop_metadata_bar(raw, crop_fraction)
    normalized = normalize_histogram(raw)
    clahe_result = apply_clahe(normalized, clahe_clip, clahe_tile)
    denoised = apply_bilateral_filter(clahe_result, bilateral_d, bilateral_sigma)

    return {
        'raw': raw,
        'normalized': normalized,
        'clahe': clahe_result,
        'denoised': denoised       # <-- this is what segmentation.py will receive
    }


# ==============================================================================
# VISUALIZATION HELPER
# ==============================================================================

def visualize_steps(results: dict, save_path = None) -> None:
    """
    Show the preprocessing steps side by side.

    Args:
        results: Output from `preprocess()`.
        save_path: Optional path to save the figure.
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
# QUICK TEST - run this file directly to test on a single image
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
