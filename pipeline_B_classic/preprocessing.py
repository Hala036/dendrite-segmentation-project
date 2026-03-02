"""
preprocessing.py - Pipeline B: Classic Computer Vision
=======================================================
PURPOSE OF THIS FILE:
    Raw SEM images are "dirty" - they have uneven lighting, noise, and low contrast.
    Before we can segment anything, we need to clean and normalize the image so that
    our segmentation algorithms work reliably across ALL images, not just well-lit ones.

    Think of this as "preparing the canvas" before doing any real work.

STEPS IN ORDER:
    1. Load image as grayscale & crop metadata bar → get a clean 2D array of pixel values
    2. Histogram Normalization  → gives all images the same brightness range
    3. CLAHE                    → boosts local contrast so thin dendrites become visible
    4. Bilateral Filter         → removes noise while KEEPING dendrite edges sharp
"""

import cv2
import numpy as np
from pathlib import Path


# ==============================================================================
# STEP 1: IMAGE LOADING + CROPPING
# ==============================================================================

def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from disk and converts it to grayscale.

    WHY GRAYSCALE?
        SEM images are inherently grayscale - they encode surface height/material
        density as brightness. There is no color information. Working in grayscale
        means we deal with a single channel (2D array) instead of 3 (RGB),
        which simplifies every operation that follows.

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

    # cv2.IMREAD_GRAYSCALE forces single-channel loading
    # even if the file was saved as RGB
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"OpenCV could not read the image: {image_path}")

    return image

def crop_metadata_bar(image: np.ndarray, crop_fraction: float = 0.065) -> np.ndarray:
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

    WHY DO WE NEED THIS?
        Different SEM images may have been captured at different exposure settings.
        One image might use pixel values in range [30, 200], another in [80, 240].
        This means the same dendrite could look bright in one image and dark in another.

        Normalization gives EVERY image the same baseline, so that any threshold
        or filter value we set later works consistently across the whole dataset.

        Think of it like adjusting the "levels" in Photoshop - we're just stretching
        the histogram to fill the full 0-255 range.

    HOW IT WORKS (linear stretch):
        new_value = (old_value - min) / (max - min) * 255

        The darkest pixel becomes 0, the brightest becomes 255,
        everything else scales linearly in between.

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
                clip_limit: float = 1.0,
                tile_size: int = 8) -> np.ndarray:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).

    WHY NOT REGULAR HISTOGRAM EQUALIZATION?
        Regular (global) histogram equalization looks at the WHOLE image at once.
        This causes two problems for SEM images:
            1. It "burns out" bright areas (electrode regions appear completely white)
            2. It amplifies background noise to look like actual structure

        This is why regular equalization is explicitly called a "common mistake"
        in your project document.

    WHAT CLAHE DOES DIFFERENTLY:
        Instead of one global histogram, CLAHE divides the image into a grid of
        small tiles (e.g., 8x8 pixels each). It computes and equalizes the
        histogram SEPARATELY for each tile.

        Result: a thin dendrite in a dark shadowed corner gets the same contrast
        boost as a thick dendrite in a bright area. Neither gets "burned."

    The reasoning for 8 specifically: dendrite branches in SEM images are thin structures 
    that occupy small regions. A tile needs to be small enough to "see" a thin branch in 
    its own neighborhood and boost it, but not so small that individual noise pixels get 
    their own enhancement zone.

    THE CLIP LIMIT:
        If a tile has very few distinct gray values (e.g., a completely flat region),
        standard equalization would massively amplify tiny noise variations.
        The clip_limit caps how much any single intensity can be boosted,
        preventing noise from being mistaken for structure.

        - Low clip (1.0-2.0): gentle enhancement, safer for noisy images
        - High clip (3.0-4.0): aggressive enhancement, risks noise amplification
    
    The clip limit controls how aggressively each tile's histogram is amplified.

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
                            sigma_space: float = 4.5) -> np.ndarray:
    """
    Applies a Bilateral Filter to remove noise while preserving edges.

    WHY NOT GAUSSIAN BLUR?
        Gaussian blur is the most common denoising filter - it averages each pixel
        with its neighbors. The problem: it doesn't care whether a neighbor is
        "same surface" or "across a dendrite boundary."

        Result: dendrite edges get blurry, thin branches lose definition,
        and you can no longer tell where the dendrite ends and background begins.

        Your project document explicitly warns: "This blurring causes loss of
        critical information about the true branch width."

    HOW BILATERAL FILTER WORKS:
        Like Gaussian blur, it averages nearby pixels. But it has TWO conditions
        for including a neighbor in the average:

            Condition 1 - SPATIAL: the neighbor must be physically close (sigma_space)
            Condition 2 - INTENSITY: the neighbor must have similar brightness (sigma_color)

        If a neighboring pixel is across a sharp edge (very different brightness),
        it gets almost ZERO weight in the average. The edge is preserved.
        If the neighbor is in the same smooth region, it contributes normally.

    PARAMETER GUIDE:
        diameter:    The pixel neighborhood size. Larger = stronger smoothing.
                     9 is a good balance. Above 15 becomes very slow.

        sigma_color: How much brightness difference is "allowed" before a neighbor
                     is excluded. Higher = smoother but less edge-aware.
                     Range: 50 (strict) to 150 (permissive)

        sigma_space: How far spatially a neighbor can be to still contribute.
                     Higher = larger smooth regions.
                     Range: 50 (local) to 150 (wide)

        NOTE: For SEM images, keeping both sigmas at 75 is a safe starting point.
        You may need to tune these in your notebook.

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
# MASTER FUNCTION - runs all steps in order
# ==============================================================================

def preprocess(image_path: str,
               clahe_clip: float = 2.0,
               clahe_tile: int = 8,
               bilateral_d: int = 9,
               bilateral_sigma: float = 75) -> dict:
    """
    Runs the full preprocessing pipeline on a single SEM image.

    Returns a dictionary with intermediate results so you can inspect
    what each step did - useful for debugging and the report visuals.

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
            'denoised'   → after bilateral filter (READY FOR SEGMENTATION)
    """
    raw = load_image(image_path)
    raw = crop_metadata_bar(raw)  # Remove SEM metadata bar before any processing
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
    Plots all preprocessing steps side by side for inspection.

    Use this in your notebook to verify each step is doing what you expect
    before moving to segmentation.

    Args:
        results:   Dictionary returned by preprocess()
        save_path: If provided, saves the figure to this path
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
