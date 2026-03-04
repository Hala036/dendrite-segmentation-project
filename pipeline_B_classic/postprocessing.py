"""
postprocessing.py - Pipeline B: Classic Computer Vision
========================================================
PURPOSE OF THIS FILE:
    The binary mask from segmentation.py is correct in its broad strokes but
    imperfect in the details. It contains:
        - Small isolated noise speckles (background bubbles, charge artifacts)
        - Holes inside dendrite bodies (gaps where threshold missed bright pixels)
        - Broken thin branches (disconnected segments of the same branch)

    This file fixes all of those problems using morphological operations —
    mathematical tools that work on the SHAPE and CONNECTIVITY of white regions,
    not on pixel brightness values.

    Think of this as the "quality control" stage before skeletonization.

STEPS IN ORDER:
    1. Remove small components   → kill noise speckles by size
    2. Morphological closing     → fill holes inside dendrite bodies
    3. Morphological reconstruction → recover thin branches lost in closing
                                      while rejecting noise not connected to main structure
"""

import cv2
import numpy as np
from skimage.morphology import reconstruction


# ==============================================================================
# STEP 1: REMOVE SMALL CONNECTED COMPONENTS
# ==============================================================================

def remove_small_components(mask: np.ndarray,
                             min_area: int = 50) -> np.ndarray:
    """
    Removes small isolated white regions (noise) from the binary mask.

    WHY THIS WORKS:
        Dendrites are physically large connected structures — a real dendrite
        branch will always occupy more than a handful of pixels at SEM resolution.
        Small isolated white blobs (< min_area pixels) are almost certainly:
            - Background bubble highlights
            - Charging artifacts
            - Noise that survived thresholding

        By finding all connected white regions and deleting the small ones,
        we clean up the majority of background noise in one pass.

    HOW CONNECTED COMPONENTS WORK:
        Imagine flooding the white regions of the mask with water.
        Each separate "island" of white pixels that you can reach without
        crossing a black pixel is one connected component.
        We measure each island's area and delete those below min_area.

    CHOOSING min_area:
        Too small (< 30):  some noise survives
        50 (default):      safe for most SEM images at standard resolution
        Too large (> 200): risk deleting real thin branch tips

        If your images are very high resolution, scale this up proportionally.

    Args:
        mask:     Binary mask (uint8, 0 or 255) from segmentation
        min_area: Minimum pixel area to keep (smaller regions are deleted)

    Returns:
        Cleaned binary mask, same shape as input
    """
    # Label every connected white region with a unique integer ID
    # num_labels = total number of regions found (including background=0)
    num_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8   # 8-connectivity: diagonal neighbors count as connected
    )

    # Build output mask — start with all zeros (black)
    cleaned = np.zeros_like(mask)

    # stats[i] contains: [x, y, width, height, area] for component i
    # Component 0 is always the background — skip it (start from 1)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labeled == i] = 255   # keep this component

    return cleaned


# ==============================================================================
# STEP 2: MORPHOLOGICAL CLOSING
# ==============================================================================

def get_adaptive_kernel_size(mask: np.ndarray) -> int:
    """
    Chooses closing kernel size based on the average white region size.
    Large dense regions (Type 2) → bigger kernel
    Small sparse regions (Type 1) → smaller kernel
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:
        return 5  # fallback
    
    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    avg_area = np.median(areas)          # median is more robust than mean
    
    if avg_area < 500:
        return 5   # small sparse structures → Type 1
    elif avg_area < 800:
        return 7   # medium → transition
    else:
        return 9   # large dense structures → Type 2


def apply_closing(mask: np.ndarray,
                  kernel_size: int = -1) -> np.ndarray:
    """
    Fills small holes inside dendrite bodies and bridges tiny gaps.

    WHY WE NEED THIS:
        After thresholding, dendrite bodies often have small black holes inside
        them — pixels where the local brightness dipped just below the threshold.
        These holes would become spurious loops in the skeleton if not filled.

        Closing also bridges tiny black gaps between two white regions that
        are physically part of the same branch but got disconnected.

    HOW CLOSING WORKS:
        Closing = Dilation followed by Erosion with the same kernel.

        Dilation first EXPANDS all white regions outward by kernel_size/2 pixels.
        This expansion fills holes and bridges gaps.

        Erosion then SHRINKS all white regions back by the same amount.
        This restores the original size/shape of dendrites, but the holes
        that were filled by dilation are now too small to re-open — they stay filled.

        Visually:
            Before:  █ █ █ _ █ █ █    (_ = hole inside dendrite)
            Dilate:  █ █ █ █ █ █ █    (gap filled)
            Erode:   █ █ █ █ █ █ █    (hole stays closed, size restored)

    KERNEL SIZE:
        Controls the maximum hole size that gets filled.
        5 = fills holes up to ~5px diameter (safe default)
        Too large: merges separate dendrite branches into one blob

    Args:
        mask:        Binary mask after small component removal
        kernel_size: Size of the structuring element (must be odd)

    Returns:
        Mask with holes filled and small gaps bridged
    """
    if kernel_size <= 0:
        kernel_size = get_adaptive_kernel_size(mask)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,          # round kernel, better than square for organic shapes
        (kernel_size, kernel_size)
    )

    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return closed


# ==============================================================================
# STEP 3: MORPHOLOGICAL RECONSTRUCTION
# ==============================================================================

def apply_morphological_reconstruction(mask: np.ndarray,
                                        erosion_size: int = 5,
                                        iters: int = 3) -> np.ndarray:
    """
    Recovers thin branches lost during closing while rejecting noise.

    THE PROBLEM WITH CLOSING ALONE:
        Closing fills holes, but it also slightly thickens and merges nearby
        thin branches. More importantly, it can't distinguish between:
            A) a thin real branch that got partially erased during thresholding
            B) a noise speckle that happens to be near a real dendrite

        Morphological reconstruction solves this by asking:
        "Is this white region REACHABLE from a definitely-real part of the dendrite?"

    HOW IT WORKS — TWO IMAGES:

        MASK (the "ceiling"):
            The closed binary mask from Step 2.
            Defines the maximum extent white regions are allowed to grow.
            Nothing can grow beyond the mask boundary.

        MARKER (the "seeds"):
            The mask after aggressive erosion — only the thick, certain core
            parts of dendrites survive. Thin branches and noise are both gone.
            These are our "guaranteed real dendrite" starting points.

        RECONSTRUCTION PROCESS:
            Iteratively dilate the marker, but never exceed the mask.
            The marker grows outward from its seeds, reclaiming thin branches
            that are connected to the main structure.
            Noise speckles that are NOT connected to any seed never get reclaimed —
            they stay dark even if they're inside the mask boundary.

        Visually:
            Mask:    ██ ████████ ████  (includes noise on the right)
            Marker:     ██████         (only thick core, noise gone)
            Result:  ████████████      (branches recovered, noise rejected)
                                ████  ← this isolated region never gets reclaimed

    EROSION SIZE:
        Controls how aggressively the marker is created.
        3 (default): removes thin branches AND noise, keeps thick cores
        Too small (1): marker too similar to mask, reconstruction doesn't help
        Too large (7+): destroys too much, thin branches never get reclaimed

    Args:
        mask:         Binary mask after closing (Step 2)
        erosion_size: Size of erosion kernel for creating the marker
        iters:         Number of iterations for reconstruction
    Returns:
        Reconstructed mask — thin branches recovered, isolated noise rejected
    """
    # Normalize to [0, 1] for skimage's reconstruction function
    mask_normalized = (mask > 0).astype(np.uint8)

    # Create marker by aggressive erosion
    erosion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (erosion_size, erosion_size)
    )
    marker = cv2.erode(mask_normalized, erosion_kernel, iterations=iters)

    # Reconstruction: grow marker under the mask constraint
    # method='dilation': marker expands toward mask but never beyond it
    reconstructed = reconstruction(
        seed=marker.astype(float),
        mask=mask_normalized.astype(float),
        method='dilation'
    )

    # Convert back to uint8 binary mask (0 or 255)
    result = np.clip(reconstructed, 0, 1)
    result = (result > 0.5).astype(np.uint8) * 255

    return result


# ==============================================================================
# MASTER FUNCTION
# ==============================================================================

def postprocess(mask: np.ndarray,
                min_area: int = 300,
                closing_kernel: int = -1,
                erosion_size: int = 5,
                iters: int = 3) -> dict:
    """
    Runs the full postprocessing pipeline on a binary segmentation mask.

    Args:
        mask:           Binary mask from segmentation.py (uint8, 0 or 255)
        min_area:       Minimum component area to keep
        closing_kernel: Kernel size for morphological closing
        erosion_size:   Erosion size for reconstruction marker
        iters:          Number of iterations for reconstruction

    Returns:
        Dictionary with intermediate results:
            'input'           → original segmentation mask
            'no_small'        → after small component removal
            'closed'          → after morphological closing
            'reconstructed'   → final clean mask (READY FOR SKELETONIZATION)
    """
    no_small = remove_small_components(mask, min_area)
    closed = apply_closing(no_small, closing_kernel)
    reconstructed = apply_morphological_reconstruction(closed, erosion_size, iters)

    return {
        'input': mask,
        'no_small': no_small,
        'closed': closed,
        'reconstructed': reconstructed    # <-- this is what skeletonization.py receives
    }


# ==============================================================================
# VISUALIZATION HELPER
# ==============================================================================

def visualize_steps(results: dict, save_path = None) -> None:
    """
    Plots all postprocessing steps side by side for inspection.

    Args:
        results:   Dictionary returned by postprocess()
        save_path: If provided, saves the figure to this path
    """
    import matplotlib.pyplot as plt

    titles = ['Segmentation Input', 'Small Components Removed',
              'After Closing', 'After Reconstruction']
    keys   = ['input', 'no_small', 'closed', 'reconstructed']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, key, title in zip(axes, keys, titles):
        ax.imshow(results[key], cmap='gray')
        ax.set_title(title, fontsize=13)
        ax.axis('off')

    plt.suptitle('Postprocessing Pipeline Steps', fontsize=15, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to {save_path}")

    plt.show()


# ==============================================================================
# QUICK TEST
# ==============================================================================

if __name__ == "__main__":
    import sys
    from preprocessing import preprocess
    from segmentation import segment

    if len(sys.argv) < 2:
        print("Usage: python postprocessing.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Processing: {image_path}")

    prep_results = preprocess(image_path)
    seg_results  = segment(prep_results['denoised'])
    post_results = postprocess(seg_results['mask'])

    print(f"  Noise components removed: visible in side-by-side")
    print(f"  Final mask ready for skeletonization")

    visualize_steps(post_results, save_path="outputs/visuals/postprocessing_check.png")
