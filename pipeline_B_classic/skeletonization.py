"""
Turn the clean dendrite mask into a 1-pixel-wide skeleton.

This file also measures simple branch information like tip count, fork count,
and total skeleton length.
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy import ndimage
from pathlib import Path


# ==============================================================================
# STEP 1: DISTANCE TRANSFORM
# ==============================================================================

def apply_distance_transform(mask: np.ndarray) -> np.ndarray:
    """
    Compute the distance from each mask pixel to the nearest background pixel.

    Args:
        mask: Clean binary mask from postprocessing.

    Returns:
        Distance transform image.
    """
    # Normalize mask to 0/1 for distance transform
    binary = (mask > 0).astype(np.uint8)

    # DIST_L2 = Euclidean distance (straight-line, not Manhattan)
    # MASK_PRECISE = highest accuracy calculation
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    return dist


# ==============================================================================
# STEP 2: WATERSHED SEPARATION
# ==============================================================================

def apply_watershed(mask: np.ndarray,
                    dist: np.ndarray,
                    peak_min_distance: int = 5) -> np.ndarray:
    """
    Use watershed to split touching regions when possible.

    Args:
        mask: Clean binary mask.
        dist: Distance transform of the mask.
        peak_min_distance: Minimum spacing between detected peaks.

    Returns:
        Labeled watershed regions.
    """
    from skimage.feature import peak_local_max

    # Find local maxima in the distance transform
    # These become the seeds (markers) for watershed
    local_max_coords = peak_local_max(
        dist,
        min_distance=peak_min_distance,
        labels=mask         # only look for peaks inside the mask
    )

    # Create a boolean peak map and label each peak as a unique seed
    peak_map = np.zeros(dist.shape, dtype=bool)
    peak_map[tuple(local_max_coords.T)] = True
    markers = label(peak_map)

    # Run watershed — negative dist because watershed floods from LOW points,
    # but we want to start from HIGH distance values (branch centers)
    from skimage.segmentation import watershed
    labels = watershed(-dist, markers, mask=(mask > 0))

    return labels


# ==============================================================================
# STEP 3: SKELETONIZATION
# ==============================================================================

def apply_skeletonization(mask: np.ndarray) -> np.ndarray:
    """
    Reduce the mask to single-pixel-wide centerlines.

    Args:
        mask: Clean binary mask.

    Returns:
        Skeleton image.
    """
    binary = (mask > 0)

    # skimage's skeletonize uses the Zhang-Suen thinning algorithm
    skeleton = skeletonize(binary)

    return skeleton.astype(np.uint8) * 255


# ==============================================================================
# STEP 4: BRANCH ANALYSIS
# ==============================================================================

def analyze_skeleton(skeleton: np.ndarray) -> dict:
    """
    Measure simple properties from the skeleton.

    Args:
        skeleton: Binary skeleton image.

    Returns:
        Dictionary with tip mask, fork mask, and summary numbers.
    """
    binary = (skeleton > 0).astype(np.uint8)

    # Count neighbors for each skeleton pixel using a convolution
    # Kernel sums all 8 neighbors (center excluded via -1 trick below)
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)

    neighbor_count = cv2.filter2D(binary, -1, neighbor_kernel)
    # neighbor_count[i,j] = sum of all 9 pixels in 3x3 window
    # subtract the center pixel itself to get neighbor count only
    neighbor_count = neighbor_count - binary

    # Apply neighbor count only to skeleton pixels
    # Background pixels should be 0 regardless of their neighbor count
    neighbor_count = neighbor_count * binary

    # Tips: skeleton pixels with exactly 1 neighbor
    tips  = ((neighbor_count == 1) & (binary == 1)).astype(np.uint8) * 255

    # Forks: skeleton pixels with 3 or more neighbors
    forks = ((neighbor_count >= 3) & (binary == 1)).astype(np.uint8) * 255

    # Tip coordinates for spatial analysis
    tip_coords = np.argwhere(tips > 0)  # shape (N, 2): [[row, col], ...]

    return {
        'tips':         tips,
        'forks':        forks,
        'tip_count':    int((tips > 0).sum()),
        'fork_count':   int((forks > 0).sum()),
        'total_length': int((binary > 0).sum()),
        'tip_coords':   tip_coords
    }


# ==============================================================================
# MASTER FUNCTION
# ==============================================================================

def skeletonize_mask(mask: np.ndarray,
                     peak_min_distance: int = 5) -> dict:
    """
    Run the full skeletonization pipeline.

    Args:
        mask: Clean mask from postprocessing.
        peak_min_distance: Minimum peak spacing for watershed.

    Returns:
        Dictionary with intermediate outputs and analysis results.
    """
    dist      = apply_distance_transform(mask)
    watershed = apply_watershed(mask, dist, peak_min_distance)
    skeleton  = apply_skeletonization(mask)
    analysis  = analyze_skeleton(skeleton)

    return {
        'dist':      dist,
        'watershed': watershed,
        'skeleton':  skeleton,
        'analysis':  analysis
    }


# ==============================================================================
# VISUALIZATION HELPER
# ==============================================================================

def visualize_steps(original_mask: np.ndarray,
                    results: dict,
                    save_path = None) -> None:
    """
    Show skeletonization results and mark tips and forks.

    Args:
        original_mask: Clean binary mask.
        results: Output from `skeletonize_mask()`.
        save_path: Optional path to save the figure.
    """
    import matplotlib.pyplot as plt

    skeleton  = results['skeleton']
    analysis  = results['analysis']

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))

    # Panel 1: distance transform
    axes[0].imshow(results['dist'], cmap='hot')
    axes[0].set_title('Distance Transform')
    axes[0].axis('off')

    # Panel 2: watershed regions
    axes[1].imshow(results['watershed'], cmap='nipy_spectral')
    axes[1].set_title(f'Watershed Separation')
    axes[1].axis('off')

    # Panel 3: skeleton alone
    axes[2].imshow(skeleton, cmap='gray')
    axes[2].set_title(f'Skeleton\n(length={analysis["total_length"]}px)')
    axes[2].axis('off')

    # Panel 4: skeleton overlaid on mask with tips (red) and forks (blue)
    overlay = cv2.cvtColor(original_mask, cv2.COLOR_GRAY2RGB)
    overlay[skeleton > 0]           = [0,   255, 0  ]  # skeleton = green
    overlay[analysis['tips'] > 0]   = [255, 0,   0  ]  # tips = red
    overlay[analysis['forks'] > 0]  = [0,   0,   255]  # forks = blue

    axes[3].imshow(overlay)
    axes[3].set_title(
        f'Tips (red): {analysis["tip_count"]}\n'
        f'Forks (blue): {analysis["fork_count"]}'
    )
    axes[3].axis('off')

    plt.suptitle('Skeletonization Pipeline Steps', fontsize=15, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved to {save_path}")

    plt.show()


def save_skeleton_overlay(original_image: np.ndarray,
                          results: dict,
                          save_path: str,
                          alpha: float = 0.4,
                          thickness: int = 2) -> None:
    """
    Save a simple overlay of the skeleton, tips, and forks.
    """
    if original_image.dtype != np.uint8:
        base = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        base = original_image.copy()

    skeleton = (results['skeleton'] > 0).astype(np.uint8) * 255
    tips = (results['analysis']['tips'] > 0).astype(np.uint8) * 255
    forks = (results['analysis']['forks'] > 0).astype(np.uint8) * 255

    if thickness > 1:
        k = np.ones((thickness, thickness), dtype=np.uint8)
        skeleton = cv2.dilate(skeleton, k, iterations=1)
        tips = cv2.dilate(tips, k, iterations=1)
        forks = cv2.dilate(forks, k, iterations=1)

    overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    overlay[skeleton > 0] = [0, 255, 0]
    overlay[tips > 0] = [255, 0, 0]
    overlay[forks > 0] = [0, 0, 255]

    blended = cv2.addWeighted(cv2.cvtColor(base, cv2.COLOR_GRAY2RGB), 1.0 - alpha, overlay, alpha, 0)

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    print(f"Saved overlay to {out}")


# ==============================================================================
# QUICK TEST
# ==============================================================================

if __name__ == "__main__":
    import sys
    from preprocessing import preprocess
    from segmentation import segment
    from postprocessing import postprocess

    if len(sys.argv) < 2:
        print("Usage: python skeletonization.py <path_to_image>")
        print("Example: python skeletonization.py data/raw/sample_01.png")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Processing: {image_path}")

    prep  = preprocess(image_path)
    seg   = segment(prep['denoised'])
    post  = postprocess(seg['mask'])
    skel  = skeletonize_mask(post['reconstructed'])

    a = skel['analysis']
    print(f"  Skeleton length : {a['total_length']} px")
    print(f"  Tips (endpoints): {a['tip_count']}")
    print(f"  Forks (junctions): {a['fork_count']}")

    visualize_steps(post['reconstructed'], skel,
                    save_path="outputs/visuals/skeletonization_check.png")
    save_skeleton_overlay(
        original_image=prep['raw'],
        results=skel,
        save_path="outputs/visuals/skeletonization_overlay.png"
    )
