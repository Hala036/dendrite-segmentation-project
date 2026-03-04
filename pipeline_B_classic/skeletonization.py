"""
skeletonization.py - Pipeline B: Classic Computer Vision
=========================================================
PURPOSE OF THIS FILE:
    Takes the clean binary mask from postprocessing and reduces every dendrite
    branch down to a single-pixel-wide centerline — the "skeleton."

    WHY DO WE NEED A SKELETON?
        The binary mask tells us WHERE dendrites are, but not their structure.
        A thick blob gives us no information about branch count, branch length,
        or tip sharpness. The skeleton extracts the pure topology:
            - How many branches are there?
            - How long is each branch?
            - Where are the tips? (dangerous sharp tips → battery failure risk)
            - Where do branches fork?

        These geometric measurements are what the battery engineers actually care about.

STEPS IN ORDER:
    1. Distance Transform   → each white pixel gets a value = distance to nearest edge
    2. Watershed            → separates touching/merged branches using distance peaks
    3. Skeletonization      → reduces each branch to a 1-pixel-wide centerline
    4. Branch analysis      → counts tips, forks, and measures branch lengths
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from scipy import ndimage


# ==============================================================================
# STEP 1: DISTANCE TRANSFORM
# ==============================================================================

def apply_distance_transform(mask: np.ndarray) -> np.ndarray:
    """
    Assigns each white pixel a value equal to its distance to the nearest
    black pixel (background).

    WHY WE NEED THIS:
        The distance transform converts a flat binary mask into a
        "topographic map" where:
            - Pixels at the CENTER of thick branches have HIGH values
              (they are far from any edge)
            - Pixels at the EDGE of branches have LOW values (~1)
            - Background pixels have value 0

        This is useful for two reasons:
            1. It feeds into Watershed to find branch centers
            2. The local maxima of the distance transform are guaranteed
               to lie on the medial axis (centerline) of each branch

        Visually for a thick branch:
            0 0 0 0 0 0 0 0
            0 1 1 1 1 1 1 0
            0 1 2 2 2 2 1 0   ← center pixels have value 2
            0 1 2 3 3 2 1 0   ← very center has value 3
            0 1 2 2 2 2 1 0
            0 1 1 1 1 1 1 0
            0 0 0 0 0 0 0 0

    Args:
        mask: Clean binary mask (uint8, 0 or 255) from postprocessing

    Returns:
        Float32 array of same shape, values = distance to nearest background pixel
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
    Separates touching or merged dendrite branches using the Watershed algorithm.

    THE PROBLEM IT SOLVES:
        After postprocessing, two branches that grew close together may be
        merged into one white blob. The skeleton of a merged blob produces
        a single thick line instead of two separate branches — losing
        topological information.

        Watershed treats the distance transform as a topographic map and
        "floods" it from local peaks (branch centers), creating boundaries
        where two flood regions meet.

    HOW IT WORKS:
        1. Find local maxima of the distance transform
           → these are the centers of branches (furthest from any edge)
        2. Label each maximum as a separate "seed"
        3. Grow each seed outward (flood uphill in the distance map)
        4. Where two growing regions meet → draw a boundary
        5. Result: each originally-merged branch gets its own labeled region

    PEAK_MIN_DISTANCE:
        Minimum number of pixels between two peaks to be considered separate.
        Too small → finds noise peaks, over-segments one branch into many
        Too large → misses real separate branches, under-segments
        5 is a safe default for SEM images at standard resolution.

    Args:
        mask:              Clean binary mask from postprocessing
        dist:              Distance transform from apply_distance_transform()
        peak_min_distance: Minimum distance between watershed seed peaks

    Returns:
        Label map (int32) where each separated region has a unique integer ID.
        Background = 0, first branch = 1, second branch = 2, etc.
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
    Reduces all white regions to single-pixel-wide centerlines.

    HOW SKELETONIZATION WORKS:
        The algorithm iteratively removes pixels from the border of white regions
        — but ONLY if removing them doesn't change the topology (connectivity).

        A pixel can be removed if:
            - It is on the border (has at least one black neighbor)
            - Removing it does NOT disconnect any remaining white pixels
            - Removing it does NOT create a hole

        This continues until no more pixels can be removed — what remains
        is the thinnest possible representation that preserves the original
        connectivity and branching structure.

        Before:              After:
        ████████             ────────
        ████████    →            │
        ████                 ────┘
        ████
        (thick blob)         (1-pixel centerlines)

    NOTE ON WATERSHED + SKELETONIZATION:
        We skeletonize the ORIGINAL mask (not the watershed labels) because
        watershed is only used to count/separate branches for analysis.
        Skeletonization on the full mask gives cleaner centerlines.

    Args:
        mask: Clean binary mask (uint8, 0 or 255)

    Returns:
        Boolean array of same shape — True = skeleton pixel, False = background
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
    Extracts geometric measurements from the skeleton.

    WHAT WE MEASURE AND WHY:

        TIPS (endpoints):
            A skeleton pixel with exactly 1 neighbor is a branch tip.
            Tips represent the growing front of dendrites — sharp, isolated tips
            are the primary puncture risk for the battery separator.
            More tips = higher risk.

        FORKS (junction points):
            A skeleton pixel with 3 or more neighbors is a branch junction.
            Fork count gives us the branching complexity of the dendrite tree.

        BRANCH COUNT:
            Approximated as: tips / 2 + forks
            (Each branch has 2 endpoints unless it's a loop)

        TOTAL LENGTH:
            Total number of skeleton pixels ≈ total centerline length in pixels.
            Multiply by the image's nm/pixel scale factor to get physical length.

    HOW NEIGHBOR COUNTING WORKS:
        For each skeleton pixel, we count how many of its 8 neighbors
        are also skeleton pixels. This gives the "connectivity" of that pixel:
            1 neighbor  → tip (dead end)
            2 neighbors → middle of a branch (pass-through)
            3+ neighbors → fork (junction)

    Args:
        skeleton: Binary skeleton array (uint8, 0 or 255)

    Returns:
        Dictionary with keys:
            'tips'         → binary mask of tip pixels
            'forks'        → binary mask of fork pixels
            'tip_count'    → integer count of tips
            'fork_count'   → integer count of forks
            'total_length' → total skeleton pixel count
            'tip_coords'   → (N, 2) array of tip pixel coordinates [row, col]
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
    Runs the full skeletonization pipeline on a postprocessed binary mask.

    Args:
        mask:              Clean binary mask from postprocessing.py
        peak_min_distance: Watershed seed separation (default 5)

    Returns:
        Dictionary with keys:
            'dist'       → distance transform (float32)
            'watershed'  → watershed label map (int32)
            'skeleton'   → single-pixel centerlines (uint8, 0 or 255)
            'analysis'   → dict from analyze_skeleton() with tip/fork counts
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
    Plots skeletonization results with tips and forks marked.

    Args:
        original_mask: The postprocessed binary mask (for overlay context)
        results:       Dictionary returned by skeletonize_mask()
        save_path:     If provided, saves the figure to this path
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
    post  = postprocess(seg['final_mask'])
    skel  = skeletonize_mask(post['reconstructed'])

    a = skel['analysis']
    print(f"  Skeleton length : {a['total_length']} px")
    print(f"  Tips (endpoints): {a['tip_count']}")
    print(f"  Forks (junctions): {a['fork_count']}")

    visualize_steps(post['reconstructed'], skel,
                    save_path="outputs/visuals/skeletonization_check.png")
