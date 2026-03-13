"""
Clean the segmentation mask before skeletonization.

This file removes small noise, fills small gaps, and gives back a cleaner mask
for the structure analysis step.
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
    Remove tiny connected regions that are probably noise.

    Args:
        mask: Binary mask from segmentation.
        min_area: Minimum component area to keep.

    Returns:
        Cleaned mask.
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
    Choose a closing kernel size from the mask content.
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
    Fill small holes and connect small gaps in the mask.

    Args:
        mask: Binary mask after removing small components.
        kernel_size: Closing kernel size. If negative, pick automatically.

    Returns:
        Mask after closing.
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
    Recover useful thin branches after closing.

    Args:
        mask: Binary mask after closing.
        erosion_size: Erosion kernel size used to build the marker.
        iters: Number of erosion iterations.
    Returns:
        Reconstructed mask.
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
    Run the full postprocessing pipeline.

    Args:
        mask: Binary mask from segmentation.
        min_area: Minimum component area to keep.
        closing_kernel: Kernel size for closing.
        erosion_size: Erosion size for the reconstruction marker.
        iters: Number of reconstruction iterations.

    Returns:
        Dictionary with the mask from each step.
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
    Show all postprocessing steps side by side.

    Args:
        results: Output from `postprocess()`.
        save_path: Optional path to save the figure.
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
