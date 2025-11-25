"""
Fundus Contrast Enhancement Example

This example demonstrates how to use the vascx_simplify library to enhance
contrast in fundus images using GPU-accelerated preprocessing.

The contrast enhancement pipeline:
1. Detects the fundus region (circular boundary)
2. Normalizes the image to standard orientation
3. Applies adaptive contrast enhancement
4. Returns enhanced image with improved vessel visibility
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from vascx_simplify.preprocess import SimpleFundusEnhance


def main():
    # Configuration
    IMG_PATH = "HRF_07_dr.jpg"  # Path to your fundus image
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {DEVICE}")

    # Step 1: Initialize contrast enhancement processor
    print("Initializing contrast enhancement...")
    enhancer = SimpleFundusEnhance(
        use_fp16=True,  # Use mixed precision for faster processing on GPU
        square_size=512,  # Optional: crop to square size
    )

    # Step 2: Load the image
    print(f"Loading image: {IMG_PATH}")
    rgb_image = Image.open(IMG_PATH)
    rgb_array = np.array(rgb_image)

    # Step 3: Convert to tensor [C, H, W] format and move to device
    img_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).to(DEVICE)

    # Step 4: Apply contrast enhancement
    print("Enhancing contrast...")
    original, enhanced, bounds = enhancer(img_tensor)

    # Step 5: Convert back to numpy for visualization
    original_np = original.cpu().permute(1, 2, 0).numpy()
    enhanced_np = enhanced.cpu().permute(1, 2, 0).numpy()

    # Step 6: Create comparison visualization
    print("Creating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Original image
    axes[0].imshow(original_np)
    axes[0].set_title("Original Fundus Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # Enhanced image
    axes[1].imshow(enhanced_np)
    axes[1].set_title("Contrast Enhanced Image", fontsize=14, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/07_contrast_enhancement_result.png", dpi=150, bbox_inches="tight")
    print("\nResult saved as 'outputs/07_contrast_enhancement_result.png'")
    plt.show(block=False)
    plt.pause(5)
    plt.close('all')

    # Step 7: Create detailed comparison with zoomed regions
    print("Creating detailed comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))

    # Full images
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(enhanced_np)
    axes[0, 1].set_title("Enhanced Image", fontsize=12, fontweight="bold")
    axes[0, 1].axis("off")

    # Zoomed regions (center region for better detail)
    h, w = original_np.shape[:2]
    crop_size = min(h, w) // 3
    center_y, center_x = h // 2, w // 2
    y1, y2 = center_y - crop_size // 2, center_y + crop_size // 2
    x1, x2 = center_x - crop_size // 2, center_x + crop_size // 2

    axes[1, 0].imshow(original_np[y1:y2, x1:x2])
    axes[1, 0].set_title("Original (Zoomed)", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(enhanced_np[y1:y2, x1:x2])
    axes[1, 1].set_title("Enhanced (Zoomed)", fontsize=12, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/07_contrast_enhancement_detailed.png", dpi=150, bbox_inches="tight")
    print("Detailed result saved as 'contrast_enhancement_detailed.png'")
    plt.show(block=False)
    plt.pause(5)
    plt.close('all')

    # Step 8: Print statistics
    print("\n" + "=" * 60)
    print("CONTRAST ENHANCEMENT STATISTICS")
    print("=" * 60)

    # Calculate intensity statistics
    original_mean = np.mean(original_np)
    enhanced_mean = np.mean(enhanced_np)
    original_std = np.std(original_np)
    enhanced_std = np.std(enhanced_np)

    print("Original Image:")
    print(f"  - Mean intensity: {original_mean:.2f}")
    print(f"  - Std deviation: {original_std:.2f}")
    print(f"  - Min/Max: {original_np.min():.0f}/{original_np.max():.0f}")

    print("\nEnhanced Image:")
    print(f"  - Mean intensity: {enhanced_mean:.2f}")
    print(f"  - Std deviation: {enhanced_std:.2f}")
    print(f"  - Min/Max: {enhanced_np.min():.0f}/{enhanced_np.max():.0f}")

    print("\nContrast Improvement:")
    print(f"  - Std deviation increase: {(enhanced_std / original_std - 1) * 100:+.1f}%")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
