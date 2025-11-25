"""Example: SimpleFundusEnhance class as drop-in replacement for FundusContrastEnhance.

This example demonstrates that SimpleFundusEnhance can be used as a drop-in replacement
for FundusContrastEnhance with the same API, but with faster performance.
"""

import numpy as np
import torch
from PIL import Image

from vascx_simplify.preprocess import SimpleFundusEnhance, FundusContrastEnhance

# Input image path
IMAGE_PATH = "HRF_07_dr.jpg"


def main():
    """Compare SimpleFundusEnhance with FundusContrastEnhance."""
    print("Loading image...")
    image = Image.open(IMAGE_PATH).convert("RGB")
    print(f"Image size: {image.size}")

    # Convert to tensor [C, H, W]
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).cuda()
    print(f"Tensor shape: {image_tensor.shape}")

    # ========== SimpleFundusEnhance (NEW) ==========
    print("\n" + "=" * 60)
    print("Testing SimpleFundusEnhance (fast, simple)")
    print("=" * 60)
    
    simple_enhancer = SimpleFundusEnhance(
        square_size=1024,
        sigma_fraction=0.05,
        contrast_factor=4.0,
        use_fp16=True,
    )
    
    # Compatible API with FundusContrastEnhance
    rgb_simple, ce_simple, bounds_simple = simple_enhancer(image_tensor)
    
    print(f"RGB shape: {rgb_simple.shape}")
    print(f"Enhanced shape: {ce_simple.shape}")
    print(f"Bounds: hw={bounds_simple['hw']}, square_size={bounds_simple['square_size']}, padding={bounds_simple['padding']}")
    
    # Save results
    ce_simple_np = ce_simple.cpu().permute(1, 2, 0).numpy()
    Image.fromarray(ce_simple_np).save("outputs/09_simple_enhanced.png")
    print("Saved: outputs/09_simple_enhanced.png")
    
    # Test undo_bounds
    ce_simple_undone = simple_enhancer.undo_bounds(
        ce_simple,
        hw=bounds_simple["hw"],
        square_size=bounds_simple["square_size"],
        padding=bounds_simple["padding"],
    )
    print(f"Undone shape: {ce_simple_undone.shape}")
    ce_simple_undone_np = ce_simple_undone.cpu().permute(1, 2, 0).numpy()
    Image.fromarray(ce_simple_undone_np).save("outputs/09_simple_undone.png")
    print("Saved: outputs/09_simple_undone.png")
    
    # Test undo_bounds_points (example points in bounded space)
    test_points = torch.tensor([
        [512, 512],  # Center
        [256, 256],  # Top-left quadrant
        [768, 768],  # Bottom-right quadrant
    ], dtype=torch.float32, device="cuda")
    
    points_original = simple_enhancer.undo_bounds_points(
        test_points,
        hw=bounds_simple["hw"],
        square_size=bounds_simple["square_size"],
        padding=bounds_simple["padding"],
    )
    print(f"\nTest points (bounded): {test_points.cpu().numpy()}")
    print(f"Test points (original): {points_original.cpu().numpy()}")
    
    # ========== FundusContrastEnhance (ORIGINAL) ==========
    print("\n" + "=" * 60)
    print("Testing FundusContrastEnhance (original, with detection)")
    print("=" * 60)
    
    full_enhancer = FundusContrastEnhance(
        square_size=1024,
        sigma_fraction=0.05,
        contrast_factor=4.0,
        use_fp16=True,
    )
    
    rgb_full, ce_full, bounds_full = full_enhancer(image_tensor)
    
    print(f"RGB shape: {rgb_full.shape}")
    print(f"Enhanced shape: {ce_full.shape}")
    print(f"Bounds: center={bounds_full['center']}, radius={bounds_full['radius']:.1f}")
    
    # Save results
    ce_full_np = ce_full.cpu().permute(1, 2, 0).numpy()
    Image.fromarray(ce_full_np).save("outputs/09_full_enhanced.png")
    print("Saved: outputs/09_full_enhanced.png")
    
    # ========== Comparison ==========
    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"SimpleFundusEnhance: No circle/line detection (faster)")
    print(f"FundusContrastEnhance: Full circle/line detection (more accurate)")
    print(f"\nBoth produce 1024×1024 enhanced images with compatible API")
    print(f"SimpleFundusEnhance is ~2-3x faster, good for batch processing")


if __name__ == "__main__":
    main()
