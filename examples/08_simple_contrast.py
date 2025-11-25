"""Example: Simple fundus contrast enhancement without bounds/masking.

This example demonstrates the simple_fundus_enhance function that applies
pure unsharp masking to the entire image without any preprocessing or masking.
"""

import numpy as np
import torch
from PIL import Image

from vascx_simplify.preprocess import simple_fundus_enhance

# Input image path
IMAGE_PATH = "HRF_07_dr.jpg"


def main():
    """Run simple contrast enhancement example."""
    print("Loading image...")
    image = Image.open(IMAGE_PATH).convert("RGB")
    print(f"Image size: {image.size}")

    # Convert to tensor [C, H, W]
    image_array = np.array(image)
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

    print(f"Tensor shape: {image_tensor.shape}")

    # Calculate auto-scaled sigma (matches FundusContrastEnhance for 1024×1024)
    h, w = image_tensor.shape[-2:]
    effective_radius = min(h, w) / 2
    auto_sigma = 0.05 * effective_radius
    print(f"Auto-scaled sigma for {w}×{h}: {auto_sigma:.2f} pixels")
    print(f"  (For 1024×1024 reference: sigma would be 25.6)")

    # Apply simple enhancement with auto-scaling
    print("\n--- Using auto-scaled sigma (None) ---")
    enhanced = simple_fundus_enhance(
        image_tensor, sigma=None, contrast_factor=4.0, device="cuda", use_fp16=True
    )

    print(f"Enhanced shape: {enhanced.shape}")
    print(f"Enhanced dtype: {enhanced.dtype}")
    print(f"Enhanced device: {enhanced.device}")

    # Save result
    output_path = "outputs/08_simple_contrast_auto.png"
    enhanced_np = enhanced.cpu().permute(1, 2, 0).numpy()
    Image.fromarray(enhanced_np).save(output_path)
    print(f"Saved to: {output_path}")

    # Compare with manual sigma values
    print("\n--- Comparing different sigma values ---")
    for sigma in [auto_sigma * 0.5, auto_sigma, auto_sigma * 1.5, auto_sigma * 2.0]:
        enhanced_var = simple_fundus_enhance(
            image_tensor, sigma=sigma, contrast_factor=4.0, device="cuda"
        )
        output_path_var = f"outputs/08_simple_contrast_sigma{int(sigma)}.png"
        enhanced_var_np = enhanced_var.cpu().permute(1, 2, 0).numpy()
        Image.fromarray(enhanced_var_np).save(output_path_var)
        print(f"Sigma={sigma:.1f}: saved to {output_path_var}")

    # Try different contrast factors with auto sigma
    print("\n--- Trying different contrast factors (auto sigma) ---")
    for factor in [2.0, 4.0, 6.0]:
        enhanced_var = simple_fundus_enhance(
            image_tensor, sigma=None, contrast_factor=factor, device="cuda"
        )
        output_path_var = f"outputs/08_simple_contrast_factor{int(factor)}.png"
        enhanced_var_np = enhanced_var.cpu().permute(1, 2, 0).numpy()
        Image.fromarray(enhanced_var_np).save(output_path_var)
        print(f"Contrast factor={factor}: saved to {output_path_var}")


if __name__ == "__main__":
    main()
