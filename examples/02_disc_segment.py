"""
Optic Disc Segmentation Example

This example demonstrates how to use the vascx_simplify library to perform
optic disc segmentation on fundus images.

The model outputs predictions with 2 classes:
- Class 0: Background
- Class 1: Optic Disc (yellow)
"""

from vascx_simplify import EnsembleSegmentation, VASCXTransform, from_huggingface
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    # Configuration
    IMG_PATH = 'HRF_07_dr.jpg'  # Path to your fundus image
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Step 1: Download and load the disc segmentation model
    print("Loading disc segmentation model...")
    model_path = from_huggingface('Eyened/vascx:disc/disc_july24.pt')
    
    # Step 2: Initialize preprocessing transform and model
    transform = VASCXTransform(size=512, have_ce=True, device=DEVICE)
    model = EnsembleSegmentation(model_path, transform, device=DEVICE)
    
    # Step 3: Load and process the image
    print(f"Loading image: {IMG_PATH}")
    rgb_image = Image.open(IMG_PATH)
    
    # Step 4: Run prediction
    print("Running segmentation...")
    prediction = model.predict(rgb_image)  # Returns tensor [B, H, W], values are class
    
    # Step 5: Process predictions
    # Get class predictions
    pred_classes = prediction[0].cpu().numpy()  # [H, W]
    
    # Get probability map for the disc class (if available from raw predictions)
    # Note: For binary mask, we just check for disc class (1)
    prob_map = (pred_classes == 1).astype(float)  # [H, W]
    
    # Create colored segmentation mask
    segmentation_mask = np.zeros((*pred_classes.shape, 3), dtype=np.uint8)
    segmentation_mask[pred_classes == 1] = [255, 255, 0]    # Optic disc in yellow
    
    # Step 6: Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original image
    axes[0, 0].imshow(rgb_image)
    axes[0, 0].set_title('Original Fundus Image')
    axes[0, 0].axis('off')
    
    # Segmentation mask
    axes[0, 1].imshow(segmentation_mask)
    axes[0, 1].set_title('Optic Disc Segmentation\n(Yellow: Optic Disc)')
    axes[0, 1].axis('off')
    
    # Probability heatmap
    im = axes[1, 0].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('Probability Heatmap')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Overlay
    rgb_array = np.array(rgb_image)
    overlay = rgb_array.copy()
    alpha = 0.4
    overlay[pred_classes == 1] = (1 - alpha) * overlay[pred_classes == 1] + alpha * np.array([255, 255, 0])
    axes[1, 1].imshow(overlay.astype(np.uint8))
    axes[1, 1].set_title('Overlay')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('disc_segmentation_result.png', dpi=150, bbox_inches='tight')
    print("Result saved as 'disc_segmentation_result.png'")
    plt.show()
    
    # Step 7: Print statistics
    disc_pixels = np.sum(pred_classes == 1)
    total_pixels = pred_classes.size
    disc_area_percentage = (disc_pixels / total_pixels) * 100
    
    # Calculate centroid
    if disc_pixels > 0:
        y_coords, x_coords = np.where(pred_classes == 1)
        centroid_y = int(np.mean(y_coords))
        centroid_x = int(np.mean(x_coords))
        
        print("\nSegmentation Statistics:")
        print(f"  Optic disc pixels: {disc_pixels:,}")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Disc area: {disc_area_percentage:.2f}%")
        print(f"  Disc centroid: ({centroid_x}, {centroid_y})")
    else:
        print("\nNo optic disc detected.")


if __name__ == "__main__":
    main()
