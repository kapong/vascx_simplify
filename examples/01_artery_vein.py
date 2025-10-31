"""
Artery/Vein Segmentation Example

This example demonstrates how to use the vascx_simplify library to perform
artery and vein segmentation on fundus images.

The model outputs predictions with 4 classes:
- Class 0: Background
- Class 1: Arteries (red)
- Class 2: Veins (blue)
- Class 3: Crossings (green)
"""

from vascx_simplify import EnsembleSegmentation, VASCXTransform, from_huggingface
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Configuration
    IMG_PATH = 'HRF_07_dr.jpg'  # Path to your fundus image
    
    # Step 1: Download and load the artery/vein segmentation model
    print("Loading artery/vein segmentation model...")
    model_path = from_huggingface('Eyened/vascx:artery_vein/av_july24.pt')
    
    # Step 2: Initialize preprocessing transform and model
    model = EnsembleSegmentation(model_path, VASCXTransform())
    
    # Step 3: Load and process the image
    print(f"Loading image: {IMG_PATH}")
    rgb_image = Image.open(IMG_PATH)
    
    # Step 4: Run prediction
    print("Running segmentation...")
    prediction = model.predict(rgb_image)  # Returns tensor [B, H, W], values are class
    
    # Step 5: Process predictions
    # Get class predictions
    pred_classes = prediction[0].cpu().numpy()  # [H, W]
    
    # Resize prediction to match original image size if needed
    rgb_array = np.array(rgb_image)
    
    # Create colored segmentation mask
    segmentation_mask = np.zeros((*pred_classes.shape, 3), dtype=np.uint8)
    segmentation_mask[pred_classes == 1] = [255, 0, 0]    # Arteries in red
    segmentation_mask[pred_classes == 2] = [0, 0, 255]    # Veins in blue
    segmentation_mask[pred_classes == 3] = [0, 255, 0]    # Crossings in green
    
    # Step 6: Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(rgb_image)
    axes[0].set_title('Original Fundus Image')
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(segmentation_mask)
    axes[1].set_title('Artery/Vein Segmentation\n(Red: Arteries, Blue: Veins, Green: Crossings)')
    axes[1].axis('off')
    
    # Overlay
    overlay = rgb_array.copy().astype(float)
    alpha = 0.5
    overlay[pred_classes == 1] = (1 - alpha) * overlay[pred_classes == 1] + alpha * np.array([255, 0, 0])
    overlay[pred_classes == 2] = (1 - alpha) * overlay[pred_classes == 2] + alpha * np.array([0, 0, 255])
    overlay[pred_classes == 3] = (1 - alpha) * overlay[pred_classes == 3] + alpha * np.array([0, 255, 0])
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('artery_vein_segmentation_result.png', dpi=150, bbox_inches='tight')
    print("Result saved as 'artery_vein_segmentation_result.png'")
    plt.show()
    
    # Step 7: Print statistics
    artery_pixels = np.sum(pred_classes == 1)
    vein_pixels = np.sum(pred_classes == 2)
    crossing_pixels = np.sum(pred_classes == 3)
    total_vessel_pixels = artery_pixels + vein_pixels + crossing_pixels
    artery_vein_ratio = artery_pixels / vein_pixels if vein_pixels > 0 else 0
    
    print("\nSegmentation Statistics:")
    print(f"  Artery pixels: {artery_pixels:,}")
    print(f"  Vein pixels: {vein_pixels:,}")
    print(f"  Crossing pixels: {crossing_pixels:,}")
    print(f"  Total vessel pixels: {total_vessel_pixels:,}")
    print(f"  Artery/Vein ratio: {artery_vein_ratio:.2f}")


if __name__ == "__main__":
    main()