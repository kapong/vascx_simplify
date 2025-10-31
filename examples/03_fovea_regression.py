"""
Fovea Detection Example

This example demonstrates how to use the vascx_simplify library to perform
fovea detection on fundus images using heatmap regression.

The model outputs predictions as (x, y) coordinates for the fovea center.
"""

from vascx_simplify import HeatmapRegressionEnsemble, VASCXTransform, from_huggingface
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    # Configuration
    IMG_PATH = 'HRF_07_dr.jpg'  # Path to your fundus image
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Step 1: Download and load the fovea detection model
    print("Loading fovea detection model...")
    model_path = from_huggingface('Eyened/vascx:fovea/fovea_july24.pt')
    
    # Step 2: Initialize preprocessing transform and model
    transform = VASCXTransform(size=1024, have_ce=True, device=DEVICE)
    model = HeatmapRegressionEnsemble(model_path, transform, device=DEVICE)
    
    # Step 3: Load and process the image
    print(f"Loading image: {IMG_PATH}")
    rgb_image = Image.open(IMG_PATH)
    
    # Step 4: Run prediction
    print("Running fovea detection...")
    prediction = model.predict(rgb_image)  # Returns tensor [B, M, 2] where M=models, 2=(x,y)
    
    # Step 5: Extract fovea coordinates
    # prediction shape: [batch, num_models, 2]
    # Get coordinates from first image, first model
    fovea_x = prediction[0, 0, 0].cpu().item()
    fovea_y = prediction[0, 0, 1].cpu().item()
    
    print(f"\nDetected Fovea Location:")
    print(f"  X coordinate: {fovea_x:.2f}")
    print(f"  Y coordinate: {fovea_y:.2f}")
    
    # If there are multiple models in the ensemble, show statistics
    num_models = prediction.shape[1]
    if num_models > 1:
        all_x = prediction[0, :, 0].cpu().numpy()
        all_y = prediction[0, :, 1].cpu().numpy()
        print(f"\nEnsemble Statistics ({num_models} models):")
        print(f"  X range: [{all_x.min():.2f}, {all_x.max():.2f}]")
        print(f"  Y range: [{all_y.min():.2f}, {all_y.max():.2f}]")
        print(f"  X std dev: {all_x.std():.2f}")
        print(f"  Y std dev: {all_y.std():.2f}")
    
    # Step 6: Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original image with fovea marker
    axes[0].imshow(rgb_image)
    axes[0].scatter(fovea_x, fovea_y, c='red', s=200, marker='x', linewidths=3, label='Detected Fovea')
    axes[0].scatter(fovea_x, fovea_y, c='red', s=800, facecolors='none', edgecolors='red', linewidths=2)
    axes[0].set_title('Fovea Detection')
    axes[0].axis('off')
    axes[0].legend(loc='upper right')
    
    # Zoomed view around fovea
    zoom_size = 200  # pixels
    rgb_array = np.array(rgb_image)
    height, width = rgb_array.shape[:2]
    
    x_min = max(0, int(fovea_x - zoom_size))
    x_max = min(width, int(fovea_x + zoom_size))
    y_min = max(0, int(fovea_y - zoom_size))
    y_max = min(height, int(fovea_y + zoom_size))
    
    zoomed = rgb_array[y_min:y_max, x_min:x_max]
    axes[1].imshow(zoomed)
    
    # Adjust marker position for zoomed view
    marker_x = fovea_x - x_min
    marker_y = fovea_y - y_min
    axes[1].scatter(marker_x, marker_y, c='red', s=200, marker='x', linewidths=3, label='Fovea Center')
    axes[1].scatter(marker_x, marker_y, c='red', s=800, facecolors='none', edgecolors='red', linewidths=2)
    axes[1].set_title(f'Zoomed View (±{zoom_size}px)')
    axes[1].axis('off')
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('fovea_detection_result.png', dpi=150, bbox_inches='tight')
    print("\nResult saved as 'fovea_detection_result.png'")
    plt.show()
    
    # Step 7: Additional visualization - simple scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(rgb_image)
    ax.scatter(prediction[0, 0, 0].cpu(), prediction[0, 0, 1].cpu(), 
               c='yellow', s=300, marker='*', edgecolors='red', linewidths=2, 
               label='Fovea', zorder=5)
    ax.set_title('Fundus Image with Fovea Location')
    ax.axis('off')
    ax.legend()
    plt.tight_layout()
    plt.savefig('fovea_simple_result.png', dpi=150, bbox_inches='tight')
    print("Simple result saved as 'fovea_simple_result.png'")
    plt.show()


if __name__ == "__main__":
    main()
