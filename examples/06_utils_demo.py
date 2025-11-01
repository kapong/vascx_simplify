"""
Utils Demo: Simplified Visualization with Helper Functions

This example demonstrates how to use the new utility functions to create
overlays and visualizations for all model types with minimal code.
"""

import matplotlib.pyplot as plt
from PIL import Image

from vascx_simplify import (
    ClassificationEnsemble,
    EnsembleSegmentation,
    HeatmapRegressionEnsemble,
    VASCXTransform,
    from_huggingface,
)
from vascx_simplify.utils import (
    calculate_class_statistics,
    calculate_vessel_ratio,
    create_artery_vein_overlay,
    create_disc_overlay,
    draw_fovea_marker,
    draw_quality_badge,
    pil_to_numpy,
    tensor_to_numpy,
)


def demo_artery_vein():
    """Demonstrate artery/vein segmentation with overlay utility."""
    print("\n=== Artery/Vein Segmentation Demo ===")
    
    # Load model and image
    model_path = from_huggingface("Eyened/vascx:artery_vein/av_july24.pt")
    model = EnsembleSegmentation(model_path, VASCXTransform())
    image = Image.open("HRF_07_dr.jpg")
    
    # Predict
    prediction = model.predict(image)
    pred_np = tensor_to_numpy(prediction[0])  # [H, W]
    
    # Create overlay with one function call!
    overlay = create_artery_vein_overlay(image, pred_np, alpha=0.5)
    
    # Calculate statistics
    stats = calculate_class_statistics(pred_np)
    av_ratio = calculate_vessel_ratio(pred_np == 1, pred_np == 2)
    
    print(f"Artery pixels: {stats.get(1, 0):,}")
    print(f"Vein pixels: {stats.get(2, 0):,}")
    print(f"A/V ratio: {av_ratio:.2f}")
    
    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Artery/Vein Overlay (A/V={av_ratio:.2f})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("demo_av_overlay.png", dpi=150, bbox_inches="tight")
    print("Saved: demo_av_overlay.png")
    plt.close()


def demo_disc():
    """Demonstrate disc segmentation with overlay utility."""
    print("\n=== Optic Disc Segmentation Demo ===")
    
    # Load model and image
    model_path = from_huggingface("Eyened/vascx:disc/disc_july24.pt")
    model = EnsembleSegmentation(model_path, VASCXTransform(512))
    image = Image.open("HRF_07_dr.jpg")
    
    # Predict
    prediction = model.predict(image)
    pred_np = tensor_to_numpy(prediction[0])
    
    # Create overlay with one function call!
    overlay = create_disc_overlay(image, pred_np, alpha=0.4)
    
    # Calculate statistics
    stats = calculate_class_statistics(pred_np)
    disc_pixels = stats.get(1, 0)
    
    print(f"Disc pixels: {disc_pixels:,}")
    
    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title(f"Disc Overlay ({disc_pixels:,} pixels)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("demo_disc_overlay.png", dpi=150, bbox_inches="tight")
    print("Saved: demo_disc_overlay.png")
    plt.close()


def demo_fovea():
    """Demonstrate fovea detection with marker utility."""
    print("\n=== Fovea Detection Demo ===")
    
    # Load model and image
    model_path = from_huggingface("Eyened/vascx:fovea/fovea_july24.pt")
    model = HeatmapRegressionEnsemble(model_path, VASCXTransform())
    image = Image.open("HRF_07_dr.jpg")
    
    # Predict
    prediction = model.predict(image)
    fovea_x = prediction[0, 0, 0].cpu().item()
    fovea_y = prediction[0, 0, 1].cpu().item()
    
    print(f"Fovea location: ({fovea_x:.1f}, {fovea_y:.1f})")
    
    # Draw marker with one function call!
    image_np = pil_to_numpy(image)
    marked = draw_fovea_marker(image_np, fovea_x, fovea_y, marker_size=150)
    
    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(marked)
    axes[1].set_title(f"Fovea: ({fovea_x:.0f}, {fovea_y:.0f})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("demo_fovea_marker.png", dpi=150, bbox_inches="tight")
    print("Saved: demo_fovea_marker.png")
    plt.close()


def demo_quality():
    """Demonstrate quality classification with badge utility."""
    print("\n=== Quality Classification Demo ===")
    
    # Load model and image
    model_path = from_huggingface("Eyened/vascx:quality/quality.pt")
    model = ClassificationEnsemble(model_path, VASCXTransform(use_ce=False))
    image = Image.open("HRF_07_dr.jpg")
    
    # Predict
    prediction = model.predict(image)
    probs = tensor_to_numpy(prediction[0])
    quality_class = int(probs.argmax())
    confidence = float(probs[quality_class])
    
    quality_labels = ["REJECT", "USABLE", "GOOD"]
    print(f"Quality: {quality_labels[quality_class]} ({confidence:.1%})")
    
    # Draw badge with one function call!
    image_np = pil_to_numpy(image)
    badged = draw_quality_badge(image_np, quality_class, confidence, position="top-left")
    
    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(badged)
    axes[1].set_title(f"{quality_labels[quality_class]} ({confidence:.0%})")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("demo_quality_badge.png", dpi=150, bbox_inches="tight")
    print("Saved: demo_quality_badge.png")
    plt.close()


def main():
    """Run all demos."""
    print("VASCX Simplify - Utils Demo")
    print("=" * 50)
    print("\nThis demo shows how the new utility functions")
    print("simplify visualization for all model types.")
    
    demo_artery_vein()
    demo_disc()
    demo_fovea()
    demo_quality()
    
    print("\n" + "=" * 50)
    print("All demos complete! Check the output images.")


if __name__ == "__main__":
    main()
