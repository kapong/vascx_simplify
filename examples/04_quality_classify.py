"""
Fundus Image Quality Classification Example

This example demonstrates how to use the vascx_simplify library to perform
quality assessment on fundus images.

The model outputs predictions with 3 quality classes aligned with EyeQ scores:
- q1: "Reject" - Poor quality, unusable for diagnosis
- q2: "Usable" - Acceptable quality, can be used with caution
- q3: "Good" - High quality, suitable for diagnosis

The output scores are already softmaxed and sum to 1.0 for probability interpretation.
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vascx_simplify import ClassificationEnsemble, VASCXTransform, from_huggingface


def main():
    # Configuration
    IMG_PATH = "HRF_07_dr.jpg"  # Path to your fundus image

    # Step 1: Download and load the quality classification model
    print("Loading quality classification model...")
    model_path = from_huggingface("Eyened/vascx:quality/quality.pt")

    # Step 2: Initialize preprocessing transform and model
    # Note: use_ce=False means no contrast enhancement for quality assessment
    model = ClassificationEnsemble(model_path, VASCXTransform(use_ce=False))

    # Step 3: Load and process the image
    print(f"Loading image: {IMG_PATH}")
    rgb_image = Image.open(IMG_PATH)

    # Step 4: Run prediction
    print("Running quality assessment...")
    prediction = model.predict(
        rgb_image
    )  # Returns tensor [B, 3] with quality scores (already softmaxed)

    # Step 5: Process predictions
    # Get probabilities (already normalized)
    probabilities = prediction[0].cpu().numpy()  # [3] - probabilities for q1, q2, q3

    quality_labels = ["Reject (q1)", "Usable (q2)", "Good (q3)"]

    print("\nQuality Assessment Results:")
    print("-" * 40)
    print(f"Probabilities:")
    for i, label in enumerate(quality_labels):
        print(f"  {label}: {probabilities[i]:.2%}")

    # Determine final quality rating
    predicted_class = np.argmax(probabilities)
    quality_rating = ["REJECT", "USABLE", "GOOD"][predicted_class]
    quality_color = ["red", "orange", "green"][predicted_class]

    print(f"\nFinal Quality Rating: {quality_rating}")
    print(f"Confidence: {probabilities[predicted_class]:.2%}")

    # Step 6: Visualize results
    fig = plt.figure(figsize=(14, 6))

    # Create grid for subplots
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])

    # Original image (spans both rows)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.imshow(rgb_image)
    ax1.set_title(
        f"Fundus Image\nQuality: {quality_rating} ({probabilities[predicted_class]:.1%})",
        fontsize=14,
        fontweight="bold",
        color=quality_color,
    )
    ax1.axis("off")

    # Add quality badge to image
    bbox_props = dict(
        boxstyle="round,pad=0.5", facecolor=quality_color, alpha=0.8, edgecolor="black", linewidth=2
    )
    ax1.text(
        0.05,
        0.95,
        quality_rating,
        transform=ax1.transAxes,
        fontsize=20,
        fontweight="bold",
        color="white",
        verticalalignment="top",
        bbox=bbox_props,
    )

    # Bar chart of probabilities
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    bars = ax2.bar(
        range(3), probabilities, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
    )
    bars[predicted_class].set_alpha(1.0)
    bars[predicted_class].set_edgecolor("black")
    bars[predicted_class].set_linewidth(3)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["Reject\n(q1)", "Usable\n(q2)", "Good\n(q3)"], fontsize=10)
    ax2.set_ylabel("Probability", fontsize=11, fontweight="bold")
    ax2.set_title("Quality Score Distribution", fontsize=12, fontweight="bold")
    ax2.set_ylim([0, 1])
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # Add percentage labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{prob:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Quality interpretation guide
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")

    guide_text = """Quality Guide:
    
🔴 Reject (q1):
   Poor quality, unusable
   for clinical diagnosis
   
🟠 Usable (q2):
   Acceptable quality,
   use with caution
   
🟢 Good (q3):
   High quality, suitable
   for diagnosis"""

    ax3.text(
        0.1,
        0.9,
        guide_text,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightgray", alpha=0.3),
    )

    plt.tight_layout()
    plt.savefig("quality_classification_result.png", dpi=150, bbox_inches="tight")
    print("\nResult saved as 'quality_classification_result.png'")
    plt.show()


if __name__ == "__main__":
    main()
