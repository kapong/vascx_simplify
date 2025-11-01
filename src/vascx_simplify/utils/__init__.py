"""Utility functions for vascx_simplify.

This module provides utilities for:
- HuggingFace model loading
- Image processing and conversion
- Visualization and overlay creation
- Statistical calculations
"""

# HuggingFace utilities
from .huggingface import from_huggingface

# Image utilities
from .image import (
    blend_images,
    create_color_mask,
    numpy_to_pil,
    pil_to_numpy,
    resize_to_original,
    tensor_to_numpy,
)

# Statistical utilities
from .stats import (
    calculate_area_percentage,
    calculate_bounding_box,
    calculate_centroid,
    calculate_class_statistics,
    calculate_vessel_ratio,
)

# Visualization utilities
from .visualization import (
    ARTERY_VEIN_COLORS,
    DISC_COLORS,
    QUALITY_COLORS,
    QUALITY_LABELS,
    create_artery_vein_overlay,
    create_comparison_grid,
    create_disc_overlay,
    create_segmentation_overlay,
    create_side_by_side,
    draw_fovea_marker,
    draw_quality_badge,
)

__all__ = [
    # HuggingFace
    "from_huggingface",
    # Image utilities
    "tensor_to_numpy",
    "pil_to_numpy",
    "numpy_to_pil",
    "resize_to_original",
    "create_color_mask",
    "blend_images",
    # Stats utilities
    "calculate_class_statistics",
    "calculate_centroid",
    "calculate_area_percentage",
    "calculate_vessel_ratio",
    "calculate_bounding_box",
    # Visualization utilities
    "create_segmentation_overlay",
    "create_artery_vein_overlay",
    "create_disc_overlay",
    "draw_fovea_marker",
    "draw_quality_badge",
    "create_side_by_side",
    "create_comparison_grid",
    # Constants
    "ARTERY_VEIN_COLORS",
    "DISC_COLORS",
    "QUALITY_COLORS",
    "QUALITY_LABELS",
]
