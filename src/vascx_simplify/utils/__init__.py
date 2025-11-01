"""Utility functions for vascx_simplify.

This module provides utilities for:
- HuggingFace model loading
- Image processing and conversion
- Visualization and overlay creation
- Statistical calculations
- Geometric operations
- Coordinate transformations
"""

# HuggingFace utilities
from .huggingface import from_huggingface

# Geometric utilities
from .geometry import (
    apply_circular_crop,
    calculate_circle_bounds,
    create_circular_mask,
    create_coordinate_grid,
    get_circle_coordinates,
    line_circle_intersection,
    polar_transform,
)

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

# Transformation utilities
from .transforms import (
    create_affine_matrix,
    crop_and_resize,
    invert_affine_matrix,
    resize_with_aspect_ratio,
    transform_point,
    transform_points_batch,
    undo_crop_and_resize,
    warp_image_affine,
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
    # Geometric utilities
    "create_coordinate_grid",
    "polar_transform",
    "calculate_circle_bounds",
    "create_circular_mask",
    "line_circle_intersection",
    "apply_circular_crop",
    "get_circle_coordinates",
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
    # Transformation utilities
    "create_affine_matrix",
    "invert_affine_matrix",
    "transform_point",
    "transform_points_batch",
    "warp_image_affine",
    "resize_with_aspect_ratio",
    "crop_and_resize",
    "undo_crop_and_resize",
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
