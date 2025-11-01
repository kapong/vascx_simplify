"""Backward compatibility module - imports from utils subpackage.

This module maintains backward compatibility for existing code that imports
from vascx_simplify.utils directly. All new code should import from the
utils subpackage (e.g., from vascx_simplify.utils import from_huggingface).
"""

# Re-export everything from the utils subpackage
from .utils import *  # noqa: F401, F403

# Explicitly list exports to satisfy linters (symbols are imported via wildcard above)
__all__ = [  # noqa: F405
    "from_huggingface",  # noqa: F405
    "tensor_to_numpy",  # noqa: F405
    "pil_to_numpy",  # noqa: F405
    "numpy_to_pil",  # noqa: F405
    "resize_to_original",  # noqa: F405
    "create_color_mask",  # noqa: F405
    "blend_images",  # noqa: F405
    "calculate_class_statistics",  # noqa: F405
    "calculate_centroid",  # noqa: F405
    "calculate_area_percentage",  # noqa: F405
    "calculate_vessel_ratio",  # noqa: F405
    "calculate_bounding_box",  # noqa: F405
    "create_segmentation_overlay",  # noqa: F405
    "create_artery_vein_overlay",  # noqa: F405
    "create_disc_overlay",  # noqa: F405
    "draw_fovea_marker",  # noqa: F405
    "draw_quality_badge",  # noqa: F405
    "create_side_by_side",  # noqa: F405
    "create_comparison_grid",  # noqa: F405
    "ARTERY_VEIN_COLORS",  # noqa: F405
    "DISC_COLORS",  # noqa: F405
    "QUALITY_COLORS",  # noqa: F405
    "QUALITY_LABELS",  # noqa: F405
]
