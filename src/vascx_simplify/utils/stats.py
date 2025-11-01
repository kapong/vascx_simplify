"""Statistical calculation utilities."""

from typing import Dict, Tuple

import numpy as np


def calculate_class_statistics(class_map: np.ndarray) -> Dict[int, int]:
    """Calculate pixel counts for each class.

    Args:
        class_map: Class predictions [H, W] with integer class IDs

    Returns:
        Dictionary mapping class ID to pixel count

    Example:
        >>> stats = calculate_class_statistics(predictions)
        >>> print(f"Artery pixels: {stats[1]}")
    """
    unique_classes, counts = np.unique(class_map, return_counts=True)
    return dict(zip(unique_classes.astype(int).tolist(), counts.astype(int).tolist()))


def calculate_centroid(binary_mask: np.ndarray) -> Tuple[float, float]:
    """Calculate centroid (center of mass) of binary mask.

    Args:
        binary_mask: Binary mask [H, W] with 1 for foreground, 0 for background

    Returns:
        Tuple of (x, y) coordinates of centroid

    Example:
        >>> centroid_x, centroid_y = calculate_centroid(disc_mask)
    """
    y_coords, x_coords = np.where(binary_mask > 0)

    if len(x_coords) == 0:
        return (0.0, 0.0)

    centroid_x = float(np.mean(x_coords))
    centroid_y = float(np.mean(y_coords))

    return (centroid_x, centroid_y)


def calculate_area_percentage(binary_mask: np.ndarray, total_pixels: int = None) -> float:
    """Calculate area percentage of binary mask.

    Args:
        binary_mask: Binary mask [H, W]
        total_pixels: Total number of pixels (if None, uses mask size)

    Returns:
        Percentage of foreground pixels (0-100)
    """
    foreground_pixels = np.sum(binary_mask > 0)

    if total_pixels is None:
        total_pixels = binary_mask.size

    return (foreground_pixels / total_pixels) * 100


def calculate_vessel_ratio(artery_mask: np.ndarray, vein_mask: np.ndarray) -> float:
    """Calculate artery-to-vein pixel ratio.

    Args:
        artery_mask: Binary mask for arteries [H, W]
        vein_mask: Binary mask for veins [H, W]

    Returns:
        Artery/vein ratio (0 if no veins detected)

    Example:
        >>> av_ratio = calculate_vessel_ratio(pred == 1, pred == 2)
    """
    artery_pixels = np.sum(artery_mask > 0)
    vein_pixels = np.sum(vein_mask > 0)

    if vein_pixels == 0:
        return 0.0

    return float(artery_pixels / vein_pixels)


def calculate_bounding_box(binary_mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Calculate bounding box of binary mask.

    Args:
        binary_mask: Binary mask [H, W]

    Returns:
        Tuple of (x_min, y_min, x_max, y_max)

    Example:
        >>> x_min, y_min, x_max, y_max = calculate_bounding_box(disc_mask)
    """
    y_coords, x_coords = np.where(binary_mask > 0)

    if len(x_coords) == 0:
        return (0, 0, 0, 0)

    x_min = int(np.min(x_coords))
    x_max = int(np.max(x_coords))
    y_min = int(np.min(y_coords))
    y_max = int(np.max(y_coords))

    return (x_min, y_min, x_max, y_max)
