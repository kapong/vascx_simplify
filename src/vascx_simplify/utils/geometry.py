"""Geometric operations and transformations."""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def create_coordinate_grid(
    h: int, w: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create coordinate grids for given dimensions.

    Args:
        h: Grid height
        w: Grid width
        device: Target device
        dtype: Data type for grid (default: float32)

    Returns:
        Tuple of (x_grid, y_grid) coordinate meshgrids [H, W]

    Example:
        >>> x_grid, y_grid = create_coordinate_grid(256, 256, device='cuda')
        >>> # x_grid[i, j] = j, y_grid[i, j] = i
    """
    y_coords = torch.arange(h, device=device, dtype=dtype)
    x_coords = torch.arange(w, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
    return (x_grid, y_grid)


def polar_transform(
    image: torch.Tensor,
    center: Tuple[float, float],
    max_radius: float,
    output_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert image to polar coordinates using GPU grid_sample.

    Args:
        image: Input image [H, W]
        center: Center point (y, x)
        max_radius: Maximum radius for sampling
        output_size: Size of output polar image (both height and width)
        device: Target device
        dtype: Data type for computation

    Returns:
        Polar transformed image [output_size, output_size]

    Example:
        >>> polar = polar_transform(image, (128, 128), 100, 256, device='cuda')
    """
    cy, cx = center

    # Create polar grid on GPU
    theta = torch.linspace(0, 2 * np.pi, output_size, device=device, dtype=dtype)
    radius = torch.linspace(0, max_radius, output_size, device=device, dtype=dtype)

    theta_grid, radius_grid = torch.meshgrid(theta, radius, indexing="ij")

    # Convert to Cartesian
    x = cx + radius_grid * torch.cos(theta_grid)
    y = cy + radius_grid * torch.sin(theta_grid)

    # Normalize to [-1, 1] for grid_sample
    h, w = image.shape[-2:]
    x_norm = 2 * x / (w - 1) - 1
    y_norm = 2 * y / (h - 1) - 1

    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    # Sample using GPU
    image_batch = image.unsqueeze(0).unsqueeze(0).to(dtype)  # [1, 1, H, W]
    polar = F.grid_sample(
        image_batch, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    return polar.squeeze(0).squeeze(0)  # [H, W]


def calculate_circle_bounds(
    center: Tuple[float, float], radius: float, image_size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """Calculate bounding box for a circle.

    Args:
        center: Circle center (y, x)
        radius: Circle radius
        image_size: Image dimensions (height, width)

    Returns:
        Tuple of (y_min, y_max, x_min, x_max) clamped to image bounds

    Example:
        >>> y_min, y_max, x_min, x_max = calculate_circle_bounds((128, 128), 50, (256, 256))
    """
    cy, cx = center
    h, w = image_size

    y_min = max(0, int(cy - radius))
    y_max = min(h, int(cy + radius))
    x_min = max(0, int(cx - radius))
    x_max = min(w, int(cx + radius))

    return (y_min, y_max, x_min, x_max)


def create_circular_mask(
    size: Tuple[int, int],
    center: Tuple[float, float],
    radius: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    scale_factor: float = 1.0,
) -> torch.Tensor:
    """Create a circular binary mask.

    Args:
        size: Mask size (height, width)
        center: Circle center (y, x)
        radius: Circle radius
        device: Target device
        dtype: Data type for computation
        scale_factor: Scale factor for radius (default: 1.0)

    Returns:
        Binary mask [H, W] with 1 inside circle, 0 outside

    Example:
        >>> mask = create_circular_mask((256, 256), (128, 128), 100, device='cuda')
    """
    h, w = size
    cy, cx = center

    x_grid, y_grid = create_coordinate_grid(h, w, device, dtype)

    r_norm = torch.sqrt(
        ((x_grid - cx) / (scale_factor * radius)) ** 2
        + ((y_grid - cy) / (scale_factor * radius)) ** 2
    )

    mask = (r_norm < 1).float()
    return mask


def line_circle_intersection(
    line_p0: np.ndarray, line_p1: np.ndarray, circle_center: np.ndarray, circle_radius: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate intersection points between a line and circle.

    Args:
        line_p0: Line start point [2]
        line_p1: Line end point [2]
        circle_center: Circle center [2]
        circle_radius: Circle radius

    Returns:
        Tuple of two intersection points as numpy arrays

    Example:
        >>> p0 = np.array([0, 0])
        >>> p1 = np.array([100, 100])
        >>> center = np.array([50, 50])
        >>> int1, int2 = line_circle_intersection(p0, p1, center, 30)
    """
    p0 = np.array(line_p0)
    p1 = np.array(line_p1)
    center = np.array(circle_center)

    d = p1 - p0
    a = d.dot(d)
    b = 2 * d.dot(p0 - center)
    c = p0.dot(p0) + center.dot(center) - 2 * p0.dot(center) - circle_radius**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        # No intersection, return line endpoints
        return (p0, p1)

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)

    intersect1 = p0 + t1 * d
    intersect2 = p0 + t2 * d

    return (intersect1, intersect2)


def apply_circular_crop(
    image: torch.Tensor, center: Tuple[float, float], radius: float, fill_value: float = 0.0
) -> torch.Tensor:
    """Apply circular crop to image (set pixels outside circle to fill_value).

    Args:
        image: Input image [C, H, W] or [H, W]
        center: Circle center (y, x)
        radius: Circle radius
        fill_value: Value for pixels outside circle (default: 0)

    Returns:
        Cropped image with same shape as input

    Example:
        >>> cropped = apply_circular_crop(image, (128, 128), 100)
    """
    device = image.device
    compute_dtype = image.dtype if image.dtype in [torch.float32, torch.float16] else torch.float32

    if image.dim() == 2:
        h, w = image.shape
        is_2d = True
    else:
        _, h, w = image.shape
        is_2d = False

    mask = create_circular_mask((h, w), center, radius, device, dtype=torch.float32)

    if is_2d:
        result = image.float() * mask + fill_value * (1 - mask)
    else:
        result = image.float() * mask.unsqueeze(0) + fill_value * (1 - mask.unsqueeze(0))

    return result.to(compute_dtype)


def get_circle_coordinates(
    center: Tuple[float, float], radius: float, num_points: int = 360
) -> Tuple[np.ndarray, np.ndarray]:
    """Get coordinates of points on a circle perimeter.

    Args:
        center: Circle center (y, x)
        radius: Circle radius
        num_points: Number of points to generate (default: 360)

    Returns:
        Tuple of (x_coords, y_coords) as numpy arrays

    Example:
        >>> x_coords, y_coords = get_circle_coordinates((128, 128), 50, num_points=100)
    """
    cy, cx = center
    theta = np.linspace(0, 2 * np.pi, num_points)
    x_coords = cx + radius * np.cos(theta)
    y_coords = cy + radius * np.sin(theta)
    return (x_coords, y_coords)
