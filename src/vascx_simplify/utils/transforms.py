"""Coordinate transformation utilities."""

from typing import Tuple

import kornia.geometry as K_geom
import kornia.geometry.conversions as Kconv
import kornia.geometry.transform as K
import numpy as np
import torch


def create_affine_matrix(
    in_size: Tuple[int, int],
    out_size: int,
    scale: float,
    center: Tuple[float, float],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create affine transformation matrix as torch tensor.

    Args:
        in_size: Input size (height, width)
        out_size: Output size (square dimension)
        scale: Scale factor
        center: Center point (y, x) for transformation
        device: Target device
        dtype: Data type for matrix (default: float32 for precision)

    Returns:
        Affine matrix [2, 3]

    Example:
        >>> M = create_affine_matrix((512, 512), 256, 0.5, (256, 256), device='cuda')
    """
    cy, cx = center
    ty, tx = out_size / 2, out_size / 2
    return torch.tensor(
        [[scale, 0, tx - scale * cx], [0, scale, ty - scale * cy]], dtype=dtype, device=device
    )


def invert_affine_matrix(M: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert 2x3 affine matrix to 3x3 and invert (float32 for precision).

    Args:
        M: 2x3 affine transformation matrix
        device: Target device

    Returns:
        3x3 inverted matrix (float32)

    Example:
        >>> M_inv = invert_affine_matrix(M, device='cuda')
    """
    M_fp32 = M.float()  # Convert to float32 for accurate inverse
    M_3x3 = torch.cat(
        [M_fp32, torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32)], dim=0
    )
    return torch.inverse(M_3x3)


def transform_point(
    point: Tuple[float, float], M_inv: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Transform a point using inverse transformation matrix.

    Args:
        point: (x, y) coordinates
        M_inv: 3x3 inverse transformation matrix (float32)
        device: Target device

    Returns:
        Transformed (x, y) coordinates as numpy array

    Example:
        >>> point_orig = transform_point((128, 128), M_inv, device='cuda')
    """
    point_homo = torch.tensor([point[0], point[1], 1.0], device=device, dtype=torch.float32)
    point_orig_homo = M_inv @ point_homo
    return (point_orig_homo[:2] / point_orig_homo[2]).cpu().numpy()


def transform_points_batch(
    points_tensor: torch.Tensor, M: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Transform a batch of points using affine transformation.

    Args:
        points_tensor: Points to transform [B, N, 2]
        M: Affine transformation matrix [1, 2, 3] or [B, 2, 3]
        inverse: If True, invert the transformation first

    Returns:
        Transformed points [B, N, 2]

    Example:
        >>> points = torch.rand(4, 100, 2)  # 4 images, 100 points each
        >>> M = create_affine_matrix((512, 512), 256, 0.5, (256, 256), device='cuda')
        >>> transformed = transform_points_batch(points, M.unsqueeze(0))
    """
    batch_size = points_tensor.shape[0]

    # Ensure M is in correct shape
    if M.dim() == 2:
        M = M.unsqueeze(0)

    if inverse:
        M = K.invert_affine_transform(M)

    # Convert to homography for point transformation
    T_3x3 = Kconv.convert_affinematrix_to_homography(M)

    # Expand to batch size if needed
    if T_3x3.shape[0] == 1 and batch_size > 1:
        T_batch = T_3x3.repeat(batch_size, 1, 1)
    else:
        T_batch = T_3x3

    # Use float32 for point transformation (precision critical)
    points_fp32 = points_tensor.float()
    transformed_points = K_geom.transform_points(T_batch, points_fp32)

    return transformed_points.to(points_tensor.dtype)


def warp_image_affine(
    image: torch.Tensor,
    M: torch.Tensor,
    output_size: Tuple[int, int],
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """Warp image using affine transformation.

    Args:
        image: Input image [C, H, W] or [B, C, H, W]
        M: Affine matrix [2, 3] or [B, 2, 3]
        output_size: Output dimensions (height, width)
        mode: Interpolation mode ('bilinear' or 'nearest')
        padding_mode: Padding mode for out-of-bounds pixels

    Returns:
        Warped image with shape [C, H_out, W_out] or [B, C, H_out, W_out]

    Example:
        >>> warped = warp_image_affine(image, M, (256, 256))
    """
    # Ensure batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Ensure M has batch dimension
    if M.dim() == 2:
        M = M.unsqueeze(0)

    # Warp
    warped = K.warp_affine(
        image,
        M,
        output_size,
        mode=mode,
        padding_mode=padding_mode,
    )

    if squeeze_output:
        warped = warped.squeeze(0)

    return warped


def resize_with_aspect_ratio(
    image: torch.Tensor, target_size: int, mode: str = "bilinear"
) -> Tuple[torch.Tensor, float]:
    """Resize image maintaining aspect ratio (fit inside target_size).

    Args:
        image: Input image [C, H, W]
        target_size: Target size for the longer dimension
        mode: Interpolation mode

    Returns:
        Tuple of (resized_image, scale_factor)

    Example:
        >>> resized, scale = resize_with_aspect_ratio(image, 256)
    """
    _, h, w = image.shape
    scale = target_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Create affine matrix for scaling
    device = image.device
    dtype = image.dtype if image.dtype in [torch.float32, torch.float16] else torch.float32

    M = torch.tensor([[scale, 0, 0], [0, scale, 0]], dtype=dtype, device=device)

    resized = warp_image_affine(image, M, (new_h, new_w), mode=mode)

    return resized, scale


def crop_and_resize(
    image: torch.Tensor,
    center: Tuple[float, float],
    crop_size: float,
    output_size: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Crop region around center and resize to output size.

    Args:
        image: Input image [C, H, W]
        center: Center point (y, x) for crop
        crop_size: Size of crop (radius in pixels)
        output_size: Output size (square dimension)
        mode: Interpolation mode

    Returns:
        Cropped and resized image [C, output_size, output_size]

    Example:
        >>> cropped = crop_and_resize(image, (256, 256), 128, 224)
    """
    device = image.device
    _, h, w = image.shape

    scale = output_size / (2 * crop_size)
    M = create_affine_matrix((h, w), output_size, scale, center, device)

    return warp_image_affine(image, M, (output_size, output_size), mode=mode)


def undo_crop_and_resize(
    cropped_image: torch.Tensor,
    original_size: Tuple[int, int],
    center: Tuple[float, float],
    crop_size: float,
    output_size: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Undo crop and resize transformation (inverse operation).

    Args:
        cropped_image: Cropped image [C, output_size, output_size]
        original_size: Original image size (height, width)
        center: Center point (y, x) used for crop
        crop_size: Size of crop (radius in pixels)
        output_size: Size of cropped image
        mode: Interpolation mode

    Returns:
        Image in original coordinate system [C, H, W]

    Example:
        >>> restored = undo_crop_and_resize(cropped, (512, 512), (256, 256), 128, 224)
    """
    device = cropped_image.device
    h, w = original_size

    scale = output_size / (2 * crop_size)
    M = create_affine_matrix((h, w), output_size, scale, center, device)

    # Invert the transformation
    if M.dim() == 2:
        M = M.unsqueeze(0)
    M_inv = K.invert_affine_transform(M)

    return warp_image_affine(cropped_image, M_inv, (h, w), mode=mode)


def compute_scale_from_bounds(bounds_size: float, target_size: int) -> float:
    """Compute scale factor from bounds to target size.

    Args:
        bounds_size: Size of bounds (e.g., diameter for circle)
        target_size: Target size in pixels

    Returns:
        Scale factor

    Example:
        >>> scale = compute_scale_from_bounds(200, 256)  # 200px -> 256px
    """
    return target_size / bounds_size
