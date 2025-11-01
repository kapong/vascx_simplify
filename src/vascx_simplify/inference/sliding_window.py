"""Sliding window inference utilities."""

from typing import Callable, Tuple

import torch

# Constants for sliding window inference
GAUSSIAN_SIGMA_FRACTION = 0.125  # Standard sigma for Gaussian importance map
MIN_WEIGHT_THRESHOLD = 1e-8  # Minimum weight threshold to prevent division by zero


def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Tuple[int, int],
    sw_batch_size: int,
    predictor: Callable,
    overlap: float = 0.5,
    mode: str = "gaussian",
) -> torch.Tensor:
    """
    Sliding window inference with proper batch handling.

    Limits total batch size to sw_batch_size by adjusting windows per iteration
    based on input batch size.
    """
    batch_size, channels, height, width = inputs.shape
    window_h, window_w = roi_size

    # MONAI uses this exact calculation
    stride_h = max(1, int(window_h * (1 - overlap)))
    stride_w = max(1, int(window_w * (1 - overlap)))

    # Calculate scan range (MONAI's exact logic)
    scan_h = range(0, height - window_h + 1, stride_h) if height > window_h else [0]
    scan_w = range(0, width - window_w + 1, stride_w) if width > window_w else [0]

    # Add final position if needed
    if height > window_h and (height - window_h) % stride_h != 0:
        scan_h = list(scan_h) + [height - window_h]
    if width > window_w and (width - window_w) % stride_w != 0:
        scan_w = list(scan_w) + [width - window_w]

    # Create importance map
    if mode == "gaussian":
        importance_map = _create_gaussian_importance_map(
            window_h, window_w, inputs.device, inputs.dtype
        )
    else:
        importance_map = torch.ones((window_h, window_w), device=inputs.device, dtype=inputs.dtype)

    # Generate all slice positions
    slices = [(h, h + window_h, w, w + window_w) for h in scan_h for w in scan_w]

    # Get first prediction for shape
    h_start, h_end, w_start, w_end = slices[0]
    with torch.no_grad():
        first_pred = predictor(inputs[:, :, h_start:h_end, w_start:w_end])
        if isinstance(first_pred, tuple):
            first_pred = torch.stack(first_pred, dim=1)

    # Initialize output using helper
    output, importance_map_exp = _initialize_output_tensor(
        first_pred, batch_size, height, width, importance_map, inputs.device, inputs.dtype
    )

    count_map = torch.zeros((batch_size, height, width), device=inputs.device, dtype=inputs.dtype)

    # Adjust windows per iteration to maintain GPU utilization
    # For single image: use sw_batch_size windows
    # For batch: reduce windows to keep total batch reasonable, but not too small
    # Target: total batch size between sw_batch_size and sw_batch_size*2
    if batch_size == 1:
        windows_per_iter = sw_batch_size
    else:
        # For multiple images, aim for sw_batch_size total patches
        windows_per_iter = max(1, sw_batch_size // batch_size)
        # But don't go too small - minimum 1 window per image batch

    # Process windows
    for i in range(0, len(slices), windows_per_iter):
        batch_slices = slices[i : i + windows_per_iter]

        # Extract windows from all batch images at each position
        # This creates [num_windows * batch_size, C, H, W]
        windows = torch.cat(
            [inputs[:, :, h_s:h_e, w_s:w_e] for h_s, h_e, w_s, w_e in batch_slices], dim=0
        )

        # Predict
        with torch.no_grad():
            batch_pred = predictor(windows)
            if isinstance(batch_pred, tuple):
                batch_pred = torch.stack(batch_pred, dim=1)

        # Split and accumulate - results are interleaved by batch
        for j, (h_start, h_end, w_start, w_end) in enumerate(batch_slices):
            # Extract predictions for this window from all batch images
            pred = batch_pred[j * batch_size : (j + 1) * batch_size]

            output[..., h_start:h_end, w_start:w_end] += pred * importance_map_exp
            count_map[:, h_start:h_end, w_start:w_end] += importance_map[None, :, :]

    # Normalize
    if output.dim() == 5:
        output = output / count_map[:, None, None, :, :].clamp(min=MIN_WEIGHT_THRESHOLD)
    else:
        output = output / count_map[:, None, :, :].clamp(min=MIN_WEIGHT_THRESHOLD)

    return output


def _create_gaussian_importance_map(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    Create 2D Gaussian importance map matching MONAI's implementation.
    MONAI uses sigma = 0.125 * roi_size
    """
    # MONAI's exact formula
    center_h = (height - 1) / 2.0
    center_w = (width - 1) / 2.0
    sigma_h = height * GAUSSIAN_SIGMA_FRACTION
    sigma_w = width * GAUSSIAN_SIGMA_FRACTION

    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)

    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

    gaussian = torch.exp(
        -((y_grid - center_h) ** 2 / (2 * sigma_h**2) + (x_grid - center_w) ** 2 / (2 * sigma_w**2))
    )

    return gaussian


def _initialize_output_tensor(
    first_pred: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    importance_map: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Initialize output tensor with correct shape based on prediction dimensions.

    Args:
        first_pred: First prediction to determine output shape
        batch_size: Number of images in batch
        height: Output height
        width: Output width
        importance_map: Importance map for weighting
        device: Target device
        dtype: Data type

    Returns:
        Tuple of (output tensor, expanded importance map)
    """
    if first_pred.dim() == 5:  # (B, M, C, H, W)
        _, n_models, n_classes, _, _ = first_pred.shape
        output = torch.zeros(
            (batch_size, n_models, n_classes, height, width), device=device, dtype=dtype
        )
        importance_map_exp = importance_map[None, None, None, :, :]
    else:  # (B, C, H, W)
        _, n_classes, _, _ = first_pred.shape
        output = torch.zeros((batch_size, n_classes, height, width), device=device, dtype=dtype)
        importance_map_exp = importance_map[None, None, :, :]

    return output, importance_map_exp
