import json
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from .preprocess import VASCXTransform

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Constants for sliding window inference
GAUSSIAN_SIGMA_FRACTION = 0.125  # MONAI's standard sigma for Gaussian importance map
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
    Sliding window inference matching MONAI's implementation exactly.
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

    # Process windows in batches
    for i in range(0, len(slices), sw_batch_size):
        batch_slices = slices[i : i + sw_batch_size]

        # Extract and concatenate windows
        windows = torch.cat(
            [inputs[:, :, h_s:h_e, w_s:w_e] for h_s, h_e, w_s, w_e in batch_slices], dim=0
        )

        # Predict
        with torch.no_grad():
            batch_pred = predictor(windows)
            if isinstance(batch_pred, tuple):
                batch_pred = torch.stack(batch_pred, dim=1)

        # Split and accumulate
        for j, (h_start, h_end, w_start, w_end) in enumerate(batch_slices):
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


class EnsembleBase(torch.nn.Module):
    def __init__(
        self,
        fpath: str,
        transforms: VASCXTransform,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
    ):
        super().__init__()
        self.device = device
        self.transforms = transforms
        self.config: Dict[str, Any] = {"inference": {}}
        self.load_torchscript(fpath)

        self.tta: bool = self.config["inference"].get("tta", False)
        self.inference_config: Dict[str, Any] = self.config["inference"]

        self.inference_fn = self.tta_inference if self.tta else self.sliding_window_inference

        self.sw_batch_size: int = 16

    def load_torchscript(self, fpath: str) -> None:
        extra_files = {"config.yaml": ""}
        extra_files = {"config.yaml": ""}
        self.ensemble = torch.jit.load(fpath, _extra_files=extra_files).eval()
        self.ensemble.to(self.device)
        self.config = json.loads(extra_files["config.yaml"])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.ensemble(x)
        x = torch.unbind(x, dim=0)
        return x

    def sliding_window_inference(
        self,
        img: torch.Tensor,
        tracing_input_size: Tuple[int, int] = (512, 512),
        overlap: float = 0.5,
        blend: str = "gaussian",
        **kwargs,
    ) -> torch.Tensor:
        with torch.no_grad():
            pred = sliding_window_inference(
                inputs=img,
                roi_size=tracing_input_size,
                sw_batch_size=self.sw_batch_size,
                predictor=self,
                overlap=overlap,
                mode=blend,
            )

            if isinstance(pred, tuple):
                pred = torch.stack(pred, dim=1)

            if pred.dim() == 4:
                pred = pred[:, None, ...]
        return pred  # Output NMCHW

    def tta_inference(self, img: torch.Tensor, tta_flips: list = None, **kwargs) -> torch.Tensor:
        if tta_flips is None:
            tta_flips = [[2], [3], [2, 3]]
        pred = self.sliding_window_inference(img, **kwargs)
        for flip_idx in tta_flips:
            flip_undo_idx = [e + 1 for e in flip_idx]  # output has extra first dim M
            pred += torch.flip(
                self.sliding_window_inference(torch.flip(img, dims=flip_idx), **kwargs),
                dims=flip_undo_idx,
            )
        pred /= len(tta_flips) + 1
        return pred  # NMCHW

    def proba_process(self, proba: torch.Tensor, bounds: Optional[Dict[str, Any]]) -> torch.Tensor:
        return proba

    def _prepare_input(self, img: torch.Tensor) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Common input preparation logic.

        Applies transforms and moves to device with batch dimension.

        Args:
            img: Input image tensor

        Returns:
            Tuple of (prepared image tensor, bounds dict or None)
        """
        img, bounds = self.transforms(img)
        return img.to(self.device).unsqueeze(dim=0), bounds

    def _run_inference(self, img: torch.Tensor) -> torch.Tensor:
        """Run inference using configured inference function.

        Args:
            img: Prepared input tensor (already batched and on device)

        Returns:
            Raw inference output
        """
        return self.inference_fn(img, **self.inference_config)

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        img, bounds = self._prepare_input(img)
        proba = self._run_inference(img)

        """Returns the output averaged over models"""
        return self.proba_process(proba, bounds)


class EnsembleSegmentation(EnsembleBase):
    def __init__(
        self,
        fpath: str,
        transforms: VASCXTransform,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
    ):
        super().__init__(fpath, transforms, device)
        self.sw_batch_size = 16

    def proba_process(self, proba: torch.Tensor, bounds: Optional[Dict[str, Any]]) -> torch.Tensor:
        proba = torch.mean(proba, dim=1)  # average over models (M)
        proba = torch.nn.functional.softmax(proba, dim=1)
        proba = self.transforms.undo_bounds(proba, bounds)
        proba = torch.argmax(proba, dim=1)
        return proba


class ClassificationEnsemble(EnsembleBase):
    def __init__(
        self,
        fpath: str,
        transforms: VASCXTransform,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
    ):
        super().__init__(fpath, transforms, device)
        self.inference_fn = None

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.ensemble(img)

    def proba_process(self, proba: torch.Tensor, bounds: Optional[Dict[str, Any]]) -> torch.Tensor:
        proba = torch.nn.functional.softmax(proba, dim=-1)
        proba = torch.mean(proba, dim=0)  # average over models
        return proba

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        img, bounds = self._prepare_input(img)
        with torch.no_grad():
            proba = self.forward(img)

        """Returns the output averaged over models"""
        return self.proba_process(proba, bounds)


class RegressionEnsemble(EnsembleBase):
    def __init__(
        self,
        fpath: str,
        transforms: VASCXTransform,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
    ):
        super().__init__(fpath, transforms, device)
        self.inference_fn = None

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.ensemble(img)

    def proba_process(self, proba: torch.Tensor, bounds: Optional[Dict[str, Any]]) -> torch.Tensor:
        return proba

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        img, bounds = self._prepare_input(img)
        with torch.no_grad():
            proba = self.forward(img)

        """Returns the output averaged over models"""
        return self.proba_process(proba, bounds)


class HeatmapRegressionEnsemble(EnsembleBase):
    def __init__(
        self,
        fpath: str,
        transforms: VASCXTransform,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
    ):
        super().__init__(fpath, transforms, device)
        self.sw_batch_size = 1

    def proba_process(self, heatmaps: torch.Tensor, bounds: Dict[str, Any]) -> torch.Tensor:
        """
        Vectorized heatmap processing - all operations on GPU with float32.

        Args:
            heatmaps: shape (B, M, K, H, W)
            bounds: bounds for undo transform

        Returns:
            outputs: shape (B, K, 2) in float32
        """
        batch_size, n_models, num_keypoints, height, width = heatmaps.shape

        # Reshape to (B*M*K, H*W) for vectorized argmax
        heatmaps_flat = heatmaps.view(batch_size * n_models * num_keypoints, -1)

        # Find argmax indices (stays on GPU, returns long tensor)
        max_indices = torch.argmax(heatmaps_flat, dim=1)  # shape (B*M*K,)

        # Convert flat indices to (col, row) coordinates
        # Original: col = max_idx % n_cols, row = max_idx // n_cols
        col = (max_indices % width).to(dtype=torch.float32) + 0.5
        row = (max_indices // width).to(dtype=torch.float32) + 0.5

        # Stack to (B*M*K, 2) with [col, row] order (x, y)
        outputs = torch.stack([col, row], dim=-1)  # (B*M*K, 2)

        # Reshape to (B, M, K, 2)
        outputs = outputs.view(batch_size, n_models, num_keypoints, 2)

        # Average over models
        outputs = torch.mean(outputs, dim=1)  # (B, K, 2)

        # Undo bounds transform
        outputs = self.transforms.undo_bounds_points(outputs, bounds)

        return outputs  # (B, K, 2) in float32
