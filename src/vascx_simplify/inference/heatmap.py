"""Heatmap regression ensemble model."""

from typing import Any, Dict, List, Union

import torch

from ..preprocess import VASCXTransform
from .base import DEFAULT_DEVICE, EnsembleBase


class HeatmapRegressionEnsemble(EnsembleBase):
    def __init__(
        self,
        fpath: str,
        transforms: VASCXTransform,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
    ):
        super().__init__(fpath, transforms, device)
        self.sw_batch_size = 1
        self.predict_batch_size = (
            1  # Default: process 1 image at a time (batch doesn't improve speed)
        )

    def proba_process(
        self, heatmaps: torch.Tensor, bounds: Union[Dict[str, Any], List], is_batch: bool = False
    ) -> torch.Tensor:
        """Vectorized heatmap processing - all operations on GPU with float32.

        Args:
            heatmaps: shape (B, M, K, H, W)
            bounds: bounds dict (single) or list of dicts (batch)
            is_batch: Whether input was a batch

        Returns:
            outputs: shape (B, K, 2) in float32
        """
        batch_size, n_models, num_keypoints, height, width = heatmaps.shape

        # Sum heatmaps across models BEFORE finding argmax (better ensemble strategy)
        # Shape: (B, M, K, H, W) -> (B, K, H, W)
        heatmaps_summed = torch.sum(heatmaps, dim=1)  # (B, K, H, W)

        # Reshape to (B*K, H*W) for vectorized argmax
        heatmaps_flat = heatmaps_summed.view(batch_size * num_keypoints, -1)

        # Find argmax indices (stays on GPU, returns long tensor)
        max_indices = torch.argmax(heatmaps_flat, dim=1)  # shape (B*K,)

        # Convert flat indices to (col, row) coordinates
        # Original: col = max_idx % n_cols, row = max_idx // n_cols
        col = (max_indices % width).to(dtype=torch.float32) + 0.5
        row = (max_indices // width).to(dtype=torch.float32) + 0.5

        # Stack to (B*K, 2) with [col, row] order (x, y)
        outputs = torch.stack([col, row], dim=-1)  # (B*K, 2)

        # Reshape to (B, K, 2)
        outputs = outputs.view(batch_size, num_keypoints, 2)

        # Undo bounds transform
        if is_batch and isinstance(bounds, list):
            results = [
                self.transforms.undo_bounds_points(outputs[i : i + 1], bounds[i])
                for i in range(len(bounds))
            ]
            outputs = torch.cat(results, dim=0)
        else:
            outputs = self.transforms.undo_bounds_points(outputs, bounds)

        return outputs  # (B, K, 2) in float32
