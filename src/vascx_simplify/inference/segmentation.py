"""Segmentation ensemble model."""

from typing import Any, Dict, List, Optional, Union

import torch

from ..preprocess import VASCXTransform
from .base import DEFAULT_DEVICE, EnsembleBase


class EnsembleSegmentation(EnsembleBase):
    def __init__(
        self,
        fpath: str,
        transforms: VASCXTransform,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
    ):
        super().__init__(fpath, transforms, device)
        self.sw_batch_size = 16
        self.predict_batch_size = 4  # Default: process 4 images at once (memory-intensive)

    def proba_process(
        self,
        proba: torch.Tensor,
        bounds: Union[Optional[Dict[str, Any]], List],
        is_batch: bool = False,
    ) -> torch.Tensor:
        """Process segmentation output.

        Args:
            proba: Raw model output [B, M, C, H, W]
            bounds: Bounds dict (single) or list of dicts (batch)
            is_batch: Whether input was a batch

        Returns:
            Class predictions [B, H, W]
        """
        proba = torch.mean(proba, dim=1)  # Average over models (M)
        proba = torch.nn.functional.softmax(proba, dim=1)

        # Handle bounds undo
        if is_batch and isinstance(bounds, list):
            results = [
                self.transforms.undo_bounds(proba[i : i + 1], bounds[i]) for i in range(len(bounds))
            ]
            proba = torch.cat(results, dim=0)
        else:
            proba = self.transforms.undo_bounds(proba, bounds)

        proba = torch.argmax(proba, dim=1)
        return proba
