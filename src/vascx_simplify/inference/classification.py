"""Classification ensemble model."""

from typing import Any, Dict, List, Optional, Union

import torch

from ..preprocess import VASCXTransform
from .base import DEFAULT_DEVICE, EnsembleBase


class ClassificationEnsemble(EnsembleBase):
    def __init__(
        self,
        fpath: str,
        transforms: VASCXTransform,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
    ):
        super().__init__(fpath, transforms, device)
        self.inference_fn = None
        self.predict_batch_size = 16  # Default: process 16 images at once (lightweight)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.ensemble(img)

    def proba_process(
        self,
        proba: torch.Tensor,
        bounds: Union[Optional[Dict[str, Any]], List],
        is_batch: bool = False,
    ) -> torch.Tensor:
        """Process classification output.

        Args:
            proba: Raw model output [B, M, C] or [B, C]
            bounds: Bounds dict (single) or list of dicts (batch) - unused for classification
            is_batch: Whether input was a batch

        Returns:
            Softmax probabilities [B, C]
        """
        proba = torch.nn.functional.softmax(proba, dim=-1)
        # Average over models if ensemble dimension exists
        if proba.dim() == 3:  # [B, M, C]
            proba = torch.mean(proba, dim=1)  # Average over models -> [B, C]
        return proba

    def predict(
        self, img: Union[torch.Tensor, Any, List], batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Predict with automatic batch splitting for classification.

        Args:
            img: Single image or batch of images
            batch_size: Maximum batch size for inference

        Returns:
            Classification probabilities [B, C]
        """
        img_tensor, bounds, is_batch = self._prepare_input(img)

        # Get effective batch size
        if batch_size is None:
            batch_size = self.predict_batch_size

        current_batch_size = img_tensor.shape[0]

        # If batch size limit not set or batch fits, process all at once
        if batch_size is None or current_batch_size <= batch_size:
            with torch.no_grad():
                proba = self.forward(img_tensor)
            return self.proba_process(proba, bounds, is_batch)

        # Split into chunks and process
        all_preds = []
        for i in range(0, current_batch_size, batch_size):
            end_idx = min(i + batch_size, current_batch_size)
            chunk_img = img_tensor[i:end_idx]

            with torch.no_grad():
                chunk_proba = self.forward(chunk_img)

            chunk_bounds = bounds[i:end_idx] if isinstance(bounds, list) else bounds
            chunk_pred = self.proba_process(chunk_proba, chunk_bounds, is_batch=True)
            all_preds.append(chunk_pred)

        return torch.cat(all_preds, dim=0)
