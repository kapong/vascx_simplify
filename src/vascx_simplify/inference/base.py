"""Base class for ensemble models."""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ..preprocess import VASCXTransform
from .sliding_window import sliding_window_inference

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

        # Default batch size for predict() - can be overridden in subclasses
        self.predict_batch_size: Optional[int] = None

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

    def proba_process(
        self,
        proba: torch.Tensor,
        bounds: Union[Optional[Dict[str, Any]], List],
        is_batch: bool = False,
    ) -> torch.Tensor:
        """Process model output. Override in subclasses.

        Args:
            proba: Raw model output
            bounds: Bounds dict (single image) or list of dicts (batch)
            is_batch: Whether input was a batch

        Returns:
            Processed predictions
        """
        return proba

    def _is_batch_input(self, img: Any) -> bool:
        """Detect if input is a batch of images.

        Args:
            img: Input to check

        Returns:
            True if input is a batch, False for single image
        """
        if isinstance(img, (list, tuple)):
            return len(img) > 0
        if isinstance(img, torch.Tensor):
            # Tensor [B, C, H, W] with B > 1 is batch
            # Tensor [C, H, W] or [1, C, H, W] is single
            return img.dim() == 4 and img.shape[0] > 1
        return False

    def _prepare_input(
        self, img: Union[torch.Tensor, Any, List]
    ) -> Tuple[torch.Tensor, Union[Optional[Dict[str, Any]], List], bool]:
        """Prepare input with batch detection.

        Args:
            img: Single image or batch of images

        Returns:
            Tuple of (batched tensor, bounds, is_batch_input)
        """
        # Detect batch input
        is_batch = self._is_batch_input(img)

        if is_batch:
            # Handle list of images
            if isinstance(img, (list, tuple)):
                results = [self.transforms(im) for im in img]
                tensors, bounds_list = zip(*results)
                batched = torch.stack([t.to(self.device) for t in tensors])
                return batched, list(bounds_list), True
            # Handle batched tensor [B, C, H, W] where B > 1
            elif isinstance(img, torch.Tensor) and img.dim() == 4:
                # Apply transforms per image in batch
                results = [self.transforms(img[i]) for i in range(img.shape[0])]
                tensors, bounds_list = zip(*results)
                batched = torch.stack([t.to(self.device) for t in tensors])
                return batched, list(bounds_list), True
        else:
            # Single image - existing behavior
            img, bounds = self.transforms(img)
            return img.to(self.device).unsqueeze(dim=0), bounds, False

    def _run_inference(self, img: torch.Tensor) -> torch.Tensor:
        """Run inference using configured inference function.

        Args:
            img: Prepared input tensor (already batched and on device)

        Returns:
            Raw inference output
        """
        return self.inference_fn(img, **self.inference_config)

    def _predict_batch(
        self, img: Union[torch.Tensor, Any, List], batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Predict with automatic batch splitting.

        Args:
            img: Single image or batch of images
            batch_size: Maximum batch size for inference. If None, uses model default.
                       If input batch exceeds this, will split and process in chunks.

        Returns:
            Predictions for all images
        """
        img_tensor, bounds, is_batch = self._prepare_input(img)

        # Get effective batch size
        if batch_size is None:
            batch_size = self.predict_batch_size

        current_batch_size = img_tensor.shape[0]

        # If batch size limit not set or batch fits, process all at once
        if batch_size is None or current_batch_size <= batch_size:
            proba = self._run_inference(img_tensor)
            return self.proba_process(proba, bounds, is_batch)

        # Split into chunks and process
        all_preds = []
        for i in range(0, current_batch_size, batch_size):
            end_idx = min(i + batch_size, current_batch_size)
            chunk_img = img_tensor[i:end_idx]

            # Get corresponding bounds for this chunk
            if isinstance(bounds, list):
                chunk_bounds = bounds[i:end_idx]
            else:
                chunk_bounds = bounds

            # Process chunk
            chunk_proba = self._run_inference(chunk_img)
            chunk_pred = self.proba_process(
                chunk_proba, chunk_bounds, is_batch=True  # Always True for chunks
            )
            all_preds.append(chunk_pred)

        # Concatenate results
        return torch.cat(all_preds, dim=0)

    def predict(
        self, img: Union[torch.Tensor, Any, List], batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Predict with automatic batch splitting.

        Args:
            img: Single image (PIL.Image or Tensor [C,H,W]) or
                 batch (List of images or Tensor [B,C,H,W])
            batch_size: Maximum batch size for inference. If None, uses self.predict_batch_size.
                       Large batches will be automatically split into chunks.

        Returns:
            Predictions tensor. Shape depends on model type:
            - Segmentation: [B, H, W]
            - Classification: [B, C]
            - Regression: [B, M, D]
            - Heatmap: [B, K, 2]
        """
        return self._predict_batch(img, batch_size)
