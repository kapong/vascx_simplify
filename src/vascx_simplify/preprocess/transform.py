"""Transform pipeline for fundus images."""

from typing import Any, Dict, Optional, Tuple, Union

import kornia.geometry.transform as K
import numpy as np
import torch

from .contrast import SimpleFundusEnhance, FundusContrastEnhance

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VASCXTransform:
    def __init__(
        self,
        size: int = 1024,
        use_ce: bool = True,
        use_fp16: bool = True,
        device: Union[torch.device, str] = DEFAULT_DEVICE,
        Enhancer=SimpleFundusEnhance,
    ):
        """
        Args:
            size: Output size
            use_ce: Whether to use contrast enhancement
            use_fp16: Use float16 for compute-intensive ops
            device: Device to run transformations on ('cuda' or 'cpu')
        """
        self.size = size
        self.device = torch.device(device)
        self.use_ce = use_ce
        self.use_fp16 = use_fp16

        if self.use_ce:
            self.contrast_enhancer = Enhancer(square_size=size, use_fp16=use_fp16)
            # 6 channels (rgb + ce)
            self.mean = torch.tensor(
                [0.485, 0.456, 0.406, 0.485, 0.456, 0.406], device=self.device, dtype=torch.float32
            ).view(6, 1, 1)
            self.std = torch.tensor(
                [0.229, 0.224, 0.225, 0.229, 0.224, 0.225], device=self.device, dtype=torch.float32
            ).view(6, 1, 1)
        else:
            self.contrast_enhancer = None
            # 3 channels (rgb only)
            self.mean = torch.tensor(
                [0.485, 0.456, 0.406], device=self.device, dtype=torch.float32
            ).view(3, 1, 1)
            self.std = torch.tensor(
                [0.229, 0.224, 0.225], device=self.device, dtype=torch.float32
            ).view(3, 1, 1)

    def __call__(self, image) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Args:
            image: Input image as numpy array, PIL Image, or torch.Tensor
        Returns:
            tuple: (processed_image, bounds) as torch.Tensors on specified device
        """
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            if hasattr(image, "shape"):  # numpy array
                image = torch.from_numpy(np.array(image))
            else:  # PIL Image
                image = torch.from_numpy(np.array(image))

        # Move to device
        image = image.to(self.device)

        # Ensure proper format [C, H, W]
        if image.ndim == 2:  # Grayscale [H, W]
            image = image.unsqueeze(0)
        elif image.ndim == 3:
            if image.shape[-1] in [1, 3, 4]:  # [H, W, C] format
                image = image.permute(2, 0, 1)
            # else already in [C, H, W] format

        # Ensure uint8
        if image.dtype != torch.uint8:
            if image.max() <= 1.0:
                image = (image * 255).clamp(0, 255).to(torch.uint8)
            else:
                image = image.clamp(0, 255).to(torch.uint8)

        if self.use_ce:
            # Apply contrast enhancement
            rgb, ce, bounds = self.contrast_enhancer(image)
            # Concatenate RGB and CE
            inputs = torch.cat([rgb, ce], dim=0)  # [6, H, W]
            # Convert to float32 [0, 1] and normalize
            inputs = inputs.float() / 255.0
            inputs = (inputs - self.mean) / self.std
            return inputs, bounds
        else:
            # Resize using kornia (GPU-accelerated)
            image_batch = image.unsqueeze(0).float()  # [1, C, H, W]
            resized = K.resize(
                image_batch, (self.size, self.size), interpolation="bilinear", align_corners=False
            )
            # Convert to [0, 1] and normalize
            resized = resized / 255.0
            resized = (resized - self.mean) / self.std
            return resized.squeeze(0), None

    def undo_bounds(self, image: torch.Tensor, bounds: Dict[str, Any]) -> torch.Tensor:
        """Reverse the cropping transformation."""
        return self.contrast_enhancer.undo_bounds(image, **bounds)

    def undo_bounds_points(self, points: torch.Tensor, bounds: Dict[str, Any]) -> torch.Tensor:
        """Reverse the cropping transformation for point coordinates."""
        return self.contrast_enhancer.undo_bounds_points(points, **bounds)
