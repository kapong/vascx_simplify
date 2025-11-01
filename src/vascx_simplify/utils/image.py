"""Image processing utilities."""

from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor on CPU or GPU
        
    Returns:
        Numpy array
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def resize_to_original(
    prediction: np.ndarray, 
    original_size: Tuple[int, int],
    interpolation: int = Image.NEAREST
) -> np.ndarray:
    """Resize prediction to match original image size.
    
    Args:
        prediction: Prediction array [H, W] or [H, W, C]
        original_size: Target size (width, height)
        interpolation: PIL interpolation mode (default: NEAREST for segmentation)
        
    Returns:
        Resized prediction array
    """
    if prediction.ndim == 2:
        pred_img = Image.fromarray(prediction.astype(np.uint8), mode='L')
    else:
        pred_img = Image.fromarray(prediction.astype(np.uint8))
    
    resized = pred_img.resize(original_size, interpolation)
    return np.array(resized)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array.
    
    Args:
        image: PIL Image
        
    Returns:
        Numpy array [H, W, C] or [H, W]
    """
    return np.array(image)


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image.
    
    Args:
        array: Numpy array [H, W, C] or [H, W]
        
    Returns:
        PIL Image
    """
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8), mode='L')
    return Image.fromarray(array.astype(np.uint8))


def create_color_mask(
    class_map: np.ndarray,
    color_dict: dict[int, Tuple[int, int, int]]
) -> np.ndarray:
    """Create RGB color mask from class predictions.
    
    Args:
        class_map: Class predictions [H, W] with integer class IDs
        color_dict: Mapping from class ID to RGB color tuple
        
    Returns:
        RGB mask [H, W, 3]
        
    Example:
        >>> colors = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 0, 255)}
        >>> mask = create_color_mask(predictions, colors)
    """
    h, w = class_map.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in color_dict.items():
        color_mask[class_map == class_id] = color
    
    return color_mask


def blend_images(
    background: np.ndarray,
    overlay: np.ndarray,
    alpha: float = 0.5,
    mask: Union[np.ndarray, None] = None
) -> np.ndarray:
    """Blend two images with optional mask.
    
    Args:
        background: Background image [H, W, 3]
        overlay: Overlay image [H, W, 3]
        alpha: Blending factor (0=background only, 1=overlay only)
        mask: Optional binary mask [H, W] - only blend where mask is True
        
    Returns:
        Blended image [H, W, 3]
    """
    result = background.copy().astype(float)
    
    if mask is None:
        result = (1 - alpha) * result + alpha * overlay.astype(float)
    else:
        result[mask] = (1 - alpha) * result[mask] + alpha * overlay[mask].astype(float)
    
    return result.astype(np.uint8)
