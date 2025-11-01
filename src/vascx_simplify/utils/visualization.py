"""Visualization utilities for model outputs."""

from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .image import blend_images, create_color_mask, pil_to_numpy


# Default color schemes for different segmentation tasks
ARTERY_VEIN_COLORS = {
    0: (0, 0, 0),        # Background - black
    1: (255, 0, 0),      # Arteries - red
    2: (0, 0, 255),      # Veins - blue
    3: (0, 255, 0),      # Crossings - green
}

DISC_COLORS = {
    0: (0, 0, 0),        # Background - black
    1: (255, 255, 0),    # Optic disc - yellow
}

QUALITY_COLORS = {
    0: (231, 76, 60),    # Reject - red (#e74c3c)
    1: (243, 156, 18),   # Usable - orange (#f39c12)
    2: (46, 204, 113),   # Good - green (#2ecc71)
}

QUALITY_LABELS = ["REJECT", "USABLE", "GOOD"]


def create_segmentation_overlay(
    image: Union[Image.Image, np.ndarray],
    prediction: np.ndarray,
    color_map: Dict[int, Tuple[int, int, int]] = None,
    alpha: float = 0.5,
    exclude_background: bool = True
) -> np.ndarray:
    """Create overlay of segmentation on original image.
    
    Args:
        image: Original image (PIL or numpy array)
        prediction: Class predictions [H, W] with integer class IDs
        color_map: Mapping from class ID to RGB color (default: artery/vein colors)
        alpha: Opacity of overlay (0=transparent, 1=opaque)
        exclude_background: If True, don't overlay class 0
        
    Returns:
        Overlay image [H, W, 3]
        
    Example:
        >>> overlay = create_segmentation_overlay(img, pred, alpha=0.5)
    """
    if isinstance(image, Image.Image):
        image = pil_to_numpy(image)
    
    if color_map is None:
        color_map = ARTERY_VEIN_COLORS
    
    # Create color mask
    color_mask = create_color_mask(prediction, color_map)
    
    # Create blending mask (exclude background if requested)
    if exclude_background:
        blend_mask = prediction > 0
    else:
        blend_mask = None
    
    # Blend
    overlay = blend_images(image, color_mask, alpha=alpha, mask=blend_mask)
    
    return overlay


def create_artery_vein_overlay(
    image: Union[Image.Image, np.ndarray],
    prediction: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """Create overlay for artery/vein segmentation.
    
    Red = Arteries, Blue = Veins, Green = Crossings
    
    Args:
        image: Original fundus image
        prediction: Class predictions [H, W] (0=bg, 1=artery, 2=vein, 3=crossing)
        alpha: Opacity of overlay
        
    Returns:
        Overlay image [H, W, 3]
    """
    return create_segmentation_overlay(
        image, prediction, 
        color_map=ARTERY_VEIN_COLORS, 
        alpha=alpha
    )


def create_disc_overlay(
    image: Union[Image.Image, np.ndarray],
    prediction: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """Create overlay for optic disc segmentation.
    
    Yellow = Optic Disc
    
    Args:
        image: Original fundus image
        prediction: Class predictions [H, W] (0=bg, 1=disc)
        alpha: Opacity of overlay
        
    Returns:
        Overlay image [H, W, 3]
    """
    return create_segmentation_overlay(
        image, prediction,
        color_map=DISC_COLORS,
        alpha=alpha
    )


def draw_fovea_marker(
    image: Union[Image.Image, np.ndarray],
    x: float,
    y: float,
    marker_size: int = 200,
    marker_color: Tuple[int, int, int] = (255, 0, 0),
    line_width: int = 3
) -> np.ndarray:
    """Draw fovea location marker on image.
    
    Args:
        image: Original fundus image
        x: X coordinate of fovea
        y: Y coordinate of fovea
        marker_size: Size of marker in pixels
        marker_color: RGB color of marker (default: red)
        line_width: Width of marker lines
        
    Returns:
        Image with marker [H, W, 3]
        
    Example:
        >>> marked = draw_fovea_marker(img, fovea_x, fovea_y)
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Make a copy
    image = image.copy()
    draw = ImageDraw.Draw(image)
    
    # Draw crosshair
    half_size = marker_size // 2
    
    # Horizontal line
    draw.line(
        [(x - half_size, y), (x + half_size, y)],
        fill=marker_color,
        width=line_width
    )
    
    # Vertical line
    draw.line(
        [(x, y - half_size), (x, y + half_size)],
        fill=marker_color,
        width=line_width
    )
    
    # Draw circle
    circle_radius = marker_size // 3
    draw.ellipse(
        [
            (x - circle_radius, y - circle_radius),
            (x + circle_radius, y + circle_radius)
        ],
        outline=marker_color,
        width=line_width
    )
    
    return pil_to_numpy(image)


def draw_quality_badge(
    image: Union[Image.Image, np.ndarray],
    quality_class: int,
    confidence: float = None,
    position: str = "top-left",
    badge_size: int = 120
) -> np.ndarray:
    """Draw quality classification badge on image.
    
    Args:
        image: Original fundus image
        quality_class: Quality class (0=reject, 1=usable, 2=good)
        confidence: Optional confidence score to display
        position: Badge position ("top-left", "top-right", "bottom-left", "bottom-right")
        badge_size: Approximate badge size in pixels
        
    Returns:
        Image with badge [H, W, 3]
        
    Example:
        >>> badged = draw_quality_badge(img, quality_class=2, confidence=0.95)
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Make a copy
    image = image.copy()
    draw = ImageDraw.Draw(image)
    
    # Get label and color
    label = QUALITY_LABELS[quality_class]
    color = QUALITY_COLORS[quality_class]
    
    # Try to load font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", badge_size // 5)
    except Exception:
        font = ImageFont.load_default()
    
    # Create badge text
    if confidence is not None:
        badge_text = f"{label}\n{confidence:.0%}"
    else:
        badge_text = label
    
    # Calculate position
    img_width, img_height = image.size
    margin = 20
    
    # Get text bbox (approximate)
    bbox = draw.textbbox((0, 0), badge_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Add padding
    padding = 15
    box_width = text_width + 2 * padding
    box_height = text_height + 2 * padding
    
    # Determine position
    if position == "top-left":
        x0, y0 = margin, margin
    elif position == "top-right":
        x0, y0 = img_width - box_width - margin, margin
    elif position == "bottom-left":
        x0, y0 = margin, img_height - box_height - margin
    else:  # bottom-right
        x0, y0 = img_width - box_width - margin, img_height - box_height - margin
    
    x1, y1 = x0 + box_width, y0 + box_height
    
    # Draw rounded rectangle background
    draw.rounded_rectangle(
        [(x0, y0), (x1, y1)],
        radius=10,
        fill=color,
        outline=(0, 0, 0),
        width=2
    )
    
    # Draw text
    text_x = x0 + padding
    text_y = y0 + padding
    draw.text(
        (text_x, text_y),
        badge_text,
        fill=(255, 255, 255),
        font=font
    )
    
    return pil_to_numpy(image)


def create_side_by_side(
    images: List[np.ndarray],
    titles: List[str] = None,
    title_fontsize: int = 20,
    spacing: int = 20
) -> np.ndarray:
    """Create side-by-side comparison of images.
    
    Args:
        images: List of images [H, W, 3] (all must have same height)
        titles: Optional list of titles for each image
        title_fontsize: Font size for titles
        spacing: Spacing between images in pixels
        
    Returns:
        Combined image [H, W_total, 3]
    """
    if not images:
        raise ValueError("Images list cannot be empty")
    
    # Convert all to same height
    target_height = images[0].shape[0]
    resized_images = []
    
    for img in images:
        if img.shape[0] != target_height:
            # Resize proportionally
            scale = target_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((new_width, target_height), Image.LANCZOS)
            img = pil_to_numpy(pil_img)
        resized_images.append(img)
    
    # Calculate total width
    total_width = sum(img.shape[1] for img in resized_images) + spacing * (len(resized_images) - 1)
    
    # Create canvas
    canvas = np.ones((target_height, total_width, 3), dtype=np.uint8) * 255
    
    # Place images
    x_offset = 0
    for img in resized_images:
        canvas[:, x_offset:x_offset + img.shape[1]] = img
        x_offset += img.shape[1] + spacing
    
    return canvas


def create_comparison_grid(
    original: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
    titles: List[str] = None
) -> np.ndarray:
    """Create standard 3-panel comparison grid (original, mask, overlay).
    
    Args:
        original: Original image [H, W, 3]
        mask: Segmentation mask [H, W, 3]
        overlay: Overlay image [H, W, 3]
        titles: Optional list of 3 titles
        
    Returns:
        Grid image [H, W*3, 3]
    """
    if titles is None:
        titles = ["Original", "Segmentation", "Overlay"]
    
    return create_side_by_side([original, mask, overlay], titles)
