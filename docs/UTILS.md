# Utils Module Documentation

The `vascx_simplify.utils` module provides helper functions for common tasks like visualization, image processing, and statistics calculation. The module is organized into specialized submodules for better organization.

## Module Structure

```
vascx_simplify/utils/
├── __init__.py           # Main exports
├── huggingface.py        # HuggingFace Hub integration
├── image.py              # Image processing utilities
├── stats.py              # Statistical calculations
└── visualization.py      # Overlay and visualization functions
```

## Quick Start

```python
from vascx_simplify import EnsembleSegmentation, VASCXTransform, from_huggingface
from vascx_simplify.utils import (
    create_artery_vein_overlay,
    calculate_class_statistics,
    tensor_to_numpy
)

# Load model and predict
model = EnsembleSegmentation(from_huggingface("Eyened/vascx:artery_vein/av_july24.pt"), VASCXTransform())
prediction = model.predict(image)

# Create overlay and calculate stats in one line each
overlay = create_artery_vein_overlay(image, tensor_to_numpy(prediction[0]))
stats = calculate_class_statistics(tensor_to_numpy(prediction[0]))
```

## HuggingFace Integration

### `from_huggingface(modelstr)`

Download models from HuggingFace Hub.

**Args:**
- `modelstr` (str): Format "repo_name:file_path"

**Returns:**
- str: Local path to downloaded file

**Example:**
```python
from vascx_simplify import from_huggingface

model_path = from_huggingface("Eyened/vascx:artery_vein/av_july24.pt")
```

## Image Processing

### `tensor_to_numpy(tensor)`

Convert PyTorch tensor to numpy array (handles GPU tensors).

**Example:**
```python
from vascx_simplify.utils import tensor_to_numpy

pred_np = tensor_to_numpy(prediction[0])  # Auto-moves from GPU if needed
```

### `pil_to_numpy(image)` / `numpy_to_pil(array)`

Convert between PIL Images and numpy arrays.

**Example:**
```python
from vascx_simplify.utils import pil_to_numpy, numpy_to_pil
from PIL import Image

# PIL to numpy
img_np = pil_to_numpy(Image.open("fundus.jpg"))

# Numpy to PIL
img_pil = numpy_to_pil(img_np)
```

### `create_color_mask(class_map, color_dict)`

Create RGB color mask from class predictions.

**Args:**
- `class_map` (np.ndarray): Class predictions [H, W]
- `color_dict` (dict): Mapping from class ID to RGB tuple

**Example:**
```python
from vascx_simplify.utils import create_color_mask

colors = {
    0: (0, 0, 0),       # Background - black
    1: (255, 0, 0),     # Class 1 - red
    2: (0, 0, 255),     # Class 2 - blue
}
mask = create_color_mask(predictions, colors)
```

### `blend_images(background, overlay, alpha, mask=None)`

Blend two images with optional mask.

**Args:**
- `background` (np.ndarray): Background image [H, W, 3]
- `overlay` (np.ndarray): Overlay image [H, W, 3]
- `alpha` (float): Blending factor (0=bg only, 1=overlay only)
- `mask` (np.ndarray, optional): Binary mask [H, W]

**Example:**
```python
from vascx_simplify.utils import blend_images

# Blend entire image
result = blend_images(original, color_mask, alpha=0.5)

# Blend only where mask is True
result = blend_images(original, color_mask, alpha=0.5, mask=pred > 0)
```

## Visualization

### Pre-defined Color Schemes

```python
from vascx_simplify.utils import (
    ARTERY_VEIN_COLORS,  # {0: black, 1: red, 2: blue, 3: green}
    DISC_COLORS,          # {0: black, 1: yellow}
    QUALITY_COLORS,       # {0: red, 1: orange, 2: green}
    QUALITY_LABELS,       # ["REJECT", "USABLE", "GOOD"]
)
```

### `create_segmentation_overlay(image, prediction, color_map=None, alpha=0.5)`

Create overlay of segmentation on original image.

**Args:**
- `image` (PIL.Image or np.ndarray): Original image
- `prediction` (np.ndarray): Class predictions [H, W]
- `color_map` (dict, optional): Color mapping (default: ARTERY_VEIN_COLORS)
- `alpha` (float): Opacity (0=transparent, 1=opaque)

**Example:**
```python
from vascx_simplify.utils import create_segmentation_overlay

overlay = create_segmentation_overlay(
    image, 
    prediction, 
    color_map={0: (0,0,0), 1: (255,0,0)},
    alpha=0.5
)
```

### `create_artery_vein_overlay(image, prediction, alpha=0.5)`

Specialized overlay for artery/vein segmentation (red=arteries, blue=veins, green=crossings).

**Example:**
```python
from vascx_simplify.utils import create_artery_vein_overlay

overlay = create_artery_vein_overlay(image, prediction, alpha=0.5)
```

### `create_disc_overlay(image, prediction, alpha=0.4)`

Specialized overlay for optic disc segmentation (yellow=disc).

**Example:**
```python
from vascx_simplify.utils import create_disc_overlay

overlay = create_disc_overlay(image, prediction, alpha=0.4)
```

### `draw_fovea_marker(image, x, y, marker_size=200)`

Draw crosshair marker for fovea location.

**Args:**
- `image` (PIL.Image or np.ndarray): Original image
- `x` (float): X coordinate
- `y` (float): Y coordinate
- `marker_size` (int): Marker size in pixels

**Example:**
```python
from vascx_simplify.utils import draw_fovea_marker

marked = draw_fovea_marker(image, fovea_x, fovea_y, marker_size=150)
```

### `draw_quality_badge(image, quality_class, confidence=None, position="top-left")`

Draw quality classification badge on image.

**Args:**
- `image` (PIL.Image or np.ndarray): Original image
- `quality_class` (int): Quality class (0=reject, 1=usable, 2=good)
- `confidence` (float, optional): Confidence score
- `position` (str): "top-left", "top-right", "bottom-left", "bottom-right"

**Example:**
```python
from vascx_simplify.utils import draw_quality_badge

badged = draw_quality_badge(image, quality_class=2, confidence=0.95)
```

### `create_side_by_side(images, titles=None)`

Create side-by-side comparison of images.

**Args:**
- `images` (list): List of images [H, W, 3]
- `titles` (list, optional): List of titles

**Example:**
```python
from vascx_simplify.utils import create_side_by_side

comparison = create_side_by_side(
    [original, mask, overlay],
    titles=["Original", "Mask", "Overlay"]
)
```

### `create_comparison_grid(original, mask, overlay, titles=None)`

Create standard 3-panel comparison grid.

**Example:**
```python
from vascx_simplify.utils import create_comparison_grid

grid = create_comparison_grid(original, mask, overlay)
```

## Statistics

### `calculate_class_statistics(class_map)`

Calculate pixel counts for each class.

**Returns:** dict mapping class ID to pixel count

**Example:**
```python
from vascx_simplify.utils import calculate_class_statistics

stats = calculate_class_statistics(prediction)
print(f"Artery pixels: {stats[1]:,}")
print(f"Vein pixels: {stats[2]:,}")
```

### `calculate_centroid(binary_mask)`

Calculate centroid (center of mass) of binary mask.

**Returns:** Tuple of (x, y) coordinates

**Example:**
```python
from vascx_simplify.utils import calculate_centroid

disc_mask = prediction == 1
centroid_x, centroid_y = calculate_centroid(disc_mask)
```

### `calculate_area_percentage(binary_mask, total_pixels=None)`

Calculate area percentage of binary mask.

**Returns:** Percentage (0-100)

**Example:**
```python
from vascx_simplify.utils import calculate_area_percentage

disc_area_pct = calculate_area_percentage(prediction == 1)
print(f"Disc covers {disc_area_pct:.2f}% of image")
```

### `calculate_vessel_ratio(artery_mask, vein_mask)`

Calculate artery-to-vein pixel ratio.

**Returns:** float (A/V ratio)

**Example:**
```python
from vascx_simplify.utils import calculate_vessel_ratio

av_ratio = calculate_vessel_ratio(prediction == 1, prediction == 2)
print(f"A/V ratio: {av_ratio:.2f}")
```

### `calculate_bounding_box(binary_mask)`

Calculate bounding box of binary mask.

**Returns:** Tuple of (x_min, y_min, x_max, y_max)

**Example:**
```python
from vascx_simplify.utils import calculate_bounding_box

x_min, y_min, x_max, y_max = calculate_bounding_box(prediction == 1)
```

## Complete Example

```python
from PIL import Image
from vascx_simplify import EnsembleSegmentation, VASCXTransform, from_huggingface
from vascx_simplify.utils import (
    create_artery_vein_overlay,
    calculate_class_statistics,
    calculate_vessel_ratio,
    tensor_to_numpy,
)

# Load model
model_path = from_huggingface("Eyened/vascx:artery_vein/av_july24.pt")
model = EnsembleSegmentation(model_path, VASCXTransform())

# Load and predict
image = Image.open("fundus.jpg")
prediction = model.predict(image)
pred_np = tensor_to_numpy(prediction[0])

# Create visualization
overlay = create_artery_vein_overlay(image, pred_np, alpha=0.5)

# Calculate statistics
stats = calculate_class_statistics(pred_np)
av_ratio = calculate_vessel_ratio(pred_np == 1, pred_np == 2)

print(f"Arteries: {stats.get(1, 0):,} pixels")
print(f"Veins: {stats.get(2, 0):,} pixels")
print(f"A/V ratio: {av_ratio:.2f}")

# Save
import matplotlib.pyplot as plt
plt.imshow(overlay)
plt.axis("off")
plt.savefig("result.png", bbox_inches="tight", dpi=150)
```

## Backward Compatibility

The old `utils.py` module is now a compatibility shim that re-exports everything from the new `utils/` subpackage. Existing code continues to work:

```python
# Both import styles work
from vascx_simplify.utils import from_huggingface  # Old style - still works
from vascx_simplify import from_huggingface        # Also works (main __init__ export)
```

## Best Practices

1. **Use specialized overlay functions** - They have sensible defaults for each task
2. **Batch operations** - Process multiple images in loops for consistency
3. **GPU handling** - `tensor_to_numpy()` automatically handles GPU tensors
4. **Type consistency** - Functions accept both PIL Images and numpy arrays

## See Also

- `examples/06_utils_demo.py` - Demonstration of all utility functions
- `examples/01_artery_vein.py` - Manual visualization (before utils)
- Main documentation: `README.md`
