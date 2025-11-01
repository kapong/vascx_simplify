# vascx_simplify

[![PyPI version](https://badge.fury.io/py/vascx-simplify.svg)](https://badge.fury.io/py/vascx-simplify)
[![Python](https://img.shields.io/pypi/pyversions/vascx-simplify.svg)](https://pypi.org/project/vascx-simplify/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/github-kapong%2Fvascx__simplify-blue.svg)](https://github.com/kapong/vascx_simplify)

A PyTorch library for vessel and fundus image analysis, providing GPU-accelerated preprocessing and inference utilities for medical imaging tasks.

**Note:** This is a simplified rewrite of [rtnls_vascx_models](https://github.com/Eyened/rtnls_vascx_models) by Eyened.

## AI Usage Disclaimer

This project was developed with significant assistance from AI tools (GitHub Copilot, ChatGPT, Claude) for code organization, refactoring, documentation, and packaging.

## Features

- **GPU-Accelerated Preprocessing**: Fast fundus image contrast enhancement with mixed precision support
- **Sliding Window Inference**: Efficient inference for large images
- **Ensemble Models**: Segmentation, classification, regression, and heatmap-based models
- **HuggingFace Integration**: Easy model loading from HuggingFace Hub
- **Minimal Dependencies**: Uses fewer dependency libraries for easier installation and maintenance

## Installation

```bash
pip install vascx_simplify
```

From source:
```bash
git clone https://github.com/kapong/vascx_simplify.git
cd vascx_simplify
pip install -e .
```

## Requirements

- Python >= 3.12
- PyTorch >= 1.10.0
- kornia >= 0.6.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- numpy >= 1.21.0
- huggingface-hub >= 0.10.0

## Usage Examples

### Artery/Vein Segmentation

Segment arteries (red), veins (blue), and crossings (green) from fundus images:

```python
from vascx_simplify import EnsembleSegmentation, VASCXTransform, from_huggingface
from PIL import Image
import torch

# Load model
model_path = from_huggingface('Eyened/vascx:artery_vein/av_july24.pt')
model = EnsembleSegmentation(model_path, VASCXTransform())

# Predict
rgb_image = Image.open('fundus.jpg')
prediction = model.predict(rgb_image)  # Returns [B, H, W] with class values
```

![Artery/Vein Segmentation Result](https://github.com/kapong/vascx_simplify/raw/main/examples/artery_vein_segmentation_result.png)

**Full example**: See [`examples/01_artery_vein.py`](examples/01_artery_vein.py)

### Optic Disc Segmentation

Detect and segment the optic disc:

```python
from vascx_simplify import EnsembleSegmentation, VASCXTransform, from_huggingface
from PIL import Image

model_path = from_huggingface('Eyened/vascx:disc/disc_july24.pt')
model = EnsembleSegmentation(model_path, VASCXTransform(512))

rgb_image = Image.open('fundus.jpg')
prediction = model.predict(rgb_image)  # Returns [B, H, W] with class values
```

![Optic Disc Segmentation Result](https://github.com/kapong/vascx_simplify/raw/main/examples/disc_segmentation_result.png)

**Full example**: See [`examples/02_disc_segment.py`](examples/02_disc_segment.py)

### Fovea Detection

Locate the fovea center using heatmap regression:

```python
from vascx_simplify import HeatmapRegressionEnsemble, VASCXTransform, from_huggingface
from PIL import Image

model_path = from_huggingface('Eyened/vascx:fovea/fovea_july24.pt')
model = HeatmapRegressionEnsemble(model_path, VASCXTransform())

rgb_image = Image.open('fundus.jpg')
prediction = model.predict(rgb_image)  # Returns [B, M, 2] with (x, y) coordinates

fovea_x = prediction[0, 0, 0].item()
fovea_y = prediction[0, 0, 1].item()
```

![Fovea Detection Result](https://github.com/kapong/vascx_simplify/raw/main/examples/fovea_detection_result.png)

**Full example**: See [`examples/03_fovea_regression.py`](examples/03_fovea_regression.py)

### Image Quality Assessment

Classify fundus image quality (Reject/Usable/Good):

```python
from vascx_simplify import ClassificationEnsemble, VASCXTransform, from_huggingface
from PIL import Image

model_path = from_huggingface('Eyened/vascx:quality/quality.pt')
model = ClassificationEnsemble(model_path, VASCXTransform(use_ce=False))

rgb_image = Image.open('fundus.jpg')
prediction = model.predict(rgb_image)  # Returns [B, 3] with quality scores (already softmaxed)

# Get probabilities (already normalized)
q1_reject, q2_usable, q3_good = prediction[0].tolist()
```

![Image Quality Classification Result](https://github.com/kapong/vascx_simplify/raw/main/examples/quality_classification_result.png)

**Full example**: See [`examples/04_quality_classify.py`](examples/04_quality_classify.py)

### Contrast Enhancement

Enhance fundus image contrast with GPU-accelerated preprocessing:

```python
from vascx_simplify.preprocess import FundusContrastEnhance
from PIL import Image
import torch
import numpy as np

# Initialize enhancer
enhancer = FundusContrastEnhance(
    use_fp16=True,      # Use mixed precision for faster processing
    square_size=512,    # Optional: crop to square size
)

# Load and convert image to tensor
rgb_image = Image.open('fundus.jpg')
img_tensor = torch.from_numpy(np.array(rgb_image)).permute(2, 0, 1).cuda()  # you can use CPU/GPU to process

# Apply contrast enhancement
original, enhanced, bounds = enhancer(img_tensor)

# Convert back to numpy for saving/visualization
enhanced_np = enhanced.cpu().permute(1, 2, 0).numpy()
```

![Contrast Enhancement Detailed](https://github.com/kapong/vascx_simplify/raw/main/examples/contrast_enhancement_detailed.png)

**Full example**: See [`examples/07_contrast_enhancement.py`](examples/07_contrast_enhancement.py) for detailed visualization with before/after comparison, zoomed regions, and statistics.

### Batch Processing

Process multiple images with automatic batch splitting to prevent out-of-memory errors:

⚠️ **Important**: Batch processing speed benefit depends on model type:
- **Segmentation/Classification/Regression**: 2-3x faster than sequential
- **Heatmap Regression (fovea)**: NO speed improvement (due to sliding window complexity)

```python
from vascx_simplify import EnsembleSegmentation, VASCXTransform, from_huggingface
from PIL import Image

# Load model once
model_path = from_huggingface('Eyened/vascx:artery_vein/av_july24.pt')
model = EnsembleSegmentation(model_path, VASCXTransform())

# Load multiple images
images = [Image.open(f'fundus_{i}.jpg') for i in range(10)]

# Batch prediction (much faster for segmentation/classification)
predictions = model.predict(images)  # Returns [10, H, W]
# Automatically splits into chunks of 4 (default for segmentation)

# Process each result
for i, pred in enumerate(predictions):
    # pred is [H, W] - no need for [0] indexing
    save_prediction(pred, f'result_{i}.png')

# For large datasets, automatic splitting prevents OOM
large_images = [Image.open(f'img_{i}.jpg') for i in range(100)]
predictions = model.predict(large_images)  # Auto-splits into chunks
# Returns [100, H, W] - seamlessly handles large batches

# Override batch size for your GPU memory
predictions = model.predict(images, batch_size=8)  # Process 8 at once
```

**Key Features**:
- ️ **Automatic OOM prevention** - large batches split automatically
- 🔄 **100% backward compatible** - existing single-image code works unchanged
- ⚙️ **Configurable** - override batch size per call or use smart defaults
- 📊 **Flexible inputs** - accepts PIL.Image, torch.Tensor, or lists of either

**Default Batch Sizes by Model Type**:
- **All Models**: `batch_size=1` (no speed benefit from batching observed in testing)

**Usage Patterns**:

```python
# Pattern 1: Batch processing a directory
import glob
from pathlib import Path

image_paths = glob.glob('fundus_images/*.jpg')
images = [Image.open(p) for p in image_paths]
predictions = model.predict(images)  # Efficient batch processing

# Pattern 2: Process with torch tensors
tensors = [torch.randn(3, 512, 512) for _ in range(20)]
predictions = model.predict(tensors)  # Works with tensors too

# Pattern 3: Memory-constrained environment
predictions = model.predict(images, batch_size=2)  # Smaller batches

# Pattern 4: High-memory GPU
predictions = model.predict(images, batch_size=16)  # Larger batches
```

**Backward Compatibility**: All existing single-image code works without modification:
```python
# This still works exactly as before
image = Image.open('fundus.jpg')
pred = model.predict(image)  # Returns [1, H, W]
result = pred[0]  # Access with [0] as before
```

**Performance Tips**:
- Default batch size is set to 1 for all models (no speed benefit observed from batching)
- You can override batch_size parameter for memory management if processing large numbers of images
- Increase batch_size for high-memory GPUs (24GB+) to process more images simultaneously
- Decrease batch_size if you encounter OOM errors
- List of tensors is slightly faster than list of PIL.Images

**Full examples**: 
- [`examples/05_batch_fovea.py`](examples/05_batch_fovea.py) - Batch processing for heatmap regression
- [`examples/06_utils_demo.py`](examples/06_utils_demo.py) - Utility functions demonstration

## License

MIT License - see LICENSE file for details.

## Author

Phongphan Phienphanich <garpong@gmail.com>

## Acknowledgments

This is a simplified rewrite of [rtnls_vascx_models](https://github.com/Eyened/rtnls_vascx_models) by Eyened.
