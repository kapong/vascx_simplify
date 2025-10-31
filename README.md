# vascx_simplify

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

- Python >= 3.8
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
transform = VASCXTransform(size=1024, have_ce=True, device='cuda')
model = EnsembleSegmentation(model_path, transform, device='cuda')

# Predict
rgb_image = Image.open('fundus.jpg')
prediction = model.predict(rgb_image)  # Returns [B, H, W] with class values
```

### Optic Disc Segmentation

Detect and segment the optic disc:

```python
from vascx_simplify import EnsembleSegmentation, VASCXTransform, from_huggingface
from PIL import Image

model_path = from_huggingface('Eyened/vascx:disc/disc_july24.pt')
transform = VASCXTransform(size=512, have_ce=True, device='cuda')
model = EnsembleSegmentation(model_path, transform, device='cuda')

rgb_image = Image.open('fundus.jpg')
prediction = model.predict(rgb_image)  # Returns [B, H, W] with class values
```

### Fovea Detection

Locate the fovea center using heatmap regression:

```python
from vascx_simplify import HeatmapRegressionEnsemble, VASCXTransform, from_huggingface
from PIL import Image

model_path = from_huggingface('Eyened/vascx:fovea/fovea_july24.pt')
transform = VASCXTransform(size=1024, have_ce=True, device='cuda')
model = HeatmapRegressionEnsemble(model_path, transform, device='cuda')

rgb_image = Image.open('fundus.jpg')
prediction = model.predict(rgb_image)  # Returns [B, M, 2] with (x, y) coordinates

fovea_x = prediction[0, 0, 0].item()
fovea_y = prediction[0, 0, 1].item()
```

### Image Quality Assessment

Classify fundus image quality (Reject/Usable/Good):

```python
from vascx_simplify import ClassificationEnsemble, VASCXTransform, from_huggingface
from PIL import Image
import torch.nn.functional as F

model_path = from_huggingface('Eyened/vascx:quality/quality.pt')
transform = VASCXTransform(size=1024, have_ce=False, device='cuda')  # No contrast enhancement
model = ClassificationEnsemble(model_path, transform, device='cuda')

rgb_image = Image.open('fundus.jpg')
prediction = model.predict(rgb_image)  # Returns [B, M, 3] with quality scores

# Convert to probabilities
probs = F.softmax(prediction[0, 0, :], dim=0)
q1_reject, q2_usable, q3_good = probs.tolist()
```

## License

MIT License - see LICENSE file for details.

## Author

Phongphan Phienphanich <garpong@gmail.com>

## Acknowledgments

This is a simplified rewrite of [rtnls_vascx_models](https://github.com/Eyened/rtnls_vascx_models) by Eyened.
