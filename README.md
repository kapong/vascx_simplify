# vascx_simplify

**⚠️ IMPORTANT: This project is NOT original work.**

This is a simplified rewrite of the original [rtnls_vascx_models](https://github.com/Eyened/rtnls_vascx_models) project by Eyened. All core algorithms, preprocessing logic, and model architectures are derived from that original work.

## AI Usage Disclaimer

**This project was developed with significant assistance from AI tools** (GitHub Copilot, ChatGPT, Claude). AI was used extensively for:
- Code organization and refactoring
- Documentation and comments
- Testing infrastructure
- Packaging and build configuration
- Docker setup
- CI/CD pipeline

The underlying algorithms and models are from the original work referenced above.

---

A PyTorch library for vessel and fundus image analysis, providing GPU-accelerated preprocessing, contrast enhancement, and inference utilities for medical imaging tasks.

## Features

- **GPU-Accelerated Preprocessing**: Fast fundus image contrast enhancement with mixed precision (FP16/FP32) support
- **Sliding Window Inference**: Efficient inference for large images with MONAI-compatible implementation
- **Ensemble Models**: Support for segmentation, classification, regression, and heatmap-based models
- **HuggingFace Integration**: Easy model loading from HuggingFace Hub
- **Flexible Transformations**: Built-in image preprocessing and normalization

## Installation

### From PyPI (once published)

```bash
pip install vascx_simplify
```

### From source

```bash
git clone https://github.com/kapong/vascx_simplify.git
cd vascx_simplify
pip install -e .
```

### With uv (recommended)

```bash
uv pip install vascx_simplify
```

## Quick Start

```python
import torch
from vascx_simplify import VASCXTransform, EnsembleSegmentation, from_huggingface

# Download model from HuggingFace
model_path = from_huggingface("username/model-name:model.pt")

# Setup preprocessing
transform = VASCXTransform(size=1024, have_ce=True, device='cuda')

# Load model
model = EnsembleSegmentation(
    fpath=model_path,
    transforms=transform,
    device='cuda'
)

# Process an image
import numpy as np
from PIL import Image

image = Image.open("fundus_image.jpg")
prediction = model.predict(image)
```

## Usage Examples

### Preprocessing Only

```python
from vascx_simplify import FundusContrastEnhance
import torch

# Initialize enhancer
enhancer = FundusContrastEnhance(
    square_size=1024,
    sigma_fraction=0.05,
    contrast_factor=4,
    use_fp16=True  # Use FP16 for faster processing on GPU
)

# Process image (torch.Tensor [C, H, W])
rgb, enhanced, bounds = enhancer(image)
```

### Sliding Window Inference

```python
from vascx_simplify import sliding_window_inference
import torch

# Your model
class MyModel(torch.nn.Module):
    def forward(self, x):
        # Your implementation
        return x

model = MyModel()
inputs = torch.randn(1, 3, 2048, 2048).cuda()

# Perform inference
output = sliding_window_inference(
    inputs=inputs,
    roi_size=(512, 512),
    sw_batch_size=4,
    predictor=model,
    overlap=0.5,
    mode='gaussian'
)
```

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/kapong/vascx_simplify.git
cd vascx_simplify

# Install with development dependencies
pip install -e ".[dev,test]"
```

### Running Tests

Tests have been removed from this simplified version.

### Code Quality

```bash
# Format code
black src/

# Sort imports
isort src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- kornia >= 0.6.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- numpy >= 1.21.0
- huggingface-hub >= 0.10.0

## Contributing

This is a simplified rewrite for personal use. For contributions to the original work, please visit [rtnls_vascx_models](https://github.com/Eyened/rtnls_vascx_models).

## License

MIT License - see LICENSE file for details.

## Citation

If you use this library in your research, please cite the original work:

```bibtex
@software{vascx_simplify,
  title = {vascx_simplify: Simplified rewrite of rtnls_vascx_models},
  author = {kapong},
  year = {2025},
  url = {https://github.com/kapong/vascx_simplify},
  note = {Simplified rewrite of https://github.com/Eyened/rtnls_vascx_models}
}
```

## Acknowledgments

**Original Work**: [rtnls_vascx_models](https://github.com/Eyened/rtnls_vascx_models) by Eyened - All core algorithms and models are from this original work.

**AI Assistance**: This rewrite was created with extensive assistance from AI tools (GitHub Copilot, ChatGPT, Claude) for code organization, documentation, and packaging.
