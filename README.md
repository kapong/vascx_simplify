# VASCX Simplify

A PyTorch library for vessel and fundus image analysis, providing GPU-accelerated preprocessing, contrast enhancement, and inference utilities for medical imaging tasks.

> **Note**: This is a simplified rewrite of the original [rtnls_vascx_models](https://github.com/Eyened/rtnls_vascx_models) project by Eyened. This version focuses on making the core functionality more accessible and easier to use as a pip-installable library.

## AI Usage Disclaimer

Parts of this codebase were developed with assistance from AI tools (GitHub Copilot, ChatGPT, Claude) for code organization, documentation, testing infrastructure, and packaging setup. The core algorithms and models are based on the original work referenced above.

## Features

- **GPU-Accelerated Preprocessing**: Fast fundus image contrast enhancement with mixed precision (FP16/FP32) support
- **Sliding Window Inference**: Efficient inference for large images with MONAI-compatible implementation
- **Ensemble Models**: Support for segmentation, classification, regression, and heatmap-based models
- **HuggingFace Integration**: Easy model loading from HuggingFace Hub
- **Flexible Transformations**: Built-in image preprocessing and normalization

## Installation

### From PyPI (once published)

```bash
pip install vascx-simplify
```

### From source

```bash
git clone https://github.com/kapong/vascx_simplify.git
cd vascx_simplify
pip install -e .
```

### With uv (recommended)

```bash
uv pip install vascx-simplify
```

## Quick Start

```python
import torch
from simple_vascx import VASCXTransform, EnsembleSegmentation, from_huggingface

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
from simple_vascx import FundusContrastEnhance
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
from simple_vascx import sliding_window_inference
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
git clone https://github.com/YOUR-USERNAME/simple_vascx.git
cd simple_vascx

# Install with development dependencies
pip install -e ".[dev,test]"
```

### Running Tests

#### Local Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=simple_vascx --cov-report=html

# Run specific test file
pytest tests/test_inference.py
```

#### Docker Testing

```bash
# CPU testing
docker-compose up test-cpu

# GPU testing (requires nvidia-docker)
docker-compose up test-gpu

# Development shell
docker-compose run dev

# Build package
docker-compose up build
```

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

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

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Authors

<!-- TODO: Add author information -->

## Citation

If you use this library in your research, please cite:

```bibtex
@software{vascx_simplify,
  title = {VASCX Simplify: A PyTorch library for vessel and fundus image analysis},
  author = {TODO: Add authors},
  year = {2025},
  url = {https://github.com/kapong/vascx_simplify},
  note = {Simplified rewrite of https://github.com/Eyened/rtnls_vascx_models}
}
```

## Acknowledgments

- **Original Work**: This project is a simplified rewrite of [rtnls_vascx_models](https://github.com/Eyened/rtnls_vascx_models) by Eyened
- Built with PyTorch and Kornia
- Sliding window inference implementation based on MONAI
- Development assisted by AI tools for packaging, documentation, and testing infrastructure
