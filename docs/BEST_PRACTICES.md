# Best Practices Guide

## Table of Contents
1. [Code Organization](#code-organization)
2. [Naming Conventions](#naming-conventions)
3. [Type Hints and Documentation](#type-hints-and-documentation)
4. [Error Handling](#error-handling)
5. [Design Patterns](#design-patterns)
6. [Testing Patterns](#testing-patterns)
7. [When NOT to Abstract](#when-not-to-abstract)

## Code Organization

### Principle: Keep It Simple

**Good code is:**
- Easy to read and understand
- Self-documenting with clear names
- Focused on solving one problem well
- As simple as possible, but no simpler

**Avoid:**
- Over-engineering simple tasks
- Premature abstraction
- Unnecessary complexity
- "Clever" code that's hard to understand

### File Structure

```python
# ✅ GOOD: Clear separation of concerns

# inference.py - Inference utilities
def sliding_window_inference(...):
    """Sliding window inference."""
    pass

def ensemble_predict(...):
    """Ensemble prediction."""
    pass

# preprocess.py - Preprocessing pipelines
class FundusContrastEnhance:
    """Contrast enhancement."""
    pass

class VASCXTransform:
    """Standard transform pipeline."""
    pass

# utils.py - Helper functions
def from_huggingface(modelstr: str):
    """Download from HuggingFace."""
    pass

def load_checkpoint(path: str):
    """Load model checkpoint."""
    pass
```

```python
# ❌ BAD: Everything in one file
# vascx_simplify.py (3000 lines!)
# - Hard to navigate
# - Difficult to maintain
# - Unclear responsibilities
```

### Function Organization

```python
# ✅ GOOD: Single Responsibility Principle
def load_image(path: str) -> np.ndarray:
    """Load image from disk."""
    return np.array(Image.open(path))

def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Preprocess image for model."""
    tensor = torch.from_numpy(image).float()
    return tensor / 255.0

def predict(model, image: torch.Tensor) -> torch.Tensor:
    """Run model prediction."""
    with torch.no_grad():
        return model(image)

# ❌ BAD: Doing too many things
def load_preprocess_and_predict(path: str, model):
    """Load, preprocess, and predict (too much!)."""
    image = Image.open(path)
    array = np.array(image)
    tensor = torch.from_numpy(array).float() / 255.0
    with torch.no_grad():
        output = model(tensor)
    return output.cpu().numpy()
```

## Naming Conventions

### Principle: Names Should Reveal Intent

```python
# ✅ GOOD: Descriptive names
def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: tuple,
    sw_batch_size: int,
    predictor: callable
) -> torch.Tensor:
    """Inference with sliding window."""
    pass

# ❌ BAD: Abbreviated or unclear names
def swi(inp: torch.Tensor, roi: tuple, bs: int, pred: callable) -> torch.Tensor:
    """What do these mean?"""
    pass
```

### Conventions

**Functions and Variables:**
```python
# Use snake_case for functions and variables
def preprocess_image(image: torch.Tensor) -> torch.Tensor:
    window_size = 256
    gaussian_kernel = create_kernel(window_size)
    return apply_kernel(image, gaussian_kernel)
```

**Classes:**
```python
# Use PascalCase for classes
class EnsembleSegmentation:
    """Ensemble segmentation model."""
    pass

class VASCXTransform:
    """VASCX preprocessing transform."""
    pass
```

**Constants:**
```python
# Use UPPER_CASE for constants
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_RESOLUTION = 2048
MIN_CONFIDENCE = 0.5
```

**Private Methods:**
```python
class MyProcessor:
    def process(self, image: torch.Tensor) -> torch.Tensor:
        """Public method."""
        return self._internal_process(image)
    
    def _internal_process(self, image: torch.Tensor) -> torch.Tensor:
        """Private method (single underscore)."""
        return self._compute_result(image)
    
    def _compute_result(self, image: torch.Tensor) -> torch.Tensor:
        """Private helper."""
        return image * 2
```

### Naming Examples

```python
# ✅ GOOD: Clear and descriptive
def create_gaussian_importance_map(height: int, width: int) -> torch.Tensor:
    """Create Gaussian importance map for sliding window."""
    pass

def calculate_sliding_window_positions(
    image_size: tuple,
    window_size: tuple,
    overlap: float
) -> List[tuple]:
    """Calculate positions for sliding windows."""
    pass

# ❌ BAD: Unclear or abbreviated
def create_map(h: int, w: int) -> torch.Tensor:
    """What kind of map?"""
    pass

def calc_pos(sz: tuple, win: tuple, ovr: float) -> List[tuple]:
    """Hard to understand."""
    pass
```

## Type Hints and Documentation

### Type Hints

```python
from typing import Optional, Union, List, Tuple, Callable
import torch
from PIL import Image

# ✅ GOOD: Complete type hints
def predict(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    model: torch.nn.Module,
    device: Optional[str] = None,
    batch_size: int = 1
) -> torch.Tensor:
    """Predict with flexible input types."""
    pass

def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Tuple[int, int],
    predictor: Callable[[torch.Tensor], torch.Tensor],
    overlap: float = 0.5
) -> torch.Tensor:
    """Sliding window inference."""
    pass

# ❌ BAD: No type hints
def predict(image, model, device=None, batch_size=1):
    """Predict (unclear what types are expected)."""
    pass
```

### Docstrings

```python
# ✅ GOOD: Clear, informative docstring
def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Tuple[int, int],
    sw_batch_size: int,
    predictor: Callable[[torch.Tensor], torch.Tensor],
    overlap: float = 0.5,
    mode: str = 'gaussian'
) -> torch.Tensor:
    """
    Perform sliding window inference on large images.
    
    Divides input into overlapping windows, runs prediction on each window,
    and blends results using Gaussian importance weighting.
    
    Args:
        inputs: Input tensor of shape [B, C, H, W]
        roi_size: Window size (height, width)
        sw_batch_size: Number of windows to process in parallel
        predictor: Model prediction function
        overlap: Overlap ratio between windows (0-1)
        mode: Blending mode ('gaussian' or 'constant')
    
    Returns:
        Predicted tensor of shape [B, num_classes, H, W]
    
    Example:
        >>> image = torch.randn(1, 3, 2048, 2048)
        >>> output = sliding_window_inference(
        ...     image,
        ...     roi_size=(512, 512),
        ...     sw_batch_size=4,
        ...     predictor=model
        ... )
    """
    pass

# ❌ BAD: Minimal or missing docstring
def sliding_window_inference(inputs, roi_size, sw_batch_size, predictor, overlap=0.5, mode='gaussian'):
    """Do sliding window inference."""  # Not helpful!
    pass
```

### When to Skip Docstrings

```python
# Skip docstrings for:
# 1. Obvious private helpers
def _validate_shape(tensor: torch.Tensor, expected: tuple) -> None:
    if tensor.shape != expected:
        raise ValueError(f"Expected {expected}, got {tensor.shape}")

# 2. Simple getters/setters
@property
def device(self) -> str:
    return self._device

# 3. Very simple functions
def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().numpy()
```

## Error Handling

### Principle: Fail Fast with Clear Messages

```python
# ✅ GOOD: Specific exceptions with context
def sliding_window_inference(inputs: torch.Tensor, roi_size: tuple, ...):
    if inputs.ndim != 4:
        raise ValueError(
            f"Expected 4D tensor [B,C,H,W], got shape {inputs.shape} "
            f"with {inputs.ndim} dimensions"
        )
    
    if roi_size[0] > inputs.shape[2] or roi_size[1] > inputs.shape[3]:
        raise ValueError(
            f"Window size {roi_size} larger than image size "
            f"{inputs.shape[2:]} (H×W)"
        )
    
    if not 0 <= overlap < 1:
        raise ValueError(
            f"Overlap must be in range [0, 1), got {overlap}"
        )

# ❌ BAD: Generic exceptions
def sliding_window_inference(inputs: torch.Tensor, roi_size: tuple, ...):
    if inputs.ndim != 4:
        raise Exception("Wrong shape")  # Not helpful!
    
    if roi_size[0] > inputs.shape[2]:
        raise Exception("Error")  # What error?
```

### Exception Types

```python
# Use appropriate exception types
def load_model(path: str) -> torch.nn.Module:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    if not path.endswith('.pt'):
        raise ValueError(f"Expected .pt file, got {path}")
    
    try:
        model = torch.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}")
    
    return model
```

### Validation Helpers

```python
# ✅ GOOD: Reusable validation functions
def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_ndim: int,
    name: str = "tensor"
) -> None:
    """Validate tensor shape."""
    if tensor.ndim != expected_ndim:
        raise ValueError(
            f"{name} must be {expected_ndim}D, got shape {tensor.shape}"
        )

def validate_range(
    value: float,
    min_val: float,
    max_val: float,
    name: str = "value",
    inclusive: bool = True
) -> None:
    """Validate value is in range."""
    if inclusive:
        if not min_val <= value <= max_val:
            raise ValueError(
                f"{name} must be in range [{min_val}, {max_val}], got {value}"
            )
    else:
        if not min_val < value < max_val:
            raise ValueError(
                f"{name} must be in range ({min_val}, {max_val}), got {value}"
            )

# Usage
def my_function(tensor: torch.Tensor, overlap: float):
    validate_tensor_shape(tensor, expected_ndim=4, name="input tensor")
    validate_range(overlap, 0, 1, name="overlap", inclusive=False)
```

## Design Patterns

### Pattern 1: GPU-Accelerated Class with Pre-computed Constants

```python
class EfficientProcessor:
    """GPU-accelerated processor with cached constants."""
    
    # Class-level constants (shared across instances)
    RESOLUTION = 256
    CENTER = RESOLUTION // 2
    
    # Pre-compute numpy arrays (for initialization)
    _gaussian_kernel_np = signal.windows.gaussian(5, 1)
    _mask_np = np.arange(RESOLUTION) > RESOLUTION // 2
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # Convert to GPU tensors once
        self.gaussian_kernel = torch.from_numpy(self._gaussian_kernel_np).to(device)
        self.mask = torch.from_numpy(self._mask_np).to(device)
        
    def process(self, image: torch.Tensor) -> torch.Tensor:
        """Process image using cached GPU tensors."""
        image = image.to(self.device)
        
        # Use pre-computed tensors (fast!)
        blurred = F.conv2d(image, self.gaussian_kernel)
        masked = blurred * self.mask
        
        return masked
```

### Pattern 2: Flexible Input Types

```python
def predict(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    model: torch.nn.Module,
    device: str = 'cuda'
) -> torch.Tensor:
    """Accept multiple input types."""
    
    # Convert to torch tensor
    if isinstance(image, Image.Image):
        image = torch.from_numpy(np.array(image))
    elif isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    elif not isinstance(image, torch.Tensor):
        raise TypeError(
            f"Expected Tensor, ndarray, or PIL Image, got {type(image)}"
        )
    
    # Ensure correct shape [B, C, H, W]
    if image.ndim == 3:
        image = image.unsqueeze(0)
    
    # Move to device
    image = image.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image)
    
    return output
```

### Pattern 3: Device-Aware Processing

```python
def process_batch(
    images: List[torch.Tensor],
    model: torch.nn.Module,
    device: Optional[str] = None
) -> List[torch.Tensor]:
    """Process batch with automatic device handling."""
    
    # Use device of first image if not specified
    if device is None:
        device = images[0].device
    
    # Stack and move to device
    batch = torch.stack(images).to(device)
    
    # Process
    with torch.no_grad():
        output = model(batch)
    
    # Return on same device
    return list(output)
```

### Pattern 4: Ensemble with Soft Voting

```python
class EnsembleModel:
    """Ensemble multiple models with soft voting."""
    
    def __init__(self, model_paths: List[str], device: str = 'cuda'):
        self.device = device
        self.models = [self._load_model(path) for path in model_paths]
        
    def _load_model(self, path: str) -> torch.nn.Module:
        model = torch.load(path)
        model.to(self.device)
        model.eval()
        return model
    
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Ensemble prediction with soft voting."""
        image = image.to(self.device)
        
        # Collect predictions
        predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = model(image)
                predictions.append(pred)
        
        # Average (soft voting)
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        
        return ensemble_pred
```

## Testing Patterns

### Pattern 1: Deterministic Testing

```python
def test_reproducibility():
    """Test that function produces deterministic results."""
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create test input
    input_tensor = torch.randn(1, 3, 256, 256)
    
    # Run twice
    output1 = my_function(input_tensor)
    output2 = my_function(input_tensor)
    
    # Should be identical
    assert torch.allclose(output1, output2), "Non-deterministic output!"
```

### Pattern 2: Device Testing

```python
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_handling(device):
    """Test function works on both CPU and GPU."""
    
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    output = my_function(input_tensor)
    
    # Output should be on same device
    assert output.device.type == device
```

### Pattern 3: Performance Testing

```python
def test_performance():
    """Test performance meets requirements."""
    import time
    
    input_tensor = torch.randn(1, 3, 512, 512).cuda()
    model = load_model().cuda()
    
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(100):
        _ = model(input_tensor)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    avg_time_ms = elapsed / 100 * 1000
    
    # Should be under 20ms per image
    assert avg_time_ms < 20, f"Too slow: {avg_time_ms:.1f}ms"
```

## When NOT to Abstract

### Anti-Pattern: Premature Abstraction

```python
# ❌ BAD: Over-engineered for simple task
class ImageProcessor:
    def __init__(self, strategy: ProcessingStrategy):
        self.strategy = strategy
    
    def process(self, image):
        return self.strategy.execute(image)

class GaussianStrategy(ProcessingStrategy):
    def execute(self, image):
        return gaussian_blur(image)

class MedianStrategy(ProcessingStrategy):
    def execute(self, image):
        return median_filter(image)

# Usage (too complex!)
processor = ImageProcessor(GaussianStrategy())
result = processor.process(image)

# ✅ GOOD: Keep it simple
def gaussian_blur(image: torch.Tensor) -> torch.Tensor:
    """Apply Gaussian blur."""
    return F.gaussian_blur(image, kernel_size=5)

def median_filter(image: torch.Tensor) -> torch.Tensor:
    """Apply median filter."""
    return F.median_filter(image, kernel_size=5)

# Usage (clear and simple!)
result = gaussian_blur(image)
```

### When to Create Abstractions

Create abstractions when you have:
1. **Three or more** similar implementations (Rule of Three)
2. **Complex state** that needs to be managed
3. **Clear interface** that multiple implementations share
4. **Significant code duplication** that's error-prone

```python
# ✅ GOOD: Valid abstraction (multiple models with same interface)
class BaseModel:
    """Base class for all models."""
    
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class SegmentationModel(BaseModel):
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        # Segmentation-specific logic
        pass

class ClassificationModel(BaseModel):
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        # Classification-specific logic
        pass

class RegressionModel(BaseModel):
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        # Regression-specific logic
        pass
```

## Quick Reference: Best Practices Checklist

- [ ] **Naming**
  - [ ] Functions and variables use snake_case
  - [ ] Classes use PascalCase
  - [ ] Constants use UPPER_CASE
  - [ ] Names reveal intent (no abbreviations)

- [ ] **Type Hints**
  - [ ] All function parameters have type hints
  - [ ] Return types specified
  - [ ] Use Optional for nullable types
  - [ ] Use Union for multiple types

- [ ] **Documentation**
  - [ ] Public APIs have docstrings
  - [ ] Docstrings include Args, Returns, Example
  - [ ] Skip docstrings for obvious private helpers

- [ ] **Error Handling**
  - [ ] Specific exception types used
  - [ ] Error messages include context
  - [ ] Input validation at function entry

- [ ] **Code Organization**
  - [ ] Functions do one thing well
  - [ ] Files organized by responsibility
  - [ ] No code duplication (unless 2 or fewer instances)
  - [ ] Keep it simple - avoid over-engineering

- [ ] **Performance**
  - [ ] No unnecessary CPU↔GPU transfers
  - [ ] Pre-compute constants where possible
  - [ ] Use appropriate precision (FP16 vs FP32)

## Summary

1. **Keep it simple** - Solve the problem directly, don't over-engineer
2. **Name things clearly** - Code should read like English
3. **Add types and docs** - Make intent explicit
4. **Fail fast** - Validate early with clear error messages
5. **Don't abstract until needed** - Rule of three
6. **Test what matters** - Correctness, performance, device handling
7. **Follow conventions** - Consistency makes code easier to read

**Remember:** Good code is simple code that solves real problems!
