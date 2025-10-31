# Quick Reference Cheat Sheet

## 🚀 Essential Commands

### Git Workflow
```bash
# Check status
git status                          # What changed?
git diff                           # Show changes

# Stage and commit
git add <file>                     # Stage file
git commit -m "type: message"      # Commit

# Review
git diff HEAD~1                    # Compare with last commit
git show HEAD                      # Show last commit
git log --oneline -5               # Recent commits
```

### Testing
```bash
# Run examples
python examples/01_artery_vein.py
python examples/02_disc_segment.py

# Run tests
python -m pytest tests/
python -m pytest tests/ -v         # Verbose

# Check consistency
python test_consistency.py
```

## 💡 Code Patterns

### GPU-Accelerated Class
```python
class MyProcessor:
    # Pre-compute constants
    RESOLUTION = 256
    _mask_np = np.arange(RESOLUTION) > RESOLUTION // 2
    
    def __init__(self, device='cuda'):
        self.device = device
        # Convert to GPU once
        self.mask = torch.from_numpy(self._mask_np).to(device)
    
    def process(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.device)
        return image * self.mask
```

### Device-Aware Function
```python
def predict(
    image: torch.Tensor,
    device: Optional[str] = None
) -> torch.Tensor:
    if device is None:
        device = image.device
    image = image.to(device)
    result = model(image)
    return result
```

### Type Hints
```python
from typing import Optional, Union, List, Tuple, Callable

def func(
    tensor: torch.Tensor,
    size: Tuple[int, int],
    callback: Callable[[torch.Tensor], torch.Tensor],
    device: Optional[str] = None
) -> torch.Tensor:
    pass
```

## ⚡ Performance Optimization

### Keep on GPU
```python
# ✅ GOOD: Stay on GPU
preprocessed = preprocess(image)    # GPU → GPU
features = extract(preprocessed)    # GPU → GPU
result = model(features)            # GPU → GPU
final = result.cpu()                # GPU → CPU (once at end)

# ❌ BAD: Multiple transfers
preprocessed = preprocess(image).cpu()     # GPU → CPU
features = extract(preprocessed.cuda())    # CPU → GPU
result = model(features).cpu()             # GPU → CPU
```

### Mixed Precision
```python
# FP16 for compute-intensive ops
image_fp16 = image.half()
blurred = gaussian_blur(image_fp16)    # 2-4x faster!

# FP32 for precision-critical ops
coords = compute_coordinates(blurred.float())
params = fit_ransac(coords)            # Needs precision
```

### Batch Processing
```python
# ✅ GOOD: Batch inference
predictions = model(torch.stack(images))

# ❌ BAD: Sequential
predictions = [model(img.unsqueeze(0)) for img in images]
```

## 📝 Naming Conventions

```python
# Functions and variables: snake_case
def process_image(image_tensor: torch.Tensor) -> torch.Tensor:
    window_size = 256
    gaussian_kernel = create_kernel(window_size)

# Classes: PascalCase
class EnsembleSegmentation:
    pass

# Constants: UPPER_CASE
DEFAULT_DEVICE = 'cuda'
MAX_RESOLUTION = 2048
```

## ✅ Code Quality Checklist

Before committing:
- [ ] No `.cpu()` followed by `.cuda()` in same function
- [ ] Type hints on all function parameters
- [ ] Docstrings on public APIs
- [ ] Clear, descriptive variable names
- [ ] Error messages include context
- [ ] No commented-out code
- [ ] No debug print statements
- [ ] Output matches original (if refactoring)

## 🎯 Commit Message Format

```
<type>(<scope>): <description>

<type> = feat | fix | perf | refactor | docs | test | chore
```

### Examples
```bash
git commit -m "feat: add gaussian importance map for sliding window"
git commit -m "perf: optimize preprocessing with FP16 (2.5x speedup)"
git commit -m "fix: preserve tensor device in preprocessing"
git commit -m "refactor: simplify RANSAC ellipse fitting"
git commit -m "docs: add performance optimization guide"
```

## 🔍 Debugging

### Profile Memory
```python
@profile_memory
def my_function(image):
    return model(image)

# Output:
# my_function Memory Profile:
#   Start: 150.0 MB
#   End: 180.0 MB
#   Peak: 250.0 MB
#   Delta: 30.0 MB
```

### Detect Transfers
```python
@profile_transfers
def my_pipeline(image):
    return model(preprocess(image))

# Output:
# my_pipeline:
#   Time: 0.015s
#   CPU transfers: 0
#   CUDA transfers: 1
```

### Benchmark
```python
with benchmark("Inference", n_runs=100):
    for _ in range(100):
        result = model(image)

# Output:
# Inference:
#   Total: 1.234s
#   Per run: 12.34ms
#   Throughput: 81.0 runs/s
```

## 🚨 Common Mistakes to Avoid

```python
# ❌ DON'T: Over-engineer simple tasks
class ImageNormalizer:
    def normalize(self, x):
        return (x - self.mean) / self.std

# ✅ DO: Keep it simple
def normalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x - mean) / std

# ❌ DON'T: Vague error messages
raise Exception("Wrong shape")

# ✅ DO: Clear error messages
raise ValueError(f"Expected 4D tensor [B,C,H,W], got shape {tensor.shape}")

# ❌ DON'T: Generic variable names
def process(inp, sz, bs):
    pass

# ✅ DO: Descriptive names
def process(image: torch.Tensor, window_size: int, batch_size: int):
    pass
```

## 📊 Performance Targets (RTX 3090)

| Operation | Resolution | FP32 | FP16 | Speedup |
|-----------|-----------|------|------|---------|
| Preprocessing | 512×512 | 15ms | 5ms | 3.0x |
| Segmentation | 512×512 | 25ms | 10ms | 2.5x |
| Sliding Window | 2048×2048 | 250ms | 100ms | 2.5x |

## 📚 Documentation

- **[BEST_PRACTICES.md](BEST_PRACTICES.md)** - Code organization and patterns
- **[PERFORMANCE.md](PERFORMANCE.md)** - GPU optimization techniques
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Git workflow and testing
- **[docs/README.md](README.md)** - Documentation overview

## 🔗 Quick Links

### Git
- Review changes: `git diff`
- Last commit: `git show HEAD`
- Compare: `git diff HEAD~1`
- History: `git log --oneline -5`

### Testing
- Examples: `python examples/01_artery_vein.py`
- Tests: `python -m pytest tests/`
- Consistency: `python test_consistency.py`

### Performance
- FP16 for: blur, conv, warp
- FP32 for: RANSAC, coordinates
- Batch size: 4-8 for 512×512
- Pre-compute: masks, kernels, constants

---

**Remember:**
1. Keep it simple
2. Stay on GPU
3. Test before commit
4. One task = one commit
5. Clear names and messages
