# GitHub Copilot Instructions for vascx_simplify

## Project Overview
This is a PyTorch library for vessel and fundus image analysis with GPU-accelerated preprocessing and inference utilities. The project emphasizes performance, simplicity, and minimal dependencies.

## Core Principles

### 1. Best Practices Without Over-Engineering
- **Keep it simple**: Prefer straightforward implementations over complex abstractions
- **Single responsibility**: Each function/class should do one thing well
- **Type hints**: Always use type hints for function signatures
- **Docstrings**: Include docstrings for public APIs, but skip them for obvious private helpers
- **DRY principle**: Avoid code duplication, but don't create abstractions until you have 3+ use cases
- **Naming**: Use clear, descriptive names (e.g., `sliding_window_inference` not `swi`)
- **Avoid premature optimization**: Profile first, optimize second

### 2. Performance Optimization Guidelines

#### GPU Memory Management
```python
# ✅ GOOD: Keep tensors on GPU throughout pipeline
def process_image(image: torch.Tensor) -> torch.Tensor:
    # All operations on GPU
    preprocessed = enhance(image)  # GPU tensor
    features = extract_features(preprocessed)  # GPU tensor
    result = model(features)  # GPU tensor
    return result  # Return on GPU, user decides when to move to CPU

# ❌ BAD: Unnecessary CPU↔GPU transfers
def process_image(image: torch.Tensor) -> torch.Tensor:
    preprocessed = enhance(image).cpu()  # DON'T
    features = extract_features(preprocessed.cuda())  # DON'T
    result = model(features).cpu().numpy()  # DON'T
    return torch.from_numpy(result).cuda()  # DON'T
```

#### Mixed Precision
```python
# ✅ GOOD: Use float16 for compute-intensive ops, float32 for precision-critical
def preprocess(image: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
    # float16 for blur/warp (2-4x faster with Tensor Cores)
    img_fp16 = image.to(device, dtype=torch.float16)
    blurred = gaussian_blur(img_fp16)
    warped = warp_affine(blurred)
    
    # float32 for RANSAC/coordinate transforms (precision critical)
    coords = compute_coordinates(warped.float())
    return coords

# ❌ BAD: Using float32 everywhere wastes memory and speed
```

#### Batch Operations
```python
# ✅ GOOD: Process in batches when possible
predictions = model(torch.stack(images))  # Batch inference

# ❌ BAD: Loop over individual items
predictions = [model(img.unsqueeze(0)) for img in images]
```

#### In-Place Operations
```python
# ✅ GOOD: Use in-place ops to save memory
tensor.clamp_(0, 1)  # In-place
tensor.mul_(255)  # In-place

# ⚠️ CAREFUL: Only when gradient tracking isn't needed
```

### 3. Output Result Consistency
- **Same output guarantee**: Changes should NOT alter numerical results unless explicitly fixing a bug
- **Test before commit**: Always verify output matches original with test images
- **Document breaking changes**: If output must change, document why in commit message
- **Preserve dtypes**: Return tensors in the same dtype as input unless conversion is intended

```python
# ✅ GOOD: Preserve input characteristics
def process(image: torch.Tensor) -> torch.Tensor:
    device = image.device  # Remember device
    dtype = image.dtype  # Remember dtype
    
    # Process...
    result = compute(image)
    
    # Restore if changed
    return result.to(device=device, dtype=dtype)

# ❌ BAD: Changing output characteristics unexpectedly
def process(image: torch.Tensor) -> torch.Tensor:
    return compute(image).cpu().float()  # User didn't ask for this!
```

### 4. Avoid Unnecessary CPU↔GPU Transfers

#### Common Pitfalls
```python
# ❌ BAD: Excessive transfers
for batch in dataloader:
    data = batch.cuda()
    intermediate = model(data).cpu()  # DON'T
    result = post_process(intermediate.cuda())  # DON'T
    
# ✅ GOOD: Stay on GPU
for batch in dataloader:
    data = batch.cuda()
    intermediate = model(data)  # Stay on GPU
    result = post_process(intermediate)  # Stay on GPU
    # Only transfer final result if needed
    final = result.cpu()  # Once at the end
```

#### Detection Patterns
- `.cpu()` followed by `.cuda()` within same function
- `.numpy()` when torch operations would work
- Moving tensors to CPU inside hot loops
- Unnecessary `.item()` calls in batch processing

### 5. File Organization
```
src/vascx_simplify/
├── __init__.py          # Public API exports
├── inference.py         # Inference utilities (sliding window, etc.)
├── preprocess.py        # Preprocessing pipelines
└── utils.py             # Helper functions (HuggingFace, etc.)

examples/                # Usage examples
├── 01_artery_vein.py
├── 02_disc_segment.py
└── ...

tests/                   # Unit tests (if needed)
```

**File Guidelines:**
- No temporary/intermediate files committed to repo
- No cached models in git (use HuggingFace Hub)
- No generated files (build artifacts, __pycache__, etc.)
- Keep examples focused and minimal

### 6. Git Workflow

#### Planning and Branching Strategy
When planning new features or significant changes:

```bash
# 1. Create a new branch for the plan
git checkout -b plan/feature-name

# 2. Create a plan document in docs/
# Format: docs/PLANXX_<descriptive_name>.md
# Example: docs/PLAN03_MIXED_PRECISION.md

# 3. Document the plan with:
# - Goals and motivation
# - Technical approach
# - Implementation steps
# - Success criteria
# - Testing strategy

# 4. Commit the plan
git add docs/PLAN03_*.md
git commit -m "docs: add plan for mixed precision optimization"

# 5. Implement the plan in the same branch
# ... make changes ...
git commit -m "feat: implement mixed precision in preprocessing"

# 6. Run code quality checks (REQUIRED before merge)
# Lint and fix code quality issues
black src/ examples/  # Format code
isort src/ examples/  # Sort imports
# Fix any linting errors
git add .
git commit -m "chore: lint and format code"

# 7. Manual testing (REQUIRED)
# Run all examples and verify outputs
python examples/01_artery_vein.py
python examples/02_disc_segment.py
# ... test all relevant examples ...

# 8. Merge back to main when manual testing is complete
git checkout main
git merge plan/feature-name
```

**Plan Document Template:**
```markdown
# PLANXX: Feature Name

## Goals
- Clear objectives
- Expected outcomes

## Motivation
Why this change is needed

## Technical Approach
High-level design decisions

## Implementation Steps
1. Step 1
2. Step 2
...

## Success Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Testing Strategy
How to verify the implementation
```

#### Commit Strategy
```bash
# Commit after EACH logical task completion
git add <changed files>
git commit -m "feat: add gaussian importance map for sliding window"

# Check backward compatibility
git diff HEAD~1  # Review changes
git diff HEAD~1 -- src/  # Review only source changes
```

#### Commit Message Format
```
<type>: <description>

[optional body]

<type> = feat|fix|perf|refactor|docs|test|chore
```

**Examples:**
```bash
git commit -m "perf: optimize sliding window with mixed precision"
git commit -m "fix: preserve tensor device in preprocessing pipeline"
git commit -m "refactor: simplify RANSAC ellipse fitting"
git commit -m "docs: add performance optimization guide"
```

#### Verification Process
```bash
# Before commit: Test that output is unchanged
python examples/01_artery_vein.py  # Run test
python -m pytest tests/  # If tests exist

# After commit: Verify changes
git diff HEAD~1
git log --oneline -5  # Check commit history
```

#### Version Management
When ready to create a new release version:

```bash
# Use the appropriate command based on the type of changes
/commit_major_version    # For breaking changes (X.0.0)
/commit_minor_version    # For new features (0.X.0)
/commit_patch_version    # For bug fixes (0.0.X)
```

**Version Guidelines:**
- **Major version**: Breaking API changes, major refactors
- **Minor version**: New features, backward-compatible changes
- **Patch version**: Bug fixes, documentation updates, small improvements

## Code Style

### Formatting
- **Line length**: 100 characters (Black default)
- **Imports**: Group stdlib, third-party, local (use isort)
- **Quotes**: Prefer double quotes for strings
- **Indentation**: 4 spaces

### Type Hints
```python
from typing import Optional, Union, Tuple, List
import torch
from PIL import Image

# ✅ GOOD: Clear type hints
def predict(
    image: Union[torch.Tensor, Image.Image],
    device: str = 'cuda',
    batch_size: int = 1
) -> torch.Tensor:
    ...

# Tuple for fixed-size returns
def get_dims() -> Tuple[int, int]:
    return (height, width)

# List for variable-size returns
def get_predictions() -> List[torch.Tensor]:
    return [pred1, pred2, pred3]
```

### Error Handling
```python
# ✅ GOOD: Specific exceptions with helpful messages
if not image.ndim == 4:
    raise ValueError(
        f"Expected 4D tensor [B,C,H,W], got shape {image.shape}"
    )

# ❌ BAD: Generic exceptions
if not image.ndim == 4:
    raise Exception("Wrong shape")
```

## Performance Checklist

Before committing performance changes, verify:
- [ ] No new CPU↔GPU transfers added
- [ ] Batch operations used where possible
- [ ] Mixed precision used appropriately (fp16 for compute, fp32 for precision)
- [ ] No unnecessary `.clone()` or `.copy()` calls
- [ ] Memory usage is reasonable (profile if unsure)
- [ ] Output matches original implementation (numerical test)

## Testing Guidelines

```python
# Minimal test for verification
def test_output_consistency():
    """Verify output matches expected results"""
    model = EnsembleSegmentation(model_path, transform)
    image = torch.randn(1, 3, 512, 512)
    
    output1 = model.predict(image)
    output2 = model.predict(image)
    
    # Should be deterministic
    assert torch.allclose(output1, output2)
    
    # Should preserve device
    assert output1.device == image.device

def test_no_memory_leak():
    """Verify no memory accumulation"""
    import gc
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    # Run multiple times
    for _ in range(10):
        result = model.predict(image)
    
    gc.collect()
    torch.cuda.empty_cache()
    final_memory = torch.cuda.memory_allocated()
    
    # Memory should not grow significantly
    assert final_memory < initial_memory * 1.1
```

## Common Patterns

### Pattern: GPU-Accelerated Class
```python
class MyProcessor:
    """GPU-accelerated processor with pre-computed constants."""
    
    # Pre-compute constants at class level
    RESOLUTION = 256
    CENTER = RESOLUTION // 2
    
    # Pre-compute masks (numpy for initialization)
    _mask_np = np.arange(RESOLUTION) > RESOLUTION // 2
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # Convert to GPU tensors once
        self.mask = torch.from_numpy(self._mask_np).to(device)
    
    def process(self, image: torch.Tensor) -> torch.Tensor:
        """Process image on GPU."""
        # Ensure on correct device
        image = image.to(self.device)
        
        # Use pre-computed tensors
        masked = image * self.mask
        
        return masked
```

### Pattern: Flexible Device Handling
```python
def inference(
    image: torch.Tensor,
    device: Optional[str] = None
) -> torch.Tensor:
    """Inference with flexible device handling."""
    # Use input device if not specified
    if device is None:
        device = image.device
    
    # Move to device
    image = image.to(device)
    
    # Process
    result = model(image)
    
    # Return on same device
    return result
```

## Anti-Patterns to Avoid

```python
# ❌ Don't: Create unnecessary intermediate lists
results = []
for item in items:
    results.append(process(item))
return torch.stack(results)

# ✅ Do: Use tensor operations directly
return process(torch.stack(items))

# ❌ Don't: Move tensors back and forth
for i in range(len(data)):
    item = data[i].cpu().numpy()  # DON'T
    result = process(torch.from_numpy(item).cuda())  # DON'T

# ✅ Do: Keep on GPU
results = process(data)

# ❌ Don't: Over-engineer simple operations
class ImageNormalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def denormalize(self, x):
        return x * self.std + self.mean

# ✅ Do: Keep it simple
def normalize(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return (x - mean) / std
```

## Summary
1. **Keep it simple** - Don't over-engineer
2. **Optimize wisely** - Profile first, use mixed precision, batch operations
3. **Preserve outputs** - No breaking changes without explicit intent
4. **Stay on GPU** - Avoid CPU↔GPU transfers during processing
5. **Clean commits** - One task per commit, verify with `git diff`
6. **No junk files** - Keep repo clean and minimal
