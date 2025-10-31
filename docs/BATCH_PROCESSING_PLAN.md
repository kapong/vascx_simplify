# Batch Processing Implementation Plan

## Overview
Plan to add batch processing support to all model predict() methods while maintaining 100% backward compatibility with existing code.

## Current State

### Existing Behavior
All ensemble models currently process **single images** at a time:
```python
# Current usage (single image)
image = Image.open('fundus.jpg')
prediction = model.predict(image)  # Returns [1, H, W] or [1, C] or [1, K, 2]
result = prediction[0]  # All examples use [0] indexing
```

### Model Types and Output Shapes

| Model Class | Use Case | Input | Output Shape | Output Type |
|------------|----------|-------|--------------|-------------|
| `EnsembleSegmentation` | Artery/vein, disc | PIL/Tensor | `[B, H, W]` | Class indices |
| `ClassificationEnsemble` | Quality assessment | PIL/Tensor | `[B, 3]` | Softmax probs |
| `RegressionEnsemble` | Generic regression | PIL/Tensor | `[B, M, D]` | Raw values |
| `HeatmapRegressionEnsemble` | Fovea detection | PIL/Tensor | `[B, K, 2]` | (x,y) coords |

### Current Pipeline
1. **Input**: Single PIL.Image or torch.Tensor `[C, H, W]`
2. **Transform**: `VASCXTransform(img)` → `[C, H, W]`, `bounds_dict`
3. **Prepare**: `_prepare_input()` → `[1, C, H, W]` (adds batch dim), `bounds_dict`
4. **Inference**: `_run_inference()` → `[1, M, C, H, W]` or similar
5. **Post-process**: `proba_process()` → `[1, H, W]` or `[1, C]` or `[1, K, 2]`
6. **Output**: User accesses with `prediction[0]`

## Design Goals

### Primary Goals
1. ✅ **Backward Compatibility**: All existing examples work without modification
2. ✅ **Batch Efficiency**: Process multiple images in single forward pass
3. ✅ **Type Flexibility**: Accept single image, list of images, or batched tensor
4. ✅ **GPU Optimization**: Minimize CPU↔GPU transfers, batch GPU operations
5. ✅ **Consistent API**: Same interface across all ensemble types

### Performance Targets
- **Single image**: No performance degradation vs current implementation
- **Batch processing**: Near-linear speedup (e.g., 10 images in ~2-3x time vs 10x sequential)
- **Memory**: Respect GPU memory limits, provide batch size hints

## Proposed API

### Single Image (Backward Compatible)
```python
# PIL.Image input
image = Image.open('fundus.jpg')
pred = model.predict(image)  # [1, H, W] or [1, 3] or [1, K, 2]
result = pred[0]  # Works as before

# Tensor input [C, H, W]
image_tensor = torch.randn(3, 512, 512)
pred = model.predict(image_tensor)  # [1, H, W] or [1, 3] or [1, K, 2]
result = pred[0]  # Works as before

# Tensor input [1, C, H, W] (already batched)
image_tensor = torch.randn(1, 3, 512, 512)
pred = model.predict(image_tensor)  # [1, H, W] or [1, 3] or [1, K, 2]
result = pred[0]  # Works as before
```

### Batch Processing (NEW)
```python
# List of PIL.Image
images = [Image.open(f'fundus_{i}.jpg') for i in range(10)]
preds = model.predict(images)  # [10, H, W] or [10, 3] or [10, K, 2]
for i, pred in enumerate(preds):
    process(pred)  # Each pred is [H, W] or [3] or [K, 2]

# List of tensors
images = [torch.randn(3, 512, 512) for _ in range(10)]
preds = model.predict(images)  # [10, H, W] or [10, 3] or [10, K, 2]

# Batched tensor [B, C, H, W] where B > 1
images = torch.randn(10, 3, 512, 512)
preds = model.predict(images)  # [10, H, W] or [10, 3] or [10, K, 2]

# Large batches with automatic splitting (NEW)
images = [Image.open(f'fundus_{i}.jpg') for i in range(100)]
preds = model.predict(images, batch_size=8)  # Processes in chunks of 8
# Returns [100, H, W] - automatic batching handled internally
```

## Implementation Plan

### Phase 1: Input Detection and Handling

**Location**: `inference.py::EnsembleBase`

**Changes**:
```python
def _prepare_input(self, img: Union[torch.Tensor, Any, List]) -> Tuple[torch.Tensor, Union[Dict, List[Dict]], bool]:
    """Prepare input with batch detection.
    
    Returns:
        tuple: (batched_tensor, bounds, is_batch_input)
    """
    # Detect batch input
    is_batch = self._is_batch_input(img)
    
    if is_batch:
        # Handle list of images
        if isinstance(img, (list, tuple)):
            results = [self.transforms(im) for im in img]
            tensors, bounds_list = zip(*results)
            batched = torch.stack([t.to(self.device) for t in tensors])
            return batched, list(bounds_list), True
        # Handle batched tensor [B, C, H, W] where B > 1
        elif isinstance(img, torch.Tensor) and img.dim() == 4:
            # Apply transforms per image in batch
            results = [self.transforms(img[i]) for i in range(img.shape[0])]
            tensors, bounds_list = zip(*results)
            batched = torch.stack([t.to(self.device) for t in tensors])
            return batched, list(bounds_list), True
    else:
        # Single image - existing behavior
        img, bounds = self.transforms(img)
        return img.to(self.device).unsqueeze(dim=0), bounds, False

def _is_batch_input(self, img: Any) -> bool:
    """Detect if input is batch."""
    if isinstance(img, (list, tuple)):
        return len(img) > 0
    if isinstance(img, torch.Tensor):
        # Tensor [B, C, H, W] with B > 1 is batch
        # Tensor [C, H, W] or [1, C, H, W] is single
        return img.dim() == 4 and img.shape[0] > 1
    return False

def _predict_batch(
    self, 
    img: Union[torch.Tensor, Any, List], 
    batch_size: Optional[int] = None
) -> torch.Tensor:
    """Predict with automatic batch splitting.
    
    Args:
        img: Single image or batch of images
        batch_size: Maximum batch size for inference. If None, uses model default.
                   If input batch exceeds this, will split and process in chunks.
    
    Returns:
        Predictions for all images
    """
    img_tensor, bounds, is_batch = self._prepare_input(img)
    
    # Get effective batch size
    if batch_size is None:
        batch_size = getattr(self, 'predict_batch_size', None)
    
    current_batch_size = img_tensor.shape[0]
    
    # If batch size limit not set or batch fits, process all at once
    if batch_size is None or current_batch_size <= batch_size:
        proba = self._run_inference(img_tensor)
        return self.proba_process(proba, bounds, is_batch)
    
    # Split into chunks and process
    all_preds = []
    for i in range(0, current_batch_size, batch_size):
        end_idx = min(i + batch_size, current_batch_size)
        chunk_img = img_tensor[i:end_idx]
        
        # Get corresponding bounds for this chunk
        if isinstance(bounds, list):
            chunk_bounds = bounds[i:end_idx]
        else:
            chunk_bounds = bounds
        
        # Process chunk
        chunk_proba = self._run_inference(chunk_img)
        chunk_pred = self.proba_process(
            chunk_proba, 
            chunk_bounds, 
            is_batch=True  # Always True for chunks
        )
        all_preds.append(chunk_pred)
    
    # Concatenate results
    return torch.cat(all_preds, dim=0)
```

**Files to modify**:
- `src/vascx_simplify/inference.py` - Update `_prepare_input()`, add `_is_batch_input()` and `_predict_batch()`

**Tests**:
- Single PIL.Image → `is_batch=False`, output `[1, ...]`
- Single tensor `[C, H, W]` → `is_batch=False`, output `[1, ...]`
- Single tensor `[1, C, H, W]` → `is_batch=False`, output `[1, ...]`
- List of 5 images → `is_batch=True`, output `[5, ...]`
- Tensor `[5, C, H, W]` → `is_batch=True`, output `[5, ...]`

### Phase 2: Transform Batch Support

**Location**: `preprocess.py::VASCXTransform`

**Current behavior**:
```python
def __call__(self, img) -> Tuple[torch.Tensor, Dict]:
    # Single image processing
    return tensor, bounds_dict
```

**No changes needed!** Phase 1 handles batch processing at the ensemble level by calling transforms per image.

**Alternative (optimization for future)**:
Could add native batch support to VASCXTransform for better performance:
```python
def __call__(self, img) -> Union[Tuple[torch.Tensor, Dict], Tuple[torch.Tensor, List[Dict]]]:
    if isinstance(img, (list, tuple)):
        # Batch processing
        results = [self._process_single(im) for im in img]
        tensors, bounds = zip(*results)
        return torch.stack(tensors), list(bounds)
    else:
        return self._process_single(img)
```

**Decision**: Start with Phase 1 approach (loop in ensemble), optimize later if needed.

### Phase 3: Post-Processing for Batches

**Location**: `inference.py::EnsembleBase::proba_process()` and subclass implementations

#### A. Base Class Update
```python
class EnsembleBase:
    def __init__(self, fpath, transforms, device=DEFAULT_DEVICE):
        # ... existing init code ...
        
        # Default batch size for predict() - can be overridden per model type
        self.predict_batch_size: Optional[int] = None
    
    def predict(
        self, 
        img: Union[torch.Tensor, Any, List],
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Predict with automatic batch splitting.
        
        Args:
            img: Single image (PIL.Image or Tensor [C,H,W]) or 
                 batch (List of images or Tensor [B,C,H,W])
            batch_size: Maximum batch size for inference. If None, uses self.predict_batch_size.
                       Large batches will be automatically split into chunks.
        
        Returns:
            Predictions tensor. Shape depends on model type:
            - Segmentation: [B, H, W]
            - Classification: [B, C]
            - Regression: [B, M, D]
            - Heatmap: [B, K, 2]
        """
        return self._predict_batch(img, batch_size)
    
    def proba_process(self, proba: torch.Tensor, bounds: Union[Dict, List[Dict]], is_batch: bool) -> torch.Tensor:
        """Override in subclasses. Handle both single and batch."""
        return proba
```

#### B. EnsembleSegmentation
```python
def __init__(self, fpath, transforms, device=DEFAULT_DEVICE):
    super().__init__(fpath, transforms, device)
    self.sw_batch_size = 16  # Sliding window batch size
    self.predict_batch_size = 4  # Default: process 4 images at once

def proba_process(self, proba: torch.Tensor, bounds: Union[Dict, List[Dict]], is_batch: bool) -> torch.Tensor:
    proba = torch.mean(proba, dim=1)  # Average over models (M)
    proba = torch.nn.functional.softmax(proba, dim=1)
    
    # Handle bounds undo
    if is_batch:
        results = [self.transforms.undo_bounds(proba[i:i+1], bounds[i]) for i in range(len(bounds))]
        proba = torch.cat(results, dim=0)
    else:
        proba = self.transforms.undo_bounds(proba, bounds)
    
    proba = torch.argmax(proba, dim=1)
    return proba
```

#### C. ClassificationEnsemble
```python
def __init__(self, fpath, transforms, device=DEFAULT_DEVICE):
    super().__init__(fpath, transforms, device)
    self.inference_fn = None
    self.predict_batch_size = 16  # Default: process 16 images at once (no sliding window)

def predict(
    self, 
    img: Union[torch.Tensor, Any, List],
    batch_size: Optional[int] = None
) -> torch.Tensor:
    """Predict with automatic batch splitting for classification."""
    return self._predict_batch(img, batch_size)

def proba_process(self, proba: torch.Tensor, bounds: Union[Dict, List[Dict]], is_batch: bool) -> torch.Tensor:
    proba = torch.nn.functional.softmax(proba, dim=-1)
    proba = torch.mean(proba, dim=0 if not is_batch else 1)  # Average over models
    return proba
```

**Note**: Classification/Regression ensembles have simpler inference (no sliding window), so can use `_predict_batch()` directly.

#### D. HeatmapRegressionEnsemble
```python
def __init__(self, fpath, transforms, device=DEFAULT_DEVICE):
    super().__init__(fpath, transforms, device)
    self.sw_batch_size = 1  # Conservative for memory
    self.predict_batch_size = 2  # Process 2 images at once (heatmaps are memory-intensive)

def proba_process(self, heatmaps: torch.Tensor, bounds: Union[Dict, List[Dict]], is_batch: bool) -> torch.Tensor:
    # Existing vectorized processing handles batches
    # ... (extract coordinates from heatmaps)
    
    # Undo bounds
    if is_batch:
        results = [self.transforms.undo_bounds_points(outputs[i:i+1], bounds[i]) for i in range(len(bounds))]
        outputs = torch.cat(results, dim=0)
    else:
        outputs = self.transforms.undo_bounds_points(outputs, bounds)
    
    return outputs
```

**Files to modify**:
- `src/vascx_simplify/inference.py` - Update all `predict()` and `proba_process()` methods

### Phase 4: Bounds Undo for Batches

**Location**: `preprocess.py::VASCXTransform`

**Current methods**:
- `undo_bounds(proba: Tensor, bounds: Dict) -> Tensor`
- `undo_bounds_points(points: Tensor, bounds: Dict) -> Tensor`

**Option 1: Loop in ensemble classes** (recommended for simplicity)
- Keep VASCXTransform unchanged
- Loop in `proba_process()` as shown in Phase 3

**Option 2: Add batch support to VASCXTransform**
```python
def undo_bounds(self, proba: torch.Tensor, bounds: Union[Dict, List[Dict]]) -> torch.Tensor:
    if isinstance(bounds, list):
        # Batch processing
        results = [self._undo_bounds_single(proba[i:i+1], bounds[i]) for i in range(len(bounds))]
        return torch.cat(results, dim=0)
    else:
        return self._undo_bounds_single(proba, bounds)

def _undo_bounds_single(self, proba: torch.Tensor, bounds: Dict) -> torch.Tensor:
    # Current undo_bounds implementation
    ...
```

**Decision**: Start with Option 1 (loop in ensemble), refactor to Option 2 if performance issue.

### Phase 5: Testing and Validation

#### Backward Compatibility Tests
```python
def test_single_image_backward_compat():
    """Verify existing code works unchanged."""
    model = EnsembleSegmentation(model_path, VASCXTransform())
    
    # PIL.Image
    image = Image.open('test.jpg')
    pred = model.predict(image)
    assert pred.shape[0] == 1  # Batch dim still present
    result = pred[0]  # This should work
    
    # Tensor [C, H, W]
    image = torch.randn(3, 512, 512)
    pred = model.predict(image)
    assert pred.shape[0] == 1
    
    # Tensor [1, C, H, W]
    image = torch.randn(1, 3, 512, 512)
    pred = model.predict(image)
    assert pred.shape[0] == 1

def test_numerical_consistency():
    """Single image processing should give same results."""
    model = EnsembleSegmentation(model_path, VASCXTransform())
    image = Image.open('test.jpg')
    
    # Process as single
    pred_single = model.predict(image)
    
    # Process as batch of 1
    pred_batch = model.predict([image])
    
    # Should be identical
    assert torch.allclose(pred_single, pred_batch)
```

#### Batch Processing Tests
```python
def test_batch_list_images():
    """Test batch processing with list of images."""
    model = EnsembleSegmentation(model_path, VASCXTransform())
    images = [Image.open(f'test_{i}.jpg') for i in range(5)]
    
    preds = model.predict(images)
    assert preds.shape[0] == 5
    
    # Compare with sequential processing
    preds_seq = torch.stack([model.predict(img) for img in images])
    assert preds_seq.shape == preds.shape
    # Note: May not be numerically identical due to batch norm, etc.

def test_batch_tensor():
    """Test batch processing with batched tensor."""
    model = EnsembleSegmentation(model_path, VASCXTransform())
    images = torch.randn(5, 3, 512, 512)
    
    preds = model.predict(images)
    assert preds.shape[0] == 5

def test_batch_all_models():
    """Test batch processing for all model types."""
    # Segmentation
    seg_model = EnsembleSegmentation(seg_path, VASCXTransform())
    seg_preds = seg_model.predict(images)
    assert seg_preds.shape == (5, H, W)
    
    # Classification
    cls_model = ClassificationEnsemble(cls_path, VASCXTransform(have_ce=False))
    cls_preds = cls_model.predict(images)
    assert cls_preds.shape == (5, 3)
    
    # Heatmap Regression
    reg_model = HeatmapRegressionEnsemble(reg_path, VASCXTransform())
    reg_preds = reg_model.predict(images)
    assert reg_preds.shape == (5, K, 2)

def test_automatic_batch_splitting():
    """Test that large batches are automatically split."""
    model = EnsembleSegmentation(seg_path, VASCXTransform())
    
    # Create batch larger than default batch_size (4)
    images = [Image.open(f'test_{i}.jpg') for i in range(10)]
    
    # Should auto-split into 3 batches: 4 + 4 + 2
    preds = model.predict(images)
    assert preds.shape[0] == 10
    
    # Custom batch size
    preds_custom = model.predict(images, batch_size=3)
    assert preds_custom.shape[0] == 10
    
    # Results should be close (may differ slightly due to batch norm)
    # At minimum, shapes should match
    assert preds.shape == preds_custom.shape

def test_batch_size_override():
    """Test batch_size parameter override."""
    model = EnsembleSegmentation(seg_path, VASCXTransform())
    images = [Image.open(f'test_{i}.jpg') for i in range(8)]
    
    # Use different batch sizes
    preds_2 = model.predict(images, batch_size=2)  # 4 chunks
    preds_4 = model.predict(images, batch_size=4)  # 2 chunks
    preds_8 = model.predict(images, batch_size=8)  # 1 chunk
    
    # All should return same shape
    assert preds_2.shape == preds_4.shape == preds_8.shape == (8, H, W)
```

### Phase 6: Documentation and Examples

#### Update README.md
Add batch processing section:
```markdown
### Batch Processing

Process multiple images efficiently with automatic batch splitting:

```python
from vascx_simplify import EnsembleSegmentation, VASCXTransform, from_huggingface
from PIL import Image

# Load model
model_path = from_huggingface('Eyened/vascx:artery_vein/av_july24.pt')
model = EnsembleSegmentation(model_path, VASCXTransform())

# Load multiple images
images = [Image.open(f'fundus_{i}.jpg') for i in range(10)]

# Batch prediction (much faster than loop)
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

**Performance**: Batch processing is 2-3x faster than processing images sequentially.

**Memory Management**: Each model type has sensible default batch sizes:
- Segmentation: 4 images (sliding window is memory-intensive)
- Classification: 16 images (lightweight forward pass)
- Regression: 16 images
- Heatmap: 2 images (heatmaps are very memory-intensive)
```

#### Create Batch Example
`examples/05_batch_processing.py`:
```python
"""
Batch Processing Example

Demonstrates efficient batch inference for multiple fundus images with automatic batch splitting.
"""

from vascx_simplify import EnsembleSegmentation, VASCXTransform, from_huggingface
from PIL import Image
import time
import torch

def main():
    # Load model once
    model_path = from_huggingface('Eyened/vascx:artery_vein/av_july24.pt')
    model = EnsembleSegmentation(model_path, VASCXTransform())
    
    print(f"Model default batch size: {model.predict_batch_size}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load multiple images
    image_paths = [f'HRF_{i:02d}_dr.jpg' for i in range(1, 11)]
    images = [Image.open(path) for path in image_paths]
    
    # Sequential processing (baseline)
    print("\n1. Sequential Processing:")
    start = time.time()
    preds_sequential = [model.predict(img)[0] for img in images]
    seq_time = time.time() - start
    print(f"   Time: {seq_time:.2f}s ({seq_time/len(images):.3f}s per image)")
    
    # Batch processing with default batch size
    print("\n2. Batch Processing (default batch_size=4):")
    start = time.time()
    preds_batch = model.predict(images)  # Auto-splits into chunks of 4
    batch_time = time.time() - start
    print(f"   Time: {batch_time:.2f}s ({batch_time/len(images):.3f}s per image)")
    print(f"   Speedup: {seq_time/batch_time:.2f}x")
    print(f"   Output shape: {preds_batch.shape}")
    
    # Batch processing with custom batch size
    print("\n3. Batch Processing (custom batch_size=2):")
    start = time.time()
    preds_batch_custom = model.predict(images, batch_size=2)
    custom_time = time.time() - start
    print(f"   Time: {custom_time:.2f}s ({custom_time/len(images):.3f}s per image)")
    print(f"   Speedup vs sequential: {seq_time/custom_time:.2f}x")
    
    # Large batch demonstration
    print("\n4. Large Batch (100 images with auto-splitting):")
    large_images = images * 10  # Repeat to get 100 images
    start = time.time()
    preds_large = model.predict(large_images)  # Auto-splits into chunks
    large_time = time.time() - start
    print(f"   Time: {large_time:.2f}s ({large_time/len(large_images):.3f}s per image)")
    print(f"   Output shape: {preds_large.shape}")
    print(f"   Memory efficient: Automatically split into {len(large_images)//model.predict_batch_size} batches")
    
    # Verify consistency
    print("\n5. Verification:")
    print(f"   Sequential[0] == Batch[0]: {torch.allclose(preds_sequential[0], preds_batch[0])}")
    print(f"   Batch == Custom batch: {torch.allclose(preds_batch, preds_batch_custom)}")

if __name__ == "__main__":
    main()
```

## Performance Considerations

### Memory Management
- **GPU Memory**: Batch size limited by GPU memory
- **Default batch sizes per model type**:
  - `EnsembleSegmentation`: 4 (sliding window is memory-intensive)
  - `ClassificationEnsemble`: 16 (simple forward pass, less memory)
  - `RegressionEnsemble`: 16 (similar to classification)
  - `HeatmapRegressionEnsemble`: 2 (heatmaps are very memory-intensive)
- **Auto-splitting**: Large batches automatically split into chunks
- **User override**: Can specify `batch_size` parameter in `predict()` call

### CPU↔GPU Transfers
- All transforms should operate on GPU where possible
- Minimize `.cpu()` calls until final output
- Batch all GPU operations together

### Preprocessing Optimization
- `FundusContrastEnhance` processes per-image (required for ellipse fitting)
- Consider GPU batching for blur/warp operations (Kornia supports it)
- Cache grid coordinates per batch size

### Inference Optimization
- Sliding window already batches windows efficiently
- Ensure `sw_batch_size` is tuned for GPU
- TTA can be parallelized across batch

## Migration Guide

### For Library Users
**No changes required!** All existing code works as-is:
```python
# This still works
image = Image.open('fundus.jpg')
pred = model.predict(image)
result = pred[0]
```

**To use batching** (optional):
```python
# Process multiple images
images = [Image.open(f'img_{i}.jpg') for i in range(10)]
preds = model.predict(images)

# Or with tensors
images_tensor = torch.stack([...])
preds = model.predict(images_tensor)
```

### For Contributors
When modifying ensemble classes:
1. Always test with single image first (backward compat)
2. Test with batch input (new functionality)
3. Verify numerical consistency between single and batch
4. Check GPU memory usage with large batches
5. Profile performance vs sequential processing

## Implementation Checklist

### Phase 1: Input Handling ✅
- [ ] Add `_is_batch_input()` method to EnsembleBase
- [ ] Update `_prepare_input()` to return `(tensor, bounds, is_batch)`
- [ ] Add `_predict_batch()` method with automatic splitting
- [ ] Add `predict_batch_size` attribute to all ensemble classes
- [ ] Handle List[PIL.Image] input
- [ ] Handle List[torch.Tensor] input
- [ ] Handle torch.Tensor [B, C, H, W] with B > 1
- [ ] Test single image backward compatibility
- [ ] Test batch detection logic
- [ ] Test automatic batch splitting (e.g., 10 images with batch_size=4)

### Phase 2: Transform Support ✅
- [ ] Verify VASCXTransform works per-image (no changes needed)
- [ ] Test transform pipeline with various inputs

### Phase 3: Post-Processing ✅
- [ ] Update `EnsembleBase.predict()` signature to accept `batch_size` parameter
- [ ] Set `predict_batch_size` defaults in each `__init__()` (seg=4, cls=16, reg=16, hm=2)
- [ ] Update `EnsembleSegmentation.proba_process()`
- [ ] Update `ClassificationEnsemble.predict()` and `proba_process()`
- [ ] Update `RegressionEnsemble.predict()` and `proba_process()`
- [ ] Update `HeatmapRegressionEnsemble.proba_process()`

### Phase 4: Bounds Handling ✅
- [ ] Implement batch undo_bounds (loop approach)
- [ ] Implement batch undo_bounds_points (loop approach)
- [ ] Test bounds restoration for batches

### Phase 5: Testing ✅
- [ ] Test backward compatibility (all 4 examples work unchanged)
- [ ] Test numerical consistency (single == batch[0])
- [ ] Test batch list images
- [ ] Test batch tensor
- [ ] Test all model types in batch mode
- [ ] Profile performance (verify speedup)
- [ ] Test GPU memory usage

### Phase 6: Documentation ✅
- [ ] Update README.md with batch processing section
- [ ] Create `examples/05_batch_processing.py`
- [ ] Add performance benchmarks
- [ ] Update docstrings in inference.py
- [ ] Update QUICK_REFERENCE.md

## Success Criteria

1. ✅ All existing examples run without modification
2. ✅ Single image processing produces identical outputs
3. ✅ Batch processing works for all model types
4. ✅ Batch processing is 2-3x faster than sequential for 10 images
5. ✅ No breaking changes to API
6. ✅ Documentation includes batch processing examples
7. ✅ Tests cover single and batch scenarios

## Timeline Estimate

- Phase 1 (Input Handling): 2-3 hours
- Phase 2 (Transform): 1 hour
- Phase 3 (Post-Processing): 3-4 hours
- Phase 4 (Bounds): 1-2 hours
- Phase 5 (Testing): 3-4 hours
- Phase 6 (Documentation): 2-3 hours

**Total**: 12-17 hours of development time

## Future Enhancements

1. ~~**Auto-batching**: Automatically split large batches based on GPU memory~~ ✅ IMPLEMENTED
2. **Dynamic batch sizing**: Automatically detect optimal batch size based on GPU memory
3. **Async processing**: Process batches asynchronously
4. **Native batch transforms**: Optimize VASCXTransform for batch processing
5. **Batch TTA**: Parallelize test-time augmentation across batch
6. **Mixed batch sizes**: Handle variable-size images in same batch (requires padding)
7. **Progress callback**: Add optional callback for tracking batch processing progress

## Notes

- Maintain GPU tensor throughout pipeline (no unnecessary CPU transfers)
- Use float16 where appropriate (compute-intensive ops)
- Document memory requirements per model type
- Consider adding `batch_size` parameter to predict() for auto-splitting large batches
