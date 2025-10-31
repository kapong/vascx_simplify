# Performance Optimization Guide for vascx_simplify

## Table of Contents
1. [GPU Memory Management](#gpu-memory-management)
2. [Mixed Precision Training](#mixed-precision-training)
3. [Batch Processing](#batch-processing)
4. [CPU↔GPU Transfer Optimization](#cpugpu-transfer-optimization)
5. [Memory Profiling](#memory-profiling)
6. [Performance Benchmarking](#performance-benchmarking)

## GPU Memory Management

### Rule: Stay on GPU During Processing

**The Golden Rule:** Once data is on GPU, keep it there until the final output is needed on CPU.

```python
# ✅ OPTIMAL: Single pipeline on GPU
def optimal_pipeline(image_path: str) -> np.ndarray:
    # Load and move to GPU once
    image = load_image(image_path)
    image = torch.from_numpy(image).cuda()
    
    # All processing on GPU
    preprocessed = preprocess(image)      # GPU → GPU
    features = extract_features(preprocessed)  # GPU → GPU
    predictions = model(features)         # GPU → GPU
    postprocessed = postprocess(predictions)   # GPU → GPU
    
    # Move to CPU only at the end
    return postprocessed.cpu().numpy()

# ❌ SLOW: Multiple CPU↔GPU transfers (10-100x slower!)
def slow_pipeline(image_path: str) -> np.ndarray:
    image = load_image(image_path)
    
    # Transfer 1: CPU → GPU
    image = torch.from_numpy(image).cuda()
    preprocessed = preprocess(image)
    
    # Transfer 2: GPU → CPU (unnecessary!)
    preprocessed = preprocessed.cpu().numpy()
    
    # Transfer 3: CPU → GPU (unnecessary!)
    features = extract_features(torch.from_numpy(preprocessed).cuda())
    
    # Transfer 4: GPU → CPU (unnecessary!)
    predictions = model(features).cpu()
    
    # Transfer 5: CPU → GPU (unnecessary!)
    postprocessed = postprocess(predictions.cuda())
    
    # Transfer 6: GPU → CPU
    return postprocessed.cpu().numpy()
```

**Impact:** The slow version can be **10-100x slower** depending on transfer size!

### Pre-compute and Cache GPU Tensors

```python
class EfficientProcessor:
    """Pre-compute constants on GPU for fast processing."""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # ✅ Pre-compute masks and weights on GPU once
        self.gaussian_kernel = self._create_gaussian_kernel().to(device)
        self.importance_map = self._create_importance_map().to(device)
        
    def _create_gaussian_kernel(self) -> torch.Tensor:
        # Compute once in numpy
        kernel = np.outer(signal.windows.gaussian(5, 1),
                         signal.windows.gaussian(5, 1))
        return torch.from_numpy(kernel).float()
    
    def process(self, image: torch.Tensor) -> torch.Tensor:
        # Use pre-computed GPU tensors (fast!)
        return F.conv2d(image, self.gaussian_kernel)

# ❌ SLOW: Re-computing every time
class SlowProcessor:
    def process(self, image: torch.Tensor) -> torch.Tensor:
        # Creating kernel every call is wasteful!
        kernel = self._create_gaussian_kernel().to(image.device)
        return F.conv2d(image, kernel)
```

### Memory-Efficient Sliding Window

```python
def efficient_sliding_window(
    inputs: torch.Tensor,
    roi_size: tuple,
    predictor,
    sw_batch_size: int = 4
) -> torch.Tensor:
    """
    Process sliding windows in batches to maximize GPU utilization.
    """
    # Pre-allocate output tensors (avoid dynamic allocation)
    output = torch.zeros(
        (inputs.shape[0], num_classes, *inputs.shape[2:]),
        device=inputs.device,
        dtype=inputs.dtype
    )
    importance = torch.zeros_like(output[:, 0])
    
    # Create importance map once
    window_importance = _create_gaussian_importance_map(
        *roi_size, inputs.device, inputs.dtype
    )
    
    # Process windows in batches
    window_batches = [windows[i:i+sw_batch_size] 
                      for i in range(0, len(windows), sw_batch_size)]
    
    for batch in window_batches:
        # Extract batch of windows
        batch_data = torch.stack([
            inputs[:, :, h:h+roi_size[0], w:w+roi_size[1]]
            for h, w in batch
        ])
        
        # Single batched prediction (fast!)
        with torch.no_grad():
            batch_pred = predictor(batch_data)
        
        # Accumulate results (in-place operations)
        for pred, (h, w) in zip(batch_pred, batch):
            output[:, :, h:h+roi_size[0], w:w+roi_size[1]] += pred * window_importance
            importance[:, h:h+roi_size[0], w:w+roi_size[1]] += window_importance
    
    # Normalize (in-place)
    output /= importance.unsqueeze(1)
    
    return output
```

## Mixed Precision Training

### When to Use Float16 vs Float32

```python
# Compute Intensity vs Precision Requirements
#
# ┌─────────────────────────────────────┬──────────┬───────────┐
# │ Operation Type                      │ Use FP16 │ Use FP32  │
# ├─────────────────────────────────────┼──────────┼───────────┤
# │ Convolutions                        │    ✓     │           │
# │ Matrix multiplications              │    ✓     │           │
# │ Gaussian blur / filtering           │    ✓     │           │
# │ Affine transformations              │    ✓     │           │
# │ Batch normalization                 │    ✓     │           │
# │ RANSAC / optimization               │          │     ✓     │
# │ Coordinate transformations          │          │     ✓     │
# │ Small value accumulation            │          │     ✓     │
# │ Trigonometric functions             │          │     ✓     │
# └─────────────────────────────────────┴──────────┴───────────┘

class MixedPrecisionPreprocessor:
    """Example of optimal mixed precision usage."""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Pre-compute in appropriate precision
        self.gaussian_kernel_fp16 = self._create_kernel().half().to(device)
        self.coord_transform_fp32 = self._create_transform().float().to(device)
    
    def process(self, image: torch.Tensor) -> torch.Tensor:
        # ✅ FP16 for compute-intensive blur (2-4x faster on modern GPUs)
        image_fp16 = image.half()
        blurred = F.conv2d(image_fp16, self.gaussian_kernel_fp16)
        
        # ✅ FP16 for affine warp (2-4x faster with Tensor Cores)
        warped = K.warp_affine(
            blurred,
            self.affine_matrix.half(),
            (256, 256)
        )
        
        # ✅ FP32 for precision-critical coordinate computation
        warped_fp32 = warped.float()
        coords = self._compute_precise_coordinates(warped_fp32)
        
        # ✅ FP32 for RANSAC (needs precision)
        ellipse_params = self._fit_ellipse_ransac(coords)
        
        return ellipse_params
```

### Performance Gains

**Typical speedups with FP16:**
- **Convolutions**: 2-3x faster
- **Matrix multiplications**: 2-4x faster
- **Gaussian blur**: 2-3x faster
- **Affine transforms**: 2-3x faster
- **Memory usage**: 50% reduction

**On Modern GPUs (Tensor Cores):**
- RTX 3090: 3-4x faster
- RTX 4090: 4-5x faster
- A100: 4-6x faster

## Batch Processing

### Maximize GPU Utilization

```python
# ✅ GOOD: Batch processing
def batch_inference(images: List[torch.Tensor], batch_size: int = 8) -> List[torch.Tensor]:
    """Process images in batches for maximum GPU efficiency."""
    results = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        
        # Pad batch if necessary
        if len(batch) < batch_size:
            batch = batch + [torch.zeros_like(batch[0])] * (batch_size - len(batch))
        
        # Single batched inference (much faster!)
        batch_tensor = torch.stack(batch)
        batch_results = model(batch_tensor)
        
        results.extend(batch_results)
    
    return results[:len(images)]  # Remove padding

# ❌ BAD: Sequential processing (underutilizes GPU)
def sequential_inference(images: List[torch.Tensor]) -> List[torch.Tensor]:
    return [model(img.unsqueeze(0))[0] for img in images]
```

**Performance Impact:**
- Sequential: 100 images × 10ms = 1000ms
- Batched (bs=8): 13 batches × 12ms = 156ms (6.4x faster!)

### Dynamic Batching

```python
def dynamic_batch_inference(
    images: List[torch.Tensor],
    max_batch_size: int = 8,
    max_memory_mb: float = 4096
) -> List[torch.Tensor]:
    """Automatically adjust batch size based on memory constraints."""
    
    # Estimate memory per image
    single_image_mb = images[0].element_size() * images[0].nelement() / (1024**2)
    
    # Calculate safe batch size
    safe_batch_size = min(
        max_batch_size,
        int(max_memory_mb / (single_image_mb * 4))  # 4x overhead for safety
    )
    
    print(f"Using batch size: {safe_batch_size}")
    
    return batch_inference(images, safe_batch_size)
```

## CPU↔GPU Transfer Optimization

### Detecting Transfer Bottlenecks

```python
import time

def profile_transfers(func):
    """Decorator to detect CPU↔GPU transfers."""
    def wrapper(*args, **kwargs):
        # Count transfers
        original_cpu = torch.Tensor.cpu
        original_cuda = torch.Tensor.cuda
        
        transfer_count = {'cpu': 0, 'cuda': 0}
        
        def tracked_cpu(self, *args, **kwargs):
            transfer_count['cpu'] += 1
            return original_cpu(self, *args, **kwargs)
        
        def tracked_cuda(self, *args, **kwargs):
            transfer_count['cuda'] += 1
            return original_cuda(self, *args, **kwargs)
        
        torch.Tensor.cpu = tracked_cpu
        torch.Tensor.cuda = tracked_cuda
        
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        torch.Tensor.cpu = original_cpu
        torch.Tensor.cuda = original_cuda
        
        print(f"{func.__name__}:")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  CPU transfers: {transfer_count['cpu']}")
        print(f"  CUDA transfers: {transfer_count['cuda']}")
        
        return result
    
    return wrapper

# Usage
@profile_transfers
def my_pipeline(image):
    return model(preprocess(image))
```

### Minimize Transfer Overhead

```python
# ✅ GOOD: Pin memory for faster transfers
dataloader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,  # 2-3x faster CPU → GPU transfers
    num_workers=4
)

# ✅ GOOD: Async transfers (overlap compute and transfer)
def async_inference(images: List[torch.Tensor]) -> List[torch.Tensor]:
    stream = torch.cuda.Stream()
    
    results = []
    for i, image in enumerate(images):
        with torch.cuda.stream(stream):
            # Transfer next image while processing current
            if i + 1 < len(images):
                next_image = images[i + 1].cuda(non_blocking=True)
            
            # Process current image
            result = model(image)
            results.append(result)
    
    return results

# ❌ BAD: Synchronous transfers (slow)
def sync_inference(images: List[torch.Tensor]) -> List[torch.Tensor]:
    return [model(img.cuda()) for img in images]  # Each waits for transfer
```

## Memory Profiling

### Track Memory Usage

```python
import torch

def profile_memory(func):
    """Profile GPU memory usage."""
    def wrapper(*args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        start_mem = torch.cuda.memory_allocated() / (1024**2)
        
        result = func(*args, **kwargs)
        
        end_mem = torch.cuda.memory_allocated() / (1024**2)
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
        
        print(f"{func.__name__} Memory Profile:")
        print(f"  Start: {start_mem:.1f} MB")
        print(f"  End: {end_mem:.1f} MB")
        print(f"  Peak: {peak_mem:.1f} MB")
        print(f"  Delta: {end_mem - start_mem:.1f} MB")
        
        return result
    
    return wrapper

# Usage
@profile_memory
def my_inference(image):
    return model(image)
```

### Memory Optimization Techniques

```python
# ✅ Use in-place operations
def optimize_inplace(tensor: torch.Tensor) -> torch.Tensor:
    tensor.clamp_(0, 1)  # In-place
    tensor.mul_(255)     # In-place
    return tensor

# ✅ Use checkpointing for large models
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(model, x):
    # Trade compute for memory
    return checkpoint(model, x)

# ✅ Clear cache between inferences
def batch_inference_with_cleanup(images):
    results = []
    for image in images:
        result = model(image)
        results.append(result.cpu())  # Move to CPU
        
        # Free GPU memory
        del result
        torch.cuda.empty_cache()
    
    return results

# ✅ Use gradient accumulation for large batches
def train_with_accumulation(model, dataloader, accumulation_steps=4):
    for i, (data, target) in enumerate(dataloader):
        output = model(data)
        loss = criterion(output, target)
        loss = loss / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

## Performance Benchmarking

### Benchmark Template

```python
import time
import torch
from contextlib import contextmanager

@contextmanager
def benchmark(name: str, n_runs: int = 100):
    """Context manager for benchmarking."""
    torch.cuda.synchronize()
    start = time.time()
    
    yield
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"{name}:")
    print(f"  Total: {elapsed:.3f}s")
    print(f"  Per run: {elapsed/n_runs*1000:.2f}ms")
    print(f"  Throughput: {n_runs/elapsed:.1f} runs/s")

# Usage
image = torch.randn(1, 3, 512, 512).cuda()

with benchmark("FP32 Inference", n_runs=100):
    for _ in range(100):
        result = model(image)

with benchmark("FP16 Inference", n_runs=100):
    for _ in range(100):
        result = model(image.half())
```

### Comprehensive Performance Test

```python
def performance_test():
    """Comprehensive performance test suite."""
    
    print("=== Performance Test ===\n")
    
    # Setup
    device = 'cuda'
    image = torch.randn(1, 3, 512, 512).to(device)
    model = load_model().to(device).eval()
    
    # Test 1: FP32 vs FP16
    print("1. Precision Comparison")
    with torch.no_grad():
        with benchmark("FP32", n_runs=100):
            for _ in range(100):
                _ = model(image)
        
        with benchmark("FP16", n_runs=100):
            for _ in range(100):
                _ = model(image.half())
    
    # Test 2: Batch Size Scaling
    print("\n2. Batch Size Scaling")
    for bs in [1, 2, 4, 8, 16]:
        batch = torch.randn(bs, 3, 512, 512).to(device)
        with torch.no_grad():
            with benchmark(f"Batch Size {bs}", n_runs=50):
                for _ in range(50):
                    _ = model(batch)
    
    # Test 3: Memory Usage
    print("\n3. Memory Usage")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(image)
    print(f"  Allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    print(f"  Peak: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
    
    # Test 4: CPU↔GPU Transfer Cost
    print("\n4. Transfer Cost")
    cpu_image = image.cpu()
    
    with benchmark("CPU → GPU Transfer", n_runs=100):
        for _ in range(100):
            _ = cpu_image.cuda()
    
    with benchmark("GPU → CPU Transfer", n_runs=100):
        for _ in range(100):
            _ = image.cpu()

if __name__ == "__main__":
    performance_test()
```

## Quick Reference: Performance Optimization Checklist

- [ ] **No unnecessary CPU↔GPU transfers**
  - Check for `.cpu()` followed by `.cuda()` in same function
  - Avoid `.numpy()` when torch operations work
  
- [ ] **Use mixed precision appropriately**
  - FP16 for compute-intensive ops (conv, matmul, blur)
  - FP32 for precision-critical ops (RANSAC, coordinates)
  
- [ ] **Batch operations when possible**
  - Use `torch.stack()` and batch inference
  - Avoid loops over individual items
  
- [ ] **Pre-compute and cache on GPU**
  - Move constants to GPU once in `__init__`
  - Avoid re-creating tensors in hot loops
  
- [ ] **Use in-place operations**
  - `.clamp_()`, `.mul_()`, `.add_()` when safe
  - Only when gradients not needed
  
- [ ] **Profile before optimizing**
  - Measure memory usage
  - Count CPU↔GPU transfers
  - Benchmark different approaches
  
- [ ] **Verify output unchanged**
  - Test that optimizations don't change results
  - Use `torch.allclose()` for numerical comparison

## Performance Targets

For reference, typical performance on RTX 3090:

| Operation | Resolution | FP32 | FP16 | Speedup |
|-----------|-----------|------|------|---------|
| Preprocessing | 512×512 | 15ms | 5ms | 3.0x |
| Segmentation | 512×512 | 25ms | 10ms | 2.5x |
| Sliding Window | 2048×2048 | 250ms | 100ms | 2.5x |
| Batch (bs=8) | 512×512 | 180ms | 75ms | 2.4x |

Aim to match or beat these numbers!
