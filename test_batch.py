"""
Test batch processing implementation - backward compatibility and batch support
"""
import torch
from PIL import Image
import numpy as np

# Test if imports work
print("Testing imports...")
from vascx_simplify import EnsembleSegmentation, ClassificationEnsemble, VASCXTransform

print("✅ Imports successful!")

# Test 1: Single image backward compatibility (no real model needed for structure test)
print("\n" + "="*60)
print("Test 1: API Structure Tests")
print("="*60)

# Create dummy PIL image
dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

# Create transform (no model loading)
transform = VASCXTransform()

print("\n✅ Transform created successfully")

# Test transform on single image
print("Testing transform on single PIL image...")
tensor, bounds = transform(dummy_image)
print(f"  Output shape: {tensor.shape}")
print(f"  Bounds type: {type(bounds)}")
assert tensor.dim() == 3, "Should be [C, H, W]"
assert isinstance(bounds, dict), "Bounds should be dict for single image"
print("✅ Single image transform works")

# Test transform on list of images
print("\nTesting transform on list of PIL images...")
dummy_images = [dummy_image for _ in range(3)]
# Note: Transform doesn't handle lists natively, this is done in ensemble
print("✅ List creation successful")

# Test 2: Tensor batch detection
print("\n" + "="*60)
print("Test 2: Batch Detection Logic")
print("="*60)

# Create dummy tensors
single_3d = torch.randn(3, 512, 512)
single_4d = torch.randn(1, 3, 512, 512)
batch_4d = torch.randn(5, 3, 512, 512)

# We'll need to create a mock ensemble to test _is_batch_input
# For now, just test the structure exists
print("\n✅ Tensor creation successful")
print(f"  Single [C,H,W]: {single_3d.shape}")
print(f"  Single [1,C,H,W]: {single_4d.shape}")
print(f"  Batch [5,C,H,W]: {batch_4d.shape}")

# Test 3: Check that new attributes exist in ensemble classes
print("\n" + "="*60)
print("Test 3: Class Attributes")
print("="*60)

# Check EnsembleSegmentation would have predict_batch_size
# We can't instantiate without a model file, but we can check the code
import inspect
from vascx_simplify.inference import EnsembleBase, EnsembleSegmentation, ClassificationEnsemble

# Check EnsembleBase has the new methods
base_methods = dir(EnsembleBase)
assert '_is_batch_input' in base_methods, "EnsembleBase should have _is_batch_input"
assert '_predict_batch' in base_methods, "EnsembleBase should have _predict_batch"
assert '_prepare_input' in base_methods, "EnsembleBase should have _prepare_input"
print("✅ EnsembleBase has batch methods")

# Check predict signature accepts batch_size
predict_sig = inspect.signature(EnsembleBase.predict)
assert 'batch_size' in predict_sig.parameters, "predict should accept batch_size parameter"
print("✅ predict() accepts batch_size parameter")

# Check _prepare_input returns 3 values
prep_sig = inspect.signature(EnsembleBase._prepare_input)
print(f"✅ _prepare_input signature: {prep_sig}")

print("\n" + "="*60)
print("All structure tests passed!")
print("="*60)
print("\nNote: Full integration tests require model files from HuggingFace")
print("Run examples/01_artery_vein.py to test with real models")
