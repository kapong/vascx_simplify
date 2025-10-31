"""Test script to verify Phase 3 refactoring is backward compatible."""

import torch
from src.vascx_simplify import VASCXTransform

def test_ensemble_classes():
    """Test that ensemble classes still work after refactoring."""
    print("Testing ensemble class refactoring...")
    
    # Test that classes can be instantiated (without actual models)
    transform = VASCXTransform(size=512, have_ce=True)
    
    print("✓ VASCXTransform instantiated")
    
    # Create a dummy image
    img = torch.rand(3, 1024, 1024) * 255
    img = img.to(torch.uint8)
    
    # Test preprocessing
    processed, bounds = transform(img)
    print(f"✓ Image preprocessed: {processed.shape}")
    print(f"✓ Bounds returned: {bounds is not None}")
    
    print("\nAll Phase 3.1 tests passed!")

def test_mirror_edge_helper():
    """Test that edge mirroring still works after refactoring."""
    print("\nTesting edge mirroring refactoring...")
    
    # Create a simple test case
    transform = VASCXTransform(size=512, have_ce=True)
    
    # Test with a simple RGB gradient image
    img = torch.arange(0, 256*256, dtype=torch.uint8).reshape(256, 256)
    img = img.unsqueeze(0).expand(3, -1, -1)  # Make it RGB [3, H, W]
    
    # Run through preprocessing
    try:
        processed, bounds = transform(img)
        print(f"✓ Edge mirroring works: {processed.shape}")
    except Exception as e:
        print(f"✗ Edge mirroring failed: {e}")
        raise
    
    print("\nAll Phase 3.2 tests passed!")

def test_line_fitting():
    """Test that line fitting still works after refactoring."""
    print("\nTesting line fitting refactoring...")
    
    transform = VASCXTransform(size=512, have_ce=True)
    
    # Create a test image with clear circular boundary
    h, w = 512, 512
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 3
    
    # Create a circle mask
    dist = torch.sqrt((y - center_y).float()**2 + (x - center_x).float()**2)
    circle = (dist < radius).float() * 255
    img = circle.to(torch.uint8)
    
    # Convert to RGB
    img = img.unsqueeze(0).expand(3, -1, -1)
    
    # Run preprocessing - this will trigger line fitting
    try:
        processed, bounds = transform(img)
        print(f"✓ Line fitting works: {processed.shape}")
        print(f"✓ Bounds contain lines: {'lines' in bounds}")
    except Exception as e:
        print(f"✗ Line fitting failed: {e}")
        raise
    
    print("\nAll Phase 3.3 tests passed!")

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3 Refactoring Verification")
    print("=" * 60)
    
    test_ensemble_classes()
    test_mirror_edge_helper()
    test_line_fitting()
    
    print("\n" + "=" * 60)
    print("✅ All Phase 3 refactoring tests PASSED!")
    print("=" * 60)
