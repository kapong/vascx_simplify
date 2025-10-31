"""Tests for preprocessing functionality."""
import pytest
import torch
import numpy as np


def test_vascx_transform_init():
    """Test VASCXTransform initialization."""
    from simple_vascx import VASCXTransform
    
    # Test with contrast enhancement
    transform_ce = VASCXTransform(size=512, have_ce=True, use_fp16=False)
    assert transform_ce.size == 512
    assert transform_ce.have_ce is True
    assert transform_ce.contrast_enhancer is not None
    
    # Test without contrast enhancement
    transform_no_ce = VASCXTransform(size=512, have_ce=False)
    assert transform_no_ce.have_ce is False
    assert transform_no_ce.contrast_enhancer is None


def test_vascx_transform_with_tensor():
    """Test VASCXTransform with tensor input."""
    from simple_vascx import VASCXTransform
    
    transform = VASCXTransform(size=256, have_ce=False, device='cpu')
    
    # Create a dummy RGB image tensor [3, 512, 512]
    img = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
    
    # Transform
    output, bounds = transform(img)
    
    # Check output shape (normalized, resized)
    assert output.shape == (3, 256, 256)
    assert output.dtype == torch.float32
    assert bounds is None  # No bounds when have_ce=False


def test_vascx_transform_with_numpy():
    """Test VASCXTransform with numpy input."""
    from simple_vascx import VASCXTransform
    
    transform = VASCXTransform(size=256, have_ce=False, device='cpu')
    
    # Create a dummy RGB image numpy array [512, 512, 3]
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Transform
    output, bounds = transform(img)
    
    # Check output shape
    assert output.shape == (3, 256, 256)
    assert output.dtype == torch.float32


def test_fundus_contrast_enhance_init():
    """Test FundusContrastEnhance initialization."""
    from simple_vascx import FundusContrastEnhance
    
    enhancer = FundusContrastEnhance(
        square_size=1024,
        sigma_fraction=0.05,
        contrast_factor=4,
        use_fp16=False
    )
    
    assert enhancer.square_size == 1024
    assert enhancer.sigma_fraction == 0.05
    assert enhancer.contrast_factor == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_vascx_transform_cuda():
    """Test VASCXTransform on CUDA if available."""
    from simple_vascx import VASCXTransform
    
    transform = VASCXTransform(size=256, have_ce=False, device='cuda')
    
    # Create a dummy RGB image
    img = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
    
    # Transform
    output, bounds = transform(img)
    
    # Check device
    assert output.device.type == 'cuda'
    assert output.shape == (3, 256, 256)
