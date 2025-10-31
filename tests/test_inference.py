"""Tests for inference functionality."""
import pytest
import torch


def test_sliding_window_inference_basic():
    """Test basic sliding window inference."""
    from simple_vascx import sliding_window_inference
    
    # Create dummy input [B, C, H, W]
    inputs = torch.randn(1, 3, 512, 512)
    
    # Create a simple predictor
    class DummyPredictor:
        def __call__(self, x):
            # Return [B, num_classes, H, W]
            return torch.randn(x.shape[0], 5, x.shape[2], x.shape[3])
    
    predictor = DummyPredictor()
    
    # Run sliding window inference
    output = sliding_window_inference(
        inputs=inputs,
        roi_size=(256, 256),
        sw_batch_size=4,
        predictor=predictor,
        overlap=0.5,
        mode='gaussian'
    )
    
    # Check output shape
    assert output.shape == (1, 5, 512, 512)
    assert output.dtype == inputs.dtype


def test_sliding_window_inference_constant_mode():
    """Test sliding window inference with constant importance map."""
    from simple_vascx import sliding_window_inference
    
    inputs = torch.randn(2, 3, 256, 256)
    
    class DummyPredictor:
        def __call__(self, x):
            return torch.randn(x.shape[0], 2, x.shape[2], x.shape[3])
    
    predictor = DummyPredictor()
    
    output = sliding_window_inference(
        inputs=inputs,
        roi_size=(128, 128),
        sw_batch_size=2,
        predictor=predictor,
        overlap=0.25,
        mode='constant'
    )
    
    assert output.shape == (2, 2, 256, 256)


def test_gaussian_importance_map():
    """Test Gaussian importance map creation."""
    from simple_vascx.inference import _create_gaussian_importance_map
    
    device = torch.device('cpu')
    dtype = torch.float32
    
    # Create Gaussian map
    importance_map = _create_gaussian_importance_map(128, 128, device, dtype)
    
    # Check properties
    assert importance_map.shape == (128, 128)
    assert importance_map.dtype == dtype
    assert importance_map.device.type == device.type
    
    # Center should have higher values than edges
    center_val = importance_map[64, 64]
    edge_val = importance_map[0, 0]
    assert center_val > edge_val


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sliding_window_inference_cuda():
    """Test sliding window inference on CUDA."""
    from simple_vascx import sliding_window_inference
    
    inputs = torch.randn(1, 3, 512, 512).cuda()
    
    class DummyPredictor:
        def __call__(self, x):
            return torch.randn(x.shape[0], 4, x.shape[2], x.shape[3]).cuda()
    
    predictor = DummyPredictor()
    
    output = sliding_window_inference(
        inputs=inputs,
        roi_size=(256, 256),
        sw_batch_size=4,
        predictor=predictor,
        overlap=0.5,
        mode='gaussian'
    )
    
    assert output.device.type == 'cuda'
    assert output.shape == (1, 4, 512, 512)
