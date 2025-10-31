#!/usr/bin/env python
"""Example script demonstrating vascx-simplify usage.

This is a simplified rewrite of the original rtnls_vascx_models project.
Original work: https://github.com/Eyened/rtnls_vascx_models
"""

import torch
import numpy as np
from simple_vascx import VASCXTransform, sliding_window_inference


def example_preprocessing():
    """Example: Preprocessing an image."""
    print("=" * 60)
    print("Example 1: Image Preprocessing")
    print("=" * 60)
    
    # Create a dummy fundus image (replace with real image path)
    dummy_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    
    # Initialize transform with contrast enhancement
    transform = VASCXTransform(size=512, have_ce=True, device='cpu', use_fp16=False)
    
    # Process image
    processed_image, bounds = transform(dummy_image)
    
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output shape: {processed_image.shape}")
    print(f"Output dtype: {processed_image.dtype}")
    print(f"Bounds available: {bounds is not None}")
    print()


def example_sliding_window():
    """Example: Sliding window inference."""
    print("=" * 60)
    print("Example 2: Sliding Window Inference")
    print("=" * 60)
    
    # Create a dummy predictor model
    class DummySegmentationModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 5, kernel_size=3, padding=1)
        
        def forward(self, x):
            return self.conv(x)
    
    model = DummySegmentationModel()
    model.eval()
    
    # Create dummy input image
    input_image = torch.randn(1, 3, 2048, 2048)
    
    # Run sliding window inference
    print("Running sliding window inference...")
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=input_image,
            roi_size=(512, 512),
            sw_batch_size=4,
            predictor=model,
            overlap=0.5,
            mode='gaussian'
        )
    
    print(f"Input shape: {input_image.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Processing window size: 512x512")
    print(f"Overlap: 50%")
    print()


def example_huggingface_integration():
    """Example: Loading model from HuggingFace (mock)."""
    print("=" * 60)
    print("Example 3: HuggingFace Integration")
    print("=" * 60)
    
    # This is just demonstrating the API
    # In real usage, replace with actual model repository
    model_string = "username/model-name:model.pt"
    
    print(f"Model string format: {model_string}")
    print("Usage:")
    print("  from simple_vascx import from_huggingface")
    print(f"  model_path = from_huggingface('{model_string}')")
    print("  # This will download the model from HuggingFace Hub")
    print()


def example_complete_workflow():
    """Example: Complete workflow with mock data."""
    print("=" * 60)
    print("Example 4: Complete Workflow")
    print("=" * 60)
    
    # 1. Setup preprocessing
    transform = VASCXTransform(size=1024, have_ce=False, device='cpu')
    
    # 2. Load image (using dummy data)
    dummy_image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
    print(f"Step 1: Loaded image with shape {dummy_image.shape}")
    
    # 3. Preprocess
    processed_image, bounds = transform(dummy_image)
    print(f"Step 2: Preprocessed to shape {processed_image.shape}")
    
    # 4. Create a dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 2, 3, padding=1)
        
        def forward(self, x):
            return torch.softmax(self.conv(x), dim=1)
    
    model = DummyModel()
    model.eval()
    
    # 5. Run inference
    with torch.no_grad():
        input_batch = processed_image.unsqueeze(0)  # Add batch dimension
        prediction = model(input_batch)
    
    print(f"Step 3: Model prediction shape {prediction.shape}")
    print(f"Step 4: Prediction range [{prediction.min():.3f}, {prediction.max():.3f}]")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "VASCX Simplify - Usage Examples" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    try:
        example_preprocessing()
        example_sliding_window()
        example_huggingface_integration()
        example_complete_workflow()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nFor more information, see:")
        print("  - README.md for detailed documentation")
        print("  - tests/ directory for more usage examples")
        print("  - QUICKSTART.md for installation guide")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
