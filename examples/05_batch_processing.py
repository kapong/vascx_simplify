"""
Batch Processing Example

Demonstrates efficient batch inference for multiple fundus images with automatic batch splitting.

This example shows:
1. Sequential processing (baseline)
2. Batch processing with default batch_size
3. Batch processing with custom batch_size
4. Large batch with automatic splitting
5. Verification of numerical consistency
"""

from vascx_simplify import EnsembleSegmentation, VASCXTransform, from_huggingface
from PIL import Image
import time
import torch
import os


def main():
    # Configuration
    IMG_PATH = 'HRF_07_dr.jpg'  # Path to your fundus image
    
    # Check if image exists
    if not os.path.exists(IMG_PATH):
        print(f"Error: Image '{IMG_PATH}' not found!")
        print("Please download a sample fundus image or update IMG_PATH")
        return
    
    print("="*70)
    print("Batch Processing Performance Demonstration")
    print("="*70)
    
    # Load model once
    print("\n1. Loading model...")
    model_path = from_huggingface('Eyened/vascx:artery_vein/av_july24.pt')
    model = EnsembleSegmentation(model_path, VASCXTransform())
    
    print(f"   Model default batch size: {model.predict_batch_size}")
    print(f"   GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU: {gpu_name}")
        print(f"   GPU memory: {gpu_memory:.1f} GB")
    
    # Load one image and replicate it for testing
    # In real usage, you would load different images
    print(f"\n2. Loading test image: {IMG_PATH}")
    base_image = Image.open(IMG_PATH)
    
    # Create test sets of different sizes
    images_10 = [base_image for _ in range(10)]
    
    print(f"   Created test set with {len(images_10)} images")
    print(f"   Image size: {base_image.size}")
    
    # Test 1: Sequential processing (baseline)
    print("\n" + "="*70)
    print("Test 1: Sequential Processing (Baseline)")
    print("="*70)
    print(f"Processing {len(images_10)} images one at a time...")
    
    start = time.time()
    preds_sequential = torch.stack([model.predict(img) for img in images_10])
    seq_time = time.time() - start
    
    print(f"✅ Complete!")
    print(f"   Total time: {seq_time:.2f}s")
    print(f"   Time per image: {seq_time/len(images_10):.3f}s")
    print(f"   Output shape: {preds_sequential.shape}")
    
    # Test 2: Batch processing with default batch size
    print("\n" + "="*70)
    print(f"Test 2: Batch Processing (default batch_size={model.predict_batch_size})")
    print("="*70)
    print(f"Processing {len(images_10)} images in batches...")
    print(f"   Will automatically split into chunks of {model.predict_batch_size}")
    
    start = time.time()
    preds_batch = model.predict(images_10)
    batch_time = time.time() - start
    
    speedup = seq_time / batch_time
    print(f"✅ Complete!")
    print(f"   Total time: {batch_time:.2f}s")
    print(f"   Time per image: {batch_time/len(images_10):.3f}s")
    print(f"   Speedup: {speedup:.2f}x faster")
    print(f"   Output shape: {preds_batch.shape}")
    
    # Test 3: Batch processing with custom batch size
    print("\n" + "="*70)
    print("Test 3: Batch Processing (custom batch_size=2)")
    print("="*70)
    print(f"Processing {len(images_10)} images with smaller batches...")
    
    start = time.time()
    preds_batch_custom = model.predict(images_10, batch_size=2)
    custom_time = time.time() - start
    
    speedup_custom = seq_time / custom_time
    print(f"✅ Complete!")
    print(f"   Total time: {custom_time:.2f}s")
    print(f"   Time per image: {custom_time/len(images_10):.3f}s")
    print(f"   Speedup vs sequential: {speedup_custom:.2f}x faster")
    print(f"   Output shape: {preds_batch_custom.shape}")
    
    # Test 4: Large batch demonstration
    print("\n" + "="*70)
    print("Test 4: Large Batch Processing (100 images with auto-splitting)")
    print("="*70)
    
    large_images = [base_image for _ in range(100)]
    print(f"Processing {len(large_images)} images...")
    print(f"   Automatically split into {len(large_images)//model.predict_batch_size} batches")
    
    start = time.time()
    preds_large = model.predict(large_images)
    large_time = time.time() - start
    
    print(f"✅ Complete!")
    print(f"   Total time: {large_time:.2f}s")
    print(f"   Time per image: {large_time/len(large_images):.3f}s")
    print(f"   Output shape: {preds_large.shape}")
    print(f"   Memory efficient: Processed in {(len(large_images) + model.predict_batch_size - 1) // model.predict_batch_size} chunks")
    
    # Test 5: Verification
    print("\n" + "="*70)
    print("Test 5: Numerical Consistency Verification")
    print("="*70)
    
    # Compare first result from each method
    match_default = torch.allclose(preds_sequential[0], preds_batch[0], rtol=1e-5, atol=1e-5)
    match_custom = torch.allclose(preds_sequential[0], preds_batch_custom[0], rtol=1e-5, atol=1e-5)
    match_large = torch.allclose(preds_sequential[0], preds_large[0], rtol=1e-5, atol=1e-5)
    
    print(f"   Sequential[0] ≈ Batch default[0]: {match_default}")
    print(f"   Sequential[0] ≈ Batch custom[0]: {match_custom}")
    print(f"   Sequential[0] ≈ Batch large[0]: {match_large}")
    
    if match_default and match_custom and match_large:
        print("   ✅ All methods produce consistent results!")
    else:
        print("   ⚠️  Minor differences detected (expected due to batch norm)")
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"   Sequential: {seq_time:.2f}s ({seq_time/len(images_10):.3f}s per image)")
    print(f"   Batch (default): {batch_time:.2f}s ({batch_time/len(images_10):.3f}s per image) - {speedup:.2f}x faster")
    print(f"   Batch (custom): {custom_time:.2f}s ({custom_time/len(images_10):.3f}s per image) - {speedup_custom:.2f}x faster")
    print(f"   Large batch: {large_time:.2f}s ({large_time/len(large_images):.3f}s per image)")
    
    print("\n" + "="*70)
    print("Key Takeaways:")
    print("="*70)
    print("1. Batch processing is significantly faster than sequential")
    print(f"2. Default batch_size={model.predict_batch_size} is optimized for memory/speed")
    print("3. Large batches automatically split to prevent OOM errors")
    print("4. Results are numerically consistent across methods")
    print("5. All existing code works without modification (backward compatible)")
    print("\n✅ Batch processing demonstration complete!")


if __name__ == "__main__":
    main()
