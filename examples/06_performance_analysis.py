"""
Performance Analysis: Preprocessing vs Inference

This script profiles where time is actually spent to understand
why batch processing shows minimal speedup.
"""

from vascx_simplify import EnsembleSegmentation, VASCXTransform, from_huggingface
from PIL import Image
import time
import torch
import os


def main():
    IMG_PATH = 'HRF_07_dr.jpg'
    
    if not os.path.exists(IMG_PATH):
        print(f"Error: Image '{IMG_PATH}' not found!")
        return
    
    print("="*70)
    print("Performance Analysis: Where Does Time Go?")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model_path = from_huggingface('Eyened/vascx:artery_vein/av_july24.pt')
    model = EnsembleSegmentation(model_path, VASCXTransform())
    
    # Load images
    print("Loading test images...")
    images = [Image.open(IMG_PATH) for _ in range(10)]
    
    print(f"\n{'='*70}")
    print("Analysis 1: Single Image Breakdown")
    print(f"{'='*70}")
    
    # Time preprocessing alone
    single_img = images[0]
    
    start = time.time()
    preprocessed, bounds = model.transforms(single_img)
    preprocess_time = time.time() - start
    print(f"Preprocessing: {preprocess_time:.3f}s")
    
    # Time inference alone (on already preprocessed image)
    preprocessed_batch = preprocessed.unsqueeze(0).to(model.device)
    
    start = time.time()
    with torch.no_grad():
        _ = model._run_inference(preprocessed_batch)
    inference_time = time.time() - start
    print(f"Inference: {inference_time:.3f}s")
    
    # Time postprocessing alone
    start = time.time()
    _ = model.proba_process(
        torch.randn(1, 2, 4, 1024, 1024, device=model.device),  # [B, M, C, H, W]
        bounds,
        is_batch=False
    )
    postprocess_time = time.time() - start
    print(f"Postprocessing: {postprocess_time:.3f}s")
    
    total_single = preprocess_time + inference_time + postprocess_time
    print(f"\nTotal for single image: {total_single:.3f}s")
    print(f"  Preprocessing: {100*preprocess_time/total_single:.1f}%")
    print(f"  Inference: {100*inference_time/total_single:.1f}%")
    print(f"  Postprocessing: {100*postprocess_time/total_single:.1f}%")
    
    print(f"\n{'='*70}")
    print("Analysis 2: Batch Processing Breakdown")
    print(f"{'='*70}")
    
    # Time preprocessing for 10 images (sequential - can't batch this easily)
    start = time.time()
    preprocessed_list = [model.transforms(img) for img in images]
    batch_preprocess_time = time.time() - start
    print(f"Preprocessing 10 images (sequential): {batch_preprocess_time:.3f}s")
    print(f"  Per image: {batch_preprocess_time/10:.3f}s")
    
    # Time inference for 10 images in batch
    tensors = [t for t, b in preprocessed_list]
    batched = torch.stack(tensors).to(model.device)
    
    start = time.time()
    with torch.no_grad():
        _ = model._run_inference(batched)
    batch_inference_time = time.time() - start
    print(f"\nInference on batch of 10: {batch_inference_time:.3f}s")
    print(f"  Per image: {batch_inference_time/10:.3f}s")
    print(f"  Speedup vs single: {(inference_time*10)/batch_inference_time:.2f}x")
    
    # Time postprocessing for 10 images (sequential)
    bounds_list = [b for t, b in preprocessed_list]
    start = time.time()
    proba_batch = torch.randn(10, 2, 4, 1024, 1024, device=model.device)  # [B, M, C, H, W]
    for i in range(10):
        _ = model.proba_process(
            proba_batch[i:i+1],
            bounds_list[i],
            is_batch=False
        )
    batch_postprocess_time = time.time() - start
    print(f"\nPostprocessing 10 images (sequential): {batch_postprocess_time:.3f}s")
    print(f"  Per image: {batch_postprocess_time/10:.3f}s")
    
    total_batch = batch_preprocess_time + batch_inference_time + batch_postprocess_time
    print(f"\nTotal for 10 images: {total_batch:.3f}s")
    print(f"  Preprocessing: {100*batch_preprocess_time/total_batch:.1f}%")
    print(f"  Inference: {100*batch_inference_time/total_batch:.1f}%")
    print(f"  Postprocessing: {100*batch_postprocess_time/total_batch:.1f}%")
    
    print(f"\n{'='*70}")
    print("Key Findings")
    print(f"{'='*70}")
    
    # Calculate theoretical max speedup
    sequential_total = total_single * 10
    inference_speedup = (inference_time * 10) / batch_inference_time
    max_possible_speedup = sequential_total / (
        batch_preprocess_time + batch_inference_time + batch_postprocess_time
    )
    
    print(f"\n1. Inference shows {inference_speedup:.2f}x speedup with batching")
    print(f"2. But preprocessing takes {100*batch_preprocess_time/total_batch:.1f}% of total time")
    print("3. Preprocessing is sequential (RANSAC can't be batched)")
    print(f"4. Overall speedup limited to {max_possible_speedup:.2f}x")
    
    print("\n5. To improve further, would need to:")
    print("   - Batch the RANSAC/circle detection (complex)")
    print("   - Or use simpler preprocessing (skip contrast enhance)")
    print("   - Or preprocess once and cache results")
    
    print("\n" + "="*70)
    print("Conclusion")
    print("="*70)
    print("Batch processing IS working for the inference part!")
    print("The small overall speedup is because preprocessing dominates.")
    print("This is expected behavior given the current architecture.")


if __name__ == "__main__":
    main()
