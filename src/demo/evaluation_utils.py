#!/usr/bin/env python3
"""
Evaluation and visualization utilities for TensorRT demo.
Handles model inference, keypoint detection, and result visualization.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def preprocess_image(image_path: str, model_type: str = 'hybrid_keypoint_net'):
    """Preprocess image for model input."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = (img_rgb.shape[1], img_rgb.shape[0])  # (width, height)
    
    # Determine input size based on model type
    if model_type == 'clip_heatmap_model':
        input_size = 256
    else:  # hybrid_keypoint_net, efficient_keypoint_net
        input_size = 128
    
    # Resize to model input size
    img_resized = cv2.resize(img_rgb, (input_size, input_size))
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    return img_tensor, img_rgb, original_size


def postprocess_heatmap(heatmap, model_type='hybrid_keypoint_net', original_size=None):
    """
    Postprocess heatmap exactly like inference_demo_simple.py.
    
    Args:
        heatmap: Predicted heatmap tensor
        model_type: Type of model ('clip_heatmap_model' or 'hybrid_keypoint_net')
        original_size: Original image size (width, height)
        
    Returns:
        Resized heatmap and keypoint coordinates
    """
    # Convert to numpy (handle both PyTorch tensors and numpy arrays)
    if isinstance(heatmap, torch.Tensor):
        heatmap_np = heatmap.squeeze().cpu().numpy()
    else:
        heatmap_np = heatmap.squeeze()
    
    # Use same threshold for all models - no normalization needed
    # This prevents false positives when there are actually zero keypoints
    threshold = 0.3

    # min threshold is 0.001 if model_type is hybrid_keypoint_net else 0.0005
    if model_type == "hybrid_keypoint_net" and heatmap_np.max() > 0.001:
        heatmap_np = heatmap_np / heatmap_np.max()
    elif model_type == "clip_heatmap_model" and heatmap_np.max() > 0.0005:
        heatmap_np = heatmap_np / heatmap_np.max()
    
    # Import the exact same functions used in inference_demo_simple.py
    from shared.functions import thresholded_locations, combine_nearby_peaks
    
    # Extract keypoints using EXACT same method as training evaluation
    peaks = thresholded_locations(heatmap_np, threshold=threshold)
    
    # Combine nearby peaks (same as training)
    combined_peaks = combine_nearby_peaks(peaks, distance_threshold=10)
    
    # Convert to keypoint format (same coordinate order as training: (x, y))
    # Use combined_peaks to reduce nearby duplicates
    keypoints = [(int(p[1]), int(p[0])) for p in combined_peaks]
    
    # Scale keypoints to original size if provided
    if original_size is not None:
        scale_x = original_size[0] / heatmap_np.shape[1]
        scale_y = original_size[1] / heatmap_np.shape[0]
        keypoints_scaled = [(int(x * scale_x), int(y * scale_y)) for x, y in keypoints]
        
        # Resize heatmap to original size for visualization
        heatmap_resized = cv2.resize(heatmap_np, original_size, interpolation=cv2.INTER_CUBIC)
        
        return heatmap_resized, keypoints_scaled
    else:
        return heatmap_np, keypoints


def evaluate_single_image_pytorch(model, image_path, segmenter=None, model_type='hybrid_keypoint_net'):
    """Evaluate a single image using PyTorch model."""
    # Preprocess image
    img_tensor, original_img, original_size = preprocess_image(image_path, model_type)
    
    # Apply segmentation if available
    if segmenter is not None:
        # Run segmentation
        results = segmenter(image_path)
        if len(results) > 0 and results[0].masks is not None:
            # Get the largest mask
            masks = results[0].masks.data.cpu().numpy()
            largest_mask = masks[np.argmax([np.sum(mask) for mask in masks])]
            
            # Resize mask to match image
            mask_resized = cv2.resize(largest_mask, (original_img.shape[1], original_img.shape[0]))
            
            # Apply mask to image
            masked_img = original_img.copy()
            masked_img[mask_resized < 0.5] = [0, 0, 0]  # Black out non-bedsheet areas
            
            # Resize masked image for model input
            input_size = 256 if model_type == 'clip_heatmap_model' else 128
            masked_img_resized = cv2.resize(masked_img, (input_size, input_size))
            img_tensor = torch.from_numpy(masked_img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
        else:
            masked_img = original_img
    else:
        masked_img = original_img
    
    # Run inference
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        heatmap = model(img_tensor)
    
    # Post-process heatmap using the correct method
    heatmap_resized, keypoints = postprocess_heatmap(heatmap, model_type, original_size)
    
    return keypoints, original_img, masked_img, heatmap_resized


def evaluate_single_image_tensorrt(tensorrt_inference, image_path, segmenter=None, model_type='hybrid_keypoint_net'):
    """Evaluate a single image using TensorRT model."""
    # Preprocess image
    img_tensor, original_img, original_size = preprocess_image(image_path, model_type)
    
    # Apply segmentation if available
    if segmenter is not None:
        # Run segmentation
        results = segmenter(image_path)
        if len(results) > 0 and results[0].masks is not None:
            # Get the largest mask
            masks = results[0].masks.data.cpu().numpy()
            largest_mask = masks[np.argmax([np.sum(mask) for mask in masks])]
            
            # Resize mask to match image
            mask_resized = cv2.resize(largest_mask, (original_img.shape[1], original_img.shape[0]))
            
            # Apply mask to image
            masked_img = original_img.copy()
            masked_img[mask_resized < 0.5] = [0, 0, 0]  # Black out non-bedsheet areas
            
            # Resize masked image for model input
            input_size = 256 if model_type == 'clip_heatmap_model' else 128
            masked_img_resized = cv2.resize(masked_img, (input_size, input_size))
            img_tensor = torch.from_numpy(masked_img_resized).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
        else:
            masked_img = original_img
    else:
        masked_img = original_img
    
    # Run TensorRT inference
    heatmap = tensorrt_inference.infer(img_tensor)
    
    # Post-process heatmap using the correct method
    heatmap_resized, keypoints = postprocess_heatmap(heatmap, model_type, original_size)
    
    return keypoints, original_img, masked_img, heatmap_resized


def visualize_results(original_img, keypoints, title="Keypoint Detection Results", heatmap=None):
    """Visualize keypoint detection results exactly like inference_demo_simple.py."""
    from PIL import Image
    import numpy as np
    
    # Convert numpy array to PIL Image if needed
    if isinstance(original_img, np.ndarray):
        image = Image.fromarray(original_img)
    else:
        image = original_img
    
    # Create figure with 3 subplots if heatmap is provided, otherwise 2
    if heatmap is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original, Heatmap, Overlay
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap, cmap='hot', alpha=0.8)
        axes[1].set_title('Predicted Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(image)
        axes[2].imshow(heatmap, cmap='hot', alpha=0.6)
        
        if keypoints:
            x_coords, y_coords = zip(*keypoints)
            axes[2].scatter(x_coords, y_coords, c='cyan', s=100, marker='x', linewidths=3)
            axes[2].set_title(f'{title} ({len(keypoints)} found)', fontsize=14, fontweight='bold')
        else:
            axes[2].set_title('No Keypoints Found', fontsize=14, fontweight='bold')
        
        axes[2].axis('off')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original and Overlay
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(image)
        
        if keypoints:
            x_coords, y_coords = zip(*keypoints)
            axes[1].scatter(x_coords, y_coords, c='cyan', s=100, marker='x', linewidths=3)
            axes[1].set_title(f'{title} ({len(keypoints)} found)', fontsize=14, fontweight='bold')
        else:
            axes[1].set_title('No Keypoints Found', fontsize=14, fontweight='bold')
        
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return image


def benchmark_model(model, image_path, num_runs=100, use_tensorrt=False, tensorrt_inference=None, model_type='hybrid_keypoint_net'):
    """Benchmark model inference speed."""
    import time
    
    # Warmup runs
    for _ in range(10):
        if use_tensorrt and tensorrt_inference:
            img_tensor, _, _ = preprocess_image(image_path, model_type)
            _ = tensorrt_inference.infer(img_tensor)
        else:
            img_tensor, _, _ = preprocess_image(image_path, model_type)
            device = next(model.parameters()).device
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                _ = model(img_tensor)
    
    # Benchmark runs
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        
        if use_tensorrt and tensorrt_inference:
            img_tensor, _, _ = preprocess_image(image_path, model_type)
            _ = tensorrt_inference.infer(img_tensor)
        else:
            img_tensor, _, _ = preprocess_image(image_path, model_type)
            device = next(model.parameters()).device
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                _ = model(img_tensor)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1.0 / avg_time
    
    return {
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'fps': fps,
        'num_runs': num_runs
    }


def load_segmentation_model():
    """Load YOLO segmentation model for bedsheet detection."""
    try:
        from ultralytics import YOLO
        seg_model_path = "models/yolo_finetuned/best.pt"
        if Path(seg_model_path).exists():
            print("Loading segmentation model...")
            segmenter = YOLO(seg_model_path)
            print("✓ Segmentation model loaded")
            return segmenter
        else:
            print("Segmentation model not found; proceeding without segmentation pre-processing")
            return None
    except Exception as e:
        print(f"Segmentation model load failed: {e}. Proceeding without segmentation.")
        return None
