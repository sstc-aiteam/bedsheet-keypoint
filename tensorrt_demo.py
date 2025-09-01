#!/usr/bin/env python3
"""
Demo script for TensorRT keypoint detection model inference.
Compares TensorRT vs PyTorch performance and visualizes results.
"""

import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Import TensorRT utilities
try:
    from src.utils.tensorrt_utils import TensorRTInference
    TENSORRT_AVAILABLE = True
except ImportError:
    print("Warning: TensorRT utilities not available")
    TENSORRT_AVAILABLE = False

# --- Safe save/load helpers for torch.compile-wrapped models ---
def _get_base_module(module):
    return getattr(module, "_orig_mod", module)

def save_model_safely(model, save_path: str) -> None:
    base = _get_base_module(model)
    torch.save(base.state_dict(), save_path)

def load_model_safely(model, load_path: str, map_location="cpu", strict: bool = False):
    state = torch.load(load_path, map_location=map_location)
    cleaned = {}
    for key, value in state.items():
        if key.startswith("_orig_mod."):
            cleaned[key[len("_orig_mod."):]] = value
        else:
            cleaned[key] = value
    target = _get_base_module(model)
    return target.load_state_dict(cleaned, strict=strict)

def load_pytorch_model(model_path: str):
    """Load PyTorch model using the exact same approach as post training script."""
    from src.models.hybrid_keypoint_net import HybridKeypointNet
    from src.models.efficient_keypoint_net import YoloBackbone
    from ultralytics import YOLO
    
    # Create model architecture (exact same as post training script)
    yolo_model = YOLO('yolo11l-pose.pt')
    backbone_seq = yolo_model.model.model[:12]
    backbone = YoloBackbone(backbone_seq, selected_indices=[0,1,2,3,4,5,6,7,8,9,10,11])
    
    input_dummy = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        feats = backbone(input_dummy)
    in_channels_list = [f.shape[1] for f in feats]
    
    model = HybridKeypointNet(backbone, in_channels_list)
    
    # Load trained weights safely using the same approach as post training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use the same safe loading approach as post training script
    missing_keys, unexpected_keys = load_model_safely(model, model_path, map_location=device, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {len(unexpected_keys)}")
    
    model.eval()
    
    return model

def preprocess_image(image_path: str):
    """Preprocess image for model input."""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get original dimensions
    orig_h, orig_w = img.shape[:2]
    
    # Resize to 128x128
    img_resized = cv2.resize(img_rgb, (128, 128))
    
    # Match post-training: use 0–255 float32 (no normalization)
    img_normalized = img_resized.astype(np.float32)
    
    # Convert to tensor format (B, C, H, W)
    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    return img_tensor, img_rgb, (orig_h, orig_w)

def postprocess_keypoints(heatmap: np.ndarray, threshold: float = 0.003):
    """Extract keypoints from heatmap using the same approach as post training."""
    # Remove batch dimension if present
    if heatmap.ndim == 4:
        heatmap = heatmap[0]
    
    # Use the same thresholding approach as in post training
    from src.utils.model_utils import thresholded_locations
    
    # Get peak locations in (y, x) format - same as post training
    peak_locations = thresholded_locations(heatmap[0], threshold)
    
    # Return in [y, x] format to match post training script
    return peak_locations

def create_thresholded_heatmap(heatmap: np.ndarray, threshold: float = 0.003):
    """Create a thresholded heatmap for debugging - all pixels above threshold are kept, below are set to 0."""
    # Remove batch dimension if present
    if heatmap.ndim == 4:
        heatmap = heatmap[0]
    
    # Create thresholded heatmap
    thresholded = heatmap.copy()
    thresholded[thresholded < threshold] = 0
    thresholded[thresholded > 0] = 1
    
    return thresholded

def visualize_heatmap_debug(original_img, heatmap, thresholded_heatmap, title, save_path):
    """Visualize the original heatmap and thresholded heatmap for debugging."""
    plt.figure(figsize=(20, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    # Raw heatmap
    plt.subplot(2, 3, 2)
    heatmap_2d = heatmap[0, 0] if heatmap.ndim == 4 else heatmap[0]
    plt.imshow(heatmap_2d, cmap='hot', interpolation='nearest')
    plt.title(f"Raw Heatmap\nRange: [{heatmap.min():.6f}, {heatmap.max():.6f}]")
    plt.colorbar()
    plt.axis('off')
    
    # Thresholded heatmap
    plt.subplot(2, 3, 3)
    thresholded_2d = thresholded_heatmap[0, 0] if thresholded_heatmap.ndim == 4 else thresholded_heatmap[0]
    plt.imshow(thresholded_2d, cmap='hot', interpolation='nearest')
    plt.title(f"Thresholded Heatmap (>{thresholded_heatmap.max():.6f})\nNon-zero pixels: {np.count_nonzero(thresholded_heatmap)}")
    plt.colorbar()
    plt.axis('off')
    
    # Histogram of heatmap values
    plt.subplot(2, 3, 4)
    plt.hist(heatmap.flatten(), bins=100, alpha=0.7, color='blue', label='All values')
    plt.hist(thresholded_heatmap.flatten(), bins=100, alpha=0.7, color='red', label='Above threshold')
    plt.axvline(x=0.003, color='green', linestyle='--', label='Threshold (0.003)')
    plt.xlabel('Heatmap Values')
    plt.ylabel('Frequency')
    plt.title('Heatmap Value Distribution')
    plt.legend()
    plt.yscale('log')
    
    # Top values in heatmap
    plt.subplot(2, 3, 5)
    top_values = np.sort(heatmap.flatten())[-20:]  # Top 20 values
    plt.bar(range(len(top_values)), top_values, color='red')
    plt.xlabel('Rank')
    plt.ylabel('Value')
    plt.title('Top 20 Heatmap Values')
    plt.axhline(y=0.003, color='green', linestyle='--', label='Threshold')
    plt.legend()
    
    # Thresholded heatmap overlay on original image using cv2.circle
    plt.subplot(2, 3, 6)
    
    # Create a copy of the original image to draw on
    img_with_keypoints = original_img.copy()
    
    # Scale thresholded heatmap to original image size
    orig_h, orig_w = original_img.shape[:2]
    scale_x = orig_w / 128
    scale_y = orig_h / 128
    
    # Find non-zero positions in thresholded heatmap
    non_zero_positions = np.where(thresholded_2d > 0)
    if len(non_zero_positions[0]) > 0:
        # Draw circles at each non-zero position using cv2.circle
        for i, (y, x) in enumerate(zip(non_zero_positions[0], non_zero_positions[1])):
            # Scale to original image coordinates
            orig_y = int(y * scale_y)
            orig_x = int(x * scale_x)
            
            # Draw circle (OpenCV uses BGR) - same as post training
            cv2.circle(img_with_keypoints, (orig_x, orig_y), 15, (0, 0, 255), -1)  # Red circle
            
            # Add keypoint number label
            cv2.putText(img_with_keypoints, f'{i+1}', (orig_x + 20, orig_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # White text
    
    # Convert BGR to RGB for matplotlib display
    img_with_keypoints_rgb = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
    plt.imshow(img_with_keypoints_rgb)
    plt.title(f"CV2 Circle Keypoints Overlay\n{len(non_zero_positions[0])} non-zero pixels")
    
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {title} heatmap debug visualization saved as '{save_path}'")

def combine_nearby_peaks(peaks, distance_threshold=10):
    """
    Combine nearby peaks into single keypoints using clustering.
    Exact copy from post training script.
    
    Args:
        peaks: List of peak coordinates [[y1, x1], [y2, x2], ...]
        distance_threshold: Maximum distance to consider peaks as part of same cluster
    
    Returns:
        List of combined peak coordinates
    """
    if not peaks:
        return []
    
    # Convert to numpy array for easier manipulation
    peaks = np.array(peaks)
    
    # If only one peak, return it
    if len(peaks) == 1:
        return peaks.tolist()
    
    # Calculate pairwise distances
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(peaks))
    
    # Create clusters
    clusters = []
    used = set()
    
    for i in range(len(peaks)):
        if i in used:
            continue
            
        # Start a new cluster
        cluster = [i]
        used.add(i)
        
        # Find all peaks within distance_threshold
        for j in range(i + 1, len(peaks)):
            if j not in used and distances[i, j] <= distance_threshold:
                cluster.append(j)
                used.add(j)
        
        clusters.append(cluster)
    
    # Calculate centroid for each cluster
    combined_peaks = []
    for cluster in clusters:
        cluster_peaks = peaks[cluster]
        centroid = cluster_peaks.mean(axis=0)
        combined_peaks.append(centroid)
    
    return combined_peaks

def visualize_results(original_img, keypoints, title="Keypoint Detection Results"):
    """Visualize keypoints on original image."""
    plt.figure(figsize=(12, 8))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Plot image with keypoints
    plt.subplot(1, 2, 2)
    plt.imshow(original_img)
    
    if keypoints:
        keypoints = np.array(keypoints)
        # Scale keypoints from 128x128 to original image size - same as post training
        orig_h, orig_w = original_img.shape[:2]
        scale_x = orig_w / 128
        scale_y = orig_h / 128
        
        # Scale to original image coordinates - same logic as post training
        scaled_keypoints = []
        for y, x in keypoints:  # keypoints are in [y, x] format
            orig_y = int(y * scale_y)
            orig_x = int(x * scale_x)
            scaled_keypoints.append([orig_x, orig_y])  # Convert to [x, y] for plotting
        
        scaled_keypoints = np.array(scaled_keypoints)
        
        # Plot keypoints with much larger size for better visibility
        plt.scatter(scaled_keypoints[:, 0], scaled_keypoints[:, 1], 
                   c='red', s=300, marker='o', edgecolors='white', linewidth=4)
        
        # Add a black outline circle for even better visibility
        plt.scatter(scaled_keypoints[:, 0], scaled_keypoints[:, 1], 
                   c='none', s=400, marker='o', edgecolors='black', linewidth=2)
        
        # Add keypoint numbers for identification
        for i, (x, y) in enumerate(scaled_keypoints):
            plt.text(x + 20, y - 20, f'KP{i+1}', 
                    color='white', fontsize=12, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8))
    
    plt.title(f"{title}\nDetected Keypoints: {len(keypoints)}")
    plt.axis('off')
    
    plt.tight_layout()
    return plt.gcf()

def benchmark_inference(model, input_data, num_runs=100, warmup_runs=10, model_name="Model"):
    """Benchmark inference performance."""
    print(f"\n=== {model_name} Benchmark ===")
    
    # Warmup
    print(f"Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        if isinstance(model, TensorRTInference):
            _ = model.infer(input_data)
        else:
            with torch.no_grad():
                _ = model(torch.from_numpy(input_data).cuda())
    
    # Benchmark
    print(f"Running {num_runs} benchmark iterations...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        if isinstance(model, TensorRTInference):
            _ = model.infer(input_data)
        else:
            with torch.no_grad():
                _ = model(torch.from_numpy(input_data).cuda())
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    fps = num_runs / total_time
    
    print(f"Total time: {total_time:.3f}s")
    print(f"Average inference time: {avg_time*1000:.2f}ms")
    print(f"FPS: {fps:.1f}")
    
    return {
        'total_time': total_time,
        'avg_inference_time': avg_time * 1000,
        'fps': fps
    }

def _get_largest_bbox_from_masks(result, allowed_classes=None):
    """Return xyxy bbox (ints) of the largest segmentation mask filtered by allowed_classes.
    If no masks or none match, return None.
    """
    try:
        masks = result.masks
        boxes = getattr(result, 'boxes', None)
        cls_ids = None
        if boxes is not None and getattr(boxes, 'cls', None) is not None:
            cls_ids = boxes.cls.cpu().numpy().astype(int)
        if masks is None or masks.data is None:
            return None
        # masks.data: (N, H, W) boolean/float; order aligns with boxes
        areas = []
        bboxes = []
        for idx, m in enumerate(masks.data):
            if allowed_classes is not None and cls_ids is not None:
                if cls_ids[idx] not in set(allowed_classes):
                    continue
            m_np = (m.cpu().numpy() > 0.5).astype(np.uint8)
            if m_np.sum() == 0:
                continue
            # Find contours and bbox
            ys, xs = np.where(m_np > 0)
            y1, y2 = int(ys.min()), int(ys.max())
            x1, x2 = int(xs.min()), int(xs.max())
            area = (y2 - y1 + 1) * (x2 - x1 + 1)
            areas.append(area)
            bboxes.append((x1, y1, x2, y2))
        if not bboxes:
            return None
        # Largest area bbox
        idx = int(np.argmax(areas))
        return bboxes[idx]
    except Exception:
        return None


# Removed top-k fallback to avoid biasing detections to corners


def _build_allowed_mask(result, allowed_classes=None):
    """Build a combined boolean mask for allowed classes from a YOLO result in the result image space."""
    masks = getattr(result, 'masks', None)
    boxes = getattr(result, 'boxes', None)
    if masks is None or masks.data is None:
        return None
    cls_ids = None
    if boxes is not None and getattr(boxes, 'cls', None) is not None:
        cls_ids = boxes.cls.cpu().numpy().astype(int)
    mask_accum = None
    for idx, m in enumerate(masks.data):
        if allowed_classes is not None and cls_ids is not None:
            if cls_ids[idx] not in set(allowed_classes):
                continue
        m_np = (m.cpu().numpy() > 0.5).astype(np.uint8)
        if mask_accum is None:
            mask_accum = m_np
        else:
            mask_accum = np.maximum(mask_accum, m_np)
    return mask_accum


def evaluate_single_image_like_post_training(model, image_path, segmenter=None):
    """Evaluate on a single image using post-training logic, with optional segmentation mask
    (allowed class=1) applied to zero out non-bedsheet pixels (no cropping).
    """
    from src.utils.model_utils import thresholded_locations
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Load original image (same as post training)
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Warning: Could not load original image: {image_path}")
        return None
    
    # Get original image dimensions
    orig_h, orig_w = orig_img.shape[:2]
    print(f"Original image dimensions: {orig_w} x {orig_h}")

    # Optional: segmentation to build mask (no cropping). Zero-out non-bedsheet pixels.
    masked_orig = orig_img.copy()
    if segmenter is not None:
        try:
            # Resize first for robust segmentation
            seg_max_side = 1280
            h0, w0 = orig_img.shape[:2]
            scale = min(seg_max_side / max(h0, w0), 1.0)
            if scale < 1.0:
                seg_w = int(w0 * scale)
                seg_h = int(h0 * scale)
                seg_input_img = cv2.resize(orig_img, (seg_w, seg_h), interpolation=cv2.INTER_AREA)
            else:
                seg_input_img = orig_img
                seg_h, seg_w = h0, w0

            results = segmenter(seg_input_img, verbose=False)
            if results and len(results) > 0:
                allowed_classes = [1]
                mask_small = _build_allowed_mask(results[0], allowed_classes=allowed_classes)
                if mask_small is not None:
                    # Resize mask back to original size (nearest for binary mask)
                    mask_full = cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
                    # Apply mask: keep bedsheet, zero elsewhere
                    masked_orig[mask_full == 0] = 0
                    print("Applied segmentation mask (class 1) to original image.")
                else:
                    print("No allowed-class masks found; proceeding without masking.")
        except Exception as e:
            print(f"Segmentation failed or returned no masks: {e}. Proceeding without masking.")
    
    # Preprocess image for model input (resize to 128x128 and normalize)
    img_rgb = cv2.cvtColor(masked_orig, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    # Match post-training: use 0–255 float32 (no normalization)
    img_normalized = img_resized.astype(np.float32)
    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    # Run inference
    with torch.no_grad():
        input_tensor = torch.from_numpy(img_tensor).to(device)
        outputs = model(input_tensor)
        kp = outputs.cpu().numpy()[0, 0, :, :]  # Get the heatmap
    
    print(f"Model output shape: {outputs.shape}")
    print(f"Model output range: [{kp.min():.6f}, {kp.max():.6f}]")
    
    # Suppress edges to avoid degenerate corner peaks
    edge = 3
    kp[:edge, :] = 0
    kp[-edge:, :] = 0
    kp[:, :edge] = 0
    kp[:, -edge:] = 0
    # Extract peaks using same approach as post training at threshold 0.003
    peaks = thresholded_locations(kp, 0.003)
    print(f"Raw peaks from thresholded_locations: {peaks}")
    
    # Combine nearby peaks into single keypoints (same as post training)
    combined_peaks = combine_nearby_peaks(peaks, distance_threshold=10)
    print(f"Combined peaks: {len(peaks)} -> {len(combined_peaks)}")
    # No top-k fallback; rely purely on threshold + clustering
    
    # Scale keypoint coordinates from 128x128 back to original image size (no offset)
    scale_x = orig_w / 128
    scale_y = orig_h / 128
    print(f"Scaling factors: scale_x={scale_x:.2f}, scale_y={scale_y:.2f}")
    
    # Create copies for drawing
    result_img = orig_img.copy()
    masked_vis_img = masked_orig.copy()
    
    # Draw keypoints on original image (same as post training)
    if combined_peaks:
        for i, p in enumerate(combined_peaks):
            row, col = p  # i=row, j=col in 128x128 space
            # Scale to original image coordinates
            orig_row = int(row * scale_y)
            orig_col = int(col * scale_x)
            print(f"Keypoint {i+1}: ({row:.1f}, {col:.1f}) -> ({orig_col}, {orig_row}) (mapped)")
            # Draw circle (OpenCV uses BGR) - same as post training
            cv2.circle(result_img, (orig_col, orig_row), 30, (0, 0, 255), -1)  # Red circle
            # Draw on masked visualization as well
            cv2.circle(masked_vis_img, (orig_col, orig_row), 20, (0, 0, 255), -1)
    else:
        # Fallback: draw global argmax to aid debugging (same as post training)
        max_idx = np.unravel_index(np.argmax(kp), kp.shape)
        mi, mj = int(max_idx[0]), int(max_idx[1])
        mv = float(kp[mi, mj])
        orig_row = int(mi * scale_y)
        orig_col = int(mj * scale_x)
        print(f"Fallback argmax: ({mi}, {mj}) -> ({orig_col}, {orig_row}), value={mv:.6f} (mapped)")
        cv2.circle(result_img, (orig_col, orig_row), 24, (0, 255, 0), 2)  # Green circle
        cv2.putText(result_img, f"argmax {mv:.4f}", (max(5, orig_col-60), max(25, orig_row-10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # Also draw on masked visualization
        cv2.circle(masked_vis_img, (orig_col, orig_row), 18, (0, 255, 0), 2)
    
    # Convert BGR back to RGB for matplotlib display
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    masked_vis_img_rgb = cv2.cvtColor(masked_vis_img, cv2.COLOR_BGR2RGB)

    return result_img_rgb, combined_peaks, cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), masked_vis_img_rgb


def evaluate_single_image_tensorrt(tensorrt_model, image_path, segmenter=None):
    """Evaluate TensorRT model on a single image using the same preprocessing logic."""
    from src.utils.model_utils import thresholded_locations
    
    # Load original image
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Warning: Could not load original image: {image_path}")
        return None
    
    # Get original image dimensions
    orig_h, orig_w = orig_img.shape[:2]
    print(f"TensorRT - Original image dimensions: {orig_w} x {orig_h}")
    
    # Optional: segmentation to build mask (same as PyTorch version)
    masked_orig = orig_img.copy()
    if segmenter is not None:
        try:
            # Resize first for robust segmentation
            seg_max_side = 1280
            h0, w0 = orig_img.shape[:2]
            scale = min(seg_max_side / max(h0, w0), 1.0)
            if scale < 1.0:
                seg_w = int(w0 * scale)
                seg_h = int(h0 * scale)
                seg_input_img = cv2.resize(orig_img, (seg_w, seg_h), interpolation=cv2.INTER_AREA)
            else:
                seg_input_img = orig_img
                seg_h, seg_w = h0, w0

            results = segmenter(seg_input_img, verbose=False)
            if results and len(results) > 0:
                allowed_classes = [1]
                mask_small = _build_allowed_mask(results[0], allowed_classes=allowed_classes)
                if mask_small is not None:
                    # Resize mask back to original size
                    mask_full = cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
                    masked_orig[mask_full == 0] = 0
                    print("TensorRT - Applied segmentation mask (class 1) to original image.")
                else:
                    print("TensorRT - No allowed-class masks found; proceeding without masking.")
        except Exception as e:
            print(f"TensorRT - Segmentation failed: {e}. Proceeding without masking.")
    
    # Preprocess image for model input (resize to 128x128, no normalization)
    img_rgb = cv2.cvtColor(masked_orig, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))
    img_normalized = img_resized.astype(np.float32)  # No /255
    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    # Run TensorRT inference
    outputs = tensorrt_model.infer(img_tensor)
    kp = outputs[0, 0, :, :]  # Get the heatmap
    
    print(f"TensorRT - Model output shape: {outputs.shape}")
    print(f"TensorRT - Model output range: [{kp.min():.6f}, {kp.max():.6f}]")
    
    # Suppress edges to avoid degenerate corner peaks
    edge = 3
    kp[:edge, :] = 0
    kp[-edge:, :] = 0
    kp[:, :edge] = 0
    kp[:, -edge:] = 0
    
    # Extract peaks using same approach as PyTorch
    peaks = thresholded_locations(kp, 0.003)
    print(f"TensorRT - Raw peaks from thresholded_locations: {peaks}")
    
    # Combine nearby peaks into single keypoints
    combined_peaks = combine_nearby_peaks(peaks, distance_threshold=10)
    print(f"TensorRT - Combined peaks: {len(peaks)} -> {len(combined_peaks)}")
    
    # Scale keypoint coordinates from 128x128 back to original image size
    scale_x = orig_w / 128
    scale_y = orig_h / 128
    print(f"TensorRT - Scaling factors: scale_x={scale_x:.2f}, scale_y={scale_y:.2f}")
    
    # Create copies for drawing
    result_img = orig_img.copy()
    masked_vis_img = masked_orig.copy()
    
    # Draw keypoints on original image
    if combined_peaks:
        for i, p in enumerate(combined_peaks):
            row, col = p  # i=row, j=col in 128x128 space
            # Scale to original image coordinates
            orig_row = int(row * scale_y)
            orig_col = int(col * scale_x)
            print(f"TensorRT - Keypoint {i+1}: ({row:.1f}, {col:.1f}) -> ({orig_col}, {orig_row}) (mapped)")
            # Draw circle (OpenCV uses BGR)
            cv2.circle(result_img, (orig_col, orig_row), 30, (0, 0, 255), -1)  # Red circle
            # Draw on masked visualization as well
            cv2.circle(masked_vis_img, (orig_col, orig_row), 20, (0, 0, 255), -1)
    else:
        # Fallback: draw global argmax to aid debugging
        max_idx = np.unravel_index(np.argmax(kp), kp.shape)
        mi, mj = int(max_idx[0]), int(max_idx[1])
        mv = float(kp[mi, mj])
        orig_row = int(mi * scale_y)
        orig_col = int(mj * scale_x)
        print(f"TensorRT - Fallback argmax: ({mi}, {mj}) -> ({orig_col}, {orig_row}), value={mv:.6f} (mapped)")
        cv2.circle(result_img, (orig_col, orig_row), 24, (0, 255, 0), 2)  # Green circle
        cv2.putText(result_img, f"argmax {mv:.4f}", (max(5, orig_col-60), max(25, orig_row-10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # Also draw on masked visualization
        cv2.circle(masked_vis_img, (orig_col, orig_row), 18, (0, 255, 0), 2)
    
    # Convert BGR back to RGB for matplotlib display
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    masked_vis_img_rgb = cv2.cvtColor(masked_vis_img, cv2.COLOR_BGR2RGB)

    return result_img_rgb, combined_peaks, cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), masked_vis_img_rgb


def benchmark_inference_speed(model, input_tensor, model_name="Model", num_runs=50, warmup_runs=10):
    """Benchmark inference speed for a model."""
    print(f"\n=== {model_name} Speed Benchmark ===")
    
    # Warmup
    print(f"Running {warmup_runs} warmup iterations...")
    for _ in range(warmup_runs):
        if hasattr(model, 'infer'):  # TensorRT
            _ = model.infer(input_tensor)
        else:  # PyTorch
            with torch.no_grad():
                _ = model(torch.from_numpy(input_tensor).cuda())
    
    # Benchmark
    print(f"Running {num_runs} benchmark iterations...")
    if hasattr(model, 'infer'):  # TensorRT
        start_time = time.time()
        for _ in range(num_runs):
            _ = model.infer(input_tensor)
        end_time = time.time()
    else:  # PyTorch
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(torch.from_numpy(input_tensor).cuda())
        torch.cuda.synchronize()
        end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / num_runs) * 1000
    fps = num_runs / total_time
    
    print(f"Total time: {total_time:.3f}s")
    print(f"Average inference time: {avg_time_ms:.2f}ms")
    print(f"FPS: {fps:.1f}")
    
    return {
        'total_time': total_time,
        'avg_time_ms': avg_time_ms,
        'fps': fps
    }

def main():
    """Main demo function."""
    print("=== TensorRT Keypoint Detection Demo ===")
    
    # Configuration
    tensorrt_model_path = "models/keypoint_model_vit_post.trt"
    pytorch_model_path = "models/keypoint_model_vit_post.pth"  # Use post training model
    image_dir = "image_data/RGB-images"
    
    # Check if models exist
    if not os.path.exists(pytorch_model_path):
        print(f"Error: PyTorch model not found: {pytorch_model_path}")
        return
    
    # Find a sample image
    image_files = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
    if not image_files:
        print(f"Error: No images found in {image_dir}")
        return
    
    sample_image = str(image_files[0])
    print(f"Using sample image: {sample_image}")
    
    # Load PyTorch model
    print("\nLoading PyTorch model...")
    try:
        pytorch_model = load_pytorch_model(pytorch_model_path)
        print("✓ PyTorch model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load PyTorch model: {e}")
        return
    
    # Load YOLO segmentation model (bedsheet segmenter)
    segmenter = None
    try:
        from ultralytics import YOLO
        seg_model_path = "models/yolo_finetuned/best.pt"
        if os.path.exists(seg_model_path):
            print("Loading segmentation model for pre-processing...")
            segmenter = YOLO(seg_model_path)
            print("✓ Segmentation model loaded")
        else:
            print("Segmentation model not found; proceeding without segmentation pre-processing")
    except Exception as e:
        print(f"Segmentation model load failed: {e}. Proceeding without segmentation.")

    # Evaluate PyTorch model
    print("\n=== Evaluating PyTorch Model ===")
    try:
        result_img_rgb, keypoints, original_img_rgb, masked_vis_img_rgb = evaluate_single_image_like_post_training(
            pytorch_model, sample_image, segmenter
        )
        print(f"✓ PyTorch evaluation completed. Detected {len(keypoints)} keypoints.")
        
        # Prepare input tensor for benchmarking
        img_rgb = cv2.cvtColor(cv2.imread(sample_image), cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (128, 128))
        img_normalized = img_resized.astype(np.float32)  # No /255 to match training
        benchmark_input = np.transpose(img_normalized, (2, 0, 1))
        benchmark_input = np.expand_dims(benchmark_input, axis=0)
        
        # Benchmark PyTorch speed
        pytorch_stats = benchmark_inference_speed(pytorch_model, benchmark_input, "PyTorch", num_runs=100)
        
    except Exception as e:
        print(f"✗ PyTorch evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test TensorRT if available
    if TENSORRT_AVAILABLE and os.path.exists(tensorrt_model_path):
        print("\n=== Evaluating TensorRT Model ===")
        try:
            print("Loading TensorRT model...")
            tensorrt_model = TensorRTInference(tensorrt_model_path)
            print("✓ TensorRT model loaded successfully")
            
            # Evaluate TensorRT
            trt_result_img_rgb, trt_keypoints, _, trt_masked_vis_img_rgb = evaluate_single_image_tensorrt(
                tensorrt_model, sample_image, segmenter
            )
            print(f"✓ TensorRT evaluation completed. Detected {len(trt_keypoints)} keypoints.")
            
            # Benchmark TensorRT speed
            tensorrt_stats = benchmark_inference_speed(tensorrt_model, benchmark_input, "TensorRT", num_runs=100)
            
            # Display comparison results
            plt.figure(figsize=(24, 10))
            
            # Original image
            plt.subplot(2, 4, 1)
            plt.imshow(original_img_rgb)
            plt.title("Original Image")
            plt.axis('off')
            
            # PyTorch results
            plt.subplot(2, 4, 2)
            plt.imshow(result_img_rgb)
            plt.title(f"PyTorch Results\n{len(keypoints)} keypoints")
            plt.axis('off')
            
            plt.subplot(2, 4, 3)
            plt.imshow(masked_vis_img_rgb)
            plt.title("PyTorch Segmented + Keypoints")
            plt.axis('off')
            
            # Speed comparison
            plt.subplot(2, 4, 4)
            plt.bar(['PyTorch', 'TensorRT'], [pytorch_stats['avg_time_ms'], tensorrt_stats['avg_time_ms']], 
                   color=['blue', 'orange'])
            plt.title('Inference Speed (ms)')
            plt.ylabel('Average Time (ms)')
            for i, (name, stats) in enumerate([('PyTorch', pytorch_stats), ('TensorRT', tensorrt_stats)]):
                plt.text(i, stats['avg_time_ms'] + 0.5, f"{stats['avg_time_ms']:.1f}ms", 
                        ha='center', va='bottom')
            
            # TensorRT results
            plt.subplot(2, 4, 6)
            plt.imshow(trt_result_img_rgb)
            plt.title(f"TensorRT Results\n{len(trt_keypoints)} keypoints")
            plt.axis('off')
            
            plt.subplot(2, 4, 7)
            plt.imshow(trt_masked_vis_img_rgb)
            plt.title("TensorRT Segmented + Keypoints")
            plt.axis('off')
            
            # Summary comparison
            plt.subplot(2, 4, 8)
            speedup = pytorch_stats['avg_time_ms'] / tensorrt_stats['avg_time_ms']
            plt.text(0.1, 0.8, f"PyTorch: {len(keypoints)} keypoints", fontsize=12, transform=plt.gca().transAxes)
            plt.text(0.1, 0.7, f"TensorRT: {len(trt_keypoints)} keypoints", fontsize=12, transform=plt.gca().transAxes)
            plt.text(0.1, 0.6, f"Keypoints match: {'✓' if len(keypoints) == len(trt_keypoints) else '✗'}", 
                    fontsize=12, transform=plt.gca().transAxes)
            plt.text(0.1, 0.4, f"PyTorch: {pytorch_stats['avg_time_ms']:.1f}ms", fontsize=12, transform=plt.gca().transAxes)
            plt.text(0.1, 0.3, f"TensorRT: {tensorrt_stats['avg_time_ms']:.1f}ms", fontsize=12, transform=plt.gca().transAxes)
            plt.text(0.1, 0.2, f"Speedup: {speedup:.2f}x", fontsize=12, transform=plt.gca().transAxes, 
                    color='green' if speedup > 1 else 'red', weight='bold')
            plt.text(0.1, 0.1, f"PyTorch FPS: {pytorch_stats['fps']:.1f}", fontsize=10, transform=plt.gca().transAxes)
            plt.text(0.1, 0.05, f"TensorRT FPS: {tensorrt_stats['fps']:.1f}", fontsize=10, transform=plt.gca().transAxes)
            plt.title("Performance Summary")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
            print(f"\n=== Performance Summary ===")
            print(f"PyTorch  - Avg: {pytorch_stats['avg_time_ms']:.2f}ms, FPS: {pytorch_stats['fps']:.1f}")
            print(f"TensorRT - Avg: {tensorrt_stats['avg_time_ms']:.2f}ms, FPS: {tensorrt_stats['fps']:.1f}")
            print(f"Speedup: {speedup:.2f}x")
            
        except Exception as e:
            print(f"✗ TensorRT evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Show only PyTorch results
            plt.figure(figsize=(20, 8))
            
            plt.subplot(1, 3, 1)
            plt.imshow(original_img_rgb)
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(result_img_rgb)
            plt.title(f"PyTorch Results\n{len(keypoints)} keypoints")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(masked_vis_img_rgb)
            plt.title("PyTorch Segmented + Keypoints")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
    else:
        print("\n=== TensorRT not available ===")
        # Show only PyTorch results
        plt.figure(figsize=(20, 8))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_img_rgb)
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(result_img_rgb)
        plt.title(f"PyTorch Results\n{len(keypoints)} keypoints")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(masked_vis_img_rgb)
        plt.title("PyTorch Segmented + Keypoints")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    print("\n=== Demo Completed Successfully! ===")
    print("Results and speed comparison displayed using matplotlib.")

if __name__ == "__main__":
    main()
