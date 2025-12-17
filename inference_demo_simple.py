#!/usr/bin/env python3
"""
Simplified Meta CLIP Keypoint Detection Inference Demo
Matches training evaluation exactly - under 300 lines
"""

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Optional
import argparse
import time

# Import model and utilities
from src.models.clip_heatmap_model import ClipHeatmapModel
from shared.functions import thresholded_locations, combine_nearby_peaks
from ultralytics import YOLO


class SimpleKeypointInference:
    """Simplified keypoint inference that matches training evaluation exactly."""
    
    def __init__(self, model_type: str = 'bedsheet'):
        """
        Initialize inference demo.
        
        Args:
            model_type: 'bedsheet' or 'mattress' or 'fitted_sheet'
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        # Model configurations
        if model_type == 'bedsheet':
            self.model_path = 'models/meta_clip_style_bedsheet_post_original'
            self.model_config = {
                'lora_r': 16, 'lora_alpha': 32, 'image_size': 560, 'use_text_prior': True
            }
        elif model_type == 'mattress':  # mattress
            self.model_path = 'models/meta_clip_style_mattress_post_original'
            self.model_config = {
                'lora_r': 16, 'lora_alpha': 32, 'image_size': 560, 'use_text_prior': True
            }
        elif model_type == 'fitted_sheet_inverse':  # fitted_sheet
            self.model_path = 'models/meta_clip_style_fitted_sheet_inverse_post_original'
            self.model_config = {
                'lora_r': 16, 'lora_alpha': 32, 'image_size': 560, 'use_text_prior': True
            }
        
        self.model = None
        self.yolo_model = None
        self._load_models()

    def _sync_cuda(self) -> None:
        """Synchronize CUDA for accurate timing (no-op on CPU)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
    
    def _load_models(self):
        """Load the trained model and YOLO model."""
        print(f"Loading {self.model_type} model from {self.model_path}")
        
        # Create model with same config as training
        self.model = ClipHeatmapModel(
            model_name='facebook/metaclip-b16-fullcc2.5b',
            image_size=self.model_config['image_size'],
            use_lora=True,
            lora_r=self.model_config['lora_r'],
            lora_alpha=self.model_config['lora_alpha'],
            use_text_prior=self.model_config['use_text_prior']
        )
        
        # Load complete model weights
        complete_model_path = os.path.join(self.model_path, 'complete_model.pth')
        if not os.path.exists(complete_model_path):
            raise FileNotFoundError(f"Complete model not found at: {complete_model_path}")
        
        checkpoint = torch.load(complete_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ Loaded {self.model_type} model successfully")
        
        # Load YOLO model for segmentation (same as training)
        yolo_candidates = [
            # Preferred (matches training scripts)
            "models/yolo_finetuned/sheet_without_plastic.v11i.yolov11/runs/segment/train/weights/best.pt",
            # Backward/alternate path (if older run exists)
            "models/yolo_finetuned/sheet_without_plastic.v13i.yolov11/runs/segment/train/weights/best.pt",
        ]
        yolo_path = next((p for p in yolo_candidates if os.path.exists(p)), None)
        if yolo_path is not None:
            self.yolo_model = YOLO(yolo_path)
            print(f"✅ Loaded YOLO model from {yolo_path}")
        else:
            print(f"⚠️  YOLO model not found (tried: {yolo_candidates}), proceeding without segmentation")
            self.yolo_model = None
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Tuple[int, int], Optional[float]]:
        """
        Preprocess image exactly like training script with YOLO segmentation.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed tensor, original size (width, height), and optional YOLO latency (ms)
        """
        # Load image with cv2 (same as training)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        original_size = (img_rgb.shape[1], img_rgb.shape[0])  # (width, height)


        # Resize to model input size
        target_size = self.model_config['image_size']
        img_resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        # Default: keep everything (no segmentation)
        mask_all = np.ones((target_size, target_size), dtype=np.uint8) * 255
        yolo_ms: Optional[float] = None
        
        # Apply YOLO segmentation if available (same as training)
        if self.yolo_model is not None:
            self._sync_cuda()
            t0 = time.perf_counter()
            # Run YOLO inference on resized image
            results = self.yolo_model(img_resized)
            self._sync_cuda()
            yolo_ms = (time.perf_counter() - t0) * 1000.0
            if len(results) > 0 and results[0].masks is not None:
                # Get allowed classes based on model type
                if self.model_type == "bedsheet":
                    allowed_classes = [1]
                elif self.model_type == "mattress":
                    allowed_classes = [0, 1, 2, 3]
                elif self.model_type == "fitted_sheet_inverse":
                    allowed_classes = [1]
                # Create mask for fitted_sheet regions
                mask_all = np.zeros((self.model.image_size, self.model.image_size), dtype=np.uint8)
                masks = results[0].masks.data.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                
                for mask, cls_id in zip(masks, classes):
                    if int(cls_id) in allowed_classes:
                        # Resize mask to target size (should already be correct size)
                        mask = cv2.resize(mask, (self.model.image_size, self.model.image_size), interpolation=cv2.INTER_NEAREST)
                        mask_all = cv2.bitwise_or(mask_all, (mask > 0.5).astype(np.uint8) * 255)
                
                # Apply mask to image (set non-fitted_sheet regions to black)
                img_resized[mask_all == 0] = 0
        
        # No normalization applied (use raw pixel values)
        
        # Convert to tensor and add batch dimension (normalize to 0-1 range)
        image_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device), original_size, yolo_ms
    
    def predict_keypoints(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Predict heatmap from image tensor."""
        with torch.no_grad():
            heatmap = self.model(image_tensor)
        return heatmap
    
    def visualize_results(self, image_path: str, heatmap: np.ndarray, keypoints: List[Tuple[int, int]], 
                         output_path: Optional[str] = None) -> None:
        """Visualize keypoint detection results."""
        image = Image.open(image_path).convert('RGB')
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
            axes[2].set_title(f'Keypoints ({len(keypoints)} found)', fontsize=14, fontweight='bold')
        else:
            axes[2].set_title('No Keypoints Found', fontsize=14, fontweight='bold')
        
        axes[2].axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"💾 Results saved to: {output_path}")
        else:
            plt.show()
        plt.close()
    
    def run_inference(self, image_path: str, output_path: Optional[str] = None) -> List[Tuple[int, int]]:
        """
        Run complete inference pipeline.
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save visualization
            
        Returns:
            List of detected keypoints as (x, y) tuples
        """
        print(f"🔍 Running inference on: {image_path}")

        total_t0 = time.perf_counter()
        
        # Preprocess
        image_tensor, original_size, yolo_ms = self.preprocess_image(image_path)
        print(f"📐 Input shape: {image_tensor.shape}, Original size: {original_size}")
        
        # Predict
        self._sync_cuda()
        model_t0 = time.perf_counter()
        heatmap = self.predict_keypoints(image_tensor)
        self._sync_cuda()
        model_ms = (time.perf_counter() - model_t0) * 1000.0
        
        post_t0 = time.perf_counter()
        pred_heatmap = heatmap[0, 0].detach().cpu().numpy()
        
        # Extract keypoints from predicted heatmap
        pred_peaks = thresholded_locations(pred_heatmap, threshold=0.1)
        # Combine nearby peaks to reduce duplicates
        combined_peaks = combine_nearby_peaks(pred_peaks, distance_threshold=10)
        scale_x = original_size[0] / pred_heatmap.shape[1]
        scale_y = original_size[1] / pred_heatmap.shape[0]

        # Scale keypoints to original size
        pred_keypoints = [(int(p[1] * scale_x), int(p[0] * scale_y)) for p in combined_peaks]  # Convert to (x, y)

        print(f"🎯 Found {len(pred_keypoints)} keypoints")
        heatmap_resized = cv2.resize(pred_heatmap, original_size, interpolation=cv2.INTER_CUBIC)
        keypoints = pred_keypoints
        post_ms = (time.perf_counter() - post_t0) * 1000.0
        total_ms = (time.perf_counter() - total_t0) * 1000.0

        if yolo_ms is not None:
            print(f"⏱️  Latency (single): YOLO={yolo_ms:.1f} ms | Heatmap={model_ms:.1f} ms | Post={post_ms:.1f} ms | Total={total_ms:.1f} ms")
        else:
            print(f"⏱️  Latency (single): Heatmap={model_ms:.1f} ms | Post={post_ms:.1f} ms | Total={total_ms:.1f} ms")
               
        # Visualize
        if output_path is None:
            output_path = f"inference_results/{self.model_type}_keypoints_{os.path.basename(image_path)}.png"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.visualize_results(image_path, heatmap_resized, keypoints, output_path)
        
        return keypoints


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Simple Meta CLIP Keypoint Detection Inference')
    parser.add_argument('--model', choices=['bedsheet', 'mattress', 'fitted_sheet_inverse'], default='bedsheet',
                        help='Model type to use')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', help='Path to a single input image')
    group.add_argument('--folder', help='Path to a folder of images (jpg/png/jpeg)')
    parser.add_argument('--output', help='Output path for visualization (single image). '
                                         'For folders, results are saved beside inputs in inference_results/.')
    
    args = parser.parse_args()
    
    # Create inference demo
    demo = SimpleKeypointInference(model_type=args.model)
    
    # Gather image paths
    if args.image:
        image_paths = [args.image]
    else:
        supported_ext = ('.jpg', '.jpeg', '.png')
        image_paths = [
            os.path.join(args.folder, f)
            for f in os.listdir(args.folder)
            if f.lower().endswith(supported_ext)
        ]
        if not image_paths:
            raise FileNotFoundError(f"No images with extensions {supported_ext} found in {args.folder}")
    
    # Run inference for each image
    for img_path in image_paths:
        # For batch mode, place outputs in inference_results/<model_type>/filename.png
        if args.folder and args.output is None:
            out_dir = os.path.join('inference_results', demo.model_type)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{os.path.basename(img_path)}.png")
        else:
            out_path = args.output
        
        keypoints = demo.run_inference(img_path, out_path)
        print(f"\n✅ {img_path}: {len(keypoints)} keypoints")
        if keypoints:
            print(f"📍 {keypoints}")


if __name__ == "__main__":
    main()
