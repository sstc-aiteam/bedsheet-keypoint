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
                'lora_r': 16, 'lora_alpha': 32, 'image_size': 256, 'use_text_prior': True
            }
        elif model_type == 'mattress':  # mattress
            self.model_path = 'models/meta_clip_style_mattress_post_original'
            self.model_config = {
                'lora_r': 16, 'lora_alpha': 32, 'image_size': 256, 'use_text_prior': True
            }
        elif model_type == 'fitted_sheet':  # fitted_sheet
            self.model_path = 'models/meta_clip_style_fitted_sheet_post_original'
            self.model_config = {
                'lora_r': 16, 'lora_alpha': 32, 'image_size': 256, 'use_text_prior': True
            }
        
        self.model = None
        self.yolo_model = None
        self._load_models()
    
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
        yolo_path = 'models/yolo_finetuned/sheet_without_plastic.v8i.yolov11/runs/segment/train/weights/best.pt'
        if os.path.exists(yolo_path):
            self.yolo_model = YOLO(yolo_path)
            print(f"✅ Loaded YOLO model from {yolo_path}")
        else:
            print(f"⚠️  YOLO model not found at {yolo_path}, proceeding without segmentation")
            self.yolo_model = None
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocess image exactly like training script with YOLO segmentation.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed tensor and original size (width, height)
        """
        # Load image with cv2 (same as training)
        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        original_size = (img_rgb.shape[1], img_rgb.shape[0])  # (width, height)
        
        # Apply YOLO segmentation if available (same as training)
        mask_all = None
        if self.yolo_model is not None:
            try:
                results = self.yolo_model(img_rgb, task="segment")
                if len(results) > 0 and results[0].masks is not None:
                    # Get allowed classes based on model type
                    if self.model_type == "bedsheet":
                        allowed_classes = [1,3]
                    elif self.model_type == "mattress":
                        allowed_classes = [0, 1, 2, 3]
                    elif self.model_type == "fitted_sheet":
                        allowed_classes = [1,3]
                    # Create mask for allowed regions
                    mask_all = np.zeros(original_size[::-1], dtype=np.uint8)  # (height, width)
                    masks = results[0].masks.data.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    
                    for mask, cls_id in zip(masks, classes):
                        if int(cls_id) in allowed_classes:
                            mask_binary = (mask > 0.5).astype(np.uint8) * 255
                            mask_resized = cv2.resize(mask_binary, original_size, interpolation=cv2.INTER_NEAREST)
                            mask_all = cv2.bitwise_or(mask_all, mask_resized)
                    
                    if np.any(mask_all > 0):
                        print(f"✅ Applied YOLO segmentation for classes {allowed_classes}")
                    else:
                        mask_all = None
            except Exception as e:
                print(f"⚠️  YOLO processing failed: {e}")
        
        # Resize to model input size
        target_size = self.model_config['image_size']
        img_resized = cv2.resize(img_rgb, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # Convert to float
        image_array = img_resized.astype(np.float32)
        
        # Apply mask if available
        if mask_all is not None:
            # Resize mask to target size
            mask_resized = cv2.resize(mask_all, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            # Apply mask to resized image
            image_array[mask_resized == 0] = 0
        plt.imshow(image_array)
        plt.show()
        test_image = img_rgb.copy()
        test_image[mask_all == 0] = 0
        plt.imshow(test_image)
        plt.show()
        
        # No normalization applied (use raw pixel values)
        
        # Convert to tensor and add batch dimension (normalize to 0-1 range)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device), original_size
    
    def predict_keypoints(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Predict heatmap from image tensor."""
        with torch.no_grad():
            heatmap = self.model(image_tensor)
        return heatmap
    
    def postprocess_heatmap(self, heatmap: torch.Tensor, original_size: Tuple[int, int]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Postprocess heatmap exactly like training evaluation.
        
        Args:
            heatmap: Predicted heatmap tensor
            original_size: Original image size (width, height)
            
        Returns:
            Resized heatmap and keypoint coordinates
        """
        # Convert to numpy (same as training: pred_heatmap = pred_heatmaps[0, 0].cpu().numpy())
        heatmap_np = heatmap.squeeze().cpu().numpy()
        
        # # normalize the heatmap
        # m = heatmap_np.max() if heatmap_np.size > 0 else 1.0
        # if m > 0.0005:
        #     heatmap_np = heatmap_np / m
        # else:
        #     heatmap_np = np.zeros_like(heatmap_np)
        
        # Extract keypoints using EXACT same method as training evaluation
        peaks = thresholded_locations(heatmap_np, threshold=0.3)
        
        # Combine nearby peaks (same as training)
        combined_peaks = combine_nearby_peaks(peaks, distance_threshold=10)
        
        # Convert to keypoint format (same coordinate order as training: (x, y))
        # Use combined_peaks to reduce nearby duplicates
        keypoints = [(p[1], p[0]) for p in combined_peaks]
        
        # Scale keypoints to original size
        scale_x = original_size[0] / heatmap_np.shape[1]
        scale_y = original_size[1] / heatmap_np.shape[0]
        keypoints_scaled = [(int(x * scale_x), int(y * scale_y)) for x, y in keypoints]
        
        # Resize heatmap to original size for visualization
        heatmap_resized = cv2.resize(heatmap_np, original_size, interpolation=cv2.INTER_CUBIC)
        
        return heatmap_resized, keypoints_scaled
    
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
        
        # Preprocess
        image_tensor, original_size = self.preprocess_image(image_path)
        print(f"📐 Input shape: {image_tensor.shape}, Original size: {original_size}")
        
        # Predict
        heatmap = self.predict_keypoints(image_tensor)
        print(f"🔥 Heatmap shape: {heatmap.shape}")
        
        # Postprocess
        heatmap_resized, keypoints = self.postprocess_heatmap(heatmap, original_size)
        print(f"🎯 Found {len(keypoints)} keypoints")
        
        # Visualize
        if output_path is None:
            output_path = f"inference_results/{self.model_type}_keypoints_{os.path.basename(image_path)}.png"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.visualize_results(image_path, heatmap_resized, keypoints, output_path)
        
        return keypoints


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Simple Meta CLIP Keypoint Detection Inference')
    parser.add_argument('--model', choices=['bedsheet', 'mattress', 'fitted_sheet'], default='bedsheet',
                       help='Model type to use')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--output', help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Create inference demo
    demo = SimpleKeypointInference(model_type=args.model)
    
    # Run inference
    keypoints = demo.run_inference(args.image, args.output)
    
    print(f"\n✅ Inference completed!")
    print(f"📊 Found {len(keypoints)} keypoints")
    if keypoints:
        print(f"📍 Keypoint coordinates: {keypoints}")


if __name__ == "__main__":
    main()
