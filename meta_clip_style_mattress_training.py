#!/usr/bin/env python3
"""
Post-Processing Meta CLIP-Style Keypoint Detection Model Training

This script implements post-training for the Meta CLIP heatmap model using mattress data.
It loads the pre-trained Meta CLIP model from cloth data and applies additional LoRA fine-tuning
on real mattress images with keypoint annotations.
"""

import os
import sys
import time
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import json
import pandas as pd
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from ultralytics import YOLO
from tqdm import tqdm
import torchvision.transforms as transforms

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import CLIP model and utilities
from src.models import ClipHeatmapModel, create_clip_heatmap_model
from src.utils.model_utils import kl_heatmap_loss, batch_gaussian_blur, normalize_heatmaps
from shared.functions import get_keypoints_for_image, resize_image_and_keypoints

# Import simple augmentation
from src.augmentation.simple_lighting_color_augmentation import create_simple_lighting_color_augmentation

# TensorRT utilities
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("TensorRT not available. Install with: pip install tensorrt")

# Default configuration for post-training
DEFAULT_CONFIG = {
    # Model configuration
    "model_name": "facebook/metaclip-b16-fullcc2.5b",  # Meta CLIP model
    "use_original_metaclip": False,  # Set to True to use original Meta CLIP instead of pre-trained
    "ensure_equal_params": True,  # Ensure both models have identical trainable parameters
    "output_dir": "models/meta_clip_style_mattress_post",
    "results_dir": "results_meta_clip_mattress_post",
    
    # Data configuration
    "keypoints_data_srcs": [
        "via_proj/mattress"
    ],
    "image_paths": [
        "image_data/mattress1", "image_data/mattress2"
    ],
    "yolo_model_path": "models/yolo_finetuned/sheet_without_plastic.v7i.yolov11/runs/segment/train/weights/best.pt",
    "allowed_classes": [0,1,2,3,4,5,6],  # mattress class
    "image_size": 256,
    
    # Training configuration
    "batch_size": 4,
    "num_epochs": 20,
    "learning_rate": 3e-4,  # Match original training LR for fair comparison
    "weight_decay": 1e-4,
    "use_fp16": True,
    "use_lora": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    
    # Text prior configuration
    "use_text_prior": True,
    "prior_prompts": [
        "a photo of a mattress corner",
        "mattress corner point",
        "sharp mattress corner"
    ],
    "negative_prompts": [
        "smooth bedsheet surface",
        "flat textile area",
        "bedsheet without corners",
        "fabric center area",
        "bedsheet wrinkle"
    ],
    "prior_weight": 0.5,
    
    # Enhanced augmentation configuration
    "use_augmentation": True,
    "augmentation_intensity": "medium",  # 'light', 'medium', 'strong'
    "use_lighting_augmentation": True,
    "use_color_augmentation": True,
    
    # Early stopping and saving
    "early_stopping_patience": 15,
    "save_best_model": True,
    "save_frequency": 5,
    
    # Evaluation configuration
    "evaluation_frequency": 2,
    "visualization_frequency": 5,
    
    # TensorRT configuration
    "convert_to_tensorrt": False,
    "tensorrt_precision": "fp16",
    
    # Random seed
    "seed": 42
}

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def compute_sigma(H):
    return max(1.0, 0.03 * H)

class BedsheetKeypointDataset(Dataset):
    """Dataset for mattress keypoint detection with Meta CLIP normalization."""
    
    def __init__(self, img_arr, rgb_img_arr, keypoints_img_arr, file_paths, original_sizes, 
                 image_size=256, transform=None):
        self.img_arr = img_arr
        self.rgb_img_arr = rgb_img_arr
        self.keypoints_img_arr = keypoints_img_arr
        self.file_paths = file_paths
        self.original_sizes = original_sizes
        self.image_size = image_size
        self.transform = transform
        
        # No normalization - use raw pixel values
    
    def __len__(self):
        return len(self.img_arr)
    
    def __getitem__(self, idx):
        # Get image and keypoints
        img = self.img_arr[idx]
        keypoints_img = self.keypoints_img_arr[idx]
        file_path = self.file_paths[idx]
        original_size = self.original_sizes[idx]
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        keypoints_tensor = torch.from_numpy(keypoints_img).unsqueeze(0).float()
        
        # No normalization applied
        
        # Apply augmentation if provided
        if self.transform:
            # Apply augmentation to both image and keypoints separately
            img_tensor, keypoints_tensor = self.transform(img_tensor, keypoints_tensor)
        
        return {
            'image': img_tensor,
            'keypoints': keypoints_tensor,
            'file_path': file_path,
            'original_size': original_size
        }

class BedsheetAugmentation:
    """Augmentation for mattress keypoint detection."""
    
    def __init__(self, image_size=256):
        self.image_size = image_size
    
    def __call__(self, img_tensor, keypoints_tensor):
        """Apply augmentation to image and keypoints separately."""
        # Convert to numpy for easier manipulation
        img = img_tensor.permute(1, 2, 0).numpy()
        keypoints = keypoints_tensor.squeeze(0).numpy()  # Remove channel dimension for 2D processing
        
        # Random horizontal flip
        if random.random() < 0.5:
            img = np.fliplr(img)
            keypoints = np.fliplr(keypoints)
        
        # Random rotation (±10 degrees)
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (w, h))
            keypoints = cv2.warpAffine(keypoints, rotation_matrix, (w, h))
        
        # Random brightness/contrast
        if random.random() < 0.3:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.uniform(-0.1, 0.1)  # brightness
            img = np.clip(alpha * img + beta, 0, 1)
        
        # Convert back to tensor (ensure contiguous arrays)
        img_tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float()  # (3, H, W)
        keypoints_tensor = torch.from_numpy(keypoints.copy()).unsqueeze(0).float()  # (1, H, W)
        
        return img_tensor, keypoints_tensor

def generate_bedsheet_dataset_data(keypoints_data_srcs, image_paths, yolo_model, allowed_classes, image_size):
    """Generate mattress dataset data with YOLO masking."""
    img_arr = []
    rgb_img_arr = []
    keypoints_img_arr = []
    file_paths = []
    original_sizes = []
    
    print("Loading mattress data...")
    
    for keypoints_src in keypoints_data_srcs:
        if not os.path.exists(keypoints_src):
            print(f"Warning: Keypoints source {keypoints_src} does not exist")
            continue
        
        # Get all JSON files in the keypoints directory
        if os.path.isdir(keypoints_src):
            json_files = [f for f in os.listdir(keypoints_src) if f.endswith('.json')]
            if not json_files:
                print(f"Warning: No JSON files found in {keypoints_src}")
                continue
        else:
            json_files = [os.path.basename(keypoints_src)]
            keypoints_src = os.path.dirname(keypoints_src)
            
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"Warning: Image path {image_path} does not exist")
                continue
                
            # Get all image files in the image path
            image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                # Try to find keypoints in any of the JSON files
                keypoints = None
                for json_file in json_files:
                    json_path = os.path.join(keypoints_src, json_file)
                    keypoints = get_keypoints_for_image(img_file, json_path)
                    if keypoints is not None:
                        break
                
                if keypoints is None:
                    continue
                img_path = os.path.join(image_path, img_file)
                
                if not os.path.exists(img_path):
                    continue
                
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                original_size = img.shape[:2]
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # First, resize the entire image and keypoints to target size
                img_resized, keypoints_resized = resize_image_and_keypoints(
                    img_rgb, keypoints, image_size, image_size
                )
                # Apply YOLO masking on the resized image if available
                if yolo_model is not None:
                    try:
                        # Run YOLO inference on resized image
                        results = yolo_model(img_resized)
                        if len(results) > 0 and results[0].masks is not None:
                            # Create mask for fitted_sheet regions
                            mask_all = np.zeros((image_size, image_size), dtype=np.uint8)
                            masks = results[0].masks.data.cpu().numpy()
                            classes = results[0].boxes.cls.cpu().numpy()
                            
                            for mask, cls_id in zip(masks, classes):
                                if int(cls_id) in allowed_classes:
                                    # Resize mask to target size (should already be correct size)
                                    mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                                    mask_all = cv2.bitwise_or(mask_all, (mask > 0.5).astype(np.uint8) * 255)
                            
                            # Apply mask to image (set non-fitted_sheet regions to black)
                            img_resized[mask_all == 0] = 0
                                
                    except Exception as e:
                        print(f"Warning: YOLO processing failed for {img_path}: {e}")
                
                # Create keypoint heatmap
                keypoints_img = np.zeros((image_size, image_size), dtype=np.float32)
                for kp in keypoints_resized:
                    x, y = int(kp[0]), int(kp[1])
                    if 0 <= x < image_size and 0 <= y < image_size:
                        keypoints_img[y, x] = 1.0
                
                # Add to arrays
                img_arr.append(img_resized)
                rgb_img_arr.append(img_resized)
                keypoints_img_arr.append(keypoints_img)
                file_paths.append(img_path)
                original_sizes.append(original_size)
    
    print(f"Loaded {len(img_arr)} mattress samples")
    return img_arr, rgb_img_arr, keypoints_img_arr, file_paths, original_sizes

def load_pretrained_meta_clip_model(config):
    """Load Meta CLIP model with equal trainable parameters for fair comparison."""
    
    print("=" * 60)
    print("LOADING META CLIP MODEL (WITH EQUAL PARAMETERS)")
    print("=" * 60)
    
    if config.get('use_original_metaclip', False):
        print("Creating original Meta CLIP model...")
        
        # Create original model
        model = create_clip_heatmap_model(
            model_name=config['model_name'],
            image_size=config['image_size'],
            use_lora=config['use_lora'],
            lora_r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            use_text_prior=config['use_text_prior'],
            prior_prompts=config['prior_prompts'],
            negative_prompts=config['negative_prompts'],
            prior_weight=config['prior_weight']
        )
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"Original model created:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable ratio: {trainable_params/total_params:.2%}")
        
        return model
        
    else:
        print("Creating pre-trained Meta CLIP model with equal parameters...")
        
        # First, create original model to get target parameter count
        if config.get('ensure_equal_params', True):
            print("Step 1: Creating reference original model for parameter count...")
            original_model = create_clip_heatmap_model(
                model_name=config['model_name'],
                image_size=config['image_size'],
                use_lora=config['use_lora'],
                lora_r=config['lora_r'],
                lora_alpha=config['lora_alpha'],
                lora_dropout=config['lora_dropout'],
                use_text_prior=config['use_text_prior'],
                prior_prompts=config['prior_prompts'],
                negative_prompts=config['negative_prompts'],
                prior_weight=config['prior_weight']
            )
            
            target_trainable = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
            print(f"Target trainable parameters: {target_trainable:,}")
            
            # Clean up reference model
            del original_model
        
        # Now create pre-trained model
        print("Step 2: Creating pre-trained model...")
        model = create_clip_heatmap_model(
            model_name=config['model_name'],
            image_size=config['image_size'],
            use_lora=config['use_lora'],
            lora_r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            use_text_prior=config['use_text_prior'],
            prior_prompts=config['prior_prompts'],
            negative_prompts=config['negative_prompts'],
            prior_weight=config['prior_weight']
        )
        
        # Load pre-trained weights
        pretrained_path = config['pretrained_model_path']
        
        # Load complete pretrained model (includes all trainable parameters)
        complete_model_path = os.path.join(pretrained_path, 'complete_model.pth')
        pretrained_loaded = False
        
        if os.path.exists(complete_model_path):
            try:
                print("Loading complete pretrained model...")
                pretrained_state_dict = torch.load(complete_model_path, map_location='cpu')
                model.load_state_dict(pretrained_state_dict)
                print("✓ Loaded complete pre-trained model (all trainable parameters)")
                pretrained_loaded = True
            except Exception as e:
                print(f"✗ Failed to load complete model: {e}")
                print("This might be due to model architecture differences")
        
        if not pretrained_loaded:
            print("Falling back to loading LoRA adapters and head separately...")
            
            # Load LoRA adapters
            if config['use_lora']:
                adapter_config_path = os.path.join(pretrained_path, 'adapter_config.json')
                if os.path.exists(adapter_config_path):
                    try:
                        from peft import PeftModel
                        base_model = model.clip
                        model.clip = PeftModel.from_pretrained(base_model, pretrained_path)
                        print("✓ Loaded LoRA adapters using PEFT")
                    except Exception as e:
                        print(f"✗ Failed to load LoRA adapters: {e}")
                else:
                    print("✗ No LoRA adapter config found")
            
            # Load complete model weights (includes head)
            complete_model_path = os.path.join(pretrained_path, 'complete_model.pth')
            if os.path.exists(complete_model_path):
                try:
                    model.load_state_dict(torch.load(complete_model_path, map_location='cpu'))
                    print("✓ Loaded complete model weights from pretrained model")
                except Exception as e:
                    print(f"✗ Failed to load complete model weights: {e}")
            else:
                print("✗ No complete model weights found")
        
        # Count actual trainable parameters
        actual_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"Pre-trained model created:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {actual_trainable:,}")
        print(f"  Trainable ratio: {actual_trainable/total_params:.2%}")
        
        # Report final parameter counts
        print(f"Final model state:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {actual_trainable:,}")
        print(f"  Trainable ratio: {actual_trainable/total_params:.2%}")
        
        if pretrained_loaded:
            print("✅ Using complete pretrained model with all learned parameters")
        else:
            print("⚠️  Using fallback loading (LoRA + head separately)")
            if config.get('ensure_equal_params', True):
                print(f"Target trainable parameters: {target_trainable:,}")
                if actual_trainable == target_trainable:
                    print("✅ SUCCESS: Parameter count matches target!")
                else:
                    print(f"⚠️  Parameter count difference: {abs(actual_trainable - target_trainable):,}")
        
        return model

def train_meta_clip_post_model(model, train_loader, val_loader, config):
    """Train Meta CLIP model on mattress data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimizer (only train LoRA parameters and head)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scaler = GradScaler(enabled=config['use_fp16'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    patience = 0
    
    print(f"Starting Meta CLIP post-training for {config['num_epochs']} epochs...")
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]')
        for batch in train_pbar:
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)
            
            optimizer.zero_grad()
            
            with autocast("cuda", dtype=torch.float16, enabled=config['use_fp16']):
                # Forward pass
                pred_heatmaps = model(images)
                
                # Apply Gaussian blur to ground truth
                sigma = compute_sigma(pred_heatmaps.shape[-1])
                k = int(6 * sigma + 1)
                k = k if k % 2 == 1 else k + 1
                gt_blurred = batch_gaussian_blur(keypoints, kernel_size=min(k, 61), sigma=float(sigma))
                
                # Compute loss
                loss = kl_heatmap_loss(normalize_heatmaps(pred_heatmaps), gt_blurred)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * images.size(0)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        # Validation phase
        if epoch % config['evaluation_frequency'] == 0:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]'):
                    images = batch['image'].to(device)
                    keypoints = batch['keypoints'].to(device)
                    
                    pred_heatmaps = model(images)
                    # Apply Gaussian blur to ground truth
                    sigma = compute_sigma(pred_heatmaps.shape[-1])
                    k = int(6 * sigma + 1)
                    k = k if k % 2 == 1 else k + 1
                    gt_blurred = batch_gaussian_blur(keypoints, kernel_size=min(k, 61), sigma=float(sigma))
                    loss = kl_heatmap_loss(normalize_heatmaps(pred_heatmaps), gt_blurred)
                    
                    val_loss += loss.item() * images.size(0)
            
            val_loss /= len(val_loader.dataset)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                
                # Save model
                os.makedirs(config['output_dir'], exist_ok=True)
                
                # Save complete model as .pth file (includes LoRA adapters and head)
                torch.save(model.state_dict(), os.path.join(config['output_dir'], 'complete_model.pth'))
                
                # Also save LoRA adapters separately for compatibility
                if config['use_lora']:
                    model.clip.save_pretrained(config['output_dir'])
                
                
                # Save config
                with open(os.path.join(config['output_dir'], 'training_config.json'), 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"✓ Saved best model (val_loss={val_loss:.4f})")
            else:
                patience += 1
            
            # Early stopping
            if patience >= config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rate'].append(scheduler.get_last_lr()[0])
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")
    
    return model, history

def evaluate_meta_clip_model(model, test_loader, results_dir, config):
    """Evaluate Meta CLIP model on test set."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    os.makedirs(results_dir, exist_ok=True)
    
    from src.utils.keypoint_metrics import calculate_keypoint_match_rate
    
    detailed_results = []
    total_distances = []
    matched_total = 0
    total_gt_points = 0
    total_test_loss = 0.0
    num_batches = 0
    
    print("Evaluating Meta CLIP model on test set...")
    
    # Import loss function
    from src.utils.model_utils import kl_heatmap_loss
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluation'):
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)
            file_paths = batch['file_path']
            original_sizes = batch['original_size']
            
            # Get predictions
            pred_heatmaps = model(images)
            pred_heatmap = pred_heatmaps[0, 0].cpu().numpy()
            
            # Calculate test loss
            if pred_heatmaps.dim() == 3:
                pred_heatmaps = pred_heatmaps.unsqueeze(1)
            if keypoints.dim() == 3:
                keypoints = keypoints.unsqueeze(1)

                # Apply Gaussian blur to ground truth
            sigma = compute_sigma(pred_heatmaps.shape[-1])
            k = int(6 * sigma + 1)
            k = k if k % 2 == 1 else k + 1
            gt_blurred = batch_gaussian_blur(keypoints, kernel_size=min(k, 61), sigma=float(sigma))
            
            test_loss = kl_heatmap_loss(normalize_heatmaps(pred_heatmaps), gt_blurred)
            total_test_loss += test_loss.item()
            num_batches += 1
            
            # Get ground truth heatmap
            gt_heatmap = keypoints[0, 0].cpu().numpy()
            
            # Calculate match rate using streamlined function
            match_result = calculate_keypoint_match_rate(
                gt_heatmap, pred_heatmap,
                gt_threshold=0.5, pred_threshold=0.3,
                match_threshold=10.0, combine_distance=10.0
            )
            
            matched = match_result['matched_count']
            distances = match_result['distances']
            gt_keypoints = match_result['gt_keypoints']
            pred_keypoints = match_result['pred_keypoints']
            
            total_gt_points += match_result['total_gt']
            matched_total += matched
            total_distances.extend(distances)
            
            # Save visualization
            file_name = os.path.basename(file_paths[0])
            save_keypoint_visualization(
                images[0].cpu(), pred_heatmap, gt_heatmap,
                pred_keypoints, gt_keypoints,
                os.path.join(results_dir, f"{file_name}_meta_clip_keypoints.png")
            )
            
            detailed_results.append({
                'file': file_name,
                'matched': matched,
                'total_gt': len(gt_keypoints),
                'avg_distance': np.mean(distances) if distances else None,
                'pred_keypoints': pred_keypoints,
                'gt_keypoints': gt_keypoints
            })
    
    # Compute overall metrics
    match_rate = matched_total / max(1, total_gt_points)
    avg_distance = np.mean(total_distances) if total_distances else None
    avg_test_loss = total_test_loss / max(1, num_batches)
    
    # Save results
    results = {
        'overall_match_rate': match_rate,
        'avg_distance': avg_distance,
        'avg_test_loss': avg_test_loss,
        'total_samples': len(detailed_results),
        'details': detailed_results
    }
    
    with open(os.path.join(results_dir, 'meta_clip_evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Meta CLIP Evaluation Results:")
    print(f"  Match Rate: {match_rate:.3f}")
    print(f"  Avg Distance: {avg_distance:.2f} pixels")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Total Samples: {len(detailed_results)}")
    
    return results


def save_keypoint_visualization(image, pred_heatmap, gt_heatmap, pred_keypoints, gt_keypoints, save_path):
    """Save visualization of keypoint predictions."""
    # Convert image to numpy
    image = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image with GT keypoints
    axes[0].imshow(image)
    for kp in gt_keypoints:
        axes[0].plot(kp[0], kp[1], 'go', markersize=8, markeredgecolor='white', markeredgewidth=2)
    axes[0].set_title('Ground Truth Keypoints')
    axes[0].axis('off')
    
    # Predicted heatmap
    axes[1].imshow(pred_heatmap, cmap='jet')
    axes[1].set_title('Predicted Heatmap')
    axes[1].axis('off')
    
    # Original image with predicted keypoints
    axes[2].imshow(image)
    for kp in pred_keypoints:
        axes[2].plot(kp[0], kp[1], 'ro', markersize=8, markeredgecolor='white', markeredgewidth=2)
    axes[2].set_title('Predicted Keypoints')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def setup_model_directories(config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup output directories based on model type (original vs pre-trained)."""
    if config.get('use_original_metaclip', False):
        config['output_dir'] = "models/meta_clip_style_mattress_post_original"
        config['results_dir'] = "results_meta_clip_mattress_post_original"
        print("Using original Meta CLIP model - output directories:")
    else:
        config['output_dir'] = "models/meta_clip_style_mattress_post_pretrained"
        config['results_dir'] = "results_meta_clip_mattress_post_pretrained"
        print("Using pre-trained Meta CLIP model - output directories:")
    
    print(f"  Output: {config['output_dir']}")
    print(f"  Results: {config['results_dir']}")
    return config

def compare_models_equal_params(config: Dict[str, Any]) -> bool:
    """Compare both models ensuring they have equal trainable parameters."""
    
    print("=" * 80)
    print("META CLIP MODEL COMPARISON WITH EQUAL PARAMETERS")
    print("=" * 80)
    
    # Test original model
    print("\n1. Creating Original Meta CLIP Model:")
    print("-" * 50)
    
    config_original = config.copy()
    config_original['use_original_metaclip'] = True
    
    model_original = load_pretrained_meta_clip_model(config_original)
    params_original = sum(p.numel() for p in model_original.parameters() if p.requires_grad)
    
    # Test pre-trained model
    print("\n2. Creating Pre-trained Meta CLIP Model:")
    print("-" * 50)
    
    config_pretrained = config.copy()
    config_pretrained['use_original_metaclip'] = False
    
    model_pretrained = load_pretrained_meta_clip_model(config_pretrained)
    params_pretrained = sum(p.numel() for p in model_pretrained.parameters() if p.requires_grad)
    
    # Final comparison
    print("\n3. Final Parameter Comparison:")
    print("-" * 50)
    
    print(f"Original model trainable parameters: {params_original:,}")
    print(f"Pre-trained model trainable parameters: {params_pretrained:,}")
    
    if params_original == params_pretrained:
        print("✅ PERFECT MATCH: Both models have identical trainable parameters!")
        print("This ensures a completely fair comparison.")
        success = True
    else:
        print(f"⚠️  Still have parameter difference: {abs(params_original - params_pretrained):,}")
        success = False
    
    # Verify exact parameter matching
    print("\n4. Verifying Exact Parameter Matching:")
    print("-" * 50)
    
    # Get trainable parameter names from both models
    original_trainable_names = set()
    for name, param in model_original.named_parameters():
        if param.requires_grad:
            original_trainable_names.add(name)
    
    pretrained_trainable_names = set()
    for name, param in model_pretrained.named_parameters():
        if param.requires_grad:
            pretrained_trainable_names.add(name)
    
    print(f"Original model trainable parameter groups: {len(original_trainable_names)}")
    print(f"Pre-trained model trainable parameter groups: {len(pretrained_trainable_names)}")
    
    # Check if the exact same parameters are trainable
    if original_trainable_names == pretrained_trainable_names:
        print("✅ PERFECT MATCH: Both models have identical trainable parameter groups!")
        exact_match = True
    else:
        print("⚠️  Parameter group mismatch detected (expected due to PEFT structure):")
        only_in_original = original_trainable_names - pretrained_trainable_names
        only_in_pretrained = pretrained_trainable_names - original_trainable_names
        
        if only_in_original:
            print(f"  Only in original: {len(only_in_original)} groups")
            for name in list(only_in_original)[:2]:  # Show first 2
                print(f"    - {name}")
            if len(only_in_original) > 2:
                print(f"    ... and {len(only_in_original) - 2} more")
        
        if only_in_pretrained:
            print(f"  Only in pre-trained: {len(only_in_pretrained)} groups")
            for name in list(only_in_pretrained)[:2]:  # Show first 2
                print(f"    - {name}")
            if len(only_in_pretrained) > 2:
                print(f"    ... and {len(only_in_pretrained) - 2} more")
        
        # This is expected due to PEFT creating different parameter name structures
        # The important thing is that we have the same number of trainable parameters
        print("  Note: This mismatch is expected due to PEFT's nested parameter structure.")
        print("  The key is that both models have identical trainable parameter counts.")
        exact_match = False
    
    # Test forward pass
    print("\n5. Testing Forward Pass:")
    print("-" * 50)
    
    try:
        dummy_input = torch.randn(1, 3, 256, 256)
        
        model_original.eval()
        with torch.no_grad():
            output_original = model_original(dummy_input)
        print(f"✓ Original model: {output_original.shape}")
        
        model_pretrained.eval()
        with torch.no_grad():
            output_pretrained = model_pretrained(dummy_input)
        print(f"✓ Pre-trained model: {output_pretrained.shape}")
        
        if output_original.shape == output_pretrained.shape:
            print("✅ Both models produce identical output shapes")
        else:
            print(f"⚠️  Output shape mismatch")
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        success = False
    
    print("\n" + "=" * 80)
    if success and exact_match:
        print("✅ MODELS READY FOR FAIR COMPARISON!")
        print("Both models have identical trainable parameters AND parameter groups.")
    elif success:
        print("✅ MODELS READY FOR FAIR COMPARISON!")
        print("Both models have identical trainable parameter counts.")
        print("Parameter group differences are expected due to PEFT structure.")
        print("The pre-trained model benefits from learned head weights.")
    else:
        print("⚠️  MODELS NEED FURTHER ADJUSTMENT")
    print("=" * 80)
    
    # Return success if we have matching parameter counts, even if groups differ
    # The group difference is expected due to PEFT's nested structure
    return success

def main_meta_clip_post_training_pipeline(config: Dict[str, Any]) -> Tuple[ClipHeatmapModel, Dict]:
    """
    Main post-training pipeline for Meta CLIP model on mattress data.
    
    Args:
        config: Configuration dictionary containing all training parameters
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    print("=== Post-Training Meta CLIP-Style Keypoint Detection Model on Bedsheet Data ===")
    
    # Setup output directories based on model type
    config = setup_model_directories(config)
    
    # Set random seeds
    set_random_seeds(config.get("seed", 42))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load YOLO model (if available)
    try:
        yolo_model_finetuned = YOLO(config["yolo_model_path"])
        print("YOLO model loaded for masking")
    except Exception as e:
        print(f"Warning: Could not load YOLO model: {e}")
        yolo_model_finetuned = None
    
    # Generate mattress dataset
    print("Generating mattress dataset...")
    img_arr, rgb_img_arr, keypoints_img_arr, file_paths, original_sizes = generate_bedsheet_dataset_data(
        config["keypoints_data_srcs"],
        config["image_paths"],
        yolo_model_finetuned,
        config["allowed_classes"],
        config["image_size"]
    )
    
    print(f"Bedsheet dataset generated: {len(img_arr)} samples")
    
    if len(img_arr) == 0:
        raise ValueError("No mattress data found. Check your data paths and keypoint annotations.")
    
    # Create base dataset without augmentation
    base_dataset = BedsheetKeypointDataset(
        img_arr, rgb_img_arr, keypoints_img_arr, file_paths, original_sizes,
        config["image_size"], transform=None
    )
    
    # Split dataset into train, validation, and test
    total_size = len(base_dataset)
    train_size = int(0.7 * total_size)  # 70% for training
    val_size = int(0.2 * total_size)    # 20% for validation
    test_size = total_size - train_size - val_size  # 10% for testing
    
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        range(total_size), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create datasets with enhanced augmentation
    if config.get("use_augmentation", True):
        augmentation_intensity = config.get("augmentation_intensity", "medium")
        augmentation = create_simple_lighting_color_augmentation(
            image_size=config["image_size"],
            intensity=augmentation_intensity,
            augmentation_type='mattress'
        )
        train_dataset = BedsheetKeypointDataset(
            img_arr, rgb_img_arr, keypoints_img_arr, file_paths, original_sizes,
            config["image_size"], transform=augmentation
        )
        print(f"Using enhanced augmentation with {augmentation_intensity} intensity")
    else:
        train_dataset = base_dataset
        print("No augmentation applied")
    
    # Create proper subsets using torch.utils.data.Subset
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(base_dataset, val_indices.indices)
    test_subset = torch.utils.data.Subset(base_dataset, test_indices.indices)
    
    # Create data loaders with proper subsets
    train_loader = DataLoader(
        train_subset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=config["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_subset, batch_size=1, shuffle=False
    )
    
    print(f"Dataset split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Load pre-trained Meta CLIP model
    model = load_pretrained_meta_clip_model(config)
    
    # Train model
    trained_model, history = train_meta_clip_post_model(model, train_loader, val_loader, config)
    
    # Load the trained model from saved checkpoint for evaluation
    print("\n=== Loading Trained Model from Checkpoint ===")
    reloaded_model = load_pretrained_meta_clip_model(config)
    
    # Load the trained weights
    checkpoint_path = os.path.join(config["output_dir"], "complete_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Loading trained weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # The checkpoint is saved directly as model state dict, not wrapped
        reloaded_model.load_state_dict(checkpoint)
        print("✅ Trained model loaded successfully")
    else:
        print(f"⚠️  Checkpoint not found at {checkpoint_path}, using in-memory model")
        reloaded_model = trained_model
    
    # Evaluate reloaded trained model on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_meta_clip_model(reloaded_model, test_loader, config["results_dir"], config)
    
    # Save training history
    with open(os.path.join(config["output_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nMeta CLIP post-training completed!")
    print(f"Model saved to: {config['output_dir']}")
    print(f"Results saved to: {config['results_dir']}")
    
    return trained_model, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Meta CLIP post-training with equal parameters")
    parser.add_argument("--use_original", action="store_true", 
                       help="Use original Meta CLIP instead of pre-trained")
    parser.add_argument("--compare", action="store_true",
                       help="Compare both models to verify equal parameters")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--disable_equal_params", action="store_true",
                       help="Disable equal parameters enforcement")
    
    args = parser.parse_args()
    
    # Update config based on arguments
    config = DEFAULT_CONFIG.copy()
    config['use_original_metaclip'] = args.use_original
    config['num_epochs'] = args.epochs
    config['learning_rate'] = args.lr
    config['ensure_equal_params'] = not args.disable_equal_params
    
    if args.compare:
        # Run comparison
        print("Running model comparison...")
        success = compare_models_equal_params(config)
        if not success:
            sys.exit(1)
    else:
        # Run training
        print("Running training...")
        model, history = main_meta_clip_post_training_pipeline(config)
