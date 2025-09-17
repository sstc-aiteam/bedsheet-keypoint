#!/usr/bin/env python3
"""
Post-Processing Meta CLIP-Style Keypoint Detection Model Training

This script implements post-training for the Meta CLIP heatmap model using bedsheet data.
It loads the pre-trained Meta CLIP model from cloth data and applies additional LoRA fine-tuning
on real bedsheet images with keypoint annotations.
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
from src.utils.model_utils import kl_heatmap_loss, batch_gaussian_blur
from shared.functions import get_keypoints_for_image, resize_image_and_keypoints

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
    "pretrained_model_path": "models/meta_clip_style_cloth",  # Path to pre-trained Meta CLIP model
    "output_dir": "models/meta_clip_style_bedsheet_post",
    "results_dir": "results_meta_clip_bedsheet_post",
    
    # Data configuration
    "keypoints_data_srcs": [
        "via_proj/bedsheets"
    ],
    "image_paths": [
        "image_data/RGB-images",
        "image_data/RGB-images2"
    ],
    "yolo_model_path": "models/yolo_finetuned/best.pt",
    "allowed_classes": [1],  # bedsheet class
    "image_size": 256,
    
    # Training configuration
    "batch_size": 4,
    "num_epochs": 20,
    "learning_rate": 1e-4,  # Lower LR for post-training
    "weight_decay": 1e-4,
    "use_fp16": True,
    "use_lora": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    
    # Text prior configuration
    "use_text_prior": True,
    "prior_prompts": [
        "a photo of a bedsheet corner",
        "bedsheet corner point",
        "sharp bedsheet corner",
        "textile edge corner",
        "fabric fold corner",
        "bedsheet seam corner"
    ],
    "negative_prompts": [
        "smooth bedsheet surface",
        "flat textile area",
        "bedsheet without corners",
        "fabric center area",
        "bedsheet wrinkle"
    ],
    "prior_weight": 0.5,
    
    # Augmentation configuration
    "use_augmentation": True,
    "use_stronger_augmentation": False,  # More conservative for post-training
    
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

class BedsheetKeypointDataset(Dataset):
    """Dataset for bedsheet keypoint detection with Meta CLIP normalization."""
    
    def __init__(self, img_arr, rgb_img_arr, keypoints_img_arr, file_paths, original_sizes, 
                 image_size=256, transform=None):
        self.img_arr = img_arr
        self.rgb_img_arr = rgb_img_arr
        self.keypoints_img_arr = keypoints_img_arr
        self.file_paths = file_paths
        self.original_sizes = original_sizes
        self.image_size = image_size
        self.transform = transform
        
        # Meta CLIP normalization (same as CLIP for compatibility)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    
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
        
        # Apply Meta CLIP normalization
        img_tensor = (img_tensor - self.mean) / self.std
        
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
    """Augmentation for bedsheet keypoint detection."""
    
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
    """Generate bedsheet dataset data with YOLO masking."""
    img_arr = []
    rgb_img_arr = []
    keypoints_img_arr = []
    file_paths = []
    original_sizes = []
    
    print("Loading bedsheet data...")
    
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
                
                # # Apply YOLO masking on the resized image if available
                # if yolo_model is not None:
                #     try:
                #         # Run YOLO inference on resized image
                #         results = yolo_model(img_resized, task="segment")
                #         if len(results) > 0 and results[0].masks is not None:
                #             # Create mask for bedsheet regions
                #             mask_all = np.zeros((image_size, image_size), dtype=np.uint8)
                #             masks = results[0].masks.data.cpu().numpy()
                #             classes = results[0].boxes.cls.cpu().numpy()
                            
                #             for mask, cls_id in zip(masks, classes):
                #                 if int(cls_id) in allowed_classes:
                #                     # Resize mask to target size (should already be correct size)
                #                     mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
                #                     mask_all = cv2.bitwise_or(mask_all, (mask > 0.5).astype(np.uint8) * 255)
                            
                #             # Apply mask to image (set non-bedsheet regions to black)
                #             if np.any(mask_all > 0):
                #                 img_resized[mask_all == 0] = 0
                                
                #     except Exception as e:
                #         print(f"Warning: YOLO processing failed for {img_path}: {e}")
                
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
    
    print(f"Loaded {len(img_arr)} bedsheet samples")
    return img_arr, rgb_img_arr, keypoints_img_arr, file_paths, original_sizes

def load_pretrained_meta_clip_model(config):
    """Load pre-trained Meta CLIP model from cloth training."""
    print(f"Loading pre-trained Meta CLIP model from {config['pretrained_model_path']}")
    
    # Create model with same configuration as cloth training
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
    
    # Load LoRA adapters if available
    if config['use_lora'] and os.path.exists(os.path.join(pretrained_path, 'adapter_config.json')):
        try:
            from peft import PeftModel
            base_model = model.clip
            model.clip = PeftModel.from_pretrained(base_model, pretrained_path)
            print("✓ Loaded LoRA adapters from pre-trained model")
        except Exception as e:
            print(f"Warning: Could not load LoRA adapters: {e}")
    
    # Load head weights
    head_path = os.path.join(pretrained_path, 'head.pth')
    if os.path.exists(head_path):
        model.head.load_state_dict(torch.load(head_path, map_location='cpu'))
        print("✓ Loaded head weights from pre-trained model")
    else:
        print("Warning: No pre-trained head weights found")
    
    return model

def train_meta_clip_post_model(model, train_loader, val_loader, config):
    """Train Meta CLIP model on bedsheet data."""
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
                gt_blurred = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
                
                # Compute loss
                loss = kl_heatmap_loss(pred_heatmaps, gt_blurred)
            
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
                    gt_blurred = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
                    loss = kl_heatmap_loss(pred_heatmaps, gt_blurred)
                    
                    val_loss += loss.item() * images.size(0)
            
            val_loss /= len(val_loader.dataset)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                
                # Save model
                os.makedirs(config['output_dir'], exist_ok=True)
                
                if config['use_lora']:
                    model.clip.save_pretrained(config['output_dir'])
                torch.save(model.head.state_dict(), os.path.join(config['output_dir'], 'head.pth'))
                
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
    
    from shared.functions import thresholded_locations
    
    detailed_results = []
    total_distances = []
    matched_total = 0
    total_gt_points = 0
    
    print("Evaluating Meta CLIP model on test set...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluation'):
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)
            file_paths = batch['file_path']
            original_sizes = batch['original_size']
            
            # Get predictions
            pred_heatmaps = model(images)
            pred_heatmap = pred_heatmaps[0, 0].cpu().numpy()
            
            # Normalize heatmap
            if pred_heatmap.max() > 0:
                pred_heatmap = pred_heatmap / pred_heatmap.max()
            
            # Extract keypoints
            peaks = thresholded_locations(pred_heatmap, threshold=0.3)
            pred_keypoints = [(int(p[1]), int(p[0])) for p in peaks]
            
            # Get ground truth keypoints
            gt_heatmap = keypoints[0, 0].cpu().numpy()
            gt_peaks = thresholded_locations(gt_heatmap, threshold=0.5)
            gt_keypoints = [(int(p[1]), int(p[0])) for p in gt_peaks]
            
            # Compute matching
            matched, distances = match_keypoints(gt_keypoints, pred_keypoints, threshold=10.0)
            
            total_gt_points += len(gt_keypoints)
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
    
    # Save results
    results = {
        'overall_match_rate': match_rate,
        'avg_distance': avg_distance,
        'total_samples': len(detailed_results),
        'details': detailed_results
    }
    
    with open(os.path.join(results_dir, 'meta_clip_evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Meta CLIP Evaluation Results:")
    print(f"  Match Rate: {match_rate:.3f}")
    print(f"  Avg Distance: {avg_distance:.2f} pixels")
    print(f"  Total Samples: {len(detailed_results)}")
    
    return results

def match_keypoints(gt_keypoints, pred_keypoints, threshold=10.0):
    """Match predicted keypoints to ground truth keypoints."""
    matched = 0
    distances = []
    used_pred = set()
    
    for gt_kp in gt_keypoints:
        best_dist = float('inf')
        best_pred_idx = -1
        
        for i, pred_kp in enumerate(pred_keypoints):
            if i in used_pred:
                continue
            
            dist = np.sqrt((gt_kp[0] - pred_kp[0])**2 + (gt_kp[1] - pred_kp[1])**2)
            if dist < best_dist:
                best_dist = dist
                best_pred_idx = i
        
        if best_pred_idx != -1 and best_dist < threshold:
            matched += 1
            used_pred.add(best_pred_idx)
            distances.append(best_dist)
    
    return matched, distances

def save_keypoint_visualization(image, pred_heatmap, gt_heatmap, pred_keypoints, gt_keypoints, save_path):
    """Save visualization of keypoint predictions."""
    # Denormalize image
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    image = (image * std + mean).clamp(0, 1)
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

def main_meta_clip_post_training_pipeline(config: Dict[str, Any]) -> Tuple[ClipHeatmapModel, Dict]:
    """
    Main post-training pipeline for Meta CLIP model on bedsheet data.
    
    Args:
        config: Configuration dictionary containing all training parameters
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    print("=== Post-Training Meta CLIP-Style Keypoint Detection Model on Bedsheet Data ===")
    
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
    
    # Generate bedsheet dataset
    print("Generating bedsheet dataset...")
    img_arr, rgb_img_arr, keypoints_img_arr, file_paths, original_sizes = generate_bedsheet_dataset_data(
        config["keypoints_data_srcs"],
        config["image_paths"],
        yolo_model_finetuned,
        config["allowed_classes"],
        config["image_size"]
    )
    
    print(f"Bedsheet dataset generated: {len(img_arr)} samples")
    
    if len(img_arr) == 0:
        raise ValueError("No bedsheet data found. Check your data paths and keypoint annotations.")
    
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
    
    # Create datasets
    if config.get("use_augmentation", True):
        augmentation = BedsheetAugmentation(config["image_size"])
        train_dataset = BedsheetKeypointDataset(
            img_arr, rgb_img_arr, keypoints_img_arr, file_paths, original_sizes,
            config["image_size"], transform=augmentation
        )
    else:
        train_dataset = base_dataset
    
    val_dataset = base_dataset
    test_dataset = base_dataset
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=False,  # Don't shuffle when using sampler
        sampler=torch.utils.data.SubsetRandomSampler(train_indices.indices)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices.indices)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        sampler=torch.utils.data.SubsetRandomSampler(test_indices.indices)
    )
    
    print(f"Dataset split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Load pre-trained Meta CLIP model
    model = load_pretrained_meta_clip_model(config)
    
    # Train model
    trained_model, history = train_meta_clip_post_model(model, train_loader, val_loader, config)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = evaluate_meta_clip_model(trained_model, test_loader, config["results_dir"], config)
    
    # Save training history
    with open(os.path.join(config["output_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\nMeta CLIP post-training completed!")
    print(f"Model saved to: {config['output_dir']}")
    print(f"Results saved to: {config['results_dir']}")
    
    return trained_model, history

if __name__ == "__main__":
    # Run with default configuration
    config = DEFAULT_CONFIG.copy()
    
    # You can modify config here if needed
    # config["num_epochs"] = 30
    # config["learning_rate"] = 5e-5
    
    model, history = main_meta_clip_post_training_pipeline(config)
