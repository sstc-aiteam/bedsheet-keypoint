#!/usr/bin/env python3
"""
Post-Processing CLIP-Style Keypoint Detection Model Training for Mattress Data

This script implements post-training for the CLIP heatmap model using mattress data.
It loads the pre-trained CLIP model from cloth data and applies additional LoRA fine-tuning
on real mattress images with keypoint annotations in a hospital bedroom environment.
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

# Default configuration for post-training on mattress data
DEFAULT_CONFIG = {
    "seed": 42,
    "pretrained_model_path": "models/clip_style_cloth",  # Path to pre-trained CLIP model
    "yolo_model_path": "models/yolo_finetuned/best.pt",
    "keypoints_data_srcs": [
        "via_proj/mattress/via_project_10Sep2025_15h14m22s.json"
    ],
    "image_paths": ["image_data/mattress1/"],  # Mattress image directory
    "allowed_classes": [1],
    "image_size": 512,  # Increased for better keypoint precision
    "batch_size": 2,  # Reduced for higher resolution
    "learning_rate": 5e-5,  # Even lower learning rate for better precision
    "weight_decay": 1e-4,
    "num_epochs": 50,
    "lr_step_size": 20,
    "lr_gamma": 0.5,
    "warmup_epochs": 5,
    "model_save_path": "models/clip_style_mattress_post.pth",
    "save_interval": 10,
    "results_dir": "results_clip_mattress_post",
    "freeze_backbone": False,
    "use_augmentation": True,
    "use_fp16": True,
    "gradient_clip_val": 1.0,
    "gradient_accumulation_steps": 1,
    "early_stopping_patience": 15,
    "use_text_prior": True,
    "prior_prompts": [
        "the mattress corner points in hospital bedroom",
        "mattress corner points that might be occluded by other objects"
    ],
    "negative_prompts": [
        "in the middle of the bed"
    ],
    "prior_weight": 0.5,  # Increased weight for better text guidance
    "lora_r": 8,  # Smaller LoRA rank for fine-tuning
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    # Keypoint count regularization
    "keypoint_count_weight": 0.1,  # Weight for penalizing excess keypoints
    "max_keypoints": 4,  # Maximum allowed keypoints (fallback if use_gt_keypoint_count=False)
    "use_gt_keypoint_count": True,  # Use actual GT keypoint count as target
    # TensorRT configuration
    "enable_tensorrt_conversion": True,
    "tensorrt_precision": "fp16",
    "tensorrt_workspace_size": 1 << 30,  # 1GB
    "tensorrt_benchmark": True,
    "tensorrt_num_runs": 100,
    # Visualization configuration
    "keypoint_threshold": 0.3,  # Threshold for keypoint detection in visualization
    "keypoint_size": 30,  # Size of predicted keypoint circles
    "gt_keypoint_size": 30,  # Size of ground truth keypoint circles (same size)
}

# Set random seeds for reproducibility
def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MattressAugmentation:
    """Data augmentation for mattress keypoint detection using heatmap-based spatial transformations."""
    
    def __init__(self, image_size: int = 256):
        self.image_size = image_size
        
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply augmentation to a sample using heatmap-based spatial transformations.
        
        Args:
            sample: Dictionary containing 'pixel_values', 'gt_heatmap', 'gt_points', and 'original_heatmap'
            
        Returns:
            Augmented sample
        """
        pixel_values = sample['pixel_values']  # (3, H, W)
        original_heatmap = sample['original_heatmap']  # Original one-hot encoded heatmap
        
        # Convert to numpy for easier manipulation
        img = pixel_values.numpy().transpose(1, 2, 0).copy()  # (H, W, 3)
        heatmap = original_heatmap.copy()  # (H, W) - already numpy array
        h, w = img.shape[:2]
        
        # Apply spatial transformations to both image and heatmap
        img, heatmap = self._apply_spatial_augmentations(img, heatmap, h, w)
        
        # Apply photometric augmentations to image only
        img = self._apply_photometric_augmentations(img)
        
        # Convert back to torch tensors
        pixel_values = torch.from_numpy(img.transpose(2, 0, 1)).float()
        gt_heatmap = torch.from_numpy(heatmap).unsqueeze(0).float()
        
        sample['pixel_values'] = pixel_values
        sample['gt_heatmap'] = gt_heatmap
        
        return sample
    
    def _apply_spatial_augmentations(self, img: np.ndarray, heatmap: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply spatial transformations to both image and heatmap."""
        
        # Random horizontal flip (50% chance)
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
            heatmap = np.fliplr(heatmap).copy()
        
        # Random rotation (±15 degrees)
        if random.random() < 0.7:  # 70% chance
            angle = random.uniform(-15, 15)
            img, heatmap = self._rotate_image_and_heatmap(img, heatmap, angle)
        
        # Random scaling (0.9 to 1.1)
        if random.random() < 0.6:  # 60% chance
            scale = random.uniform(0.9, 1.1)
            img, heatmap = self._scale_image_and_heatmap(img, heatmap, scale)
        
        return img, heatmap
    
    def _apply_photometric_augmentations(self, img: np.ndarray) -> np.ndarray:
        """Apply photometric augmentations (brightness, contrast, color jitter)."""
        
        # Random brightness adjustment (±20%)
        if random.random() < 0.7:  # 70% chance
            brightness_factor = random.uniform(0.8, 1.2)
            img = np.clip(img * brightness_factor, 0, 1)
        
        # Random contrast adjustment (±20%)
        if random.random() < 0.7:  # 70% chance
            contrast_factor = random.uniform(0.8, 1.2)
            mean = np.mean(img)
            img = np.clip((img - mean) * contrast_factor + mean, 0, 1)
        
        # Random color jitter (small adjustments)
        if random.random() < 0.5:  # 50% chance
            for c in range(3):  # RGB channels
                jitter = random.uniform(-0.1, 0.1)
                img[:, :, c] = np.clip(img[:, :, c] + jitter, 0, 1)
        
        # Random Gaussian noise
        if random.random() < 0.3:  # 30% chance
            noise = np.random.normal(0, 0.02, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        return img
    
    def _rotate_image_and_heatmap(self, img: np.ndarray, heatmap: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate image and heatmap using the same rotation matrix."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation to image
        rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Apply same rotation to heatmap
        rotated_heatmap = cv2.warpAffine(heatmap, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return rotated_img.copy(), rotated_heatmap.copy()
    
    def _scale_image_and_heatmap(self, img: np.ndarray, heatmap: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """Scale image and heatmap and return the results."""
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Resize heatmap
        heatmap_scaled = cv2.resize(heatmap, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        if scale > 1.0:
            # Crop from center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            img_result = img_scaled[start_h:start_h+h, start_w:start_w+w].copy()
            heatmap_result = heatmap_scaled[start_h:start_h+h, start_w:start_w+w].copy()
        else:
            # Pad with reflection
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            img_result = np.pad(img_scaled, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w), (0, 0)), mode='reflect')
            heatmap_result = np.pad(heatmap_scaled, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='reflect')
        
        return img_result.copy(), heatmap_result.copy()

def spatial_klloss(pred_map, target_map, eps=1e-8):
    """Spatial KL loss function"""
    # pred_map: after spatial softmax, (B, 1, H, W)
    # target_map: one-hot or few-hot, (B, H, W)
    B, _, H, W = pred_map.shape
    pred = pred_map.view(B, -1) + eps  # avoid log(0)
    target = target_map.view(B, -1) + eps
    pred_log = pred.log()
    target = target / target.sum(dim=1, keepdim=True)  # ensure sum-to-1; safe for multi-keypoint
    return (target * (target.log() - pred_log)).sum(dim=1).mean()

def generate_mattress_dataset_data(
    keypoints_data_srcs: List[str],
    image_paths: List[str],
    yolo_model_finetuned,
    allowed_classes: List[int],
    image_size: int = 256
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[Tuple[int, int]]]:
    """
    Generate mattress dataset data using functional approach from multiple data sources.
    
    Args:
        keypoints_data_srcs: List of paths to keypoints JSON files
        image_paths: List of paths to image directories
        yolo_model_finetuned: YOLO model for masking
        allowed_classes: List of allowed class IDs
        image_size: Target image size
    
    Returns:
        Tuple of (images, rgb_images, keypoints, file_paths, original_sizes)
    """
    img_arr = []
    rgb_img_arr = []
    keypoints_img_arr = []
    file_paths = []
    original_sizes = []
    
    # Ensure we have matching pairs of keypoints and image paths
    if len(keypoints_data_srcs) != len(image_paths):
        raise ValueError(f"Number of keypoints sources ({len(keypoints_data_srcs)}) must match number of image paths ({len(image_paths)})")
    
    # Process each data source pair
    for keypoints_data_src, image_path in zip(keypoints_data_srcs, image_paths):
        print(f"Processing mattress data source: {keypoints_data_src} with images from: {image_path}")
        
        for filename in os.listdir(image_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            # Load and process image
            img = cv2.imread(os.path.join(image_path, filename))
            if img is None:
                continue
                
            # Store original image dimensions
            orig_h, orig_w = img.shape[:2]
            original_sizes.append((orig_h, orig_w))
            
            color_img = img.copy()
            
            # Get keypoints from the corresponding data source
            orig_keypoints = get_keypoints_for_image(filename, keypoints_data_src)
            if orig_keypoints is None:
                # For now, create dummy keypoints if no annotations available
                # This will be updated when mattress annotations are available
                print(f"Warning: No keypoints found for {filename}, creating dummy keypoints")
                orig_keypoints = [
                    [orig_w * 0.1, orig_h * 0.1],  # Top-left
                    [orig_w * 0.9, orig_h * 0.1],  # Top-right
                    [orig_w * 0.1, orig_h * 0.9],  # Bottom-left
                    [orig_w * 0.9, orig_h * 0.9],  # Bottom-right
                ]
            
            # Resize both images and keypoints to target size (256x256 for CLIP model) FIRST
            img, keypoints = resize_image_and_keypoints(img, orig_keypoints, image_size, image_size)
            color_img, _ = resize_image_and_keypoints(color_img, orig_keypoints, image_size, image_size)
            
            # Create keypoint heatmap
            kp_img = np.zeros((image_size, image_size))
            for kp in keypoints:
                x, y = int(kp[0]), int(kp[1])
                if 0 <= x < image_size and 0 <= y < image_size:
                    kp_img[y, x] = 1
            
            # Store data
            img_arr.append(img)
            rgb_img_arr.append(color_img)
            keypoints_img_arr.append(kp_img)
            file_paths.append(os.path.join(image_path, filename))
    
    print(f"Combined mattress dataset from {len(keypoints_data_srcs)} sources: {len(img_arr)} samples")
    return (
        np.array(img_arr),
        np.array(rgb_img_arr),
        np.array(keypoints_img_arr),
        file_paths,
        original_sizes
    )

class MattressKeypointDataset(Dataset):
    """Dataset for mattress keypoint detection with CLIP preprocessing."""
    
    def __init__(
        self, 
        images: np.ndarray, 
        rgb_images: np.ndarray, 
        keypoints: np.ndarray, 
        file_paths: List[str], 
        original_sizes: List[Tuple[int, int]],
        image_size: int = 256,
        transform=None
    ):
        self.images = images.astype(np.float32) / 255.0
        self.rgb_images = rgb_images.astype(np.float32) / 255.0
        self.keypoints = keypoints.astype(np.float32)
        self.file_paths = file_paths
        self.original_sizes = original_sizes
        self.image_size = image_size
        self.transform = transform
        
        # CLIP normalization
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 1, 3)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 1, 3)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = self.images[idx]
        rgb_img = self.rgb_images[idx]
        kp = self.keypoints[idx]
        
        # Convert to channels-first format
        img = np.transpose(img, (2, 0, 1))
        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        
        # Apply CLIP normalization
        rgb_img = (rgb_img - self.mean.transpose(2, 0, 1)) / self.std.transpose(2, 0, 1)
        
        # Extract ground truth keypoint coordinates from heatmap
        gt_keypoints = self._extract_keypoints_from_heatmap(kp)
        
        sample = {
            'pixel_values': torch.from_numpy(rgb_img).float(),
            'gt_heatmap': torch.from_numpy(kp).unsqueeze(0).float(),
            'image_id': self.file_paths[idx],
            'original_size': self.original_sizes[idx],
            'gt_points': gt_keypoints,
            'original_heatmap': kp  # Store original heatmap for coordinate-based augmentation
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _extract_keypoints_from_heatmap(self, heatmap: np.ndarray, threshold: float = 0.1) -> List[Tuple[int, int]]:
        """Extract keypoint coordinates from ground truth heatmap."""
        # Find local maxima in the heatmap
        from scipy.ndimage import maximum_filter
        
        # Find local maxima
        local_maxima = maximum_filter(heatmap, size=5) == heatmap
        local_maxima = local_maxima & (heatmap > threshold)
        
        # Get coordinates of local maxima
        peak_coords = np.where(local_maxima)
        keypoints = list(zip(peak_coords[1], peak_coords[0]))  # (x, y) format
        
        # Sort by intensity and return top 4
        keypoints.sort(key=lambda kp: heatmap[kp[1], kp[0]], reverse=True)
        return keypoints[:4]

def collate_mattress_batch(batch):
    """Collate function for mattress dataset."""
    pixel_values = torch.stack([b['pixel_values'] for b in batch], dim=0)
    gt_heatmap = torch.stack([b['gt_heatmap'] for b in batch], dim=0)
    image_id = [b['image_id'] for b in batch]
    original_size = [b['original_size'] for b in batch]
    gt_points = [b['gt_points'] for b in batch]
    return {
        'pixel_values': pixel_values,
        'gt_heatmap': gt_heatmap,
        'image_id': image_id,
        'original_size': original_size,
        'gt_points': gt_points,
    }

def load_pretrained_clip_model(pretrained_path: str, config: Dict) -> ClipHeatmapModel:
    """
    Load the pre-trained CLIP model and prepare it for post-training.
    
    Args:
        pretrained_path: Path to the pre-trained model directory
        config: Configuration dictionary
        
    Returns:
        Loaded and configured ClipHeatmapModel
    """
    print(f"Loading pre-trained CLIP model from: {pretrained_path}")
    
    if os.path.isdir(pretrained_path):
        # Load from directory (preferred method)
        model = ClipHeatmapModel.from_pretrained(
            pretrained_path,
            model_name=config.get('model_name', 'openai/clip-vit-base-patch16'),
            image_size=config['image_size'],
            use_lora=config.get('use_lora', True),
            lora_r=config.get('lora_r', 8),
            lora_alpha=config.get('lora_alpha', 16),
            lora_dropout=config.get('lora_dropout', 0.1),
            use_text_prior=config.get('use_text_prior', True),
            prior_prompts=config.get('prior_prompts'),
            negative_prompts=config.get('negative_prompts'),
            prior_weight=config.get('prior_weight', 0.3)
        )
    else:
        # Fallback: create new model with pre-trained weights
        print("Warning: Pretrained path is not a directory, creating new model")
        model = create_clip_heatmap_model(
            model_name=config.get('model_name', 'openai/clip-vit-base-patch16'),
            image_size=config['image_size'],
            use_lora=config.get('use_lora', True),
            lora_r=config.get('lora_r', 8),
            lora_alpha=config.get('lora_alpha', 16),
            lora_dropout=config.get('lora_dropout', 0.1),
            use_text_prior=config.get('use_text_prior', True),
            prior_prompts=config.get('prior_prompts'),
            negative_prompts=config.get('negative_prompts'),
            prior_weight=config.get('prior_weight', 0.3)
        )
    
    return model

def train_model_post(
    model: ClipHeatmapModel, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    test_loader: DataLoader,
    config: Dict
) -> Tuple[ClipHeatmapModel, Dict]:
    """
    Post-train the CLIP model on mattress data.
    
    Args:
        model: Pre-trained CLIP model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Configuration dictionary
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Setup optimization
    torch.backends.cudnn.benchmark = True
    scaler = GradScaler(enabled=config.get('use_fp16', True))
    
    # Optimizer - only train LoRA parameters and head
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step_size'], gamma=config['lr_gamma'])
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = {'train_loss': [], 'val_loss': []}
    
    num_epochs = config['num_epochs']
    
    for epoch in range(num_epochs):
        time_start = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, batch in enumerate(train_pbar):
            pixel_values = batch["pixel_values"].to(device)
            gt_heatmap = batch["gt_heatmap"].to(device)
            
            optimizer.zero_grad()
            
            with autocast("cuda", dtype=torch.float16, enabled=config.get('use_fp16', True)):
                outputs = model(pixel_values)
                # Apply Gaussian blur to ground truth for KL loss
                gt_blurred = batch_gaussian_blur(gt_heatmap, kernel_size=31, sigma=3)
                loss = kl_heatmap_loss(outputs, gt_blurred)
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.get('gradient_clip_val', 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, config['gradient_clip_val'])
            
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * pixel_values.size(0)
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch_idx, batch in enumerate(val_pbar):
                pixel_values = batch["pixel_values"].to(device)
                gt_heatmap = batch["gt_heatmap"].to(device)
                
                with autocast("cuda", dtype=torch.float16, enabled=config.get('use_fp16', True)):
                    outputs = model(pixel_values)
                    gt_blurred = batch_gaussian_blur(gt_heatmap, kernel_size=31, sigma=3)
                    loss = kl_heatmap_loss(outputs, gt_blurred)
                
                val_loss += loss.item() * pixel_values.size(0)
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss = val_loss / len(val_loader.dataset)
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Time: {time.time() - time_start:.2f}s')
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), config['model_save_path'])
            print(f"New best model saved with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss didn't improve. Patience: {patience_counter}/{config['early_stopping_patience']}")
            
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load(config['model_save_path'], map_location=device))
    
    # Save training history
    history_path = config['model_save_path'].replace('.pth', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    return model, training_history

def evaluate_model_post(model: ClipHeatmapModel, test_loader: DataLoader, results_dir: str = 'results', config: Dict = None):
    """Evaluate the post-trained model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    test_loss = 0.0
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            pixel_values = batch["pixel_values"].to(device)
            gt_heatmap = batch["gt_heatmap"].to(device)
            image_ids = batch["image_id"]
            original_sizes = batch["original_size"]
            
            with autocast("cuda", dtype=torch.float16):
                outputs = model(pixel_values)
                gt_blurred = batch_gaussian_blur(gt_heatmap, kernel_size=31, sigma=3)
                loss = kl_heatmap_loss(outputs, gt_blurred)
            
            test_loss += loss.item() * pixel_values.size(0)
            
            # Save visualization for each image
            for i, (pred_heat, gt_heat, img_id, orig_size) in enumerate(zip(
                outputs.cpu(), gt_heatmap.cpu(), image_ids, original_sizes
            )):
                # Read original image from filepath
                orig_img = cv2.imread(img_id)
                if orig_img is None:
                    print(f"Warning: Could not load original image: {img_id}")
                    continue
                
                # Convert BGR to RGB for consistency
                orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = orig_img_rgb.shape[:2]
                
                # Get predicted heatmap and resize to original image size
                pred_heat_np = pred_heat.squeeze(0).numpy()
                pred_heat_resized = cv2.resize(pred_heat_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                
                # Find keypoint locations from predicted heatmap
                # Use configurable threshold to find peaks
                threshold = config.get('keypoint_threshold', 0.1) if config else 0.1
                pred_heat_thresh = (pred_heat_resized > threshold).astype(np.uint8)
                
                # Find contours to get keypoint locations
                contours, _ = cv2.findContours(pred_heat_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Create overlay image
                overlay = orig_img_rgb.copy()
                
                # First draw ground truth keypoints (so they appear behind predicted points)
                gt_heat_np = gt_heat.squeeze(0).numpy()
                gt_heat_resized = cv2.resize(gt_heat_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                gt_heat_thresh = (gt_heat_resized > 0.5).astype(np.uint8)  # GT is binary
                
                gt_contours, _ = cv2.findContours(gt_heat_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in gt_contours:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Draw circle for ground truth keypoint (same size as predicted)
                        gt_size = config.get('gt_keypoint_size', 30) if config else 30
                        cv2.circle(overlay, (cx, cy), gt_size, (0, 255, 0), -1)  # Green circle
                        cv2.circle(overlay, (cx, cy), gt_size + 5, (255, 255, 255), 4)  # White border
                
                # Then draw predicted keypoints (so they appear on top and are bigger)
                for contour in contours:
                    # Get centroid of contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Draw circle for predicted keypoint (same size as GT)
                        keypoint_size = config.get('keypoint_size', 30) if config else 30
                        cv2.circle(overlay, (cx, cy), keypoint_size, (255, 0, 0), -1)  # Red circle
                        cv2.circle(overlay, (cx, cy), keypoint_size + 5, (255, 255, 255), 4)  # White border
                
                # Add legend
                legend_height = 60
                legend = np.zeros((legend_height, orig_w, 3), dtype=np.uint8)
                legend[:] = (50, 50, 50)  # Dark gray background
                
                # Add text
                cv2.putText(legend, "Red: Predicted Keypoints", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(legend, "Green: Ground Truth Keypoints", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Combine overlay with legend
                final_vis = np.vstack([overlay, legend])
                
                # Save result
                filename = os.path.basename(img_id)
                name_without_ext = os.path.splitext(filename)[0]
                result_path = os.path.join(results_dir, f'{name_without_ext}_post_clip_mattress_keypoints.png')
                cv2.imwrite(result_path, cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR))
    
    print(f'Test Loss: {test_loss / len(test_loader.dataset):.4f}')
    print(f'Keypoint visualizations saved to {results_dir}/ directory')
    return test_loss / len(test_loader.dataset)

def main_post_training_pipeline(config: Dict[str, Any]) -> Tuple[ClipHeatmapModel, Dict]:
    """
    Main post-training pipeline for CLIP model on mattress data.
    
    Args:
        config: Configuration dictionary containing all training parameters
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    print("=== Post-Training CLIP-Style Keypoint Detection Model on Mattress Data ===")
    
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
    img_arr, rgb_img_arr, keypoints_img_arr, file_paths, original_sizes = generate_mattress_dataset_data(
        config["keypoints_data_srcs"],
        config["image_paths"],
        yolo_model_finetuned,
        config["allowed_classes"],
        config["image_size"]
    )
    
    print(f"Mattress dataset generated: {len(img_arr)} samples")
    
    if len(img_arr) == 0:
        raise ValueError("No mattress data found. Check your data paths and keypoint annotations.")
    
    # Create base dataset without augmentation
    base_dataset = MattressKeypointDataset(
        img_arr, rgb_img_arr, keypoints_img_arr, file_paths, original_sizes,
        config["image_size"], transform=None
    )
    
    # Split dataset into train, validation, and test
    total_size = len(base_dataset)
    train_size = int(0.7 * total_size)  # 70% for training
    val_size = int(0.2 * total_size)    # 20% for validation
    test_size = total_size - train_size - val_size  # 10% for testing
    
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        range(total_size), [train_size, val_size, test_size]
    )
    
    # Create training dataset with augmentation (if enabled)
    if config.get("use_augmentation", False):
        print("Applying data augmentation to training set only")
        augmentation_transform = MattressAugmentation(config["image_size"])
        train_dataset = MattressKeypointDataset(
            img_arr, rgb_img_arr, keypoints_img_arr, file_paths, original_sizes,
            config["image_size"], transform=augmentation_transform
        )
        # Create subset with only training indices
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    else:
        print("No augmentation applied")
        train_dataset = torch.utils.data.Subset(base_dataset, train_indices)
    
    # Create validation and test datasets without augmentation
    val_dataset = torch.utils.data.Subset(base_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(base_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        pin_memory=True,
        collate_fn=collate_mattress_batch
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        pin_memory=True,
        collate_fn=collate_mattress_batch
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        pin_memory=True,
        collate_fn=collate_mattress_batch
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load pre-trained CLIP model
    print("Loading pre-trained CLIP model...")
    model = load_pretrained_clip_model(config["pretrained_model_path"], config)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Post-train model
    print("Starting post-training...")
    trained_model, training_history = train_model_post(
        model, 
        train_loader, 
        val_loader,
        test_loader, 
        config
    )
    
    # Evaluate post-trained model
    print("Evaluating post-trained model...")
    test_loss = evaluate_model_post(trained_model, test_loader, config['results_dir'], config)
    print(f"Post-trained model test loss: {test_loss:.4f}")
    
    print("Post-training completed!")
    
    return trained_model, training_history

def main():
    """Main function for standalone execution"""
    # Use default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Optionally disable TensorRT conversion for testing
    # config["enable_tensorrt_conversion"] = False
    
    # Run post-training pipeline
    model, history = main_post_training_pipeline(config)
    
    return model, history

if __name__ == "__main__":
    main()
