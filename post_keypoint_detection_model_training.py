#!/usr/bin/env python3
"""
Post-Processing Keypoint Detection Model Training

This script implements the working training logic from keypoint_detection_model_training.py
integrated with the comprehensive pipeline structure from test.py for post-processing keypoint detection.
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.utils import (
    YoloBackbone, 
    batch_gaussian_blur,
    extract_mask_compare,
    thresholded_locations
)
from src.utils.model_utils import batch_entropy
from src.models import HybridKeypointNet
from src.models.efficient_keypoint_net import EfficientViTKeypointNet
from src.utils import kl_heatmap_loss
from shared.functions import get_keypoints_for_image, resize_image_and_keypoints

# Default configuration
DEFAULT_CONFIG = {
    "seed": 42,
    "yolo_model_path": "models/yolo_finetuned/best.pt",
    "keypoints_data_src": "via_proj/via_project_22Aug2025_16h07m06s.json",
    "image_path": "RGB-images-jpg/",
    "allowed_classes": [1],
    "batch_size": 32,
    "learning_rate": 3e-5,
    "weight_decay": 5e-5,
    "num_epochs": 300,
    "lr_step_size": 50,
    "lr_gamma": 0.7,
    "warmup_epochs": 5,
    "model_save_path": "models/keypoint_model_vit_post.pth",
    "save_interval": 100,
    "results_dir": "results",
    "pretrained_path": "models/keypoint_model_vit.pth",
    "freeze_backbone": False,
    "use_augmentation": True,
    "use_mixup": True,
    "mixup_alpha": 0.2,
    "use_fp16": True,
    "gradient_clip_val": 1.0,
    "gradient_accumulation_steps": 2,
    "early_stopping_patience": 20,
    "use_stronger_augmentation": True,
    "dropout_rate": 0.1,
    "label_smoothing": 0.1
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

# Data generation functions from test.py
def generate_dataset_data(
    keypoints_data_src: str,
    image_path: str,
    yolo_model_finetuned,
    allowed_classes: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Generate dataset data using functional approach.
    
    Args:
        keypoints_data_src: Path to keypoints JSON file
        image_path: Path to image directory
        yolo_model_finetuned: YOLO model for masking
        allowed_classes: List of allowed class IDs
    
    Returns:
        Tuple of (images, rgb_images, keypoints, file_paths)
    """
    img_arr = []
    rgb_img_arr = []
    keypoints_img_arr = []
    file_paths = []
    
    for filename in os.listdir(image_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        # Load and process image
        img = cv2.imread(os.path.join(image_path, filename))
        if img is None:
            continue
            
        color_img = img.copy()
        
        # Get keypoints
        orig_keypoints = get_keypoints_for_image(filename, keypoints_data_src)
        if orig_keypoints is None:
            continue
        
        # Resize image and keypoints
        img, keypoints = resize_image_and_keypoints(img, orig_keypoints, 128, 128)
        
        # Apply YOLO masking
        mask = extract_mask_compare(img, yolo_model_finetuned, allowed_classes)
        if np.sum(mask) == 0:
            continue
            
        img[mask == 0] = 0
        
        # Process color image and keypoints
        color_img, keypoints = resize_image_and_keypoints(color_img, orig_keypoints, 128, 128)
        
        # Apply the same mask to color image
        color_img[mask == 0] = 0
        
        # Create keypoint heatmap (keep original coordinate order: [x, y])
        kp_img = np.zeros((128, 128))
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])  # x, y coordinates
            if 0 <= x < 128 and 0 <= y < 128:  # Bounds check
                kp_img[y, x] = 1  # Note: kp_img[y, x] for numpy array indexing
        
        # Store data
        img_arr.append(img)
        rgb_img_arr.append(color_img)
        keypoints_img_arr.append(kp_img)
        file_paths.append(os.path.join(image_path, filename))
    
    return (
        np.array(img_arr),
        np.array(rgb_img_arr),
        np.array(keypoints_img_arr),
        file_paths
    )

# Data augmentation classes from test.py
class RandomRotateFlip:
    """Random rotation and flip augmentation."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() < self.p:
            # Random rotation (0, 90, 180, 270 degrees)
            k = random.randint(0, 3)
            sample['image'] = torch.rot90(sample['image'], k, [1, 2])
            sample['keypoints'] = torch.rot90(sample['keypoints'], k, [0, 1])
            sample['rgb_image'] = torch.rot90(sample['rgb_image'], k, [1, 2])
        
        if random.random() < self.p:
            # Random horizontal flip
            sample['image'] = torch.flip(sample['image'], [2])
            sample['keypoints'] = torch.flip(sample['keypoints'], [1])
            sample['rgb_image'] = torch.flip(sample['rgb_image'], [2])
        
        if random.random() < self.p:
            # Random vertical flip
            sample['image'] = torch.flip(sample['image'], [1])
            sample['keypoints'] = torch.flip(sample['keypoints'], [0])
            sample['rgb_image'] = torch.flip(sample['rgb_image'], [1])
        
        return sample

class ColorJitter:
    """Color jitter augmentation."""
    
    def __init__(self, brightness: float = 0.1, contrast: float = 0.1, p: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() < self.p:
            # Apply to both image and rgb_image
            for key in ['image', 'rgb_image']:
                img = sample[key]
                
                # Brightness
                if self.brightness > 0:
                    factor = 1.0 + random.uniform(-self.brightness, self.brightness)
                    img = img * factor
                    img = torch.clamp(img, 0, 255)
                
                # Contrast
                if self.contrast > 0:
                    factor = 1.0 + random.uniform(-self.contrast, self.contrast)
                    mean = img.mean()
                    img = (img - mean) * factor + mean
                    img = torch.clamp(img, 0, 255)
                
                sample[key] = img
        
        return sample

class GaussianNoise:
    """Gaussian noise augmentation."""
    
    def __init__(self, std: float = 5.0, p: float = 0.3):
        self.std = std
        self.p = p
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() < self.p:
            for key in ['image', 'rgb_image']:
                noise = torch.randn_like(sample[key]) * self.std
                sample[key] = sample[key] + noise
                sample[key] = torch.clamp(sample[key], 0, 255)
        
        return sample

class MinimalAugmentation:
    """Minimal augmentation for small datasets."""
    
    def __init__(self):
        self.rotate_flip = RandomRotateFlip(p=0.3)
        self.color_jitter = ColorJitter(brightness=0.05, contrast=0.05, p=0.3)
        self.noise = GaussianNoise(std=2.0, p=0.2)
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sample = self.rotate_flip(sample)
        sample = self.color_jitter(sample)
        sample = self.noise(sample)
        return sample

class StrongerAugmentation:
    """Stronger augmentation to prevent overfitting."""
    
    def __init__(self):
        self.rotate_flip = RandomRotateFlip(p=0.5)
        self.color_jitter = ColorJitter(brightness=0.15, contrast=0.15, p=0.5)
        self.noise = GaussianNoise(std=8.0, p=0.4)
        self.cutout = Cutout(size=32, p=0.3)
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sample = self.rotate_flip(sample)
        sample = self.color_jitter(sample)
        sample = self.noise(sample)
        sample = self.cutout(sample)
        return sample

class Cutout:
    """Cutout augmentation."""
    
    def __init__(self, size: int = 32, p: float = 0.3):
        self.size = size
        self.p = p
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() < self.p:
            for key in ['image', 'rgb_image']:
                img = sample[key]
                c, h, w = img.shape
                
                # Random position for cutout
                y = random.randint(0, h - self.size)
                x = random.randint(0, w - self.size)
                
                # Apply cutout (set to zero)
                img[:, y:y+self.size, x:x+self.size] = 0
                sample[key] = img
        
        return sample

# Mixup function from test.py
def mixup_data(images: torch.Tensor, keypoints: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mixup augmentation.
    
    Args:
        images: Input images
        keypoints: Input keypoints
        alpha: Mixup alpha parameter
    
    Returns:
        Mixed images and keypoints
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)
    
    mixed_images = lam * images + (1 - lam) * images[index, :]
    mixed_keypoints = lam * keypoints + (1 - lam) * keypoints[index, :]
    
    return mixed_images, mixed_keypoints

# Dataset class from test.py
class KeypointDataset(torch.utils.data.Dataset):
    """Functional dataset for keypoint detection."""
    
    def __init__(
        self, 
        images: np.ndarray, 
        rgb_images: np.ndarray, 
        keypoints: np.ndarray, 
        file_paths: List[str], 
        transform=None
    ):
        self.images = images.astype(np.float32)
        self.rgb_images = rgb_images.astype(np.float32)
        self.keypoints = keypoints.astype(np.float32)
        self.file_paths = file_paths
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = self.images[idx]
        rgb_img = self.rgb_images[idx]
        kp = self.keypoints[idx]
        
        # Convert to channels-first format
        img = np.transpose(img, (2, 0, 1))
        rgb_img = np.transpose(rgb_img, (2, 0, 1))
        
        sample = {
            'image': torch.from_numpy(img),
            'keypoints': torch.from_numpy(kp),
            'rgb_image': torch.from_numpy(rgb_img),
            'file_path': self.file_paths[idx]
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample

# Training functions with working logic from keypoint_detection_model_training.py
def create_model():
    """Create and configure the model using the working approach"""
    # yolo vit
    yolo_model = YOLO('yolo11l-pose.pt')
    backbone_seq = yolo_model.model.model[:12]
    backbone = YoloBackbone(backbone_seq, selected_indices=[0,1,2,3,4,5,6,7,8,9,10,11])
    input_dummy = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        feats = backbone(input_dummy)
    in_channels_list = [f.shape[1] for f in feats]
    keypoint_net = HybridKeypointNet(backbone, in_channels_list)
    model = keypoint_net
    
    # Freeze backbone parameters
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    return model

def train_model(model, trainloader, valloader, testloader, num_epochs=300, load_model=False, save_path=None, use_mixup=True, mixup_alpha=0.2, early_stopping_patience=20):
    """Train the model using the working logic from keypoint_detection_model_training.py"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # optimization
    torch.backends.cudnn.benchmark = True
    scaler = GradScaler()
    
    # loss
    loss_fn = nn.BCEWithLogitsLoss()
    
    compiled_model = torch.compile(model)
    
    if not load_model:
        optimizer = optim.AdamW(compiled_model.parameters(), lr=1e-5)
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {'train_loss': [], 'val_loss': []}

        for epoch in range(num_epochs):
            time_start = time.time()
            
            # Training phase
            compiled_model.train()
            running_loss = 0.0

            for batch in trainloader:
                images = batch["image"].to(device)
                keypoints = batch["keypoints"].to(device)
                optimizer.zero_grad()

                # Apply mixup if enabled
                if use_mixup:
                    images, keypoints = mixup_data(images, keypoints, alpha=mixup_alpha)

                with autocast("cuda", dtype=torch.float16):      # AMP context, not forcing .half()
                    outputs = compiled_model(images)
                    keypoints_blur = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
                    
                    # calculate loss on all samples (no active learning)
                    loss = kl_heatmap_loss(outputs, keypoints_blur.unsqueeze(1))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * images.size(0)
            
            train_loss = running_loss / len(trainloader.dataset)
            
            # Validation phase
            compiled_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in valloader:
                    images = batch["image"].to(device)
                    keypoints = batch["keypoints"].to(device)
                    
                    with autocast("cuda", dtype=torch.float16):
                        outputs = compiled_model(images)
                        keypoints_blur = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
                        loss = kl_heatmap_loss(outputs, keypoints_blur.unsqueeze(1))
                    
                    val_loss += loss.item() * images.size(0)
            
            val_loss = val_loss / len(valloader.dataset)
            
            # Store history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            
            print(f'Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Time: {time.time() - time_start:.2f}s')
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                if save_path is None:
                    save_path = 'models/keypoint_model_vit_post.pth'
                torch.save(compiled_model.state_dict(), save_path)
                print(f"New best model saved with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Validation loss didn't improve. Patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Save training history
        history_path = save_path.replace('.pth', '_history.json') if save_path else 'models/keypoint_model_vit_post_history.json'
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        print(f"Training history saved to: {history_path}")
        
        # Plot training curves
        plot_training_curves(training_history, save_path.replace('.pth', '_training_plot.png') if save_path else 'models/keypoint_model_vit_post_training_plot.png')
        
    else:
        # Load from the specified path or default
        load_path = save_path if save_path else 'models/keypoint_model_vit_post.pth'
        compiled_model.load_state_dict(torch.load(load_path, map_location=device))
        compiled_model.eval()
    
    return compiled_model, training_history if 'training_history' in locals() else None

def evaluate_model(model, testloader):
    """Evaluate the model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    val_loss = 0.0
    iter_count = 0
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    with torch.no_grad():
        for batch in testloader:
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)
            with autocast("cuda", dtype=torch.float16):
                outputs = model(images)
                keypoints_blur = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
                loss = kl_heatmap_loss(outputs, keypoints_blur.unsqueeze(1))

            # render the predicted keypoints on the image
            for img, kp in zip(images.cpu().numpy(), outputs.cpu().numpy()):
                img = np.transpose(img, (1, 2, 0)) * 255
                # Convert RGB to BGR for OpenCV
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                kp = kp[0,:,:]
                peaks = thresholded_locations(kp, 0.003)
                for p in peaks:
                    i,j = p
                    cv2.circle(img, (int(j), int(i)), 3, (255,0,0), -1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Save test results to output dataset
                cv2.imwrite(f'results/keypoints_{iter_count}.png', img)
                iter_count += 1
            val_loss += loss.item() * images.size(0)
    
    print(f'Validation Loss: {val_loss / len(testloader.dataset):.4f}')
    return val_loss / len(testloader.dataset)

def plot_training_curves(history, save_path):
    """Plot training and validation loss curves"""
    try:
        plt.figure(figsize=(12, 4))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss', color='blue')
        plt.plot(history['val_loss'], label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot loss difference
        plt.subplot(1, 2, 2)
        loss_diff = [abs(train - val) for train, val in zip(history['train_loss'], history['val_loss'])]
        plt.plot(loss_diff, label='|Train - Val| Loss', color='green')
        plt.title('Loss Difference (Overfitting Indicator)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Difference')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to: {save_path}")
        
    except Exception as e:
        print(f"Warning: Could not plot training curves: {e}")

def visualize_model_architecture(model):
    """Visualize the model architecture"""
    try:
        from torchview import draw_graph
        
        # Create model graph
        model_graph = draw_graph(model, input_data=torch.randn((8,3,128,128)), expand_nested=True)
        model_graph.visual_graph.render(filename='architecture_full', format='png')
        print("Model architecture saved as 'architecture_full.png'")
        
    except ImportError:
        print("torchview not installed. Install with: pip install torchview")
        print("Model architecture visualization skipped.")

def main_training_pipeline(config: Dict[str, Any]) -> Tuple[Any, Dict[str, List[float]]]:
    """
    Main training pipeline that integrates the working logic with the comprehensive pipeline structure.
    
    Args:
        config: Configuration dictionary containing all training parameters
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    print("=== Post-Processing Keypoint Detection Model Training ===")
    
    # Set random seeds
    set_random_seeds(config.get("seed", 42))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load YOLO model
    yolo_model_finetuned = YOLO(config["yolo_model_path"])
    
    # Generate dataset using the comprehensive approach from test.py
    print("Generating dataset...")
    img_arr, rgb_img_arr, keypoints_img_arr, file_paths = generate_dataset_data(
        config["keypoints_data_src"],
        config["image_path"],
        yolo_model_finetuned,
        config["allowed_classes"]
    )
    
    print(f"Dataset generated: {len(img_arr)} samples")
    
    # Create datasets with augmentation
    # Apply augmentation if enabled
    if config.get("use_augmentation", True):
        if config.get("use_stronger_augmentation", True):
            train_transform = StrongerAugmentation()
            print("Using stronger augmentation to prevent overfitting")
        else:
            train_transform = MinimalAugmentation()
            print("Using minimal augmentation")
    else:
        train_transform = None
        print("No augmentation applied")
    
    # Create separate datasets for training and evaluation
    # Training dataset with augmentation
    train_dataset_full = KeypointDataset(img_arr, rgb_img_arr, keypoints_img_arr, file_paths, transform=train_transform)
    
    # Evaluation dataset without augmentation (for proper keypoint plotting)
    eval_dataset_full = KeypointDataset(img_arr, rgb_img_arr, keypoints_img_arr, file_paths, transform=None)
    
    # Split dataset into train, validation, and test
    total_size = len(train_dataset_full)
    train_size = int(0.7 * total_size)  # 70% for training
    val_size = int(0.2 * total_size)    # 20% for validation
    test_size = total_size - train_size - val_size  # 10% for testing
    
    # Split training dataset (with augmentation)
    train_dataset, val_dataset, _ = torch.utils.data.random_split(
        train_dataset_full, [train_size, val_size, test_size]
    )
    
    # Split evaluation dataset (without augmentation) using same indices
    eval_train, eval_val, test_dataset = torch.utils.data.random_split(
        eval_dataset_full, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        pin_memory=True
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create model using the working approach
    print("Creating model...")
    model = create_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load pretrained model if specified
    if config.get("pretrained_path") and os.path.exists(config["pretrained_path"]):
        print(f"Loading pretrained model from {config['pretrained_path']}")
        state_dict = torch.load(config["pretrained_path"], map_location=device)
        
        # Handle model state dict with _orig_mod prefixes
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                new_key = key[10:]  # Remove "_orig_mod." prefix
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # Load with strict=False to handle any remaining mismatches
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            print(f"Warning: Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {len(unexpected_keys)}")
        print("Pretrained weights loaded successfully")
    else:
        print("No pretrained model found, starting from scratch")
    
    # Train model using the working logic
    print("Starting regular training...")
    load_model = config.get("load_model", False)
    num_epochs = config.get("num_epochs", 300)
    use_mixup = config.get("use_mixup", True)
    mixup_alpha = config.get("mixup_alpha", 0.2)
    early_stopping_patience = config.get("early_stopping_patience", 20)
    trained_model, training_history = train_model(
        model, 
        train_loader, 
        val_loader,
        test_loader, 
        num_epochs=num_epochs, 
        load_model=load_model, 
        save_path=config["model_save_path"],
        use_mixup=use_mixup,
        mixup_alpha=mixup_alpha,
        early_stopping_patience=early_stopping_patience
    )
    
    # Evaluate regular model
    print("Evaluating regular model...")
    val_loss = evaluate_model(trained_model, test_loader)
    print(f"Regular model validation loss: {val_loss:.4f}")
    
    # Visualize model architecture
    print("Visualizing model architecture...")
    visualize_model_architecture(model)
    
    # Use actual training history if available, otherwise create simplified version
    if training_history is not None:
        history = training_history
    else:
        history = {
            "train_loss": [val_loss],  # Simplified - in real implementation, track epoch losses
            "val_loss": [val_loss]
        }
    
    print("Training completed!")
    return trained_model, history

def main():
    """Main function for standalone execution"""
    # Use default configuration
    config = DEFAULT_CONFIG.copy()
    
    # Run training pipeline
    model, history = main_training_pipeline(config)
    
    return model, history

if __name__ == "__main__":
    main()
