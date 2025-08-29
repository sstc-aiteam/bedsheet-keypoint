#!/usr/bin/env python3
"""
Functional Keypoint Detection Model Training Script

This script implements quantization-aware training for keypoint detection
using functional programming principles and clean code organization.
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
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ..models.hybrid_keypoint_net import HybridKeypointNet
from ..utils.model_utils import (
    YoloBackbone, 
    batch_gaussian_blur,
    extract_mask_compare,
    thresholded_locations,
    kl_heatmap_loss
)

from ..utils.quantization_utils import (
    create_quantized_model_structure,
    prepare_model_for_qat,
    convert_to_quantized,
    export_model_pipeline
)
from shared.functions import get_keypoints_for_image, resize_image_and_keypoints

# Set random seeds for reproducibility
def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data generation functions
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

# Data augmentation classes
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

# Mixup function
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

# Dataset class
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

# Training functions
def create_training_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: callable,
    device: torch.device,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2,
    use_fp16: bool = False,
    gradient_clip_val: float = 1.0,
    gradient_accumulation_steps: int = 1
) -> callable:
    """
    Create a functional training step.
    
    Args:
        model: Model to train
        optimizer: Optimizer
        loss_fn: Loss function
        device: Target device
        use_mixup: Whether to use mixup augmentation
        mixup_alpha: Mixup alpha parameter
        use_fp16: Whether to use FP16 training
    
    Returns:
        Training step function
    """
    scaler = GradScaler() if use_fp16 else None
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    
    def training_step(
        images: torch.Tensor, 
        keypoints: torch.Tensor,
        step: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step."""
        # Only zero gradients at the start of accumulation
        if step % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        # Apply mixup if enabled
        if use_mixup and images.size(0) > 1:
            images, keypoints = mixup_data(images, keypoints, alpha=mixup_alpha)
        
        # Forward pass with FP16 if enabled
        if use_fp16:
            with autocast(device_type=device_type, dtype=torch.float16):
                outputs = model(images)
                keypoints_blur = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
                loss = loss_fn(outputs, keypoints_blur.unsqueeze(1))
            
            # Backward pass with gradient scaling
            scaler.scale(loss / gradient_accumulation_steps).backward()
            
            # Only step optimizer at the end of accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                
                scaler.step(optimizer)
                scaler.update()
        else:
            # Regular FP32 training
            outputs = model(images)
            keypoints_blur = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
            loss = loss_fn(outputs, keypoints_blur.unsqueeze(1))
            
            # Backward pass
            (loss / gradient_accumulation_steps).backward()
            
            # Only step optimizer at the end of accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
                
                optimizer.step()
        
        # Return loss and metrics
        metrics = {
            "loss": loss.item(),
            "learning_rate": optimizer.param_groups[0]["lr"]
        }
        
        return loss, metrics
    
    return training_step

def train_model_functional(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,  # Added validation loader
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device,
    save_path: str,
    save_interval: int = 10,
    early_stopping_patience: int = 20,  # Added early stopping
    use_stronger_augmentation: bool = True  # Added stronger augmentation option
) -> Dict[str, List[float]]:
    """
    Functional training loop.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Target device
        save_path: Path to save model
        save_interval: Save model every N epochs
    
    Returns:
        Training history
    """
    model = model.to(device)
    # Get training parameters from function arguments or defaults
    use_mixup = True
    mixup_alpha = 0.2
    use_fp16 = False
    gradient_clip_val = 1.0
    gradient_accumulation_steps = 1
    
    training_step = create_training_step(
        model, 
        optimizer, 
        kl_heatmap_loss, 
        device,
        use_mixup=use_mixup,
        mixup_alpha=mixup_alpha,
        use_fp16=use_fp16,
        gradient_clip_val=gradient_clip_val,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    history = {
        "train_loss": [],
        "val_loss": [],  # Added validation loss tracking
        "learning_rate": [],
        "epoch_times": []
    }
    
    # Initialize anti-overfitting callbacks
    early_stopping = create_early_stopping(patience=early_stopping_patience)
    save_best_model = create_model_checkpoint(save_path)
    

    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            keypoints = batch["keypoints"].to(device)
            
            # Calculate global step for gradient accumulation
            global_step = epoch * len(train_loader) + batch_idx
            
            loss, metrics = training_step(images, keypoints, global_step)
            epoch_losses.append(metrics["loss"])
        
        # Update learning rate
        scheduler.step()
        
        # Record metrics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start
        
        history["train_loss"].append(avg_loss)
        history["learning_rate"].append(current_lr)
        history["epoch_times"].append(epoch_time)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                keypoints = batch["keypoints"].to(device)
                
                outputs = model(images)
                keypoints_blur = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
                loss = kl_heatmap_loss(outputs, keypoints_blur.unsqueeze(1))
                val_loss += loss.item() * images.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        history["val_loss"].append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss {avg_loss:.4f}, Val Loss {avg_val_loss:.4f}, LR: {current_lr:.2e}, Time: {epoch_time:.2f}s')
        
        # Anti-overfitting strategies
        is_best = save_best_model(avg_val_loss, model, epoch)
        should_stop = early_stopping(avg_val_loss)
        
        # Save model periodically
        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), f"{save_path}_epoch_{epoch+1}.pth")
        
        # Early stopping
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save final model
    torch.save(model.state_dict(), f"{save_path}_final.pth")
    
    return history

# Evaluation functions
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    results_dir: str = "results",
    use_fp16: bool = False
) -> float:
    """
    Evaluate model on test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Target device
        results_dir: Directory to save results
        use_fp16: Whether to use FP16 for evaluation
    
    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0.0
    os.makedirs(results_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)
            file_paths = batch['file_path']
            
            # Forward pass with FP16 if enabled
            if use_fp16:
                device_type = 'cuda' if device.type == 'cuda' else 'cpu'
                with autocast(device_type=device_type, dtype=torch.float16):
                    outputs = model(images)
                    keypoints_blur = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
                    loss = kl_heatmap_loss(outputs, keypoints_blur.unsqueeze(1))
            else:
                outputs = model(images)
                keypoints_blur = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
                loss = kl_heatmap_loss(outputs, keypoints_blur.unsqueeze(1))
            
            val_loss += loss.item() * images.size(0)
            
            # Save visualization results
            if batch_idx < 10:  # Save first 10 batches
                for sample_idx, (img, output, file_path) in enumerate(zip(images.cpu().numpy(), outputs.cpu().numpy(), file_paths)):
                    # Load original image at full size
                    orig_img = cv2.imread(file_path)
                    if orig_img is None:
                        print(f"Warning: Could not load image {file_path}")
                        continue
                    
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                    orig_h, orig_w = orig_img.shape[:2]
                    
                    # Extract keypoint heatmap (output shape: [batch, 1, H, W])
                    kp_heatmap = output[0, :, :]  # Remove batch and channel dims
                    
                    # Use thresholded_locations to get single keypoint per region
                    peaks = thresholded_locations(kp_heatmap, threshold=0.003)
                    
                    # Scale keypoint coordinates from 128x128 to original image size
                    for peak in peaks:
                        orig_x, orig_y = fix_keypoint_coordinates(peak, orig_h, orig_w)
                        cv2.circle(orig_img, (orig_x, orig_y), 15, (255, 0, 0), -1)  # Larger circle for visibility
                    
                    # Save result with correct indexing
                    result_path = os.path.join(results_dir, f"keypoints_{batch_idx}_{sample_idx}.png")
                    plt.imsave(result_path, orig_img)
    
    avg_val_loss = val_loss / len(test_loader.dataset)
    print(f'Validation Loss: {avg_val_loss:.4f}')
    
    return avg_val_loss

def evaluate_quantized_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    results_dir: str = "results"
) -> float:
    """
    Evaluate quantized model on test set.
    
    Args:
        model: Quantized model to evaluate
        test_loader: Test data loader
        device: Target device
        results_dir: Directory to save results
    
    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0.0
    os.makedirs(results_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].to(device)
            file_paths = batch['file_path']
            
            # For quantized models, we need to ensure proper input format
            outputs = model(images)
            keypoints_blur = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
            loss = kl_heatmap_loss(outputs, keypoints_blur.unsqueeze(1))
            
            val_loss += loss.item() * images.size(0)
            
            # Save visualization results
            if batch_idx < 10:  # Save first 10 batches
                for sample_idx, (img, output, file_path) in enumerate(zip(images.cpu().numpy(), outputs.cpu().numpy(), file_paths)):
                    # Load original image at full size
                    orig_img = cv2.imread(file_path)
                    if orig_img is None:
                        print(f"Warning: Could not load image {file_path}")
                        continue
                    
                    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                    orig_h, orig_w = orig_img.shape[:2]
                    
                    # Extract keypoint heatmap (output shape: [batch, 1, H, W])
                    kp_heatmap = output[0, :, :]  # Remove batch and channel dims
                    
                    # Use thresholded_locations to get single keypoint per region
                    peaks = thresholded_locations(kp_heatmap, threshold=0.003)
                    
                    # Scale keypoint coordinates from 128x128 to original image size
                    for peak in peaks:
                        orig_x, orig_y = fix_keypoint_coordinates(peak, orig_h, orig_w)
                        cv2.circle(orig_img, (orig_x, orig_y), 15, (255, 0, 0), -1)  # Larger circle for visibility
                    
                    # Save result with correct indexing
                    result_path = os.path.join(results_dir, f"keypoints_quantized_{batch_idx}_{sample_idx}.png")
                    plt.imsave(result_path, orig_img)
    
    avg_val_loss = val_loss / len(test_loader.dataset)
    print(f'Quantized Model Validation Loss: {avg_val_loss:.4f}')
    
    return avg_val_loss

# Fixed threshold function for single keypoint detection
def fixed_threshold_locations(heatmap, threshold=0.1):
    """
    Find single peak location in heatmap above threshold.
    
    Args:
        heatmap: (H, W) tensor or numpy array
        threshold: Threshold value
    
    Returns:
        List of (y, x) coordinates (single point per keypoint)
    """
    # Convert numpy array to tensor if needed
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)
    
    # Find regions above threshold
    mask = heatmap > threshold
    
    if not torch.any(mask):
        return []
    
    # Find connected components
    from scipy.ndimage import label
    mask_np = mask.numpy().astype(np.uint8)
    labeled, num_features = label(mask_np, structure=np.ones((3, 3), dtype=int))
    
    peaks = []
    for i in range(1, num_features + 1):
        # Get coordinates of this component
        coords = np.argwhere(labeled == i)
        
        if len(coords) == 0:
            continue
        
        # Find the peak (maximum value) within this component
        component_mask = (labeled == i)
        component_values = heatmap.numpy()[component_mask]
        component_coords = coords
        
        # Find the coordinate with maximum value
        max_idx = np.argmax(component_values)
        peak_coord = component_coords[max_idx]
        
        peaks.append((int(peak_coord[0]), int(peak_coord[1])))
    
    return peaks

# Helper function to fix coordinate mapping
def fix_keypoint_coordinates(peak, orig_h, orig_w):
    """
    Fix coordinate mapping from model output to original image.
    
    Args:
        peak: [y, x] coordinates from thresholded_locations
        orig_h: Original image height
        orig_w: Original image width
    
    Returns:
        (x, y) coordinates for cv2.circle
    """
    peak_y, peak_x = peak  # peak is [y, x] from thresholded_locations
    # Scale coordinates from model output (128x128) to original image size
    orig_x = int(peak_x * orig_w / 128)
    orig_y = int(peak_y * orig_h / 128)
    return orig_x, orig_y

# Anti-overfitting strategies
def create_early_stopping(patience: int = 20, min_delta: float = 1e-4):
    """Create early stopping callback."""
    best_val_loss = float('inf')
    patience_counter = 0
    
    def early_stopping(val_loss: float) -> bool:
        nonlocal best_val_loss, patience_counter
        
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            return False  # Don't stop
        else:
            patience_counter += 1
            return patience_counter >= patience  # Stop if patience exceeded
    
    return early_stopping

def create_model_checkpoint(save_path: str):
    """Create model checkpoint callback."""
    best_val_loss = float('inf')
    
    def save_best_model(val_loss: float, model: nn.Module, epoch: int):
        nonlocal best_val_loss
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, f"{save_path}_best.pth")
            print(f"New best model saved! Val Loss: {val_loss:.4f}")
            return True
        return False
    
    return save_best_model

# Enhanced regularization
class Dropout2d(nn.Module):
    """2D Dropout for spatial dropout."""
    
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training and self.p > 0:
            # Create dropout mask
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p)) / (1 - self.p)
            return x * mask
        return x

# Enhanced augmentation for better generalization
class StrongerAugmentation:
    """Stronger augmentation to prevent overfitting."""
    
    def __init__(self):
        self.rotate_flip = RandomRotateFlip(p=0.5)  # Increased probability
        self.color_jitter = ColorJitter(brightness=0.1, contrast=0.1, p=0.5)  # Stronger
        self.noise = GaussianNoise(std=3.0, p=0.4)  # More noise
        self.cutout = Cutout(p=0.3)  # New augmentation
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        sample = self.rotate_flip(sample)
        sample = self.color_jitter(sample)
        sample = self.noise(sample)
        sample = self.cutout(sample)
        return sample

class Cutout:
    """Cutout augmentation to prevent overfitting."""
    
    def __init__(self, p: float = 0.3, size: int = 16):
        self.p = p
        self.size = size
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() < self.p:
            for key in ['image', 'rgb_image']:
                img = sample[key]
                h, w = img.shape[1], img.shape[2]
                
                # Random position for cutout
                y = random.randint(0, h - self.size)
                x = random.randint(0, w - self.size)
                
                # Apply cutout (set to zero)
                img[:, y:y+self.size, x:x+self.size] = 0
                sample[key] = img
        
        return sample

# Main training function
def main_training_pipeline(
    config: Dict[str, Any]
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Main training pipeline.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Tuple of (trained_model, training_history)
    """
    # Set random seeds
    set_random_seeds(config.get("seed", 42))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load YOLO model
    from ultralytics import YOLO
    yolo_model_finetuned = YOLO(config["yolo_model_path"])
    
    # Generate dataset
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
    
    # Create model
    print("Creating model...")
    
    yolo_model = YOLO('yolo11l-pose.pt')
    backbone_seq = yolo_model.model.model[:12]
    backbone = YoloBackbone(backbone_seq, selected_indices=list(range(12)))
    
    input_dummy = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        feats = backbone(input_dummy)
    in_channels_list = [f.shape[1] for f in feats]
    
    # Always start with the regular model first
    model = HybridKeypointNet(backbone, in_channels_list)
    print("Created regular model")
    
    # Load pretrained weights (required for quantization fine-tuning)
    if config.get("pretrained_path") and os.path.exists(config["pretrained_path"]):
        print(f"Loading pretrained weights from {config['pretrained_path']}")
        state_dict = torch.load(config["pretrained_path"], map_location=device)
        
        # Handle quantized model state dict with _orig_mod prefixes
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
        raise ValueError("Pretrained model path is required for quantization fine-tuning!")
    
    # Convert to quantized model for fine-tuning if requested
    if config.get("use_quantization", False):
        print("Converting to quantized model for fine-tuning...")
        model = create_quantized_model_structure(backbone, in_channels_list)
        model = prepare_model_for_qat(model)
        
        # Load the pretrained weights into the quantized model
        try:
            model.load_state_dict(torch.load(config["pretrained_path"], map_location=device), strict=False)
            print("Pretrained weights loaded into quantized model")
        except Exception as e:
            print(f"Warning: Could not load all pretrained weights into quantized model: {e}")
            print("Continuing with partial weight loading...")
        
        print("Using quantized model for fine-tuning")
    else:
        print("Using regular model (no quantization)")
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"], 
        weight_decay=config.get("weight_decay", 1e-4)
    )
    # Create adaptive learning rate scheduler for small datasets
    warmup_epochs = config.get("warmup_epochs", 5)
    total_epochs = config.get("num_epochs", 300)
    
    # Multi-stage learning rate schedule
    def lr_lambda(epoch):
        # Stage 1: Warmup (0 to warmup_epochs)
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        
        # Stage 2: High learning rate plateau (warmup_epochs to 50)
        elif epoch < 50:
            return 1.0
        
        # Stage 3: Gradual decay (50 to 150)
        elif epoch < 150:
            progress = float(epoch - 50) / float(max(1, 100))
            return 1.0 - 0.5 * progress
        
        # Stage 4: Fine-tuning with very low LR (150 to end)
        else:
            progress = float(epoch - 150) / float(max(1, total_epochs - 150))
            return 0.5 * (1.0 - 0.8 * progress)  # Decay to 10% of original
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Train model
    print("Starting training...")
    
    if config.get("use_quantization", False):
        # Quantization-aware training workflow
        print("Starting quantization-aware training...")
        
        # Freeze backbone if requested
        if config.get("freeze_backbone", True):
            for param in model.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen for QAT")
        
        # Use lower learning rate for QAT
        qat_lr = config.get("qat_learning_rate", 1e-5)
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=qat_lr, 
            weight_decay=config.get("weight_decay", 1e-4)
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.get("lr_step_size", 30), 
            gamma=config.get("lr_gamma", 0.5)
        )
        
        # Train for QAT epochs
        qat_epochs = config.get("qat_epochs", 50)
        history = train_model_functional(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,  # Added validation loader
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=qat_epochs,
            device=device,
            save_path=f"{config['model_save_path']}_qat",
            save_interval=config.get("save_interval", 10),
            early_stopping_patience=config.get("early_stopping_patience", 20),
            use_stronger_augmentation=config.get("use_stronger_augmentation", True)
        )
    else:
        # Regular training
        history = train_model_functional(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,  # Added validation loader
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config["num_epochs"],
            device=device,
            save_path=config["model_save_path"],
            save_interval=config.get("save_interval", 10),
            early_stopping_patience=config.get("early_stopping_patience", 20),
            use_stronger_augmentation=config.get("use_stronger_augmentation", True)
        )
    
    # Evaluate model (regular model evaluation)
    print("Evaluating model...")
    val_loss = evaluate_model(
        model, 
        test_loader, 
        device, 
        config.get("results_dir", "results"),
        use_fp16=config.get("use_fp16", False)
    )
    print(f"Regular model validation loss: {val_loss:.4f}")
    
    # Convert to quantized if using QAT
    if config.get("use_quantization", False):
        print("Converting to final quantized model...")
        model = convert_to_quantized(model)
        torch.save(model.state_dict(), f"{config['model_save_path']}_quantized.pth")
        
        # Evaluate quantized model
        print("Evaluating quantized model...")
        quantized_val_loss = evaluate_quantized_model(
            model, 
            test_loader, 
            device, 
            config.get("results_dir", "results")
        )
        print(f"Quantized model validation loss: {quantized_val_loss:.4f}")
    
    # Export model
    if config.get("export_model", False):
        print("Exporting model...")
        export_results = export_model_pipeline(
            model,
            config.get("export_name", "keypoint_model"),
            config.get("export_dir", "models"),
            formats=config.get("export_formats", ["onnx"])
        )
        print(f"Export results: {export_results}")
    
    return model, history

# Configuration
DEFAULT_CONFIG = {
    "seed": 42,
    "yolo_model_path": "models/yolo_finetuned/best.pt",  # Use finetuned model for our dataset
    "keypoints_data_src": "via_proj/via_project_22Aug2025_16h07m06s.json",
    "image_path": "RGB-images-jpg/",
    "allowed_classes": [1],
    "batch_size": 32,
    "learning_rate": 3e-5,  # Lower LR for better convergence on small dataset
    "weight_decay": 5e-5,  # More conservative weight decay for small dataset
    "num_epochs": 300,
    "lr_step_size": 50,  # Longer steps for more stable training
    "lr_gamma": 0.7,  # Less aggressive decay
    "warmup_epochs": 5,  # Shorter warmup for better convergence
    "model_save_path": "models/keypoint_model_vit_depth",
    "save_interval": 10,
    "results_dir": "results",
    "use_quantization": False,
    "pretrained_path": "models/keypoint_model_vit.pth",  # Required for quantization fine-tuning
    "export_model": True,
    "export_name": "keypoint_model",
    "export_dir": "models",
    "export_formats": ["onnx"],
    "qat_epochs": 50,  # Number of epochs for quantization-aware training
    "freeze_backbone": True,  # Freeze backbone during QAT
    "qat_learning_rate": 1e-5,  # Lower learni    "use_augmentation": True,  # Use data augmentation
    "use_mixup": True,  # Use mixup augmentation
    "mixup_alpha": 0.2,  # Mixup alpha parameter
    "use_fp16": True,  # Use FP16 training for speed
    "gradient_clip_val": 1.0,  # Add gradient clipping for stability
    "gradient_accumulation_steps": 2,  # Accumulate gradients for stability
    "early_stopping_patience": 20,  # Early stopping patience
    "use_stronger_augmentation": True,  # Use stronger augmentation to prevent overfitting
    "dropout_rate": 0.1,  # Dropout rate for regularization
    "label_smoothing": 0.1  # Label smoothing for better generalization
}

if __name__ == "__main__":
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    # Override with command line arguments if provided
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config.update(json.load(f))
    
    # Run training pipeline
    model, history = main_training_pipeline(config)
    
    print("Training completed successfully!")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
