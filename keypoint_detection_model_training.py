#!/usr/bin/env python3
"""
Keypoint Detection Model Training with Enhanced YOLO Backbone
Migrated from keypoint_detection_model_training.ipynb
Updated to use the new src/ structure and Enhanced YOLO Backbone

Enhanced YOLO Backbone Benefits:
- Full architecture utilization (backbone + neck)
- Proper skip connection handling
- Strategic feature selection for better performance
- Multi-scale feature pyramid network (FPN) features

Usage Options:
- create_model(use_enhanced_yolo=True) for enhanced YOLO (default, recommended)
- create_model(use_enhanced_yolo=False) for original YOLO (fallback)
"""

import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random
import torch.optim as optim
import torch.nn as nn
import time
from torch.amp import autocast, GradScaler
from ultralytics import YOLO

# Import from the new src structure
from shared.functions import *
from src.models.hybrid_keypoint_net import HybridKeypointNet
from src.models.efficient_keypoint_net import EfficientViTKeypointNet
from src.utils.model_utils import (
    YoloBackbone,
    EnhancedYoloBackbone, 
    batch_gaussian_blur, 
    batch_entropy, 
    thresholded_locations,
    normalize_heatmaps
)

# Import enhanced augmentation
from src.augmentation.lighting_color_augmentation import create_lighting_color_augmentation

# Import UNet if needed (commented out in original)
# from models.unet import UNet

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

def kl_heatmap_loss(pred_hm, gt_hm, mask=None, reduction='mean'):
    """KL divergence heatmap loss function"""
    # pred_hm: (B, 1, H, W)
    # gt_hm:   (B, 1, H, W)
    # mask:    (B, 1, H, W) or None

    eps = 1e-8

    # Force positive
    pred_probs = pred_hm.clamp(min=eps)
    gt_probs = gt_hm.clamp(min=eps)

    # Optionally apply mask
    if mask is not None:
        pred_probs = pred_probs * mask
        gt_probs = gt_probs * mask

    # Sum per sample
    pred_sum = pred_probs.sum(dim=(2, 3), keepdim=True)
    gt_sum = gt_probs.sum(dim=(2, 3), keepdim=True)

    # Identify gt_hm slices that are all zeros (or close enough)
    gt_zero_mask = (gt_sum < eps).squeeze(1).squeeze(1)  # (B,) boolean: True means skip or zero out

    # Safe normalization (avoids divide by zero)
    pred_probs = pred_probs / pred_sum.clamp(min=eps)
    gt_probs = torch.where(gt_sum < eps, torch.zeros_like(gt_probs), gt_probs / gt_sum.clamp(min=eps))

    # Compute KL divergence per sample
    log_pred = pred_probs.log()
    kl_div = F.kl_div(log_pred, gt_probs, reduction='none').sum(dim=(2, 3))  # shape (B,1)
    kl_div = kl_div.squeeze(1)  # (B,)

    # For samples where gt_hm is all zeros, set loss to 0 (no supervision there)
    kl_div = kl_div.masked_fill(gt_zero_mask, 0.)

    if reduction == 'mean':
        num = (~gt_zero_mask).float().sum().clamp(min=1)
        return kl_div.sum() / num
    elif reduction == 'sum':
        return kl_div.sum()
    else:
        return kl_div

def combine_nearby_peaks(heatmap, threshold=0.003, min_distance=5):
    """
    Find peaks in heatmap above threshold and combine nearby peaks into single keypoints.
    
    Args:
        heatmap: 2D numpy array of heatmap values
        threshold: Minimum value to consider as a peak
        min_distance: Minimum distance between peaks to keep them separate
    
    Returns:
        List of (row, col) tuples representing combined peak locations
    """
    # Find all locations above threshold
    above_threshold = np.where(heatmap > threshold)
    
    if len(above_threshold[0]) == 0:
        return []
    
    # Get coordinates and values
    coords = list(zip(above_threshold[0], above_threshold[1]))
    values = [heatmap[y, x] for y, x in coords]
    
    # Sort by value (highest first)
    sorted_indices = np.argsort(values)[::-1]
    sorted_coords = [coords[i] for i in sorted_indices]
    
    # Combine nearby peaks
    combined_peaks = []
    for coord in sorted_coords:
        # Check if this peak is far enough from existing combined peaks
        if not combined_peaks:
            combined_peaks.append(coord)
        else:
            # Calculate distances to existing peaks
            distances = cdist([coord], combined_peaks, metric='euclidean')[0]
            if np.min(distances) >= min_distance:
                combined_peaks.append(coord)
    
    return combined_peaks

class RandomRotateFlip:
    """
    Randomly applies:
    - A rotation by any angle in [0, 360)
    - Optionally, a horizontal flip with 50% chance after rotation
    """
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        # image: (C, H, W)
        # keypoints: (N, H, W) or (H, W)

        # --- Random rotation ---
        angle = random.uniform(0, 360)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        # For keypoints as heatmaps, use same rotate (assume keypoints is Tensor [N,H,W] or [H,W])
        # If N, treat each as a channel
        if keypoints.ndim == 3:
            keypoints = TF.rotate(keypoints, angle, interpolation=TF.InterpolationMode.BILINEAR)
        else:
            keypoints = TF.rotate(keypoints.unsqueeze(0), angle, interpolation=TF.InterpolationMode.BILINEAR).squeeze(0)

        # --- Random flip after rotation ---
        if random.random() < 0.5:
            image = TF.hflip(image)
            keypoints = TF.hflip(keypoints)
        if random.random() < 0.5:
            image = TF.vflip(image)
            keypoints = TF.vflip(keypoints)

        return {'image': image, 'keypoints': keypoints}

class KeypointDataset(Dataset):
    def __init__(self, images, keypoints, transform=None):
        self.images = images.astype(np.float32) / 255
        self.keypoints = keypoints.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # shape (400, 400, 3)
        kp = self.keypoints[idx]  # shape (4, 2)
        img = np.transpose(img, (2, 0, 1))  # channels first
        sample = {'image': torch.from_numpy(img), 'keypoints': torch.from_numpy(kp)}
        if self.transform:
            sample = self.transform(sample)
        return sample

def load_data():
    """Load and prepare the dataset"""
    image_data_dir = "cloth_data_gen/output/images"
    keypoint_data_dir = "cloth_data_gen/output/keypoints"

    img_arr = []
    keypoints_img_arr = []
    for img_file in os.listdir(image_data_dir):
        if img_file.endswith('.png'):
            name = img_file.split('.')[0]
            keypoint_file = os.path.join(keypoint_data_dir, name + '.txt')
            image_path = os.path.join(image_data_dir, img_file)
            img = cv2.imread(image_path)
            keypoints = pd.read_csv(keypoint_file)
            pixels_coords = keypoints[['x_pixel', 'y_pixel']].values
            kimg = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            karr = []
            # check if all pixels coordinates are within the image bounds
            if pixels_coords.shape[0] > 0 and np.all((pixels_coords[:, 0] >= 0) & (pixels_coords[:, 0] < img.shape[1]) &
                                                      (pixels_coords[:, 1] >= 0) & (pixels_coords[:, 1] < img.shape[0])):
                kp_img = np.zeros((128, 128))
                for point in pixels_coords:
                    kp_img[int(point[1]), int(point[0])] = 1
                keypoints_img_arr.append(kp_img)
                img_arr.append(img)
    
    img_arr = np.array(img_arr)
    keypoints_img_arr = np.array(keypoints_img_arr)
    
    return img_arr, keypoints_img_arr

def test_keypoint_visualization(full_dataset, index=9):
    """Test keypoint visualization on a sample"""
    pair = full_dataset.__getitem__(index)
    img = pair["image"].numpy().copy()
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    kp = pair["keypoints"].numpy()
    print(f"Keypoint shape: {kp.shape}")
    for i in range(kp.shape[0]):
        for j in range(kp.shape[1]):
            if kp[i,j] > 0.1:
                cv2.circle(img, (j, i), 1, (0,0,255), -1)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Keypoint Visualization - Sample {index}")
    plt.axis('off')
    plt.show()

def create_model(use_enhanced_yolo=True):
    """Create and configure the model"""
    
    if use_enhanced_yolo:
        # Enhanced YOLO + ViT (recommended)
        print("Creating Enhanced YOLO + ViT model...")
        yolo_model = YOLO('yolo11l-pose.pt')
        backbone = EnhancedYoloBackbone(
            yolo_model, 
            include_neck=True,  # Include neck features for better multi-scale representation
            selected_indices=[2, 4, 6, 8, 10, 13, 16, 19, 22]  # Strategic feature selection
        )
        
        input_dummy = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            feats = backbone(input_dummy)
            in_channels_list = [f.shape[1] for f in feats]
            print(f"Enhanced YOLO backbone features: {len(feats)}")
            print(f"Input channels: {in_channels_list}")
        
        keypoint_net = HybridKeypointNet(backbone, in_channels_list)
        model = keypoint_net
        
    else:
        # Original YOLO + ViT (fallback)
        print("Creating Original YOLO + ViT model...")
        yolo_model = YOLO('yolo11l-pose.pt')
        backbone_seq = yolo_model.model.model[:12]
        backbone = YoloBackbone(backbone_seq, selected_indices=[0,1,2,3,4,5,6,7,8,9,10,11])
        
        input_dummy = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            feats = backbone(input_dummy)
            in_channels_list = [f.shape[1] for f in feats]
            print(f"Original YOLO backbone features: {len(feats)}")
            print(f"Input channels: {in_channels_list}")
        
        keypoint_net = HybridKeypointNet(backbone, in_channels_list)
        model = keypoint_net
    
    # Freeze backbone parameters
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    return model

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

def train_model(model, trainloader, valloader, testloader, num_epochs=300, load_model=False, early_stopping_patience: int = 100):
    """Train the model"""
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
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            time_start = time.time()
            compiled_model.train()
            running_loss = 0.0

            total_train_batches = len(trainloader)
            for batch_idx, batch in enumerate(trainloader):
                images = batch["image"].to(device)
                keypoints = batch["keypoints"].to(device)
                optimizer.zero_grad()

                with autocast("cuda", dtype=torch.float16):      # AMP context, not forcing .half()
                    outputs = compiled_model(images)
                    keypoints_blur = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
                    
                    # active learning: Uncertainty Sampling using entropy as the uncertainty metric
                    entropies = batch_entropy(outputs)
                    k = max(1, images.size(0) // 2)
                    topk_vals, topk_idx = torch.topk(entropies, k, largest=True)  # highest entropy first
                    selected_outputs = outputs[topk_idx]
                    selected_keypoints_blur = keypoints_blur[topk_idx]

                    # calculate loss
                    loss = kl_heatmap_loss(normalize_heatmaps(selected_outputs), selected_keypoints_blur.unsqueeze(1))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * images.size(0)
                # Progress output
                print(f"Epoch {epoch+1}/{num_epochs} [Train {batch_idx+1}/{total_train_batches}] loss: {loss.item():.4f}", end='\r')
            print()

            train_epoch_loss = running_loss / len(trainloader.dataset)

            # Validation
            compiled_model.eval()
            val_running_loss = 0.0
            total_val_batches = len(valloader)
            with torch.no_grad():
                for batch_idx, batch in enumerate(valloader):
                    images = batch["image"].to(device)
                    keypoints = batch["keypoints"].to(device)
                    with autocast("cuda", dtype=torch.float16):
                        outputs = compiled_model(images)
                        keypoints_blur = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
                        vloss = kl_heatmap_loss(normalize_heatmaps(outputs), keypoints_blur.unsqueeze(1))
                    val_running_loss += vloss.item() * images.size(0)
                    print(f"Epoch {epoch+1}/{num_epochs} [Val {batch_idx+1}/{total_val_batches}] loss: {vloss.item():.4f}", end='\r')
            print()

            val_epoch_loss = val_running_loss / len(valloader.dataset)
            print(f'Epoch {epoch+1}: Train Loss {train_epoch_loss:.4f}, Val Loss {val_epoch_loss:.4f}, Time: {time.time() - time_start:.2f}s')

            # Early stopping
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                patience_counter = 0
                save_model_safely(compiled_model, 'models/keypoint_model_vit.pth')
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"Validation loss didn't improve. Patience: {patience_counter}/{early_stopping_patience}")
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
    else:
        # load safely into compiled model
        load_model_safely(compiled_model, 'models/keypoint_model_vit.pth', map_location=device, strict=False)
        compiled_model.eval()
    
    return compiled_model

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
                loss = kl_heatmap_loss(normalize_heatmaps(outputs), keypoints_blur.unsqueeze(1))

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

def main():
    """Main training pipeline"""
    print("=== Keypoint Detection Model Training ===")
    
    # Load data
    print("Loading data...")
    img_arr, keypoints_img_arr = load_data()
    print(f"Loaded {len(img_arr)} images with keypoints")
    
    # Create datasets with enhanced augmentation
    print("Creating datasets...")
    
    # Use enhanced augmentation instead of basic rotation/flip
    enhanced_transform = create_lighting_color_augmentation(
        image_size=128,  # Default image size for this model
        intensity='medium',
        augmentation_type='cloth'  # Use cloth augmentation for general keypoint detection
    )
    
    # Create the full dataset without transform
    full_dataset = KeypointDataset(img_arr, keypoints_img_arr, transform=None)
    
    # Split into train/val/test (70/10/20)
    total_len = len(full_dataset)
    train_size = int(0.7 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size
    train_indices, val_indices, test_indices = torch.utils.data.random_split(range(total_len), [train_size, val_size, test_size])
    
    # Create datasets with enhanced transforms
    train_dataset = torch.utils.data.Subset(KeypointDataset(img_arr, keypoints_img_arr, transform=enhanced_transform), train_indices)
    val_dataset = torch.utils.data.Subset(KeypointDataset(img_arr, keypoints_img_arr, transform=None), val_indices)
    test_dataset = torch.utils.data.Subset(KeypointDataset(img_arr, keypoints_img_arr, transform=None), test_indices)
    
    trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
    valloader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test keypoint visualization
    print("Testing keypoint visualization...")
    test_keypoint_visualization(full_dataset)
    
    # Create model
    print("Creating model...")
    # Options: create_model(use_enhanced_yolo=True) for enhanced YOLO (default)
    #          create_model(use_enhanced_yolo=False) for original YOLO
    model = create_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    print("Starting training...")
    load_model = False  # Set to True to load existing model
    trained_model = train_model(model, trainloader, valloader, testloader, num_epochs=300, load_model=load_model, early_stopping_patience=100)
    
    # Evaluate model
    print("Evaluating model...")
    val_loss = evaluate_model(trained_model, testloader)
    print(f"Final validation loss: {val_loss:.4f}")
    
    # Visualize model architecture
    print("Visualizing model architecture...")
    visualize_model_architecture(model)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
