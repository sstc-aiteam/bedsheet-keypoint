#!/usr/bin/env python3
"""
Keypoint Detection Model Training
Migrated from keypoint_detection_model_training.ipynb
Updated to use the new src/ structure
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
    batch_gaussian_blur, 
    batch_entropy, 
    thresholded_locations
)

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

def create_model():
    """Create and configure the model"""
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

def train_model(model, trainloader, testloader, num_epochs=300, load_model=False):
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

        for epoch in range(num_epochs):
            time_start = time.time()
            compiled_model.train()
            running_loss = 0.0

            for batch in trainloader:
                images = batch["image"].to(device)
                keypoints = batch["keypoints"].to(device)
                optimizer.zero_grad()

                with autocast("cuda", dtype=torch.float16):      # AMP context, not forcing .half()
                    outputs = compiled_model(images)
                    keypoints_blur = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
                    
                    # active learning: Uncertainty Sampling using entropy as the uncertainty metric
                    entropies = batch_entropy(outputs)
                    k = images.size(0) // 2
                    topk_vals, topk_idx = torch.topk(entropies, k, largest=True)  # highest entropy first
                    selected_outputs = outputs[topk_idx]
                    selected_keypoints_blur = keypoints_blur[topk_idx]

                    # calculate loss
                    loss = kl_heatmap_loss(selected_outputs, selected_keypoints_blur.unsqueeze(1))

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item() * images.size(0)
            
            print(f'Epoch {epoch+1}: Loss {running_loss / len(trainloader.dataset):.4f} time seconds: {time.time() - time_start}')

        # save the model
        torch.save(compiled_model.state_dict(), 'models/keypoint_model_vit.pth')
    else:
        compiled_model.load_state_dict(torch.load('models/keypoint_model_vit.pth', map_location=device))
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
    
    # Create datasets
    print("Creating datasets...")
    rotate_transform = RandomRotateFlip()
    
    # Create the full dataset without transform
    full_dataset = KeypointDataset(img_arr, keypoints_img_arr, transform=None)
    
    # Split indices for train/test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_indices, test_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, test_size])
    
    # Create train and test datasets with/without transform
    train_dataset = torch.utils.data.Subset(KeypointDataset(img_arr, keypoints_img_arr, transform=rotate_transform), train_indices)
    test_dataset = torch.utils.data.Subset(KeypointDataset(img_arr, keypoints_img_arr, transform=None), test_indices)
    
    trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
    testloader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Test keypoint visualization
    print("Testing keypoint visualization...")
    test_keypoint_visualization(full_dataset)
    
    # Create model
    print("Creating model...")
    model = create_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    print("Starting training...")
    load_model = False  # Set to True to load existing model
    trained_model = train_model(model, trainloader, testloader, num_epochs=300, load_model=load_model)
    
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
