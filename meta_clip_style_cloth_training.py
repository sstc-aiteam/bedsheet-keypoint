#!/usr/bin/env python3
"""
Meta CLIP-style cloth keypoint training with a single-channel heatmap target.

Design:
- Backbone: Meta CLIP ViT-B/16 vision encoder (Facebook)
- Head: lightweight upsampling conv head to full-resolution 1xHxW heatmap
- Target: single heatmap with Gaussians at all visible corners (some images may have <4)
- Loss: KL heatmap loss vs on-the-fly blurred GT (single channel)
- Augmentations: horizontal/vertical flips with keypoint correction; mild photometrics
- Optional LoRA on vision attention projections (PEFT)

Outputs:
- Saves best adapters (if LoRA) and head weights in models/meta_clip_style_cloth
- Persists training_config.json with image_size, etc.
"""

import os
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from PIL import Image

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import CLIP model from src.models
from src.models import ClipHeatmapModel, ClipHeatmapHead, create_clip_heatmap_model

try:
    from transformers import CLIPModel, AutoTokenizer
    HF_AVAILABLE = True
except Exception as e:
    print(f"Transformers not available: {e}")
    HF_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except Exception as e:
    print(f"PEFT not available: {e}")
    PEFT_AVAILABLE = False

# Project utils
from src.utils.model_utils import kl_heatmap_loss, batch_gaussian_blur


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_cloth_keypoints(keypoint_file: str) -> List[Dict]:
    keypoints = []
    try:
        with open(keypoint_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line and ',' in line:
                parts = line.split(',')
                if len(parts) >= 2:
                    x_pixel, y_pixel = float(parts[0]), float(parts[1])
                    # Only add keypoints that are visible (not -1, -1)
                    if x_pixel >= 0 and y_pixel >= 0:
                        keypoints.append({'x': x_pixel, 'y': y_pixel})
    except Exception as e:
        print(f"Error loading keypoints from {keypoint_file}: {e}")
    return keypoints


def extract_keypoints_from_heatmap(heatmap: np.ndarray, max_keypoints: int = 4, min_distance: int = 20) -> List[Tuple[int, int]]:
    """Extract keypoint coordinates from heatmap using intelligent detection."""
    from scipy.ndimage import maximum_filter, gaussian_filter
    import cv2
    
    smoothed_heatmap = gaussian_filter(heatmap, sigma=1.0)
    max_val = smoothed_heatmap.max()
    threshold = max(0.1, max_val * 0.3)
    
    # Local maxima detection
    local_maxima = maximum_filter(smoothed_heatmap, size=7) == smoothed_heatmap
    local_maxima = local_maxima & (smoothed_heatmap > threshold)
    peak_coords = np.where(local_maxima)
    peak_keypoints = [(int(x), int(y)) for y, x in zip(peak_coords[0], peak_coords[1])]
    
    # Contour-based detection
    binary_map = (smoothed_heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_keypoints = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            if 10 < cx < heatmap.shape[1] - 10 and 10 < cy < heatmap.shape[0] - 10:
                contour_keypoints.append((cx, cy))
    
    # Top-k highest values
    flat_indices = np.argsort(smoothed_heatmap.flatten())[-max_keypoints*2:]
    top_k_coords = []
    for idx in flat_indices:
        y, x = np.unravel_index(idx, smoothed_heatmap.shape)
        if smoothed_heatmap[y, x] > threshold:
            if 10 < x < heatmap.shape[1] - 10 and 10 < y < heatmap.shape[0] - 10:
                top_k_coords.append((x, y))
    
    # Combine and deduplicate
    all_keypoints = peak_keypoints + contour_keypoints + top_k_coords
    unique_keypoints = []
    for kp in all_keypoints:
        is_duplicate = False
        for existing_kp in unique_keypoints:
            if np.sqrt((kp[0] - existing_kp[0])**2 + (kp[1] - existing_kp[1])**2) < min_distance:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_keypoints.append(kp)
    
    # Sort by intensity and return top N
    unique_keypoints.sort(key=lambda kp: smoothed_heatmap[kp[1], kp[0]], reverse=True)
    
    # Fallback: if no keypoints found, use top-k values
    if len(unique_keypoints) == 0:
        flat_indices = np.argsort(smoothed_heatmap.flatten())[-max_keypoints:]
        for idx in flat_indices:
            y, x = np.unravel_index(idx, smoothed_heatmap.shape)
            if 10 < x < heatmap.shape[1] - 10 and 10 < y < heatmap.shape[0] - 10:
                unique_keypoints.append((x, y))
    
    return unique_keypoints[:max_keypoints]


class MetaClipClothHeatmapDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int = 256, max_samples: int = None, augment: bool = True,
                 pairs: List[Tuple[Path, Path]] | None = None):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.max_samples = max_samples
        self.augment = augment

        if pairs is None:
            images = sorted(list((self.data_dir / 'imgs').glob('*.png')) +
                            list((self.data_dir / 'imgs').glob('*.jpg')))
            if max_samples:
                images = images[:max_samples]
            pairs = []
            for img in images:
                kp_file = self.data_dir / 'keypoints' / f"{img.stem}.txt"
                if kp_file.exists():
                    pairs.append((img, kp_file))
        self.pairs = pairs
        print(f"Dataset pairs: {len(self.pairs)} | augment={self.augment}")

        # Meta CLIP normalization (same as CLIP for compatibility)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, kp_path = self.pairs[idx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img_rgb.shape[:2]

        # Load keypoints and convert to pixel coordinates
        kps = load_cloth_keypoints(str(kp_path))
        gt_keypoints = [(int(kp['x']), int(kp['y'])) for kp in kps if 0 <= kp['x'] < W0 and 0 <= kp['y'] < H0]

        # Build initial heatmap at original size
        heat = np.zeros((H0, W0), dtype=np.float32)
        for x, y in gt_keypoints:
            if 0 <= x < W0 and 0 <= y < H0:
                heat[y, x] = 1.0

        # Resize to model input size
        img_rgb = cv2.resize(img_rgb, (self.image_size, self.image_size))
        heat = cv2.resize(heat, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        H = W = self.image_size

        # Scale keypoint coordinates to new size
        scale_x = W / W0
        scale_y = H / H0
        gt_keypoints = [(int(x * scale_x), int(y * scale_y)) for x, y in gt_keypoints]

        # To tensors
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - self.mean) / self.std
        heat_t = torch.from_numpy(heat).unsqueeze(0)  # (1,H,W)

        # Create sample dictionary
        sample = {
            'pixel_values': img_t,
            'gt_heatmap': heat_t,
            'image_id': img_path.stem,
            'gt_points': gt_keypoints,
        }

        # Apply heatmap-based augmentation if enabled
        if self.augment:
            augmentation = ClothAugmentation(self.image_size)
            sample = augmentation(sample)

        return sample


def collate_meta_clip_batch(batch):
    """Custom collate to handle variable-length gt_points.

    Stacks tensors and keeps image_id and gt_points as python lists.
    """
    pixel_values = torch.stack([b['pixel_values'] for b in batch], dim=0)
    gt_heatmap = torch.stack([b['gt_heatmap'] for b in batch], dim=0)
    image_id = [b['image_id'] for b in batch]
    gt_points = [b['gt_points'] for b in batch]
    return {
        'pixel_values': pixel_values,
        'gt_heatmap': gt_heatmap,
        'image_id': image_id,
        'gt_points': gt_points,
    }


class ClothAugmentation:
    """Data augmentation for cloth keypoint detection using heatmap-based spatial transformations."""
    
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
        original_heatmap = sample['gt_heatmap'].numpy().squeeze(0)  # Original one-hot encoded heatmap
        
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
        import random
        
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
    
    def _apply_photometric_augmentations(self, img: np.ndarray) -> np.ndarray:
        """Apply photometric augmentations to the image."""
        import random
        
        # Random brightness adjustment
        if random.random() < 0.5:  # 50% chance
            brightness_factor = random.uniform(0.8, 1.2)
            img = np.clip(img * brightness_factor, 0, 1)
        
        # Random contrast adjustment
        if random.random() < 0.5:  # 50% chance
            contrast_factor = random.uniform(0.8, 1.2)
            mean = img.mean()
            img = np.clip((img - mean) * contrast_factor + mean, 0, 1)
        
        # Random color jitter
        if random.random() < 0.3:  # 30% chance
            for c in range(3):  # RGB channels
                jitter = random.uniform(-0.1, 0.1)
                img[:, :, c] = np.clip(img[:, :, c] + jitter, 0, 1)
        
        # Random Gaussian noise
        if random.random() < 0.3:  # 30% chance
            noise = np.random.normal(0, 0.02, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        return img


def adjust_for_vram(cfg):
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_mem_gb = props.total_memory / (1024 ** 3)
            print(f"Detected GPU VRAM: {total_mem_gb:.1f} GiB")
            if cfg.get('auto_image_size', False):
                if total_mem_gb >= 20:
                    cfg['image_size'] = max(cfg['image_size'], 384)
                    cfg['batch_size'] = max(cfg['batch_size'], 8)
                elif total_mem_gb >= 12:
                    cfg['image_size'] = max(cfg['image_size'], 320)
                    cfg['batch_size'] = max(cfg['batch_size'], 4)
                else:
                    cfg['image_size'] = max(cfg['image_size'], 256)
                    cfg['batch_size'] = max(cfg['batch_size'], 2)
    except Exception as e:
        print(f"VRAM check failed: {e}")
    return cfg


def save_config(cfg: Dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'training_config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)


def train_meta_clip_heatmap():
    set_seed(42)

    if not HF_AVAILABLE:
        print("Transformers not available; install transformers to proceed.")
        return

    # Meta CLIP model configuration
    config = {
        'model_name': 'facebook/metaclip-b16-fullcc2.5b',  # Meta CLIP model
        'data_dir': 'cloth_data_gen/bedsheet_dataset_3000',
        'output_dir': 'models/meta_clip_style_cloth',
        'image_size': 256,
        'auto_image_size': True,
        'batch_size': 4,
        'num_epochs': 20,  # Increased epochs for larger dataset
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'use_fp16': True,
        'use_lora': True,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'use_text_prior': True,
        'prior_prompts': [
            "a photo of a cloth corner",
            "fabric corner point",
            "sharp cloth corner",
            "textile edge corner",  # Enhanced prompts for Meta CLIP
            "fabric fold corner",
            "cloth seam corner"
        ],
        'negative_prompts': [
            "smooth fabric surface",
            "flat textile area", 
            "cloth without corners",
            "fabric center area"
        ],
        'prior_weight': 0.5,
        'max_samples': None,  # Use all 3000 samples
        'early_stopping_patience': 15,  # Increased patience for longer training
        'splits': {'train': 0.8, 'val': 0.1, 'test': 0.1},
        'results_dir': 'results_meta_clip',
    }

    config = adjust_for_vram(config)
    print(f"Training with Meta CLIP model: {config['model_name']}")
    print(f"Training with image_size={config['image_size']} batch_size={config['batch_size']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create base dataset without augmentation
    base_dataset = MetaClipClothHeatmapDataset(config['data_dir'], config['image_size'], config['max_samples'], augment=False)
    
    # Split dataset into train, validation, and test using PyTorch's random_split
    total_size = len(base_dataset)
    train_size = int(config['splits']['train'] * total_size)
    val_size = int(config['splits']['val'] * total_size)
    test_size = total_size - train_size - val_size
    
    train_indices, val_indices, test_indices = torch.utils.data.random_split(
        range(total_size), [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create datasets with proper splitting
    train_dataset = MetaClipClothHeatmapDataset(config['data_dir'], config['image_size'], config['max_samples'], augment=True)
    
    # Create proper subsets using torch.utils.data.Subset
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(base_dataset, val_indices.indices)
    test_subset = torch.utils.data.Subset(base_dataset, test_indices.indices)
    
    print(f"Dataset split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")

    # Save a few augmented previews to verify transforms
    try:
        os.makedirs(config['output_dir'], exist_ok=True)
        mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 1, 3)
        std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 1, 3)
        for i in range(min(3, len(train_subset))):
            sample = train_subset[i]
            img = sample['pixel_values'].permute(1, 2, 0).numpy()  # HWC in CLIP norm
            img = (img * std + mean).clip(0, 1)
            img = (img * 255).astype(np.uint8)
            heat = sample['gt_heatmap'].squeeze(0).numpy()
            heat_u8 = (np.clip(heat / (heat.max() + 1e-6), 0, 1) * 255).astype(np.uint8)
            heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
            heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(img, 0.7, heat_color, 0.3, 0)
            # Draw points if provided
            for (x, y) in sample['gt_points']:
                cv2.circle(overlay, (int(x), int(y)), 3, (255, 255, 255), -1)
            cv2.imwrite(os.path.join(config['output_dir'], f'meta_clip_aug_preview_{i}.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved Meta CLIP augmentation previews to {config['output_dir']}/meta_clip_aug_preview_*.png")
    except Exception as e:
        print(f"Meta CLIP augmentation preview failed: {e}")

    # Create data loaders with proper subsets
    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=0, collate_fn=collate_meta_clip_batch)
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_meta_clip_batch)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_meta_clip_batch)

    # Model - using Meta CLIP
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
    ).to(device)

    # Optim
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scaler = GradScaler(enabled=config['use_fp16'])

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    os.makedirs(config['output_dir'], exist_ok=True)
    save_config(config, config['output_dir'])

    best_val = float('inf')
    patience = 0
    global_step = 0

    # Optional TensorBoard logging
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(config['output_dir'], 'tb')
        writer = SummaryWriter(tb_dir)
        print(f"TensorBoard logging to {tb_dir}")
    except Exception as e:
        print(f"TensorBoard unavailable: {e}")

    # CSV fallback logging
    csv_path = os.path.join(config['output_dir'], 'training_log.csv')
    try:
        if not os.path.exists(csv_path):
            with open(csv_path, 'w') as f:
                f.write('step,epoch,split,loss\n')
    except Exception as e:
        print(f"CSV log init failed: {e}")

    def compute_sigma(H):
        return max(1.0, 0.03 * H)

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc='Train')
        for batch in pbar:
            pix = batch['pixel_values'].to(device)
            gt = batch['gt_heatmap'].to(device)

            optimizer.zero_grad()
            with autocast(enabled=config['use_fp16']):
                pred = model(pix)  # (B,1,H,W), positive via softplus
                # On-the-fly blur on GT
                sigma = compute_sigma(pred.shape[-1])
                k = int(6 * sigma + 1)
                k = k if k % 2 == 1 else k + 1
                gt_blur = batch_gaussian_blur(gt, kernel_size=min(k, 61), sigma=float(sigma))
                loss = kl_heatmap_loss(pred, gt_blur, reduction='mean')

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item()) * pix.size(0)
            global_step += 1

            # Live progress
            avg = running / max(1, (global_step * train_loader.batch_size))
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg:.4f}")

            # Loggers
            if writer is not None:
                writer.add_scalar('train/loss', float(loss.item()), global_step)
            try:
                with open(csv_path, 'a') as f:
                    f.write(f"{global_step},{epoch+1},train,{float(loss.item())}\n")
            except Exception:
                pass

        scheduler.step()
        train_loss = running / len(train_loader.dataset)
        print(f"Train loss: {train_loss:.4f}")

        # Validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Val'):
                pix = batch['pixel_values'].to(device)
                gt = batch['gt_heatmap'].to(device)
                pred = model(pix)
                sigma = compute_sigma(pred.shape[-1])
                k = int(6 * sigma + 1)
                k = k if k % 2 == 1 else k + 1
                gt_blur = batch_gaussian_blur(gt, kernel_size=min(k, 61), sigma=float(sigma))
                vloss = kl_heatmap_loss(pred, gt_blur, reduction='mean')
                val_running += float(vloss.item()) * pix.size(0)

        val_loss = val_running / len(val_loader.dataset)
        print(f"Val loss: {val_loss:.4f}")

        if writer is not None:
            writer.add_scalar('val/loss', float(val_loss), global_step)
        try:
            with open(csv_path, 'a') as f:
                f.write(f"{global_step},{epoch+1},val,{float(val_loss)}\n")
        except Exception:
            pass

        # Early stopping + save best
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            # Save complete model state (includes all trainable parameters)
            torch.save(model.state_dict(), os.path.join(config['output_dir'], 'complete_model.pth'))
            
            # Also save LoRA adapters (if used) and head weights separately for compatibility
            if config['use_lora'] and PEFT_AVAILABLE:
                # Save adapters from the CLIP PEFT wrapper
                model.clip.save_pretrained(config['output_dir'])
            torch.save(model.head.state_dict(), os.path.join(config['output_dir'], 'head.pth'))
            print(f"Saved best Meta CLIP model (val={best_val:.4f}) to {config['output_dir']}")
        else:
            patience += 1
            if patience >= config['early_stopping_patience']:
                print("Early stopping triggered")
                break

    print("Meta CLIP training complete.")
    if writer is not None:
        writer.close()

    # --- Evaluation on test set with best checkpoint ---
    with torch.no_grad():
        # Use the already trained model (it already has the best weights loaded)
        eval_model = model
        eval_model.eval()

        # Prepare results dir
        results_dir = Path(config['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)

        from src.utils.keypoint_metrics import match_keypoints
        from shared.functions import thresholded_locations

        detailed = []
        total_dist = []
        matched_total = 0
        total_gt_points = 0

        for batch in tqdm(test_loader, desc='Test'):
            pix = batch['pixel_values'].to(device)
            gt_points = batch['gt_points'][0]  # list of tuples
            image_id = batch['image_id'][0]

            pred = eval_model(pix)  # (1,1,H,W)
            heat = pred.squeeze(0).squeeze(0).detach().cpu().numpy()
            m = heat.max() if heat.size > 0 else 1.0
            if m > 0:
                heat = heat / m

            peaks = thresholded_locations(heat, threshold=0.3)
            peaks_xy = [(int(p[1]), int(p[0])) for p in peaks]

            # Use streamlined matching function
            matched, dists = match_keypoints(gt_points, peaks_xy, threshold=10.0)

            total_gt_points += len(gt_points)
            matched_total += matched
            total_dist.extend(dists)

            # Overlay
            vis = pix.detach().cpu()[0]
            # unnormalize CLIP
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            vis = (vis * std + mean).clamp(0, 1)
            vis = (vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # Ensure array is contiguous for OpenCV
            vis = np.ascontiguousarray(vis)
            for (x, y) in gt_points:
                cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1)
            for (pxx, pyy) in peaks_xy:
                cv2.circle(vis, (int(pxx), int(pyy)), 3, (255, 0, 0), -1)
            cv2.imwrite(str(results_dir / f"{image_id}_meta_clip_heatmap.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            detailed.append({
                'image': f"{image_id}.png",
                'matched': int(matched),
                'total_gt': int(len(gt_points)),
                'avg_distance': float(np.mean(dists)) if dists else None,
                'peaks': peaks_xy,
                'gt_pixels': gt_points,
            })

        summary = {
            'overall_match_rate': float(matched_total / max(1, total_gt_points)),
            'avg_distance_all': float(np.mean(total_dist)) if total_dist else None,
            'samples': len(detailed),
            'details': detailed,
        }
        with open(results_dir / 'meta_clip_evaluation_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved Meta CLIP test evaluation to {results_dir}/meta_clip_evaluation_results.json")


if __name__ == '__main__':
    train_meta_clip_heatmap()
