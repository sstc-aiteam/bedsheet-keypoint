#!/usr/bin/env python3
"""
CLIP-style cloth keypoint training with a single-channel heatmap target.

Design:
- Backbone: CLIP ViT-B/16 vision encoder (Hugging Face)
- Head: lightweight upsampling conv head to full-resolution 1xHxW heatmap
- Target: single heatmap with Gaussians at all visible corners (some images may have <4)
- Loss: KL heatmap loss vs on-the-fly blurred GT (single channel)
- Augmentations: horizontal/vertical flips with keypoint correction; mild photometrics
- Optional LoRA on vision attention projections (PEFT)

Outputs:
- Saves best adapters (if LoRA) and head weights in models/clip_style_cloth
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
        for line in lines[1:]:
            line = line.strip()
            if line and ',' in line:
                parts = line.split(',')
                if len(parts) >= 5:
                    x_pixel, y_pixel = float(parts[3]), float(parts[4])
                    keypoints.append({'x': x_pixel, 'y': y_pixel})
    except Exception as e:
        print(f"Error loading keypoints from {keypoint_file}: {e}")
    return keypoints


class ClipClothHeatmapDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int = 256, max_samples: int = None, augment: bool = True,
                 pairs: List[Tuple[Path, Path]] | None = None):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.max_samples = max_samples
        self.augment = augment

        if pairs is None:
            images = sorted(list((self.data_dir / 'images').glob('*.png')) +
                            list((self.data_dir / 'images').glob('*.jpg')))
            if max_samples:
                images = images[:max_samples]
            pairs = []
            for img in images:
                kp_file = self.data_dir / 'keypoints' / f"{img.stem}.txt"
                if kp_file.exists():
                    pairs.append((img, kp_file))
        self.pairs = pairs
        print(f"Dataset pairs: {len(self.pairs)} | augment={self.augment}")

        # CLIP normalization (OpenAI)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    def __len__(self):
        return len(self.pairs)

    def _apply_aug(self, img_rgb: np.ndarray, heat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Geometric augmentation applied jointly to image and heatmap.

        Simpler and consistent with the other script: rotate/flip the rasterized heatmap
        using the same ops as the image, so no manual keypoint math or renormalization.
        """
        # Convert to torch tensors for torchvision geometric ops
        img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3,H,W)
        heat_t = torch.from_numpy(heat.astype(np.float32)).unsqueeze(0)  # (1,H,W)

        # Geometric flips
        do_h = np.random.rand() < 0.2 if self.augment else False
        do_v = np.random.rand() < 0.2 if self.augment else False
        if do_h:
            img_t = TF.hflip(img_t)
            heat_t = TF.hflip(heat_t)
        if do_v:
            img_t = TF.vflip(img_t)
            heat_t = TF.vflip(heat_t)

        # Random rotation (small angles)
        if self.augment and np.random.rand() < 0.3:
            angle = float(np.random.uniform(-20.0, 20.0))
            img_t = TF.rotate(img_t, angle, interpolation=InterpolationMode.BILINEAR, fill=0.0)
            heat_t = TF.rotate(heat_t, angle, interpolation=InterpolationMode.NEAREST, fill=0.0)

        # Back to numpy
        img_rgb = (img_t.clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        heat = heat_t.squeeze(0).numpy().astype(np.float32)

        # Photometric jitter (mild) — image only
        if self.augment and np.random.rand() < 0.3:
            alpha = float(np.random.uniform(0.9, 1.1))
            img_rgb = cv2.convertScaleAbs(img_rgb, alpha=alpha, beta=0)
        return img_rgb, heat

    def __getitem__(self, idx):
        img_path, kp_path = self.pairs[idx]
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H0, W0 = img_rgb.shape[:2]

        # Load keypoints and convert to normalized coords
        kps = load_cloth_keypoints(str(kp_path))
        kp_norm = [{'x': kp['x'] / W0, 'y': kp['y'] / H0} for kp in kps]

        # Build heatmap at original size first (simplifies rotation handling)
        heat = np.zeros((H0, W0), dtype=np.float32)
        for kp in kp_norm:
            x0 = int(round(kp['x'] * (W0 - 1)))
            y0 = int(round(kp['y'] * (H0 - 1)))
            if 0 <= x0 < W0 and 0 <= y0 < H0:
                heat[y0, x0] = 1.0

        # Apply geometric augmentations jointly to image and heatmap
        img_rgb, heat = self._apply_aug(img_rgb, heat)

        # Resize both to model input size
        img_rgb = cv2.resize(img_rgb, (self.image_size, self.image_size))
        heat = cv2.resize(heat, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        H = W = self.image_size

        # To tensors
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        img_t = (img_t - self.mean) / self.std
        heat_t = torch.from_numpy(heat).unsqueeze(0)  # (1,H,W)

        # For test/val (no aug), we can still derive points from original normalized coords
        # For augmented samples, derive points from rasterized heatmap (nonzero locations)
        if not self.augment:
            gt_points = [(int(round(kp['x'] * (W - 1))), int(round(kp['y'] * (H - 1)))) for kp in kp_norm]
        else:
            ys, xs = np.where(heat > 0.5)
            gt_points = list(zip(xs.tolist(), ys.tolist()))

        return {
            'pixel_values': img_t,
            'gt_heatmap': heat_t,
            'image_id': img_path.stem,
            'gt_points': gt_points
        }


def collate_clip_batch(batch):
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


class ClipHeatmapHead(nn.Module):
    def __init__(self, in_dim: int, out_size: int):
        super().__init__()
        self.out_size = out_size
        self.proj = nn.Conv2d(in_dim, 256, kernel_size=1)
        self.block = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
        # feat_2d: (B, D, h, w)
        x = self.proj(feat_2d)
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
        x = self.block(x)
        x = self.out(x)
        # Ensure positive for KL loss
        return F.softplus(x)


class ClipHeatmapModel(nn.Module):
    def __init__(self, model_name: str, image_size: int, use_lora: bool = True, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05,
                 use_text_prior: bool = True, prior_prompts: List[str] = None, prior_weight: float = 0.5):
        super().__init__()
        # Load full CLIP model; we will use its vision submodule directly.
        clip = CLIPModel.from_pretrained(model_name)
        self.clip = clip
        self.vision = clip.vision_model
        self.hidden_size = self.vision.config.hidden_size
        self.patch_size = self.vision.config.patch_size
        self.image_size = image_size
        self.head = ClipHeatmapHead(self.hidden_size, image_size)
        self.use_lora = use_lora and PEFT_AVAILABLE
        self.use_text_prior = use_text_prior
        self.prior_weight = float(prior_weight)
        if prior_prompts is None:
            prior_prompts = [
                "a photo of a cloth corner",
                "fabric corner point",
                "sharp cloth corner"
            ]
        self.prior_prompts = prior_prompts
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.use_lora:
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
            )
            # Apply LoRA to the full CLIP model; vision submodules are shared.
            self.clip = get_peft_model(self.clip, lora_cfg)
            self.vision = self.clip.get_submodule('vision_model')
        else:
            # Freeze vision encoder if not using LoRA
            for p in self.vision.parameters():
                p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: (B,3,H,W)
        # Enable positional embedding interpolation for non-224 inputs
        outputs = self.vision(pixel_values=pixel_values, interpolate_pos_encoding=True)
        tokens = outputs.last_hidden_state  # (B, 1+P, D)
        # Remove CLS token and reshape to 2D feature map
        patch_tokens = tokens[:, 1:, :]  # (B,P,D)
        h = w = self.image_size // self.patch_size
        feat_2d = patch_tokens.transpose(1, 2).contiguous().view(pixel_values.size(0), self.hidden_size, h, w)

        # Optional text prior: compute patch-text similarity and gate features
        if self.use_text_prior:
            device = pixel_values.device
            enc = self.tokenizer(self.prior_prompts, padding=True, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                text_feats = self.clip.get_text_features(**enc)  # (P_text, d)
            # Normalize and average prompts
            text_feats = F.normalize(text_feats, dim=-1)
            text_vec = text_feats.mean(dim=0)  # (d,)
            text_vec = F.normalize(text_vec, dim=-1)

            # Project patch tokens to CLIP embed dim and normalize
            patch_proj = self.clip.visual_projection(patch_tokens)  # (B,P,d)
            patch_proj = F.normalize(patch_proj, dim=-1)
            # Similarity (B,P)
            sim = torch.matmul(patch_proj, text_vec.unsqueeze(-1)).squeeze(-1)
            # Scale to [0,1]
            sim = (sim + 1.0) * 0.5
            sim_hw = sim.view(pixel_values.size(0), 1, h, w)
            # Gate features
            feat_2d = feat_2d * (1.0 + self.prior_weight * sim_hw)
        heat = self.head(feat_2d)
        return heat


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


def train_clip_heatmap():
    set_seed(42)

    if not HF_AVAILABLE:
        print("Transformers not available; install transformers to proceed.")
        return

    config = {
        'model_name': 'openai/clip-vit-base-patch16',
        'data_dir': 'cloth_data_gen/output',
        'output_dir': 'models/clip_style_cloth',
        'image_size': 256,
        'auto_image_size': True,
        'batch_size': 4,
        'num_epochs': 10,
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
            "sharp cloth corner"
        ],
        'prior_weight': 0.5,
        'max_samples': None,
        'early_stopping_patience': 10,
        'splits': {'train': 0.8, 'val': 0.1, 'test': 0.1},
        'results_dir': 'results_clip'
    }

    config = adjust_for_vram(config)
    print(f"Training with image_size={config['image_size']} batch_size={config['batch_size']}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build full list of pairs for deterministic split
    scan_ds = ClipClothHeatmapDataset(config['data_dir'], config['image_size'], config['max_samples'], augment=False)
    pairs = scan_ds.pairs
    rng = np.random.RandomState(42)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(config['splits']['train'] * n)
    n_val = int(config['splits']['val'] * n)
    n_test = n - n_train - n_val
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    test_pairs = [pairs[i] for i in test_idx]

    train_ds = ClipClothHeatmapDataset(config['data_dir'], config['image_size'], augment=True, pairs=train_pairs)
    val_ds = ClipClothHeatmapDataset(config['data_dir'], config['image_size'], augment=False, pairs=val_pairs)
    test_ds = ClipClothHeatmapDataset(config['data_dir'], config['image_size'], augment=False, pairs=test_pairs)

    # Save a few augmented previews to verify transforms
    try:
        os.makedirs(config['output_dir'], exist_ok=True)
        mean = np.array([0.48145466, 0.4578275, 0.40821073]).reshape(1, 1, 3)
        std = np.array([0.26862954, 0.26130258, 0.27577711]).reshape(1, 1, 3)
        for i in range(min(3, len(train_ds))):
            sample = train_ds[i]
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
            cv2.imwrite(os.path.join(config['output_dir'], f'aug_preview_{i}.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved augmentation previews to {config['output_dir']}/aug_preview_*.png")
    except Exception as e:
        print(f"Augmentation preview failed: {e}")

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=0, collate_fn=collate_clip_batch)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_clip_batch)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_clip_batch)

    # Model
    model = ClipHeatmapModel(
        model_name=config['model_name'],
        image_size=config['image_size'],
        use_lora=config['use_lora'],
        lora_r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        use_text_prior=config['use_text_prior'],
        prior_prompts=config['prior_prompts'],
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
            # Save LoRA adapters (if used) and head weights
            if config['use_lora'] and PEFT_AVAILABLE:
                # Save adapters from the CLIP PEFT wrapper
                model.clip.save_pretrained(config['output_dir'])
            torch.save(model.head.state_dict(), os.path.join(config['output_dir'], 'head.pth'))
            print(f"Saved best model (val={best_val:.4f}) to {config['output_dir']}")
        else:
            patience += 1
            if patience >= config['early_stopping_patience']:
                print("Early stopping triggered")
                break

    print("Training complete.")
    if writer is not None:
        writer.close()

    # --- Evaluation on test set with best checkpoint ---
    from peft import PeftModel
    with torch.no_grad():
        # Rebuild model for best weights
        eval_model = ClipHeatmapModel(
            model_name=config['model_name'],
            image_size=config['image_size'],
            use_lora=config['use_lora'],
            lora_r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            use_text_prior=config['use_text_prior'],
            prior_prompts=config['prior_prompts'],
            prior_weight=config['prior_weight']
        ).to(device)
        # Load adapters
        if config['use_lora'] and PEFT_AVAILABLE and os.path.exists(os.path.join(config['output_dir'], 'adapter_config.json')):
            try:
                base = CLIPModel.from_pretrained(config['model_name'])
                base = PeftModel.from_pretrained(base, config['output_dir'])
                base = base.to(device)  # Move to device
                eval_model.clip = base
                eval_model.vision = base.get_submodule('vision_model')
            except Exception as e:
                print(f"Warning: failed to load adapters for eval: {e}")
        # Load head
        head_path = os.path.join(config['output_dir'], 'head.pth')
        if os.path.exists(head_path):
            eval_model.head.load_state_dict(torch.load(head_path, map_location=device))
        eval_model.eval()

        # Prepare results dir
        results_dir = Path(config['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)

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

            # Greedy matching
            matched = 0
            dists = []
            used = set()
            for (gx, gy) in gt_points:
                best = None
                best_d = 1e9
                best_j = -1
                for j, (pxx, pyy) in enumerate(peaks_xy):
                    if j in used:
                        continue
                    d = ((gx - pxx) ** 2 + (gy - pyy) ** 2) ** 0.5
                    if d < best_d:
                        best_d, best, best_j = d, (pxx, pyy), j
                if best is not None:
                    used.add(best_j)
                    dists.append(best_d)
                    if best_d < 10.0:
                        matched += 1

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
            cv2.imwrite(str(results_dir / f"{image_id}_clip_heatmap.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

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
        with open(results_dir / 'evaluation_results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved test evaluation to {results_dir}/evaluation_results.json")


if __name__ == '__main__':
    train_clip_heatmap()
