#!/usr/bin/env python3
"""
CLIP Heatmap Keypoint Detection Model

This module contains the CLIP-based heatmap prediction model for keypoint detection.
It uses CLIP's vision encoder with LoRA fine-tuning and optional text prior gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import os
try:
    from ..utils.model_utils import *
except ImportError:
    # Fallback for when running from root directory
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.utils.model_utils import *

# Import CLIP and PEFT dependencies
try:
    from transformers import CLIPModel, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")

try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not available. Install with: pip install peft")


class ClipHeatmapHead(nn.Module):
    """
    Custom head for converting CLIP features to keypoint heatmaps.
    
    Args:
        in_dim: Input feature dimension from CLIP vision encoder
        out_size: Output heatmap size (e.g., 256 for 256x256 heatmap)
    """
    
    def __init__(self, in_dim: int, out_size: int):
        super().__init__()
        self.out_size = out_size
        self.proj = nn.Conv2d(in_dim, 256, kernel_size=1)
        self.block = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1), 
            nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the heatmap head.
        
        Args:
            feat_2d: 2D feature map from CLIP vision encoder (B, D, h, w)
            
        Returns:
            Heatmap tensor (B, 1, out_size, out_size)
        """
        # feat_2d: (B, D, h, w)
        x = self.proj(feat_2d)
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
        x = self.block(x)
        x = self.out(x)
        # Ensure positive for KL loss
        return spatial_softmax(x)


class ClipHeatmapModel(nn.Module):
    """
    CLIP-based heatmap keypoint detection model.
    
    This model uses CLIP's vision encoder with LoRA fine-tuning and optional text prior gating
    to predict keypoint locations as heatmaps.
    
    Args:
        model_name: CLIP model name (e.g., 'openai/clip-vit-base-patch16')
        image_size: Input image size (e.g., 256)
        use_lora: Whether to use LoRA fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        use_text_prior: Whether to use text prior gating
        prior_prompts: List of text prompts for prior gating
        prior_weight: Weight for text prior gating
    """
    
    def __init__(
        self, 
        model_name: str, 
        image_size: int, 
        use_lora: bool = True, 
        lora_r: int = 16, 
        lora_alpha: int = 32, 
        lora_dropout: float = 0.05,
        use_text_prior: bool = True, 
        prior_prompts: Optional[List[str]] = None, 
        negative_prompts: Optional[List[str]] = None,
        prior_weight: float = 0.5
    ):
        super().__init__()
        
        if not HF_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers")
        
        # Store model name for saving
        self.model_name = model_name
        
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
        
        # Default text prompts for cloth corner detection
        if prior_prompts is None:
            prior_prompts = [
                "a photo of a cloth corner",
                "fabric corner point",
                "sharp cloth corner"
            ]
        self.prior_prompts = prior_prompts
        
        # Default negative prompts to avoid non-corner features
        if negative_prompts is None:
            negative_prompts = [
                "cloth fold line",
                "fabric crease",
                "cloth wrinkle"
            ]
        self.negative_prompts = negative_prompts
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError("peft library is required for LoRA. Install with: pip install peft")
            
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
        """
        Forward pass through the CLIP heatmap model.
        
        Args:
            pixel_values: Input images (B, 3, H, W)
            
        Returns:
            Keypoint heatmap (B, 1, image_size, image_size)
        """
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
            
            # Process positive prompts
            pos_enc = self.tokenizer(self.prior_prompts, padding=True, return_tensors='pt')
            pos_enc = {k: v.to(device) for k, v in pos_enc.items()}
            with torch.no_grad():
                pos_text_feats = self.clip.get_text_features(**pos_enc)  # (P_text, d)
            pos_text_feats = F.normalize(pos_text_feats, dim=-1)
            pos_text_vec = pos_text_feats.mean(dim=0)  # (d,)
            pos_text_vec = F.normalize(pos_text_vec, dim=-1)
            
            # Process negative prompts
            neg_enc = self.tokenizer(self.negative_prompts, padding=True, return_tensors='pt')
            neg_enc = {k: v.to(device) for k, v in neg_enc.items()}
            with torch.no_grad():
                neg_text_feats = self.clip.get_text_features(**neg_enc)  # (N_text, d)
            neg_text_feats = F.normalize(neg_text_feats, dim=-1)
            neg_text_vec = neg_text_feats.mean(dim=0)  # (d,)
            neg_text_vec = F.normalize(neg_text_vec, dim=-1)

            # Project patch tokens to CLIP embed dim and normalize
            patch_proj = self.clip.visual_projection(patch_tokens)  # (B,P,d)
            patch_proj = F.normalize(patch_proj, dim=-1)
            
            # Compute positive similarity (B,P)
            pos_sim = torch.matmul(patch_proj, pos_text_vec.unsqueeze(-1)).squeeze(-1)
            pos_sim = (pos_sim + 1.0) * 0.5  # Scale to [0,1]
            
            # Compute negative similarity (B,P)
            neg_sim = torch.matmul(patch_proj, neg_text_vec.unsqueeze(-1)).squeeze(-1)
            neg_sim = (neg_sim + 1.0) * 0.5  # Scale to [0,1]
            
            # Combine positive and negative similarities
            # Higher positive similarity and lower negative similarity = stronger corner signal
            combined_sim = pos_sim - 0.3 * neg_sim  # Reduce negative influence
            combined_sim = torch.clamp(combined_sim, 0.0, 1.0)  # Clamp to [0,1]
            
            sim_hw = combined_sim.view(pixel_values.size(0), 1, h, w)
            # Gate features
            feat_2d = feat_2d * (1.0 + self.prior_weight * sim_hw)
        
        heat = self.head(feat_2d)
        return heat

    def save_pretrained(self, save_directory: str):
        """
        Save the model to a directory.
        
        Args:
            save_directory: Directory to save the model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the full model state dict
        torch.save(self.state_dict(), os.path.join(save_directory, 'pytorch_model.bin'))
        
        # Save LoRA adapters if using LoRA
        if self.use_lora and hasattr(self.clip, 'save_pretrained'):
            self.clip.save_pretrained(os.path.join(save_directory, 'lora_adapters'))
        
        # Save model configuration
        config = {
            'model_name': self.model_name,  # Use the actual model name from the instance
            'image_size': self.image_size,
            'use_lora': self.use_lora,
            'use_text_prior': self.use_text_prior,
            'prior_prompts': self.prior_prompts,
            'prior_weight': self.prior_weight,
            'hidden_size': self.hidden_size,
            'patch_size': self.patch_size
        }
        
        import json
        with open(os.path.join(save_directory, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_directory: str, **kwargs):
        """
        Load a pretrained model from a directory.
        
        Args:
            model_directory: Directory containing the saved model
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded ClipHeatmapModel instance
        """
        import json
        
        # Load configuration
        config_path = os.path.join(model_directory, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Update with any provided kwargs
            config.update(kwargs)
        else:
            # Use default config with provided kwargs
            config = {
                'model_name': 'openai/clip-vit-base-patch16',
                'image_size': 256,
                'use_lora': True,
                'use_text_prior': True,
                'prior_prompts': [
                    "a photo of a cloth corner",
                    "fabric corner point",
                    "sharp cloth corner"
                ],
                'prior_weight': 0.5
            }
            config.update(kwargs)
        
        # Create model instance
        model = cls(**config)
        
        # Load model weights
        model_path = os.path.join(model_directory, 'pytorch_model.bin')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
        
        # Load LoRA adapters if they exist
        lora_path = os.path.join(model_directory, 'lora_adapters')
        if os.path.exists(lora_path) and model.use_lora:
            try:
                from peft import PeftModel
                model.clip = PeftModel.from_pretrained(model.clip, lora_path)
            except ImportError:
                print("Warning: Could not load LoRA adapters. peft library may not be available.")
        
        return model


def create_clip_heatmap_model(
    model_name: str = 'openai/clip-vit-base-patch16',
    image_size: int = 256,
    use_lora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_text_prior: bool = True,
    prior_prompts: Optional[List[str]] = None,
    negative_prompts: Optional[List[str]] = None,
    prior_weight: float = 0.5
) -> ClipHeatmapModel:
    """
    Factory function to create a CLIP heatmap model.
    
    Args:
        model_name: CLIP model name
        image_size: Input image size
        use_lora: Whether to use LoRA fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        use_text_prior: Whether to use text prior gating
        prior_prompts: List of text prompts for prior gating
        prior_weight: Weight for text prior gating
        
    Returns:
        ClipHeatmapModel instance
    """
    return ClipHeatmapModel(
        model_name=model_name,
        image_size=image_size,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_text_prior=use_text_prior,
        prior_prompts=prior_prompts,
        negative_prompts=negative_prompts,
        prior_weight=prior_weight
    )
