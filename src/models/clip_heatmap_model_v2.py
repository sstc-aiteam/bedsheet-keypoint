"""
CLIP Heatmap Model V2 - Multi-Tier Pipeline

This module implements a two-stage keypoint detection system:
1. Stage 1: Count prediction model (predicts number of keypoints)
2. Stage 2: Location prediction model (predicts keypoint locations given the count)

This design allows for handling variable numbers of keypoints across different datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


class ClipCountHead(nn.Module):
    """
    Head for predicting the number of keypoints in an image.
    """
    
    def __init__(self, hidden_size: int, max_keypoints: int = 10):
        super().__init__()
        self.max_keypoints = max_keypoints
        
        # Global average pooling + MLP for count prediction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.count_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, max_keypoints + 1)  # +1 for 0 keypoints
        )
    
    def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_2d: (B, D, H, W) feature map
        Returns:
            count_logits: (B, max_keypoints + 1) logits for count prediction
        """
        # Global average pooling
        global_feat = self.global_pool(feat_2d).flatten(1)  # (B, D)
        
        # Predict count
        count_logits = self.count_head(global_feat)
        
        return count_logits


class CountConditionedLocationHead(nn.Module):
    """
    Count-conditioned head for predicting keypoint locations.
    Uses count information as conditioning input instead of gating.
    """
    
    def __init__(self, hidden_size: int, max_keypoints: int, out_size: int = 256):
        super().__init__()
        self.out_size = out_size
        self.max_keypoints = max_keypoints
        
        # Count embedding
        self.count_embedding = nn.Embedding(max_keypoints + 1, hidden_size // 4)
        
        # Convolutional layers for location prediction
        self.proj = nn.Conv2d(hidden_size, hidden_size // 2, 1)
        self.count_proj = nn.Conv2d(hidden_size // 4, hidden_size // 4, 1)
        
        # Combined processing
        combined_size = hidden_size // 2 + hidden_size // 4
        self.block = nn.Sequential(
            nn.Conv2d(combined_size, hidden_size // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size // 2, hidden_size // 4, 3, padding=1),
            nn.ReLU(),
        )
        self.out = nn.Conv2d(hidden_size // 4, 1, 1)
    
    def forward(self, feat_2d: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_2d: (B, D, H, W) feature map
            count: (B,) count tensor
        Returns:
            heatmap: (B, 1, out_size, out_size) heatmap
        """
        # Process visual features
        visual_feat = self.proj(feat_2d)
        
        # Process count information
        count_emb = self.count_embedding(count)  # (B, hidden_size//4)
        count_emb = count_emb.unsqueeze(-1).unsqueeze(-1)  # (B, hidden_size//4, 1, 1)
        count_emb = count_emb.expand(-1, -1, visual_feat.size(2), visual_feat.size(3))  # (B, hidden_size//4, H, W)
        count_feat = self.count_proj(count_emb)
        
        # Combine visual and count features
        combined_feat = torch.cat([visual_feat, count_feat], dim=1)
        
        # Process combined features
        x = F.interpolate(combined_feat, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False)
        x = self.block(x)
        x = self.out(x)
        # Ensure positive for KL loss
        return F.softplus(x)


class ClipHeatmapModelV2(nn.Module):
    """
    Multi-tier CLIP-based keypoint detection model.
    
    Stage 1: Predict number of keypoints
    Stage 2: Predict keypoint locations given the count
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
        prior_weight: float = 0.5,
        max_keypoints: int = 10,
        freeze_backbone: bool = False
    ):
        super().__init__()
        
        # Store configuration parameters
        self.model_name = model_name
        self.image_size = image_size
        self.max_keypoints = max_keypoints
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_text_prior = use_text_prior
        self.prior_weight = prior_weight
        self.prior_prompts = prior_prompts or [
            "a photo of a cloth corner",
            "fabric corner point",
            "sharp cloth corner",
            "textile edge corner",
            "fabric fold corner",
            "cloth seam corner"
        ]
        self.negative_prompts = negative_prompts or [
            "smooth fabric surface",
            "flat textile area",
            "fabric background",
            "cloth interior",
            "textile center"
        ]
        
        # Load CLIP model
        self.clip = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.vision = self.clip.vision_model
        self.hidden_size = self.vision.config.hidden_size
        self.patch_size = self.vision.config.patch_size
        
        # Freeze text model (we only need it for text priors, not training)
        for param in self.clip.text_model.parameters():
            param.requires_grad = False
        
        # Freeze CLIP projections (we don't need to train these)
        for param in self.clip.visual_projection.parameters():
            param.requires_grad = False
        for param in self.clip.text_projection.parameters():
            param.requires_grad = False
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.vision.parameters():
                param.requires_grad = False
        
        # Stage 1: Count prediction head
        self.count_head = ClipCountHead(self.hidden_size, max_keypoints)
        
        # Stage 2: Count-conditioned location prediction head
        self.location_head = CountConditionedLocationHead(self.hidden_size, max_keypoints, image_size)
        
        # LoRA configuration
        if use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # Configure LoRA for vision attention layers
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                modules_to_save=[]
            )
            
            # Apply LoRA to the full CLIP model (like V1)
            self.clip = get_peft_model(self.clip, lora_config)
            self.vision = self.clip.get_submodule('vision_model')
    
    def forward(
        self, 
        pixel_values: torch.Tensor,
        return_count: bool = True,
        return_locations: bool = True,
        target_count: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both stages.
        
        Args:
            pixel_values: (B, 3, H, W) input images
            return_count: Whether to return count predictions
            return_locations: Whether to return location predictions
            target_count: (B,) target counts for location prediction (if provided)
        
        Returns:
            Dictionary containing:
            - 'count_logits': (B, max_keypoints + 1) count prediction logits
            - 'count_probs': (B, max_keypoints + 1) count prediction probabilities
            - 'predicted_count': (B,) predicted counts
            - 'location_heatmap': (B, 1, H, W) location heatmap
        """
        device = pixel_values.device
        batch_size = pixel_values.size(0)
        
        # Get vision features
        # Enable positional embedding interpolation for non-224 inputs (like V1)
        outputs = self.vision(pixel_values=pixel_values, interpolate_pos_encoding=True)
        tokens = outputs.last_hidden_state  # (B, 1+P, D)
        patch_tokens = tokens[:, 1:, :]  # (B, P, D)
        
        # Reshape to spatial format
        h = w = self.image_size // self.patch_size
        feat_2d = patch_tokens.transpose(1, 2).contiguous().view(batch_size, self.hidden_size, h, w)
        
        # Apply text prior gating if enabled
        if self.use_text_prior:
            feat_2d = self._apply_text_prior_gating(feat_2d, device)
        
        results = {}
        
        # Stage 1: Count prediction
        if return_count:
            count_logits = self.count_head(feat_2d)
            count_probs = F.softmax(count_logits, dim=-1)
            predicted_count = torch.argmax(count_probs, dim=-1)
            
            results.update({
                'count_logits': count_logits,
                'count_probs': count_probs,
                'predicted_count': predicted_count
            })
        
        # Stage 2: Location prediction
        if return_locations:
            # Use target count if provided, otherwise use predicted count
            if target_count is not None:
                count_to_use = target_count
            elif return_count:
                count_to_use = predicted_count
            else:
                # Default to maximum count if no count information available
                count_to_use = torch.full((batch_size,), self.max_keypoints, device=device)
            
            # Predict locations using count conditioning
            location_heatmap = self.location_head(feat_2d, count_to_use)
            results['location_heatmap'] = location_heatmap
        
        return results
    
    def save_pretrained(self, save_directory: str):
        """
        Save the model to a directory.
        
        Args:
            save_directory: Directory to save the model
        """
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model configuration
        config = {
            'model_name': self.model_name,
            'image_size': self.image_size,
            'max_keypoints': self.max_keypoints,
            'use_lora': self.use_lora,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'use_text_prior': self.use_text_prior,
            'prior_prompts': self.prior_prompts,
            'negative_prompts': self.negative_prompts,
            'prior_weight': self.prior_weight
        }
        
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save custom head weights
        head_state_dict = {
            'count_head.count_head.0.weight': self.count_head.count_head[0].weight.data,
            'count_head.count_head.0.bias': self.count_head.count_head[0].bias.data,
            'count_head.count_head.3.weight': self.count_head.count_head[3].weight.data,
            'count_head.count_head.3.bias': self.count_head.count_head[3].bias.data,
            'count_head.count_head.6.weight': self.count_head.count_head[6].weight.data,
            'count_head.count_head.6.bias': self.count_head.count_head[6].bias.data,
            'location_head.count_embedding.weight': self.location_head.count_embedding.weight.data,
            'location_head.proj.weight': self.location_head.proj.weight.data,
            'location_head.proj.bias': self.location_head.proj.bias.data,
            'location_head.count_proj.weight': self.location_head.count_proj.weight.data,
            'location_head.count_proj.bias': self.location_head.count_proj.bias.data,
            'location_head.block.0.weight': self.location_head.block[0].weight.data,
            'location_head.block.0.bias': self.location_head.block[0].bias.data,
            'location_head.block.2.weight': self.location_head.block[2].weight.data,
            'location_head.block.2.bias': self.location_head.block[2].bias.data,
            'location_head.out.weight': self.location_head.out.weight.data,
            'location_head.out.bias': self.location_head.out.bias.data,
        }
        
        head_path = os.path.join(save_directory, 'head_weights.pth')
        torch.save(head_state_dict, head_path)
        
        # Save LoRA adapters if using LoRA
        if self.use_lora and hasattr(self.clip, 'save_pretrained'):
            lora_path = os.path.join(save_directory, 'lora_adapters')
            self.clip.save_pretrained(lora_path)
            print(f"✓ Saved LoRA adapters to {lora_path}")
        
        print(f"✓ Saved model configuration to {config_path}")
        print(f"✓ Saved head weights to {head_path}")
    
    @classmethod
    def from_pretrained(cls, save_directory: str, device: str = 'cuda'):
        """
        Load a saved model from a directory.
        
        Args:
            save_directory: Directory containing the saved model
            device: Device to load the model on
        
        Returns:
            ClipHeatmapModelV2 instance
        """
        import os
        import json
        
        # Load configuration
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model instance without LoRA first
        model = cls(
            model_name=config['model_name'],
            image_size=config['image_size'],
            max_keypoints=config['max_keypoints'],
            use_lora=False,  # Don't initialize LoRA yet
            lora_r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            use_text_prior=config['use_text_prior'],
            prior_prompts=config['prior_prompts'],
            negative_prompts=config['negative_prompts'],
            prior_weight=config['prior_weight']
        )
        
        # Load head weights
        head_path = os.path.join(save_directory, 'head_weights.pth')
        if os.path.exists(head_path):
            head_state_dict = torch.load(head_path, map_location=device)
            model.count_head.count_head[0].weight.data = head_state_dict['count_head.count_head.0.weight']
            model.count_head.count_head[0].bias.data = head_state_dict['count_head.count_head.0.bias']
            model.count_head.count_head[3].weight.data = head_state_dict['count_head.count_head.3.weight']
            model.count_head.count_head[3].bias.data = head_state_dict['count_head.count_head.3.bias']
            model.count_head.count_head[6].weight.data = head_state_dict['count_head.count_head.6.weight']
            model.count_head.count_head[6].bias.data = head_state_dict['count_head.count_head.6.bias']
            
            # Load location head weights (with backward compatibility)
            if 'location_head.count_embedding.weight' in head_state_dict:
                model.location_head.count_embedding.weight.data = head_state_dict['location_head.count_embedding.weight']
                model.location_head.count_proj.weight.data = head_state_dict['location_head.count_proj.weight']
                model.location_head.count_proj.bias.data = head_state_dict['location_head.count_proj.bias']
            
            model.location_head.proj.weight.data = head_state_dict['location_head.proj.weight']
            model.location_head.proj.bias.data = head_state_dict['location_head.proj.bias']
            model.location_head.block[0].weight.data = head_state_dict['location_head.block.0.weight']
            model.location_head.block[0].bias.data = head_state_dict['location_head.block.0.bias']
            model.location_head.block[2].weight.data = head_state_dict['location_head.block.2.weight']
            model.location_head.block[2].bias.data = head_state_dict['location_head.block.2.bias']
            model.location_head.out.weight.data = head_state_dict['location_head.out.weight']
            model.location_head.out.bias.data = head_state_dict['location_head.out.bias']
            print(f"✓ Loaded head weights from {head_path}")
        
        # Load LoRA adapters if they exist
        lora_path = os.path.join(save_directory, 'lora_adapters')
        if os.path.exists(lora_path) and config['use_lora']:
            try:
                from peft import PeftModel
                model.clip = PeftModel.from_pretrained(model.clip, lora_path)
                model.vision = model.clip.get_submodule('vision_model')
                print(f"✓ Loaded LoRA adapters from {lora_path}")
            except ImportError:
                print("Warning: Could not load LoRA adapters. peft library may not be available.")
        
        # Move model to device
        model = model.to(device)
        
        # Ensure the model is in eval mode
        model.eval()
        
        return model
    
    def _apply_text_prior_gating(self, feat_2d: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Apply text prior gating to features."""
        batch_size = feat_2d.size(0)
        B, D, H, W = feat_2d.shape
        
        # Reshape features to patch tokens format for projection
        feat_flat = feat_2d.view(B, D, H * W).transpose(1, 2)  # (B, H*W, D)
        
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
        
        # Project patch tokens to CLIP embed dim and normalize (same as original model)
        patch_proj = self.clip.visual_projection(feat_flat)  # (B, H*W, d)
        patch_proj = F.normalize(patch_proj, dim=-1)
        
        # Compute positive similarity (B, H*W)
        pos_sim = torch.matmul(patch_proj, pos_text_vec.unsqueeze(-1)).squeeze(-1)
        pos_sim = (pos_sim + 1.0) * 0.5  # Scale to [0,1]
        
        # Compute negative similarity (B, H*W)
        neg_sim = torch.matmul(patch_proj, neg_text_vec.unsqueeze(-1)).squeeze(-1)
        neg_sim = (neg_sim + 1.0) * 0.5  # Scale to [0,1]
        
        # Combine positive and negative similarities
        combined_sim = pos_sim - 0.3 * neg_sim  # Reduce negative influence
        combined_sim = torch.clamp(combined_sim, 0.0, 1.0)  # Clamp to [0,1]
        
        # Apply gating
        gate = combined_sim.view(B, 1, H, W)
        gated_feat = feat_2d * (1 + self.prior_weight * gate)
        
        return gated_feat
    
    
    def predict_keypoints(
        self, 
        pixel_values: torch.Tensor,
        count_threshold: float = 0.5,
        location_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Predict keypoints using both stages.
        
        Args:
            pixel_values: (B, 3, H, W) input images
            count_threshold: Threshold for count prediction confidence
            location_threshold: Threshold for location prediction
        
        Returns:
            Dictionary containing predictions for each image in the batch
        """
        self.eval()
        with torch.no_grad():
            results = self.forward(pixel_values, return_count=True, return_locations=True)
            
            batch_predictions = []
            for i in range(pixel_values.size(0)):
                # Get count prediction
                count_probs = results['count_probs'][i]
                predicted_count = results['predicted_count'][i].item()
                count_confidence = count_probs[predicted_count].item()
                
                # Get location prediction
                location_heatmap = results['location_heatmap'][i, 0].cpu().numpy()
                
                # Normalize heatmap
                if location_heatmap.max() > 0:
                    location_heatmap = location_heatmap / location_heatmap.max()
                
                # Extract keypoints
                from shared.functions import thresholded_locations
                peaks = thresholded_locations(location_heatmap, threshold=location_threshold)
                pred_keypoints = [(int(p[1]), int(p[0])) for p in peaks]
                
                # Limit to predicted count
                if len(pred_keypoints) > predicted_count:
                    # Keep the strongest peaks
                    peak_strengths = [location_heatmap[int(p[0]), int(p[1])] for p in peaks]
                    sorted_indices = np.argsort(peak_strengths)[::-1]
                    pred_keypoints = [pred_keypoints[i] for i in sorted_indices[:predicted_count]]
                
                batch_predictions.append({
                    'predicted_count': predicted_count,
                    'count_confidence': count_confidence,
                    'keypoints': pred_keypoints,
                    'heatmap': location_heatmap
                })
            
            return {
                'predictions': batch_predictions,
                'count_logits': results['count_logits'],
                'count_probs': results['count_probs'],
                'location_heatmap': results['location_heatmap']
            }


def create_clip_heatmap_model_v2(
    model_name: str = "facebook/metaclip-b16-fullcc2.5b",
    image_size: int = 256,
    max_keypoints: int = 10,
    **kwargs
) -> ClipHeatmapModelV2:
    """
    Factory function to create a ClipHeatmapModelV2.
    
    Args:
        model_name: CLIP model name
        image_size: Input image size
        max_keypoints: Maximum number of keypoints to predict
        **kwargs: Additional arguments for ClipHeatmapModelV2
    
    Returns:
        ClipHeatmapModelV2 instance
    """
    return ClipHeatmapModelV2(
        model_name=model_name,
        image_size=image_size,
        max_keypoints=max_keypoints,
        **kwargs
    )
