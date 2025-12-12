#!/usr/bin/env python3
"""
Model loading utilities for TensorRT demo.
Handles loading different types of models with automatic detection.
"""

import os
import torch
from pathlib import Path


def detect_model_type(model_path: str) -> str:
    """Detect model type based on model path and available files."""
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_dir)
    
    # Check for CLIP-based models
    if 'clip' in model_name.lower() or 'meta_clip' in model_name.lower():
        # Check if it has CLIP-specific files
        if os.path.exists(os.path.join(model_dir, 'adapter_config.json')):
            return 'clip_heatmap_model'
        elif os.path.exists(os.path.join(model_dir, 'complete_model.pth')):
            return 'clip_heatmap_model'
    
    # Check for hybrid keypoint models
    if 'keypoint' in model_name.lower() or 'hybrid' in model_name.lower():
        return 'hybrid_keypoint_net'
    
    # Check for efficient models
    if 'efficient' in model_name.lower() or 'mobile' in model_name.lower():
        return 'efficient_keypoint_net'
    
    # Default to hybrid keypoint net for backward compatibility
    return 'hybrid_keypoint_net'


def load_model_safely(model, load_path: str, map_location="cpu", strict: bool = False):
    """Safe model loading that handles torch.compile-wrapped models."""
    def _get_base_module(module):
        return getattr(module, "_orig_mod", module)
    
    state = torch.load(load_path, map_location=map_location)
    cleaned = {}
    for key, value in state.items():
        if key.startswith("_orig_mod."):
            cleaned[key[len("_orig_mod."):]] = value
        else:
            cleaned[key] = value
    target = _get_base_module(model)
    return target.load_state_dict(cleaned, strict=strict)


def load_clip_heatmap_model(model_path: str):
    """Load CLIP heatmap model.

    Prefers loading `training_config.json` (or `config.json`) from the model directory so we don't hardcode
    model_name/image_size/lora settings in multiple places.
    """
    from src.models.clip_heatmap_model import ClipHeatmapModel
    
    # Try to load from complete model first
    complete_model_path = os.path.join(os.path.dirname(model_path), 'complete_model.pth')
    if os.path.exists(complete_model_path):
        model_path = complete_model_path

    model_dir = os.path.dirname(model_path)
    cfg = None
    for cfg_name in ("training_config.json", "config.json"):
        cfg_path = os.path.join(model_dir, cfg_name)
        if os.path.exists(cfg_path):
            try:
                import json
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                break
            except Exception:
                cfg = None
    
    # Create model with config from disk (fallback defaults)
    # Note: MetaCLIP2 L/14 is patch14, so 224 is the safe default image_size.
    model_name = (cfg or {}).get("model_name", "facebook/metaclip-2-worldwide-l14")
    image_size = int((cfg or {}).get("image_size", 224))
    use_lora = bool((cfg or {}).get("use_lora", True))
    lora_r = int((cfg or {}).get("lora_r", 16))
    lora_alpha = int((cfg or {}).get("lora_alpha", 32))
    lora_dropout = float((cfg or {}).get("lora_dropout", 0.05))
    use_text_prior = bool((cfg or {}).get("use_text_prior", True))
    prior_prompts = (cfg or {}).get("prior_prompts", None)
    negative_prompts = (cfg or {}).get("negative_prompts", None)
    prior_weight = float((cfg or {}).get("prior_weight", 0.5))

    model = ClipHeatmapModel(
        model_name=model_name,
        image_size=image_size,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_text_prior=use_text_prior,
        prior_prompts=prior_prompts,
        negative_prompts=negative_prompts,
        prior_weight=prior_weight,
    )
    
    # Load state dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    return model


def load_hybrid_keypoint_model(model_path: str):
    """Load hybrid keypoint model (YOLO + ViT)."""
    from src.models.hybrid_keypoint_net import HybridKeypointNet
    from src.utils.model_utils import EnhancedYoloBackbone
    from ultralytics import YOLO
    
    # Create model architecture (exact same as post training script)
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
    
    model = HybridKeypointNet(backbone, in_channels_list)
    
    # Load trained weights safely using the same approach as post training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Use the same safe loading approach as post training script
    missing_keys, unexpected_keys = load_model_safely(model, model_path, map_location=device, strict=False)
    if missing_keys:
        print(f"Warning: Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {len(unexpected_keys)}")
    
    model.eval()
    
    return model


def load_efficient_keypoint_model(model_path: str):
    """Load efficient keypoint model."""
    from src.models.efficient_keypoint_net import EfficientKeypointNet
    from src.utils.model_utils import YoloBackbone
    from ultralytics import YOLO
    
    # Create backbone
    yolo_model = YOLO('yolov8s.pt')
    backbone_seq = yolo_model.model.model[:8]
    backbone = YoloBackbone(backbone_seq, selected_indices=[0,1,2,3,4,5,6,7])
    
    # Get input channels list
    input_dummy = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        feats = backbone(input_dummy)
    in_channels_list = [f.shape[1] for f in feats]
    
    # Create model
    model = EfficientKeypointNet(backbone, in_channels_list)
    
    # Load state dict
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def load_pytorch_model(model_path: str, model_type: str = None):
    """Load PyTorch model with automatic type detection."""
    if model_type is None:
        model_type = detect_model_type(model_path)
    
    print(f"Loading {model_type} model from: {model_path}")
    
    if model_type == 'clip_heatmap_model':
        return load_clip_heatmap_model(model_path)
    elif model_type == 'hybrid_keypoint_net':
        return load_hybrid_keypoint_model(model_path)
    elif model_type == 'efficient_keypoint_net':
        return load_efficient_keypoint_model(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def list_available_models():
    """List all available models in the models directory."""
    print("\nAvailable models:")
    models_dir = Path("models")
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            model_files = list(model_dir.glob("*.pth")) + list(model_dir.glob("complete_model.pth"))
            if model_files:
                model_type = detect_model_type(str(model_files[0]))
                print(f"  {model_dir.name} ({model_type})")
