"""
Utility functions for the bed sheet folding robot.
"""

from .model_utils import (
    YoloBackbone,
    EnhancedYoloBackbone, 
    MultiScaleFusion, 
    soft_argmax, 
    spatial_softmax,
    batch_gaussian_blur,
    normalize_heatmaps,
    batch_entropy,
    mixup_data,
    load_state_dict_safely,
    extract_mask_compare,
    thresholded_locations,
    kl_heatmap_loss
)
# Quantization utilities removed - functionality deprecated

__all__ = [
    # Model utilities
    'YoloBackbone',
    'EnhancedYoloBackbone',
    'MultiScaleFusion', 
    'soft_argmax',
    'spatial_softmax',
    'batch_gaussian_blur',
    'normalize_heatmaps',
    'batch_entropy',
    'mixup_data',
    'load_state_dict_safely',
    'extract_mask_compare',
    'thresholded_locations',
    
    # Losses
    'kl_heatmap_loss'
]
