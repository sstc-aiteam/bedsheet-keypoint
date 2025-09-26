#!/usr/bin/env python3
"""
Configuration and argument parsing for TensorRT demo.
"""

import argparse
from pathlib import Path


def create_argument_parser():
    """Create argument parser for TensorRT demo."""
    parser = argparse.ArgumentParser(description="TensorRT Keypoint Detection Demo")
    
    parser.add_argument("--pytorch_model", type=str, 
                       help="Path to PyTorch model")
    parser.add_argument("--tensorrt_model", type=str,
                       help="Path to TensorRT model")
    parser.add_argument("--image_dir", type=str,
                       default="image_data/RGB-images",
                       help="Directory containing test images")
    parser.add_argument("--model_type", type=str,
                       choices=['hybrid_keypoint_net', 'clip_heatmap_model', 'efficient_keypoint_net', 'auto'],
                       default='auto',
                       help="Model type (auto-detect if not specified)")
    parser.add_argument("--list_models", action="store_true",
                       help="List available models and exit")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark comparison")
    parser.add_argument("--num_runs", type=int, default=100,
                       help="Number of runs for benchmarking")
    parser.add_argument("--demo_clip", action="store_true",
                       help="Run CLIP-specific demo")
    
    return parser


def get_clip_demo_config():
    """Get configuration for CLIP demo."""
    return {
        'pytorch_model': "models/meta_clip_style_bedsheet_post_pretrained/complete_model.pth",
        'tensorrt_model': "models/meta_clip_style_bedsheet_post_pretrained/complete_model.trt",
        'image_dir': "image_data/RGB-images",
        'model_type': 'clip_heatmap_model'
    }


def validate_config(args):
    """Validate configuration and check file existence."""
    # Check PyTorch model
    if not Path(args.pytorch_model).exists():
        print(f"Error: PyTorch model not found: {args.pytorch_model}")
        return False
    
    # Check image directory
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"Error: Image directory not found: {args.image_dir}")
        return False
    
    # Check for images
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    if not image_files:
        print(f"Error: No images found in {args.image_dir}")
        return False
    
    return True


def get_sample_image(image_dir):
    """Get a sample image from the directory."""
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    return str(image_files[0]) if image_files else None
