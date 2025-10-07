#!/usr/bin/env python3
"""
TensorRT Model Conversion Script

This script provides a simple interface for converting various PyTorch models to TensorRT format
using the generalized TensorRT utilities. It supports multiple model types and configurations.

Usage:
    python tensorrt_convert.py --model_type hybrid_keypoint_net --model_path models/model.pth --output_path models/model.trt
    python tensorrt_convert.py --model_type clip_heatmap_model --model_path models/clip_model.pth --output_path models/clip_model.trt --precision fp16
    python tensorrt_convert.py --model_type efficient_keypoint_net --model_path models/efficient_model.pth --output_path models/efficient_model.trt --input_shape 1 3 128 128
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.utils.tensorrt_utils import (
    convert_any_model_to_tensorrt,
    benchmark_any_model,
    ModelConfig,
    GeneralizedTensorRTConverter,
    ModelRegistry
)


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from: {config_path}")
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return {}


def save_config_to_file(config: Dict[str, Any], config_path: str):
    """Save configuration to JSON file."""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved configuration to: {config_path}")
    except Exception as e:
        print(f"Error saving config file {config_path}: {e}")


def validate_model_path(model_path: str) -> bool:
    """Validate that the model path exists and is a valid PyTorch model."""
    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        return False
    
    if not model_path.endswith(('.pth', '.pt')):
        print(f"Warning: Model path doesn't have .pth or .pt extension: {model_path}")
    
    return True


def validate_output_path(output_path: str) -> bool:
    """Validate and create output directory if needed."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return False
    
    if not output_path.endswith('.trt'):
        print(f"Warning: Output path doesn't have .trt extension: {output_path}")
    
    return True


def get_model_info(model_type: str) -> Dict[str, Any]:
    """Get information about a model type."""
    registry = ModelRegistry()
    
    if model_type not in registry.list_available_types():
        return {"error": f"Unknown model type: {model_type}"}
    
    # Create a dummy config to get model info
    config = ModelConfig(
        model_type=model_type,
        model_path="dummy.pth",
        input_shape=(1, 3, 256, 256)
    )
    
    try:
        loader = registry.get_loader(model_type)
        info = loader.get_model_info(config)
        return info
    except Exception as e:
        return {"error": f"Error getting model info: {e}"}


def convert_model(
    model_type: str,
    model_path: str,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    precision: str = "fp16",
    workspace_size: int = 1 << 30,
    benchmark: bool = False,
    **kwargs
) -> str:
    """
    Convert a model to TensorRT format.
    
    Args:
        model_type: Type of model to convert
        model_path: Path to PyTorch model
        output_path: Path to save TensorRT model
        input_shape: Input tensor shape
        precision: Precision mode
        workspace_size: Workspace size in bytes
        benchmark: Whether to run benchmark after conversion
        **kwargs: Additional model-specific parameters
        
    Returns:
        Path to converted TensorRT model
    """
    print("=" * 60)
    print("TENSORRT MODEL CONVERSION")
    print("=" * 60)
    
    # Validate inputs
    if not validate_model_path(model_path):
        raise ValueError("Invalid model path")
    
    if not validate_output_path(output_path):
        raise ValueError("Invalid output path")
    
    # Get model info
    model_info = get_model_info(model_type)
    if "error" in model_info:
        raise ValueError(model_info["error"])
    
    print(f"Model Type: {model_type}")
    print(f"Model Path: {model_path}")
    print(f"Output Path: {output_path}")
    print(f"Input Shape: {input_shape}")
    print(f"Precision: {precision}")
    print(f"Workspace Size: {workspace_size / (1 << 30):.1f} GB")
    print(f"Model Info: {model_info}")
    print()
    
    # Convert model
    print("Starting TensorRT conversion...")
    try:
        tensorrt_path = convert_any_model_to_tensorrt(
            model_type=model_type,
            model_path=model_path,
            output_path=output_path,
            input_shape=input_shape,
            precision=precision,
            workspace_size=workspace_size,
            **kwargs
        )
        
        print(f"✅ Conversion completed successfully!")
        print(f"TensorRT model saved to: {tensorrt_path}")
        
        # Get file sizes
        pytorch_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        tensorrt_size = os.path.getsize(tensorrt_path) / (1024 * 1024)  # MB
        compression_ratio = pytorch_size / tensorrt_size if tensorrt_size > 0 else 0
        
        print(f"PyTorch model size: {pytorch_size:.1f} MB")
        print(f"TensorRT model size: {tensorrt_size:.1f} MB")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        raise
    
    # Run benchmark if requested
    if benchmark:
        print("\n" + "=" * 60)
        print("RUNNING BENCHMARK")
        print("=" * 60)
        
        try:
            results = benchmark_any_model(
                model_type=model_type,
                pytorch_model_path=model_path,
                tensorrt_model_path=tensorrt_path,
                input_shape=input_shape,
                num_runs=100,
                **kwargs
            )
            
            print("✅ Benchmark completed successfully!")
            print(f"PyTorch avg time: {results['pytorch']['avg_inference_time']:.2f} ms")
            print(f"TensorRT avg time: {results['tensorrt']['avg_inference_time']:.2f} ms")
            print(f"Speedup: {results['speedup']:.2f}x")
            print(f"FPS improvement: {results['tensorrt']['fps']:.1f} FPS")
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            print("Note: Model conversion was successful, but benchmark failed")
    
    return tensorrt_path


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to TensorRT format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert hybrid keypoint model
  python tensorrt_convert.py --model_type hybrid_keypoint_net --model_path models/model.pth --output_path models/model.trt
  
  # Convert CLIP heatmap model with custom settings
  python tensorrt_convert.py --model_type clip_heatmap_model --model_path models/clip_model.pth --output_path models/clip_model.trt --precision fp16 --input_shape 1 3 256 256
  
  # Convert with benchmark
  python tensorrt_convert.py --model_type efficient_keypoint_net --model_path models/efficient_model.pth --output_path models/efficient_model.trt --benchmark
  
  # Use config file
  python tensorrt_convert.py --config config.json --model_path models/model.pth --output_path models/model.trt
        """
    )
    
    # Model configuration
    parser.add_argument("--model_type", type=str, 
                       choices=['hybrid_keypoint_net', 'clip_heatmap_model', 'efficient_keypoint_net'],
                       help="Type of model to convert")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to PyTorch model (.pth or .pt file)")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save TensorRT model (.trt file)")
    
    # Conversion parameters
    parser.add_argument("--input_shape", type=int, nargs=4, default=[1, 3, 256, 256],
                       help="Input shape (batch, channels, height, width)")
    parser.add_argument("--precision", type=str, default="fp16", 
                       choices=["fp32", "fp16", "int8"],
                       help="Precision mode for TensorRT conversion")
    parser.add_argument("--workspace_size", type=int, default=1,
                       help="Workspace size in GB (default: 1)")
    
    # Model-specific parameters
    parser.add_argument("--use_enhanced_yolo", action="store_true", default=True,
                       help="Use Enhanced YOLO backbone (default: True)")
    parser.add_argument("--model_name", type=str, default="facebook/metaclip-b16-fullcc2.5b",
                       help="CLIP model name (for clip_heatmap_model)")
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA fine-tuning (for clip_heatmap_model)")
    parser.add_argument("--model_variant", type=str, default="EfficientKeypointNet",
                       choices=["EfficientKeypointNet", "EfficientViTKeypointNet", "MobileKeypointNet"],
                       help="Model variant (for efficient_keypoint_net)")
    
    # Additional options
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark after conversion")
    parser.add_argument("--config", type=str,
                       help="Path to JSON configuration file")
    parser.add_argument("--save_config", type=str,
                       help="Save current configuration to JSON file")
    parser.add_argument("--list_models", action="store_true",
                       help="List available model types and exit")
    
    args = parser.parse_args()
    
    # List available models
    if args.list_models:
        print("Available model types:")
        registry = ModelRegistry()
        for model_type in registry.list_available_types():
            info = get_model_info(model_type)
            print(f"  - {model_type}: {info}")
        return
    
    # Load config from file if provided
    config = {}
    if args.config:
        config = load_config_from_file(args.config)
    
    # Override config with command line arguments
    if args.model_type:
        config['model_type'] = args.model_type
    if args.model_path:
        config['model_path'] = args.model_path
    if args.output_path:
        config['output_path'] = args.output_path
    if args.input_shape:
        config['input_shape'] = tuple(args.input_shape)
    if args.precision:
        config['precision'] = args.precision
    if args.workspace_size:
        config['workspace_size'] = args.workspace_size * (1 << 30)  # Convert GB to bytes
    if args.benchmark:
        config['benchmark'] = args.benchmark
    
    # Model-specific parameters
    config['use_enhanced_yolo'] = args.use_enhanced_yolo
    config['model_name'] = args.model_name
    config['use_lora'] = args.use_lora
    config['model_variant'] = args.model_variant
    
    # Save config if requested
    if args.save_config:
        save_config_to_file(config, args.save_config)
    
    # Validate required parameters
    if not config.get('model_type'):
        parser.error("--model_type is required")
    
    # Convert model
    try:
        tensorrt_path = convert_model(
            model_type=config['model_type'],
            model_path=config['model_path'],
            output_path=config['output_path'],
            input_shape=config.get('input_shape', (1, 3, 256, 256)),
            precision=config.get('precision', 'fp16'),
            workspace_size=config.get('workspace_size', 1 << 30),
            benchmark=config.get('benchmark', False),
            use_enhanced_yolo=config.get('use_enhanced_yolo', True),
            model_name=config.get('model_name', 'facebook/metaclip-b16-fullcc2.5b'),
            use_lora=config.get('use_lora', True),
            model_variant=config.get('model_variant', 'EfficientKeypointNet')
        )
        
        print(f"\n🎉 Conversion completed successfully!")
        print(f"TensorRT model: {tensorrt_path}")
        
    except Exception as e:
        print(f"\n💥 Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
