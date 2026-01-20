#!/usr/bin/env python3
"""
TensorRT Conversion Script for BedSheet Keypoint Models

This script allows converting PyTorch models to TensorRT engines.
It supports:
1. Full conversion: PyTorch -> ONNX -> TensorRT (default)
2. Export only: PyTorch -> ONNX (useful for cross-platform workflows)
3. Build only: ONNX -> TensorRT (useful for running on target device)

Usage:
    python convert.py --model_type clip_heatmap_model --model_path path/to/model.pth --output_path model.trt
    python convert.py --mode export_only ...
    python convert.py --mode build_only ...
"""

import argparse
import logging
import os
import sys

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.tensorrt_utils import convert_any_model_to_tensorrt, benchmark_any_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="TensorRT Conversion Tool")
    
    # Required arguments
    parser.add_argument("--model_type", type=str, required=True, 
                       choices=['hybrid_keypoint_net', 'clip_heatmap_model', 'efficient_keypoint_net', 'yolo_model'],
                       help="Type of model to convert")
    parser.add_argument("--output_path", type=str, required=True, 
                       help="Path to save output (TensorRT engine or ONNX model)")
    
    # Mode arguments
    parser.add_argument("--mode", type=str, default="full", 
                       choices=["full", "export_only", "build_only"],
                       help="Conversion mode: 'full' (PyTorch->TRT), 'export_only' (PyTorch->ONNX), 'build_only' (ONNX->TRT)")
    
    # Input arguments
    parser.add_argument("--model_path", type=str, help="Path to PyTorch model (.pth). Required for 'full' and 'export_only' modes.")
    parser.add_argument("--onnx_path", type=str, help="Path to ONNX model. Required for 'build_only' mode. Optional for others (as intermediate path).")
    
    # Configuration
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"], 
                       help="Precision mode for TensorRT")
    parser.add_argument("--input_shape", type=int, nargs=4, default=None,
                       help="Input shape (batch, channels, height, width). Defaults: [1, 3, 256, 256] or [1, 3, 560, 560] depending on model.")
    parser.add_argument("--workspace_size", type=str, default="1", 
                       help="Workspace size in GB (default: 1)")
    
    # Model specific args
    parser.add_argument("--model_name", type=str, default="facebook/metaclip-b16-fullcc2.5b",
                       help="CLIP model name (for clip_heatmap_model)")
    
    # Actions
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after conversion")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate arguments
    if args.mode in ["full", "export_only"] and not args.model_path:
        logger.error("--model_path is required for 'full' and 'export_only' modes")
        sys.exit(1)
        
    if args.mode == "build_only" and not args.onnx_path:
        # If output_path is .onnx, maybe they meant that? No, output is TRT in build_only.
        # But we can try to infer onnx_path from default locations if needed, but safer to require it.
        # Actually, let's allow onnx_path to be inferred or passed as model_path if users are confused, 
        # but strictest is best.
        if args.model_path and args.model_path.endswith('.onnx'):
            args.onnx_path = args.model_path
        else:
            logger.error("--onnx_path (or --model_path pointing to .onnx) is required for 'build_only' mode")
            sys.exit(1)

    # Set default input shape if not provided
    if args.input_shape is None:
        if args.model_type == "yolo_model":
            # Ultralytics YOLO segmentation defaults to imgsz=640 in our codepaths.
            args.input_shape = [1, 3, 640, 640]
            logger.info(f"Using YOLO default input shape: {args.input_shape}")
        else:
            # Check both model_path and onnx_path for keywords
            img_size = 256
            
            paths_to_check = []
            if args.model_path: paths_to_check.append(args.model_path)
            if args.onnx_path: paths_to_check.append(args.onnx_path)
            
            # Check heuristics
            is_560 = False
            for p in paths_to_check:
                p_lower = p.lower()
                if 'mattress' in p_lower or 'fitted' in p_lower:
                    is_560 = True
                    break
            
            if is_560:
                 args.input_shape = [1, 3, 560, 560]
                 logger.info(f"Inferred input shape from path keywords ('mattress'/'fitted'): {args.input_shape}")
            else:
                 args.input_shape = [1, 3, 256, 256]
                 logger.info(f"Using standard default input shape: {args.input_shape}")

    # Convert workspace size to bytes
    try:
        workspace_size = int(float(args.workspace_size) * (1 << 30))
    except ValueError:
        logger.error("Invalid workspace size")
        sys.exit(1)
        
    # Setup paths
    export_only = (args.mode == "export_only")
    from_onnx = args.onnx_path if args.mode == "build_only" else None
    
    # If using build_only, model_path might be None, but ModelConfig needs a model_path.
    # We can pass a dummy path if we are loading from ONNX, as the loader won't be called for PyTorch.
    # Wait, ModelConfig is used by loader. 
    # In `convert_model`:
    #   if from_onnx:
    #       onnx_path = from_onnx
    #   else:
    #       loader.load_model(config) ...
    # So if from_onnx is set, load_model is NOT called. So model_path in config can be dummy.
    model_path = args.model_path if args.model_path else "dummy_path_for_build_mode.pth"

    try:
        logger.info(f"Starting conversion in mode: {args.mode}")
        
        result_path = convert_any_model_to_tensorrt(
            model_type=args.model_type,
            model_path=model_path,
            output_path=args.output_path,
            input_shape=tuple(args.input_shape),
            precision=args.precision,
            workspace_size=workspace_size,
            export_only=export_only,
            from_onnx=from_onnx,
            # Extra kwargs
            model_name=args.model_name
        )
        
        logger.info(f"Process completed successfully. Output: {result_path}")
        
        if args.benchmark and args.mode != "export_only":
            logger.info("Running benchmark...")
            if args.mode == "build_only":
                # Benchmark TRT only since we don't have PyTorch model loaded necessarily
                from src.utils.tensorrt_utils import GeneralizedTensorRTInference
                trt_inf = GeneralizedTensorRTInference(result_path)
                bench_res = trt_inf.benchmark()
                print(f"TRT Benchmark Results: {bench_res}")
            else:
                 # Full benchmark
                results = benchmark_any_model(
                    model_type=args.model_type,
                    pytorch_model_path=args.model_path,
                    tensorrt_model_path=result_path,
                    input_shape=tuple(args.input_shape),
                    model_name=args.model_name
                )
                print(f"Benchmark Results: {results}")

    except ImportError as e:
        logger.error(f"Import Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
