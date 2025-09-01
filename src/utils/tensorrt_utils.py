"""
TensorRT utilities for keypoint detection model optimization.

This module provides functions to convert PyTorch models to TensorRT format
for faster inference on NVIDIA GPUs.
"""

import os
import torch
import numpy as np
import tensorrt as trt
from typing import Optional, Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorRTConverter:
    """TensorRT model converter for keypoint detection models."""
    
    def __init__(self, workspace_size: int = 1 << 30):
        """
        Initialize TensorRT converter.
        
        Args:
            workspace_size: Maximum workspace size in bytes (default: 1GB)
        """
        self.workspace_size = workspace_size
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.config = self.builder.create_builder_config()
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        
    def convert_pytorch_to_tensorrt(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, int, int, int],
        output_path: str,
        precision: str = "fp16",
        max_batch_size: int = 1,
        dynamic_axes: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Convert PyTorch model to TensorRT format.
        
        Args:
            model: PyTorch model to convert
            input_shape: Input tensor shape (batch_size, channels, height, width)
            output_path: Path to save TensorRT model
            precision: Precision mode ("fp32", "fp16", "int8")
            max_batch_size: Maximum batch size for TensorRT model
            dynamic_axes: Dynamic axes configuration
            
        Returns:
            Path to saved TensorRT model
        """
        logger.info(f"Converting model to TensorRT with precision: {precision}")
        
        # Set precision
        if precision == "fp16" and self.builder.platform_has_fast_fp16:
            self.config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 precision enabled")
        elif precision == "int8" and self.builder.platform_has_fast_int8:
            self.config.set_flag(trt.BuilderFlag.INT8)
            logger.info("INT8 precision enabled")
        
        # Add optimization profile for dynamic inputs
        profile = self.builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 128, 128), (1, 3, 128, 128), (1, 3, 128, 128))
        self.config.add_optimization_profile(profile)
        
        # Create network
        network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # Create ONNX model first
        onnx_path = output_path.replace('.trt', '.onnx')
        self._export_to_onnx(model, input_shape, onnx_path, dynamic_axes)
        
        # Parse ONNX to TensorRT
        parser = trt.OnnxParser(network, self.logger)
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                for error in range(parser.num_errors):
                    logger.error(f"ONNX parsing error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")
        
        # Build TensorRT engine
        serialized_engine = self.builder.build_serialized_network(network, self.config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save TensorRT model
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)
        
        logger.info(f"TensorRT model saved to: {output_path}")
        
        # Clean up ONNX file
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        
        return output_path
    
    def _export_to_onnx(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, int, int, int],
        output_path: str,
        dynamic_axes: Optional[Dict[str, Any]] = None
    ):
        """Export PyTorch model to ONNX format."""
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes or {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        logger.info(f"ONNX model exported to: {output_path}")


class TensorRTInference:
    """TensorRT inference engine for keypoint detection."""
    
    def __init__(self, model_path: str):
        """
        Initialize TensorRT inference engine.
        
        Args:
            model_path: Path to TensorRT model file
        """
        self.model_path = model_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load TensorRT engine
        with open(model_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Get input/output information - use the correct TensorRT API
        try:
            # Try to get tensor names first
            input_name = self.engine.get_tensor_name(0)
            output_name = self.engine.get_tensor_name(1)
            
            # Get shapes using the correct method and convert to tuple
            input_dims = self.engine.get_tensor_shape(input_name)
            output_dims = self.engine.get_tensor_shape(output_name)
            self.input_shape = tuple(input_dims)
            self.output_shape = tuple(output_dims)
            
        except Exception as e:
            # Fallback: try to get shapes directly
            try:
                self.input_shape = self.engine.get_tensor_shape(0)
                self.output_shape = self.engine.get_tensor_shape(1)
            except Exception as e2:
                print(f"Warning: Could not get tensor shapes: {e2}")
                # Use default shapes
                self.input_shape = (1, 3, 128, 128)
                self.output_shape = (1, 1, 128, 128)
        
        logger.info(f"TensorRT model loaded: {model_path}")
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Output shape: {self.output_shape}")
    
    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Input tensor with shape (batch_size, channels, height, width)
            
        Returns:
            Model output
        """
        # Allocate GPU memory
        input_size = trt.volume(self.input_shape) * input_data.dtype.itemsize
        output_size = trt.volume(self.output_shape) * np.float32().itemsize
        
        d_input = torch.cuda.FloatTensor(input_data).contiguous()
        d_output = torch.empty(self.output_shape, dtype=torch.float32, device='cuda')
        
        # Create bindings
        bindings = [d_input.data_ptr(), d_output.data_ptr()]
        
        # Run inference
        self.context.execute_v2(bindings)
        
        # Get output
        output = d_output.cpu().numpy()
        
        return output
    
    def benchmark(self, num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary with performance metrics
        """
        import time
        
        # Create dummy input
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup_runs):
            _ = self.infer(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_runs):
            _ = self.infer(dummy_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        fps = num_runs / total_time
        
        return {
            'total_time': total_time,
            'avg_inference_time': avg_time * 1000,  # Convert to ms
            'fps': fps,
            'num_runs': num_runs
        }


def convert_keypoint_model_to_tensorrt(
    model_path: str,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 128, 128),
    precision: str = "fp16",
    workspace_size: int = 1 << 30
) -> str:
    """
    Convert keypoint detection model to TensorRT format.
    
    Args:
        model_path: Path to PyTorch model (.pth file)
        output_path: Path to save TensorRT model (.trt file)
        input_shape: Input tensor shape
        precision: Precision mode ("fp32", "fp16", "int8")
        workspace_size: Maximum workspace size in bytes
        
    Returns:
        Path to saved TensorRT model
    """
    logger.info(f"Converting keypoint model: {model_path}")
    
    # Import model architecture and utilities
    from src.models.hybrid_keypoint_net import HybridKeypointNet
    from src.utils.model_utils import YoloBackbone
    from ultralytics import YOLO
    
    # Create model with proper initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create YOLO backbone (use the same model as in training)
    yolo_model = YOLO('yolo11l-pose.pt')
    backbone_seq = yolo_model.model.model[:12]
    backbone = YoloBackbone(backbone_seq, selected_indices=[0,1,2,3,4,5,6,7,8,9,10,11])
    
    # Get input channels list
    input_dummy = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        feats = backbone(input_dummy)
    in_channels_list = [f.shape[1] for f in feats]
    
    # Create model
    model = HybridKeypointNet(backbone, in_channels_list)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle model state dict with _orig_mod prefixes
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key[10:]  # Remove "_orig_mod." prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Load with strict=False to handle any remaining mismatches
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
    
    model.eval()
    
    # Convert to TensorRT
    converter = TensorRTConverter(workspace_size=workspace_size)
    tensorrt_path = converter.convert_pytorch_to_tensorrt(
        model=model,
        input_shape=input_shape,
        output_path=output_path,
        precision=precision
    )
    
    logger.info(f"Conversion completed: {tensorrt_path}")
    return tensorrt_path


def benchmark_tensorrt_vs_pytorch(
    pytorch_model_path: str,
    tensorrt_model_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 128, 128),
    num_runs: int = 100
) -> Dict[str, Any]:
    """
    Benchmark TensorRT vs PyTorch inference performance.
    
    Args:
        pytorch_model_path: Path to PyTorch model
        tensorrt_model_path: Path to TensorRT model
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info("Running TensorRT vs PyTorch benchmark")
    
    # Import model architecture and utilities
    from src.models.hybrid_keypoint_net import HybridKeypointNet
    from src.utils.model_utils import YoloBackbone
    from ultralytics import YOLO
    
    # Create PyTorch model with proper initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create YOLO backbone (use the same model as in training)
    yolo_model = YOLO('yolo11l-pose.pt')
    backbone_seq = yolo_model.model.model[:12]
    backbone = YoloBackbone(backbone_seq, selected_indices=[0,1,2,3,4,5,6,7,8,9,10,11])
    
    # Get input channels list
    input_dummy = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        feats = backbone(input_dummy)
    in_channels_list = [f.shape[1] for f in feats]
    
    # Create model
    pytorch_model = HybridKeypointNet(backbone, in_channels_list)
    
    # Load state dict
    state_dict = torch.load(pytorch_model_path, map_location=device)
    
    # Handle model state dict with _orig_mod prefixes
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            new_key = key[10:]  # Remove "_orig_mod." prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Load with strict=False to handle any remaining mismatches
    missing_keys, unexpected_keys = pytorch_model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {len(unexpected_keys)}")
    
    # Ensure model and inputs are on the same device
    pytorch_model = pytorch_model.to(device)
    pytorch_model.eval()
    
    # Load TensorRT model
    tensorrt_inference = TensorRTInference(tensorrt_model_path)
    
    # Create dummy input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    dummy_input_torch = torch.from_numpy(dummy_input).to(device)
    
    # Benchmark PyTorch
    import time
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = pytorch_model(dummy_input_torch)
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = pytorch_model(dummy_input_torch)
    
    torch.cuda.synchronize()
    pytorch_time = time.time() - start_time
    
    # Benchmark TensorRT
    tensorrt_results = tensorrt_inference.benchmark(num_runs=num_runs)
    
    # Calculate speedup
    pytorch_avg_time = pytorch_time / num_runs * 1000  # Convert to ms
    tensorrt_avg_time = tensorrt_results['avg_inference_time']
    speedup = pytorch_avg_time / tensorrt_avg_time
    
    results = {
        'pytorch': {
            'total_time': pytorch_time,
            'avg_inference_time': pytorch_avg_time,
            'fps': num_runs / pytorch_time
        },
        'tensorrt': tensorrt_results,
        'speedup': speedup
    }
    
    logger.info(f"PyTorch avg time: {pytorch_avg_time:.2f} ms")
    logger.info(f"TensorRT avg time: {tensorrt_avg_time:.2f} ms")
    logger.info(f"Speedup: {speedup:.2f}x")
    
    return results


def create_tensorrt_config(
    model_path: str,
    output_path: str,
    precision: str = "fp16",
    max_batch_size: int = 1,
    workspace_size: int = 1 << 30
) -> Dict[str, Any]:
    """
    Create TensorRT conversion configuration.
    
    Args:
        model_path: Path to PyTorch model
        output_path: Path to save TensorRT model
        precision: Precision mode
        max_batch_size: Maximum batch size
        workspace_size: Workspace size in bytes
        
    Returns:
        Configuration dictionary
    """
    return {
        'model_path': model_path,
        'output_path': output_path,
        'precision': precision,
        'max_batch_size': max_batch_size,
        'workspace_size': workspace_size,
        'input_shape': (max_batch_size, 3, 128, 128),
        'dynamic_axes': {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="TensorRT conversion for keypoint detection model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to PyTorch model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save TensorRT model")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"], help="Precision mode")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after conversion")
    
    args = parser.parse_args()
    
    # Convert model
    tensorrt_path = convert_keypoint_model_to_tensorrt(
        model_path=args.model_path,
        output_path=args.output_path,
        precision=args.precision
    )
    
    # Run benchmark if requested
    if args.benchmark:
        results = benchmark_tensorrt_vs_pytorch(
            pytorch_model_path=args.model_path,
            tensorrt_model_path=tensorrt_path
        )
        print(f"Benchmark results: {results}")
