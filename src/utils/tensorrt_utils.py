"""
Generalized TensorRT utilities for multiple model types.

This module provides a flexible system to convert various PyTorch models to TensorRT format
for faster inference on NVIDIA GPUs. It supports multiple model architectures through
a registry-based system.
"""

import os
import torch
import numpy as np
try:
    import tensorrt as trt
except ImportError:
    trt = None
    
from typing import Optional, Tuple, Dict, Any, Union, Callable, Type
from abc import ABC, abstractmethod
import logging
import importlib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for model conversion."""
    
    def __init__(
        self,
        model_type: str,
        model_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
        precision: str = "fp16",
        workspace_size: int = 1 << 30,
        max_batch_size: int = 1,
        dynamic_axes: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.model_type = model_type
        self.model_path = model_path
        self.input_shape = input_shape
        self.precision = precision
        self.workspace_size = workspace_size
        self.max_batch_size = max_batch_size
        self.dynamic_axes = dynamic_axes or {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        self.extra_kwargs = kwargs


class ModelLoader(ABC):
    """Abstract base class for model loaders."""
    
    @abstractmethod
    def load_model(self, config: ModelConfig) -> torch.nn.Module:
        """Load and return a PyTorch model."""
        pass
    
    @abstractmethod
    def get_model_info(self, config: ModelConfig) -> Dict[str, Any]:
        """Get model information for debugging."""
        pass


class HybridKeypointNetLoader(ModelLoader):
    """Loader for HybridKeypointNet models."""
    
    def load_model(self, config: ModelConfig) -> torch.nn.Module:
        """Load HybridKeypointNet model."""
        from src.models.hybrid_keypoint_net import HybridKeypointNet
        from src.utils.model_utils import YoloBackbone, EnhancedYoloBackbone
        from ultralytics import YOLO
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        use_enhanced_yolo = config.extra_kwargs.get('use_enhanced_yolo', True)
        
        if use_enhanced_yolo:
            logger.info("Using Enhanced YOLO backbone")
            yolo_model = YOLO('yolo11l-pose.pt')
            backbone = EnhancedYoloBackbone(
                yolo_model, 
                include_neck=True,
                selected_indices=[2, 4, 6, 8, 10, 13, 16, 19, 22]
            )
        else:
            logger.info("Using Original YOLO backbone")
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
        state_dict = torch.load(config.model_path, map_location=device)
        
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
        return model
    
    def get_model_info(self, config: ModelConfig) -> Dict[str, Any]:
        return {
            'model_type': 'HybridKeypointNet',
            'use_enhanced_yolo': config.extra_kwargs.get('use_enhanced_yolo', True),
            'input_shape': config.input_shape
        }


class ClipHeatmapModelLoader(ModelLoader):
    """Loader for CLIP-based heatmap models."""
    
    def load_model(self, config: ModelConfig) -> torch.nn.Module:
        """Load CLIP heatmap model."""
        from src.models.clip_heatmap_model import create_clip_heatmap_model
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        model = create_clip_heatmap_model(
            model_name=config.extra_kwargs.get('model_name', 'facebook/metaclip-b16-fullcc2.5b'),
            image_size=config.input_shape[-1],
            use_lora=config.extra_kwargs.get('use_lora', True),
            lora_r=config.extra_kwargs.get('lora_r', 16),
            lora_alpha=config.extra_kwargs.get('lora_alpha', 32),
            lora_dropout=config.extra_kwargs.get('lora_dropout', 0.05),
            use_text_prior=config.extra_kwargs.get('use_text_prior', True),
            prior_prompts=config.extra_kwargs.get('prior_prompts', None),
            negative_prompts=config.extra_kwargs.get('negative_prompts', None),
            prior_weight=config.extra_kwargs.get('prior_weight', 0.5),
            # Unified decoder spec (supported):
            # - examples:
            #   {"kind":"gcnn","mode":"so2","hidden":32,"so2_num_angles":8,"so2_num_gconvs":2}
            #   {"kind":"gcnn","mode":"c4","hidden":64}
            #   "standard"
            head_decoder=config.extra_kwargs.get('head_decoder', None),
        )
        
        # Load state dict
        state_dict = torch.load(config.model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    def get_model_info(self, config: ModelConfig) -> Dict[str, Any]:
        return {
            'model_type': 'ClipHeatmapModel',
            'model_name': config.extra_kwargs.get('model_name', 'facebook/metaclip-2-worldwide-l14'),
            'use_lora': config.extra_kwargs.get('use_lora', True),
            'input_shape': config.input_shape
        }


class EfficientKeypointNetLoader(ModelLoader):
    """Loader for EfficientKeypointNet models."""
    
    def load_model(self, config: ModelConfig) -> torch.nn.Module:
        """Load EfficientKeypointNet model."""
        from src.models.efficient_keypoint_net import EfficientKeypointNet, EfficientViTKeypointNet, MobileKeypointNet
        from src.utils.model_utils import YoloBackbone
        from ultralytics import YOLO
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_variant = config.extra_kwargs.get('model_variant', 'EfficientKeypointNet')
        
        # Create backbone
        yolo_model = YOLO('yolov8s.pt')
        backbone_seq = yolo_model.model.model[:8]
        backbone = YoloBackbone(backbone_seq, selected_indices=[0,1,2,3,4,5,6,7])
        
        # Get input channels list
        input_dummy = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            feats = backbone(input_dummy)
        in_channels_list = [f.shape[1] for f in feats]
        
        # Create model based on variant
        if model_variant == 'EfficientKeypointNet':
            model = EfficientKeypointNet(backbone, in_channels_list)
        elif model_variant == 'EfficientViTKeypointNet':
            model = EfficientViTKeypointNet(backbone, in_channels_list)
        elif model_variant == 'MobileKeypointNet':
            model = MobileKeypointNet(backbone, in_channels_list)
        else:
            raise ValueError(f"Unknown model variant: {model_variant}")
        
        # Load state dict
        state_dict = torch.load(config.model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    
    def get_model_info(self, config: ModelConfig) -> Dict[str, Any]:
        return {
            'model_type': 'EfficientKeypointNet',
            'model_variant': config.extra_kwargs.get('model_variant', 'EfficientKeypointNet'),
            'input_shape': config.input_shape
        }


class ModelRegistry:
    """Registry for different model loaders."""
    
    def __init__(self):
        self._loaders: Dict[str, ModelLoader] = {}
        self._register_default_loaders()
    
    def _register_default_loaders(self):
        """Register default model loaders."""
        self.register_loader('hybrid_keypoint_net', HybridKeypointNetLoader())
        self.register_loader('clip_heatmap_model', ClipHeatmapModelLoader())
        self.register_loader('efficient_keypoint_net', EfficientKeypointNetLoader())
    
    def register_loader(self, model_type: str, loader: ModelLoader):
        """Register a model loader."""
        self._loaders[model_type] = loader
        logger.info(f"Registered loader for model type: {model_type}")
    
    def get_loader(self, model_type: str) -> ModelLoader:
        """Get a model loader by type."""
        if model_type not in self._loaders:
            raise ValueError(f"No loader registered for model type: {model_type}")
        return self._loaders[model_type]
    
    def list_available_types(self) -> list:
        """List all available model types."""
        return list(self._loaders.keys())


class GeneralizedTensorRTConverter:
    """Generalized TensorRT converter for multiple model types."""
    
    
    def __init__(self, workspace_size: int = 1 << 30):
        """
        Initialize TensorRT converter.
        
        Args:
            workspace_size: Maximum workspace size in bytes (default: 1GB)
        """
        if trt is None:
            # We allow initialization without TRT if we only want to export ONNX
            # However, if we try to use builder methods, we will need to check again
            pass
            
        self.workspace_size = workspace_size
        self.registry = ModelRegistry()
        
        # Initialize builder only if TRT is available
        if trt is not None:
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.builder = trt.Builder(self.logger)
            self.config = self.builder.create_builder_config()
            self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
        else:
            self.logger = None
            self.builder = None
            self.config = None
        
    def convert_model(
        self,
        config: ModelConfig,
        output_path: str,
        export_only: bool = False,
        from_onnx: Optional[str] = None
    ) -> str:
        """
        Convert any supported model to TensorRT format.
        
        Args:
            config: Model configuration
            output_path: Path to save TensorRT model (or ONNX if export_only=True)
            export_only: If True, only export to ONNX and stop
            from_onnx: If provided, use this ONNX file instead of exporting from PyTorch
            
        Returns:
            Path to saved model
        """
        if export_only:
             logger.info(f"Exporting {config.model_type} model to ONNX...")
             
             # Get model loader
             loader = self.registry.get_loader(config.model_type)
             
             # Load model
             model = loader.load_model(config)
             
             # Export
             self._export_to_onnx(model, config, output_path)
             return output_path

        if trt is None:
            raise ImportError("TensorRT is not installed. Cannot build TensorRT engine. Use export_only=True to just export ONNX.")

        logger.info(f"Converting {config.model_type} model to TensorRT with precision: {config.precision}")
        
        # Set precision
        if config.precision == "fp16" and self.builder.platform_has_fast_fp16:
            self.config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 precision enabled")
        elif config.precision == "int8" and self.builder.platform_has_fast_int8:
            self.config.set_flag(trt.BuilderFlag.INT8)
            logger.info("INT8 precision enabled")
        
        # Add optimization profile for dynamic inputs
        profile = self.builder.create_optimization_profile()
        min_shape = (1, config.input_shape[1], config.input_shape[2], config.input_shape[3])
        opt_shape = config.input_shape
        max_shape = (config.max_batch_size, config.input_shape[1], config.input_shape[2], config.input_shape[3])
        profile.set_shape("input", min_shape, opt_shape, max_shape)
        self.config.add_optimization_profile(profile)

        # Create network
        network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # Create ONNX model or use existing
        if from_onnx:
            onnx_path = from_onnx
            logger.info(f"Using existing ONNX model: {onnx_path}")
        else:
            # Determine ONNX path
            if output_path.endswith('.trt'):
                onnx_path = output_path.replace('.trt', '.onnx')
            else:
                onnx_path = output_path + '.onnx'
                
            # Get model loader
            loader = self.registry.get_loader(config.model_type)
            
            # Load model
            model = loader.load_model(config)
            model_info = loader.get_model_info(config)
            logger.info(f"Model info: {model_info}")
            
            self._export_to_onnx(model, config, onnx_path)
        
        # Parse ONNX to TensorRT
        
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
        
        # Clean up ONNX file ONLY if we generated it
        if not from_onnx and os.path.exists(onnx_path):
            os.remove(onnx_path)
        
        return output_path
    
    def _export_to_onnx(
        self,
        model: torch.nn.Module,
        config: ModelConfig,
        output_path: str
    ):
        """Export PyTorch model to ONNX format."""
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(config.input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=20,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=config.dynamic_axes
        )
        logger.info(f"ONNX model exported to: {output_path}")



class GeneralizedTensorRTInference:
    """Generalized TensorRT inference engine."""
    
    def __init__(self, model_path: str):
        """
        Initialize TensorRT inference engine.
        
        Args:
            model_path: Path to TensorRT model file
        """
        if trt is None:
            raise ImportError("TensorRT is not installed. Cannot run inference.")
            
        self.model_path = model_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load TensorRT engine
        with open(model_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Get input/output information
        try:
            input_name = self.engine.get_tensor_name(0)
            output_name = self.engine.get_tensor_name(1)
            
            input_dims = self.engine.get_tensor_shape(input_name)
            output_dims = self.engine.get_tensor_shape(output_name)
            self.input_shape = tuple(input_dims)
            self.output_shape = tuple(output_dims)
            
        except Exception as e:
            logger.warning(f"Could not get tensor shapes: {e}")
                # Use default shapes
            self.input_shape = (1, 3, 256, 256)
            self.output_shape = (1, 1, 256, 256)
        
        logger.info(f"TensorRT model loaded: {model_path}")
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Output shape: {self.output_shape}")
    
    def infer(self, input_data) -> np.ndarray:
        """
        Run inference on input data.
        
        Args:
            input_data: Input tensor (PyTorch tensor or numpy array) with shape (batch_size, channels, height, width)
            
        Returns:
            Model output
        """
        # Handle both PyTorch tensors and numpy arrays
        if isinstance(input_data, torch.Tensor):
            # If it's a PyTorch tensor, convert to numpy first
            if input_data.is_cuda:
                input_numpy = input_data.cpu().numpy()
            else:
                input_numpy = input_data.numpy()
        else:
            # It's already a numpy array
            input_numpy = input_data
        
        # Allocate GPU memory
        input_size = trt.volume(self.input_shape) * input_numpy.dtype.itemsize
        output_size = trt.volume(self.output_shape) * np.float32().itemsize
        
        d_input = torch.cuda.FloatTensor(input_numpy).contiguous()
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


def convert_any_model_to_tensorrt(
    model_type: str,
    model_path: str,
    output_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    precision: str = "fp16",
    workspace_size: int = 1 << 30,
    export_only: bool = False,
    from_onnx: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convert any supported model to TensorRT format.
    
    Args:
        model_type: Type of model ('hybrid_keypoint_net', 'clip_heatmap_model', 'efficient_keypoint_net')
        model_path: Path to PyTorch model (.pth file)
        output_path: Path to save TensorRT model (.trt file)
        input_shape: Input tensor shape
        precision: Precision mode ("fp32", "fp16", "int8")
        workspace_size: Maximum workspace size in bytes
        export_only: If True, only export to ONNX
        from_onnx: If provided, use this ONNX file
        **kwargs: Additional model-specific parameters
        
    Returns:
        Path to saved TensorRT model
    """
    config = ModelConfig(
        model_type=model_type,
        model_path=model_path,
        input_shape=input_shape,
        precision=precision,
        workspace_size=workspace_size,
        **kwargs
    )
    
    converter = GeneralizedTensorRTConverter(workspace_size=workspace_size)
    return converter.convert_model(config, output_path, export_only=export_only, from_onnx=from_onnx)


def benchmark_any_model(
    model_type: str,
    pytorch_model_path: str,
    tensorrt_model_path: str,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    num_runs: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark any supported model (TensorRT vs PyTorch).
    
    Args:
        model_type: Type of model
        pytorch_model_path: Path to PyTorch model
        tensorrt_model_path: Path to TensorRT model
        input_shape: Input tensor shape
        num_runs: Number of benchmark runs
        **kwargs: Additional model-specific parameters
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Running TensorRT vs PyTorch benchmark for {model_type}")
    
    # Load PyTorch model
    config = ModelConfig(
        model_type=model_type,
        model_path=pytorch_model_path,
        input_shape=input_shape,
        **kwargs
    )
    
    registry = ModelRegistry()
    loader = registry.get_loader(model_type)
    pytorch_model = loader.load_model(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pytorch_model = pytorch_model.to(device)
    pytorch_model.eval()
    
    # Load TensorRT model
    tensorrt_inference = GeneralizedTensorRTInference(tensorrt_model_path)
    
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
        'model_type': model_type,
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


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Generalized TensorRT conversion")
    parser.add_argument("--model_type", type=str, required=True, 
                       choices=['hybrid_keypoint_net', 'clip_heatmap_model', 'efficient_keypoint_net'],
                       help="Type of model to convert")
    parser.add_argument("--model_path", type=str, required=True, help="Path to PyTorch model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save TensorRT model")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"], 
                       help="Precision mode")
    parser.add_argument("--input_shape", type=int, nargs=4, default=[1, 3, 256, 256],
                       help="Input shape (batch, channels, height, width)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark after conversion")
    
    args = parser.parse_args()
    
    # Convert model
    tensorrt_path = convert_any_model_to_tensorrt(
        model_type=args.model_type,
        model_path=args.model_path,
        output_path=args.output_path,
        input_shape=tuple(args.input_shape),
        precision=args.precision
    )
    
    # Run benchmark if requested
    if args.benchmark:
        results = benchmark_any_model(
            model_type=args.model_type,
            pytorch_model_path=args.model_path,
            tensorrt_model_path=tensorrt_path,
            input_shape=tuple(args.input_shape)
        )
        print(f"Benchmark results: {results}")
