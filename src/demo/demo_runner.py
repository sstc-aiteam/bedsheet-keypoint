#!/usr/bin/env python3
"""
Main demo runner classes for TensorRT demo.
"""

import os
from pathlib import Path

# Import TensorRT utilities
try:
    from src.utils.tensorrt_utils import GeneralizedTensorRTInference as TensorRTInference
    TENSORRT_AVAILABLE = True
except ImportError:
    print("Warning: TensorRT utilities not available")
    TENSORRT_AVAILABLE = False

from .model_loaders import load_pytorch_model, list_available_models
from .evaluation_utils import (
    evaluate_single_image_pytorch, 
    evaluate_single_image_tensorrt,
    visualize_results,
    benchmark_model,
    load_segmentation_model
)


class TensorRTDemo:
    """Main demo class for TensorRT keypoint detection."""
    
    def __init__(self, config):
        self.config = config
        self.pytorch_model = None
        self.tensorrt_inference = None
        self.segmenter = None
        
    def load_models(self):
        """Load PyTorch and TensorRT models."""
        # Detect model type if auto
        model_type = self.config.get('model_type', 'hybrid_keypoint_net')
        if model_type == 'auto':
            from .model_loaders import detect_model_type
            model_type = detect_model_type(self.config['pytorch_model'])
            self.config['model_type'] = model_type  # Store detected type
        
        print("\nLoading PyTorch model...")
        try:
            self.pytorch_model = load_pytorch_model(
                self.config['pytorch_model'], 
                model_type
            )
            print("✓ PyTorch model loaded successfully")
        except Exception as e:
            print(f"✗ Failed to load PyTorch model: {e}")
            return False
        
        # Load TensorRT model if available
        if os.path.exists(self.config['tensorrt_model']) and TENSORRT_AVAILABLE:
            print("\nLoading TensorRT model...")
            try:
                self.tensorrt_inference = TensorRTInference(self.config['tensorrt_model'])
                print("✓ TensorRT model loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load TensorRT model: {e}")
        
        # Load segmentation model
        self.segmenter = load_segmentation_model()
        
        return True
    
    def run_evaluation(self, image_path):
        """Run evaluation on a single image."""
        print(f"\nEvaluating image: {image_path}")
        
        # Get model type for proper preprocessing
        model_type = self.config.get('model_type', 'hybrid_keypoint_net')
        if model_type == 'auto':
            from .model_loaders import detect_model_type
            model_type = detect_model_type(self.config['pytorch_model'])
        
        # Evaluate PyTorch model
        print("\n=== Evaluating PyTorch Model ===")
        try:
            keypoints_pytorch, original_img, masked_img, heatmap_pytorch = evaluate_single_image_pytorch(
                self.pytorch_model, image_path, self.segmenter, model_type
            )
            print(f"✓ PyTorch model evaluation completed")
            print(f"Detected {len(keypoints_pytorch)} keypoints")
        except Exception as e:
            print(f"✗ PyTorch model evaluation failed: {e}")
            return False
        
        # Evaluate TensorRT model if available
        if self.tensorrt_inference:
            print("\n=== Evaluating TensorRT Model ===")
            try:
                keypoints_tensorrt, _, _, heatmap_tensorrt = evaluate_single_image_tensorrt(
                    self.tensorrt_inference, image_path, self.segmenter, model_type
                )
                print(f"✓ TensorRT model evaluation completed")
                print(f"Detected {len(keypoints_tensorrt)} keypoints")
                
                # Compare results
                print("\n=== Comparing Results ===")
                print("PyTorch Results:")
                visualize_results(original_img, keypoints_pytorch, "PyTorch Model", heatmap_pytorch)
                print("TensorRT Results:")
                visualize_results(original_img, keypoints_tensorrt, "TensorRT Model", heatmap_tensorrt)
                
            except Exception as e:
                print(f"✗ TensorRT model evaluation failed: {e}")
                print("Showing PyTorch results only")
                visualize_results(original_img, keypoints_pytorch, "PyTorch Model Results", heatmap_pytorch)
        else:
            print("\n=== Showing PyTorch Model Results ===")
            visualize_results(original_img, keypoints_pytorch, "PyTorch Model Results", heatmap_pytorch)
        
        return True
    
    def run_benchmark(self, image_path, num_runs=100):
        """Run benchmark comparison between PyTorch and TensorRT."""
        print(f"\n=== Benchmarking Models ({num_runs} runs) ===")
        
        # Get model type for proper preprocessing
        model_type = self.config.get('model_type', 'hybrid_keypoint_net')
        if model_type == 'auto':
            from .model_loaders import detect_model_type
            model_type = detect_model_type(self.config['pytorch_model'])
        
        # Benchmark PyTorch model
        print("\nBenchmarking PyTorch model...")
        pytorch_stats = benchmark_model(
            self.pytorch_model, image_path, num_runs, use_tensorrt=False, model_type=model_type
        )
        
        print(f"PyTorch Results:")
        print(f"  Average: {pytorch_stats['avg_time_ms']:.2f} ms")
        print(f"  Std Dev: {pytorch_stats['std_time_ms']:.2f} ms")
        print(f"  Min: {pytorch_stats['min_time_ms']:.2f} ms")
        print(f"  Max: {pytorch_stats['max_time_ms']:.2f} ms")
        print(f"  FPS: {pytorch_stats['fps']:.2f}")
        
        # Benchmark TensorRT model if available
        if self.tensorrt_inference:
            print("\nBenchmarking TensorRT model...")
            tensorrt_stats = benchmark_model(
                None, image_path, num_runs, use_tensorrt=True, 
                tensorrt_inference=self.tensorrt_inference, model_type=model_type
            )
            
            print(f"TensorRT Results:")
            print(f"  Average: {tensorrt_stats['avg_time_ms']:.2f} ms")
            print(f"  Std Dev: {tensorrt_stats['std_time_ms']:.2f} ms")
            print(f"  Min: {tensorrt_stats['min_time_ms']:.2f} ms")
            print(f"  Max: {tensorrt_stats['max_time_ms']:.2f} ms")
            print(f"  FPS: {tensorrt_stats['fps']:.2f}")
            
            # Calculate speedup
            speedup = pytorch_stats['avg_time_ms'] / tensorrt_stats['avg_time_ms']
            print(f"\nSpeedup: {speedup:.2f}x")
        else:
            print("\nTensorRT model not available for benchmarking")
    
    def run(self):
        """Run the complete demo."""
        print("=== TensorRT Keypoint Detection Demo ===")
        
        # Load models
        if not self.load_models():
            return False
        
        # Get sample image
        from .demo_config import get_sample_image
        sample_image = get_sample_image(self.config['image_dir'])
        if not sample_image:
            print("No sample image found")
            return False
        
        print(f"Using sample image: {sample_image}")
        
        # Run evaluation
        if not self.run_evaluation(sample_image):
            return False
        
        # Run benchmark if requested
        if self.config.get('benchmark', False):
            self.run_benchmark(sample_image, self.config.get('num_runs', 100))
        
        print("\n=== Demo Completed Successfully! ===")
        return True


class CLIPDemo(TensorRTDemo):
    """Specialized demo for CLIP-based models."""
    
    def __init__(self):
        from .demo_config import get_clip_demo_config
        super().__init__(get_clip_demo_config())
    
    def run(self):
        """Run CLIP-specific demo."""
        print("=== CLIP Model Demo ===")
        
        # Check if CLIP model exists
        if not os.path.exists(self.config['pytorch_model']):
            print(f"Error: CLIP model not found: {self.config['pytorch_model']}")
            print("Available CLIP models:")
            for model_dir in Path("models").iterdir():
                if 'clip' in model_dir.name.lower() and model_dir.is_dir():
                    print(f"  {model_dir}")
            return False
        
        # Run parent demo
        return super().run()
