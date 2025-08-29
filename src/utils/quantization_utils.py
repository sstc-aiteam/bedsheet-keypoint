# Functional approach to quantization setup
import torch
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
from typing import Dict, Any, Tuple, Optional
import functools
from .model_utils import batch_gaussian_blur, mixup_data
from .model_utils import kl_heatmap_loss

def create_quantization_stubs() -> Tuple[QuantStub, DeQuantStub]:
    """Create quantization and dequantization stubs."""
    return QuantStub(), DeQuantStub()

def create_quantized_model_structure(backbone, in_channels_list) -> torch.nn.Module:
    """Create quantized model structure using functional approach."""
    # Local import to avoid circular dependency
    from ..models.hybrid_keypoint_net import HybridKeypointNet
    
    quant_stub, dequant_stub = create_quantization_stubs()
    
    # Create base model
    base_model = HybridKeypointNet(backbone, in_channels_list)
    
    # Add quantization stubs functionally
    def quantized_forward(x: torch.Tensor) -> torch.Tensor:
        """Quantized forward pass."""
        x = quant_stub(x)
        # Use the full forward pass of the base model
        output = base_model(x)
        return dequant_stub(output)
    
    # Create a functional model wrapper
    class FunctionalQuantizedModel(torch.nn.Module):
        def __init__(self, base_model, quant_stub, dequant_stub):
            super().__init__()
            self.base_model = base_model
            self.quant = quant_stub
            self.dequant = dequant_stub
        
        def forward(self, x):
            return quantized_forward(x)
        
        # Expose backbone and decoder for compatibility
        @property
        def backbone(self):
            return self.base_model.backbone
        
        @property
        def decoder(self):
            return self.base_model.decoder
        
        @property
        def keypoint_head(self):
            # For backward compatibility, return decoder
            return self.base_model.decoder
        
        def to(self, device):
            """Move model to device."""
            self.base_model = self.base_model.to(device)
            self.quant = self.quant.to(device)
            self.dequant = self.dequant.to(device)
            
            # Update the closure to use the moved model
            def updated_quantized_forward(x: torch.Tensor) -> torch.Tensor:
                """Updated quantized forward pass with correct device."""
                x = self.quant(x)
                # Use the full forward pass of the base model
                output = self.base_model(x)
                return self.dequant(output)
            
            # Replace the forward function
            self.forward = lambda x: updated_quantized_forward(x)
            return self
    
    return FunctionalQuantizedModel(base_model, quant_stub, dequant_stub)

def prepare_model_for_qat(model: torch.nn.Module) -> torch.nn.Module:
    """Prepare model for quantization-aware training."""
    return prepare_qat(model, mapping=None, inplace=False)

def convert_to_quantized(model: torch.nn.Module) -> torch.nn.Module:
    """Convert model to final quantized format."""
    return convert(model, inplace=False)

def export_model_to_onnx(
    model: torch.nn.Module,
    model_name: str,
    output_dir: str,
    input_shape: Tuple[int, ...] = (1, 3, 128, 128)
) -> Optional[str]:
    """
    Export model to ONNX format using functional approach.
    
    Args:
        model: Model to export
        model_name: Name for the exported model
        output_dir: Output directory
        input_shape: Input tensor shape
    
    Returns:
        Path to exported ONNX file or None if failed
    """
    try:
        # Get device from model
        device = next(model.parameters()).device
        
        # Create dummy input on the same device as model
        dummy_input = torch.randn(*input_shape, device=device)
        
        # Export to ONNX
        onnx_path = f"{output_dir}/{model_name}.onnx"
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=16,  # Updated to support scaled_dot_product_attention
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"ONNX model exported to: {onnx_path}")
        return onnx_path
        
    except Exception as e:
        print(f"ONNX export failed: {e}")
        return None

def convert_onnx_to_gguf(onnx_path: str, model_name: str, output_dir: str) -> Optional[str]:
    """
    Convert ONNX model to GGUF format.
    
    Args:
        onnx_path: Path to ONNX model
        model_name: Name for the GGUF model
        output_dir: Output directory
    
    Returns:
        Path to GGUF file or None if failed
    """
    try:
        import subprocess
        
        gguf_path = f"{output_dir}/{model_name}.gguf"
        
        # Try llama.cpp conversion
        cmd = [
            "python3", "-m", "llama_cpp.convert_llama_weights_to_gguf",
            "--input", onnx_path,
            "--output", gguf_path,
            "--model-name", model_name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"GGUF model saved to: {gguf_path}")
            return gguf_path
        else:
            print(f"llama.cpp conversion failed: {result.stderr}")
            return None
            
    except ImportError:
        print("llama.cpp not available")
        return None
    except Exception as e:
        print(f"GGUF conversion failed: {e}")
        return None

def export_model_pipeline(
    model: torch.nn.Module,
    model_name: str,
    output_dir: str,
    formats: Tuple[str, ...] = ("onnx", "gguf")
) -> Dict[str, Optional[str]]:
    """
    Export model to multiple formats using functional pipeline.
    
    Args:
        model: Model to export
        model_name: Base name for exported models
        output_dir: Output directory
        formats: Tuple of formats to export to
    
    Returns:
        Dictionary mapping format names to file paths
    """
    results = {}
    
    # Export to ONNX first (required for GGUF)
    if "onnx" in formats or "gguf" in formats:
        onnx_path = export_model_to_onnx(model, model_name, output_dir)
        results["onnx"] = onnx_path
        
        # Convert ONNX to GGUF if requested
        if "gguf" in formats and onnx_path:
            gguf_path = convert_onnx_to_gguf(onnx_path, model_name, output_dir)
            results["gguf"] = gguf_path
    
    return results

def create_quantized_training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: callable
) -> callable:
    """
    Create a functional training step for quantized models.
    
    Args:
        model: Quantized model
        optimizer: Optimizer
        loss_fn: Loss function
    
    Returns:
        Training step function
    """
    def training_step(images: torch.Tensor, keypoints: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step.
        
        Args:
            images: Input images
            keypoints: Target keypoints
        
        Returns:
            Tuple of (loss, metrics_dict)
        """
        optimizer.zero_grad()

        # Mixup data
        images, keypoints = mixup_data(images, keypoints)
        
        # Forward pass
        outputs = model(images)
        keypoints_blur = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
        loss = loss_fn(outputs, keypoints_blur.unsqueeze(1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Return loss and metrics
        metrics = {
            "loss": loss.item(),
            "learning_rate": optimizer.param_groups[0]["lr"]
        }
        
        return loss, metrics
    
    return training_step

def train_quantized_model_functional(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: torch.device
) -> Dict[str, list]:
    """
    Functional training loop for quantized models.
    
    Args:
        model: Quantized model
        train_loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Target device
    
    Returns:
        Dictionary containing training history
    """
    model = model.to(device)
    training_step = create_quantized_training_step(model, optimizer, kl_heatmap_loss)
    
    history = {
        "train_loss": [],
        "learning_rate": []
    }
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            images = batch["image"].to(device)
            keypoints = batch["keypoints"].to(device)
            keypoints_blur = batch_gaussian_blur(keypoints, kernel_size=31, sigma=3)
            loss, metrics = training_step(images, keypoints_blur.unsqueeze(1))
            epoch_losses.append(metrics["loss"])
        
        # Update learning rate
        scheduler.step()
        
        # Record metrics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        current_lr = scheduler.get_last_lr()[0]
        
        history["train_loss"].append(avg_loss)
        history["learning_rate"].append(current_lr)
        
        print(f'Epoch {epoch+1}: Loss {avg_loss:.4f}, LR: {current_lr:.2e}')
    
    return history