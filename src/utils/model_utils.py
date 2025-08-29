import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Any, Tuple
import torchvision.transforms.functional as TF
import numpy as np
import cv2

class YoloBackbone(nn.Module):
    def __init__(self, backbone_seq, selected_indices=None):
        super().__init__()
        self.backbone = backbone_seq
        if selected_indices is None:
            # By default, collect all intermediate outputs
            self.selected_indices = list(range(len(backbone_seq)))
        else:
            self.selected_indices = selected_indices

    def forward(self, x):
        feats = []
        out = x
        for i, layer in enumerate(self.backbone):
            # Check if the layer is a Concat layer
            if layer.__class__.__name__.lower().startswith('concat'):
                # If not already a list/tuple, wrap it
                if not isinstance(out, (list, tuple)):
                    out = [out]
            out = layer(out)
            if i in self.selected_indices:
                # If out is a list (as in some YOLO features), add all, else add single
                if isinstance(out, (list, tuple)):
                    feats.extend(out)
                else:
                    feats.append(out)
        return feats


class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.reduce_convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, 1) for in_c in in_channels_list
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(len(in_channels_list) * out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        target_size = features[0].shape[-2:]
        upsampled = []
        for conv, feat in zip(self.reduce_convs, features):
            out = conv(feat)
            if out.shape[-2:] != target_size:
                out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
            upsampled.append(out)
        fused = torch.cat(upsampled, dim=1)
        return self.fuse(fused)

def soft_argmax(heatmap, beta=100):
    # heatmap: (B, K, H, W)
    *rest, H, W = heatmap.shape
    heatmap_flat = heatmap.reshape(-1, H*W)  # (B*K, H*W)
    y_soft = F.softmax(heatmap_flat * beta, dim=1)

    # Create meshgrid of coordinates
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=heatmap.device),
        torch.arange(W, device=heatmap.device),
        indexing='ij'
    )  # each shape (H, W)
    x_coords = x_coords.reshape(-1).float()  # (H*W,)
    y_coords = y_coords.reshape(-1).float()  # (H*W,)

    x = (y_soft * x_coords).sum(dim=1)
    y = (y_soft * y_coords).sum(dim=1)
    coords = torch.stack((x, y), dim=1)  # (B*K, 2)
    return coords.view(*rest, 2)

def spatial_softmax(heatmap):
    """
    Applies softmax over the spatial dimension of a 4D tensor (B, 1, H, W).
    Returns same shape, values sum to 1 per image.
    """
    B, C, H, W = heatmap.shape
    heatmap_flat = heatmap.view(B, -1)
    softmax_flat = F.softmax(heatmap_flat, dim=1)
    return softmax_flat.view(B, 1, H, W)
    # return heatmap


def batch_gaussian_blur(x, kernel_size=5, sigma=2.0):
    """
    Apply Gaussian blur to a batch of heatmaps and normalize each so the max is 1.
    Args:
        x: Tensor [B, H, W] or [B, 1, H, W]
    Returns:
        Tensor with same shape, blurred and with peak 1 per sample
    """
    unsqueeze = False
    if x.dim() == 3:  # [B, H, W]
        x = x.unsqueeze(1)
        unsqueeze = True
    
    blurred = TF.gaussian_blur(x, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    max_vals = blurred.amax(dim=[2, 3], keepdim=True)
    max_vals[max_vals == 0] = 1.0  # Avoid division by zero
    normalized = blurred / max_vals

    if unsqueeze:
        normalized = normalized.squeeze(1)
    return normalized

def batch_entropy(pred_heatmaps):
    """
    pred_heatmaps: [B, C, H, W]
    Returns: [B] entropy per image
    """
    # Flatten spatial dimensions (and optionally channels) for softmax
    B, C, H, W = pred_heatmaps.shape
    flat = pred_heatmaps.view(B, -1)                # [B, C*H*W]
    probs = torch.softmax(flat, dim=1)              # normalize to sum=1 per image
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # [B]
    return entropy

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    
    return mixed_x, mixed_y

# Loss function for keypoint detection
def kl_heatmap_loss(pred_hm, gt_hm, mask=None, reduction='mean'):
    """
    KL divergence loss for heatmap prediction.
    
    Args:
        pred_hm: Predicted heatmap (B, 1, H, W)
        gt_hm: Ground truth heatmap (B, 1, H, W)
        mask: Optional mask (B, 1, H, W) or None
        reduction: Reduction method ('mean', 'sum', or 'none')
    
    Returns:
        Loss value
    """
    eps = 1e-8

    # Force positive
    pred_probs = pred_hm.clamp(min=eps)
    gt_probs = gt_hm.clamp(min=eps)

    # Optionally apply mask
    if mask is not None:
        pred_probs = pred_probs * mask
        gt_probs = gt_probs * mask

    # Sum per sample
    pred_sum = pred_probs.sum(dim=(2, 3), keepdim=True)
    gt_sum = gt_probs.sum(dim=(2, 3), keepdim=True)

    # Identify gt_hm slices that are all zeros (or close enough)
    gt_zero_mask = (gt_sum < eps).squeeze(1).squeeze(1)  # (B,) boolean: True means skip or zero out

    # Safe normalization (avoids divide by zero)
    pred_probs = pred_probs / pred_sum.clamp(min=eps)
    gt_probs = torch.where(gt_sum < eps, torch.zeros_like(gt_probs), gt_probs / gt_sum.clamp(min=eps))

    # Compute KL divergence per sample
    log_pred = pred_probs.log()
    kl_div = F.kl_div(log_pred, gt_probs, reduction='none').sum(dim=(2, 3))  # shape (B,1)
    kl_div = kl_div.squeeze(1)  # (B,)

    # For samples where gt_hm is all zeros, set loss to 0 (no supervision there)
    kl_div = kl_div.masked_fill(gt_zero_mask, 0.)

    if reduction == 'mean':
        num = (~gt_zero_mask).float().sum().clamp(min=1)
        return kl_div.sum() / num
    elif reduction == 'sum':
        return kl_div.sum()
    else:
        return kl_div

def load_state_dict_safely(
    model: torch.nn.Module, 
    state_dict: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Safely load state dict with detailed error reporting.
    
    Returns:
        Tuple of (successfully_loaded_keys, failed_keys)
    """
    model_state_dict = model.state_dict()
    
    # Find compatible keys
    compatible_keys = {}
    incompatible_keys = {}
    
    for key, value in state_dict.items():
        if key in model_state_dict:
            if value.shape == model_state_dict[key].shape:
                compatible_keys[key] = value
            else:
                incompatible_keys[key] = f"Shape mismatch: saved {value.shape} vs current {model_state_dict[key].shape}"
        else:
            incompatible_keys[key] = "Key not found in current model"
    
    return compatible_keys, incompatible_keys

def load_quantized_model_functional(
    model_path: str,
    backbone,
    in_channels_list,
    device: torch.device
) -> torch.nn.Module:
    """
    Functional approach to loading quantized models.
    
    Args:
        model_path: Path to saved model
        backbone: Model backbone
        in_channels_list: List of input channels
        device: Target device
    
    Returns:
        Loaded quantized model
    """
    # Local import to avoid circular dependency
    from .quantization_utils import create_quantized_model_structure
    
    print(f"Loading quantized model from: {model_path}")
    
    # Load saved state dict
    saved_state_dict = torch.load(model_path, map_location='cpu')
    print(f"Saved model contains {len(saved_state_dict)} parameters")
    
    # Create model structure
    quantized_model = create_quantized_model_structure(backbone, in_channels_list)
    
    # Load state dict safely
    compatible_keys, incompatible_keys = load_state_dict_safely(quantized_model, saved_state_dict)
    
    # Report loading results
    print(f"Successfully loaded {len(compatible_keys)} parameters")
    if incompatible_keys:
        print(f"Failed to load {len(incompatible_keys)} parameters:")
        for key, reason in list(incompatible_keys.items())[:5]:  # Show first 5
            print(f"  {key}: {reason}")
    
    # Load compatible parameters
    quantized_model.load_state_dict(compatible_keys, strict=False)
    
    # Move to device and set to eval mode
    quantized_model = quantized_model.to(device)
    quantized_model.eval()
    
    return quantized_model

def thresholded_locations(heatmap, threshold=0.1):
    """
    Find locations in heatmap above threshold.
    
    Args:
        heatmap: (H, W) tensor or numpy array
        threshold: Threshold value
    
    Returns:
        List of (y, x) coordinates
    """
    # Convert numpy array to tensor if needed
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)
    
    # Find peaks above threshold
    peaks = torch.nonzero(heatmap > threshold)
    if len(peaks) == 0:
        return []
    
    # Convert to (y, x) format
    locations = [(int(peak[0]), int(peak[1])) for peak in peaks]
    return locations

def extract_mask_compare(img, yolo_model, allowed_classes):
    """
    Extract improved mask using finetuned YOLO model.
    
    Args:
        img: Input image
        yolo_model: Finetuned YOLO model
        allowed_classes: List of allowed class IDs
    
    Returns:
        Binary mask with improved precision
    """
    results = yolo_model(img)
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    
    for result in results:
        if result.masks is not None:
            # Use segmentation masks if available
            for i, box in enumerate(result.boxes):
                if int(box.cls) in allowed_classes:
                    # Get segmentation mask
                    seg_mask = result.masks.data[i].cpu().numpy()
                    # Resize mask to image size if needed
                    if seg_mask.shape != mask.shape:
                        seg_mask = cv2.resize(seg_mask, (mask.shape[1], mask.shape[0]))
                    # Apply threshold to get binary mask
                    seg_mask = (seg_mask > 0.5).astype(np.uint8) * 255
                    # Combine with existing mask
                    mask = np.maximum(mask, seg_mask)
        elif result.boxes is not None:
            # Create improved bounding box mask with confidence-based refinement
            for box in result.boxes:
                if int(box.cls) in allowed_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy()
                    
                    # Create basic bounding box mask
                    box_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                    box_mask[y1:y2, x1:x2] = 255
                    
                    # Apply confidence-based refinement
                    if conf > 0.7:  # High confidence - use full box
                        mask = np.maximum(mask, box_mask)
                    elif conf > 0.5:  # Medium confidence - use smaller box
                        # Shrink the box by 10%
                        w, h = x2 - x1, y2 - y1
                        shrink_w, shrink_h = int(w * 0.1), int(h * 0.1)
                        refined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                        refined_mask[y1+shrink_h:y2-shrink_h, x1+shrink_w:x2-shrink_w] = 255
                        mask = np.maximum(mask, refined_mask)
                    else:  # Low confidence - use even smaller box
                        # Shrink the box by 20%
                        w, h = x2 - x1, y2 - y1
                        shrink_w, shrink_h = int(w * 0.2), int(h * 0.2)
                        refined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                        refined_mask[y1+shrink_h:y2-shrink_h, x1+shrink_w:x2-shrink_w] = 255
                        mask = np.maximum(mask, refined_mask)
    
    return mask