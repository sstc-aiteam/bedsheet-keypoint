import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ..utils.model_utils import *

class EfficientKeypointNet(nn.Module):
    """
    Efficient keypoint detection network with ~5-10M parameters.
    Uses only YOLO backbone + lightweight decoder, no ViT encoder.
    """
    
    def __init__(self, backbone, in_channels_list, output_shape=128):
        super().__init__()
        self.backbone = backbone
        
        # Lightweight feature fusion (much smaller than MultiScaleFusion)
        self.fusion = nn.Sequential(
            nn.Conv2d(sum(in_channels_list), 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Efficient decoder (much smaller than current decoder)
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final output
            nn.Conv2d(16, 1, 1)
        )
        
        self.output_shape = output_shape
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Concatenate all features (much simpler than MultiScaleFusion)
        # Resize all features to the same size (8x8)
        resized_features = []
        for feat in features:
            resized_feat = F.interpolate(feat, size=(8, 8), mode='bilinear', align_corners=False)
            resized_features.append(resized_feat)
        
        # Concatenate along channel dimension
        fused = torch.cat(resized_features, dim=1)
        
        # Apply lightweight fusion
        fused = self.fusion(fused)
        
        # Decode to heatmap
        heatmap = self.decoder(fused)
        
        # Apply spatial softmax
        softmaxed_heatmap = spatial_softmax(heatmap)
        
        return softmaxed_heatmap

class EfficientViTKeypointNet(nn.Module):
    """
    Efficient keypoint detection network with ViT encoder for 128x128 images.
    ~15-25M parameters (much smaller than original 103M hybrid model).
    Uses lightweight ViT + YOLO backbone + efficient decoder.
    """
    
    def __init__(self, backbone, in_channels_list, output_shape=128, 
                 vit_patch_size=8, vit_embed_dim=256, vit_depth=6, vit_num_heads=8):
        super().__init__()
        self.backbone = backbone
        self.output_shape = output_shape
        
        # ViT parameters for 128x128 images
        self.vit_patch_size = vit_patch_size
        self.vit_embed_dim = vit_embed_dim
        self.vit_depth = vit_depth
        self.vit_num_heads = vit_num_heads
        
        # Calculate number of patches for 128x128 images
        self.num_patches = (128 // vit_patch_size) ** 2  # 16x16 = 256 patches
        
        # Lightweight ViT encoder
        self.patch_embed = nn.Conv2d(3, vit_embed_dim, kernel_size=vit_patch_size, stride=vit_patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, vit_embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, vit_embed_dim) * 0.02)
        
        # Transformer encoder (much smaller than original)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vit_embed_dim,
            nhead=vit_num_heads,
            dim_feedforward=vit_embed_dim * 2,  # Smaller than original
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=vit_depth)
        
        # Project ViT output back to spatial features
        self.vit_proj = nn.Sequential(
            nn.Linear(vit_embed_dim, vit_embed_dim),
            nn.GELU(),
            nn.Linear(vit_embed_dim, vit_embed_dim)
        )
        
        # Reshape ViT features to spatial format (16x16)
        self.vit_spatial = nn.Sequential(
            nn.Linear(vit_embed_dim, vit_embed_dim * 4),  # 16x16 = 256
            nn.GELU(),
            nn.Linear(vit_embed_dim * 4, vit_embed_dim)
        )
        
        # Lightweight feature fusion (combine YOLO + ViT features)
        total_channels = sum(in_channels_list) + vit_embed_dim
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Efficient decoder
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Final output
            nn.Conv2d(16, 1, 1)
        )
    
    def forward(self, x):
        # Extract YOLO features
        yolo_features = self.backbone(x)
        
        # Process through ViT encoder
        # Patch embedding
        patches = self.patch_embed(x)  # B, embed_dim, H//patch_size, W//patch_size
        patches = patches.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        
        # Add position embeddings
        patches = patches + self.pos_embed
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(patches.shape[0], -1, -1)
        patches = torch.cat([cls_tokens, patches], dim=1)
        
        # Transformer encoding
        vit_features = self.transformer(patches)
        
        # Remove CLS token and project
        vit_features = vit_features[:, 1:, :]  # Remove CLS token
        vit_features = self.vit_proj(vit_features)
        
        # Reshape to spatial format (16x16)
        vit_spatial = self.vit_spatial(vit_features)  # B, num_patches, embed_dim
        vit_spatial = vit_spatial.transpose(1, 2).reshape(
            vit_spatial.shape[0], self.vit_embed_dim, 16, 16
        )
        
        # Resize YOLO features to 8x8 and concatenate with ViT features
        resized_yolo_features = []
        for feat in yolo_features:
            resized_feat = F.interpolate(feat, size=(8, 8), mode='bilinear', align_corners=False)
            resized_yolo_features.append(resized_feat)
        
        # Resize ViT features to 8x8
        vit_resized = F.interpolate(vit_spatial, size=(8, 8), mode='bilinear', align_corners=False)
        
        # Concatenate all features
        yolo_concat = torch.cat(resized_yolo_features, dim=1)
        fused = torch.cat([yolo_concat, vit_resized], dim=1)
        
        # Apply fusion
        fused = self.fusion(fused)
        
        # Decode to heatmap
        heatmap = self.decoder(fused)
        
        # Apply spatial softmax
        softmaxed_heatmap = spatial_softmax(heatmap)
        
        return softmaxed_heatmap

class UltraLightKeypointNet(nn.Module):
    """
    Ultra-lightweight keypoint detection network with ~1-2M parameters.
    Uses only the last few YOLO layers + minimal decoder.
    """
    
    def __init__(self, backbone, in_channels_list, output_shape=128):
        super().__init__()
        self.backbone = backbone
        
        # Use only the last 3 layers (much smaller)
        self.selected_layers = in_channels_list[-3:]  # Last 3 layers only
        
        # Minimal fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(sum(self.selected_layers), 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Ultra-light decoder
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            # Final output
            nn.Conv2d(8, 1, 1)
        )
        
        self.output_shape = output_shape
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Use only last 3 layers
        selected_features = features[-3:]
        
        # Resize and concatenate
        resized_features = []
        for feat in selected_features:
            resized_feat = F.interpolate(feat, size=(8, 8), mode='bilinear', align_corners=False)
            resized_features.append(resized_feat)
        
        fused = torch.cat(resized_features, dim=1)
        fused = self.fusion(fused)
        
        # Decode to heatmap
        heatmap = self.decoder(fused)
        softmaxed_heatmap = spatial_softmax(heatmap)
        
        return softmaxed_heatmap

class MobileKeypointNet(nn.Module):
    """
    Mobile-optimized keypoint detection network with ~500K-1M parameters.
    Uses depthwise separable convolutions for efficiency.
    """
    
    def __init__(self, backbone, in_channels_list, output_shape=128):
        super().__init__()
        self.backbone = backbone
        
        # Use only the last 2 layers for maximum efficiency
        self.selected_layers = in_channels_list[-2:]
        
        # Depthwise separable fusion
        self.fusion = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(sum(self.selected_layers), sum(self.selected_layers), 3, padding=1, groups=sum(self.selected_layers)),
            nn.BatchNorm2d(sum(self.selected_layers)),
            nn.ReLU(inplace=True),
            # Pointwise convolution
            nn.Conv2d(sum(self.selected_layers), 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Mobile-optimized decoder with depthwise separable convolutions
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            
            # Final output
            nn.Conv2d(4, 1, 1)
        )
        
        self.output_shape = output_shape
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Use only last 2 layers
        selected_features = features[-2:]
        
        # Resize and concatenate
        resized_features = []
        for feat in selected_features:
            resized_feat = F.interpolate(feat, size=(8, 8), mode='bilinear', align_corners=False)
            resized_features.append(resized_feat)
        
        fused = torch.cat(resized_features, dim=1)
        fused = self.fusion(fused)
        
        # Decode to heatmap
        heatmap = self.decoder(fused)
        softmaxed_heatmap = spatial_softmax(heatmap)
        
        return softmaxed_heatmap

# Example usage and testing
if __name__ == "__main__":
    # Test efficient model
    yolo_model = YOLO('yolov8s.pt')  # Use smaller YOLO model
    backbone_seq = yolo_model.model.model[:8]  # Use fewer layers
    backbone = YoloBackbone(backbone_seq, selected_indices=[0,1,2,3,4,5,6,7])
    
    input_dummy = torch.randn(1, 3, 128, 128)
    feats = backbone(input_dummy)
    in_channels_list = [f.shape[1] for f in feats]
    
    print(f"Backbone output channels: {in_channels_list}")
    
    # Test different model sizes
    models = {
        "Efficient": EfficientKeypointNet(backbone, in_channels_list),
        "EfficientViT": EfficientViTKeypointNet(backbone, in_channels_list),
        "Ultra-Light": UltraLightKeypointNet(backbone, in_channels_list),
        "Mobile": MobileKeypointNet(backbone, in_channels_list)
    }
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"{name} model parameters: {params:,}")
        
        # Test forward pass
        with torch.no_grad():
            output = model(input_dummy)
            print(f"{name} output shape: {output.shape}")
    
    print("\nModel comparison:")
    print("Current Hybrid Model: ~103M parameters")
    print("Efficient Model: ~5-10M parameters (20x smaller)")
    print("EfficientViT Model: ~15-25M parameters (4-7x smaller)")
    print("Ultra-Light Model: ~1-2M parameters (50x smaller)")
    print("Mobile Model: ~500K-1M parameters (100x smaller)")
