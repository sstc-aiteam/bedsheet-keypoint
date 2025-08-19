import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from ultralytics import YOLO
from .utils import *

class FlashAttentionBlock(nn.Module):
    def __init__(self, in_channels, embed_dim, num_heads, dropout_p=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout_p

        self.q_proj = nn.Conv2d(in_channels, embed_dim, 1)
        self.k_proj = nn.Conv2d(in_channels, embed_dim, 1)
        self.v_proj = nn.Conv2d(in_channels, embed_dim, 1)
        self.out_proj = nn.Conv2d(embed_dim, in_channels, 1)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W
        out_list = []

        for i in range(B):
            # Single example (C, H, W)
            xi = x[i].unsqueeze(0)  # Shape: (1, C, H, W)

            # Projections: (1, embed_dim, H, W)
            q = self.q_proj(xi)
            k = self.k_proj(xi)
            v = self.v_proj(xi)

            # Flatten spatial, move channels last: (1, H*W, embed_dim)
            q = q.permute(0, 2, 3, 1).reshape(1, N, self.embed_dim)
            k = k.permute(0, 2, 3, 1).reshape(1, N, self.embed_dim)
            v = v.permute(0, 2, 3, 1).reshape(1, N, self.embed_dim)

            # Split heads: (1, N, num_heads, head_dim)
            q = q.view(1, N, self.num_heads, self.head_dim)
            k = k.view(1, N, self.num_heads, self.head_dim)
            v = v.view(1, N, self.num_heads, self.head_dim)

            # Pack Q, K, V: (1, N, 3, num_heads, head_dim)
            qkv = torch.stack([q, k, v], dim=2)

            # Attention: attn_out (1, N, num_heads, head_dim)
            attn_out = flash_attn_qkvpacked_func(
                qkv, dropout_p=self.dropout_p, causal=False
            )

            # Merge heads: (1, N, embed_dim)
            attn_out = attn_out.contiguous().view(1, N, self.embed_dim)
            # Restore spatial: (1, H, W, embed_dim) -> (1, embed_dim, H, W)
            attn_out = attn_out.view(1, H, W, self.embed_dim).permute(0, 3, 1, 2)
            out = self.out_proj(attn_out)   # (1, in_channels, H, W)
            out_list.append(out)

        # Stack all outputs back into batch: (B, in_channels, H, W)
        out_final = torch.cat(out_list, dim=0)
        return out_final

class EnhancedYoloKeypointNet(nn.Module):
    def __init__(self, backbone, in_channels_list, num_keypoints=4, output_shape=(128,128)):
        super().__init__()
        self.backbone = backbone
        self.fusion = MultiScaleFusion(in_channels_list, 256)
        self.flashattn = FlashAttentionBlock(in_channels=256, embed_dim=256, num_heads=8)

        self.head = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, 3, padding=1)
        )
        self.output_shape = output_shape

    def forward(self, x):
        features = self.backbone(x)
        fused_features = self.fusion(features)
        attn_features = self.flashattn(fused_features)
        y = self.head(attn_features)  # (B, 4, H, W)
        if y.shape[-2:] != self.output_shape:
            y = F.interpolate(y, size=self.output_shape, mode='bilinear', align_corners=False)
        return y  # (B, 4, H, W)

# if __name__ == "__main__":
#     # Load YOLOv8l and slice the first 10 layers
#     yolo11 = YOLO('yolo11l-seg.pt')  # Or yolo11m-seg.pt, yolo11x-seg.pt, etc.
#     backbone_seq = yolo11.model.model[:12]
#     # Initialize the backbone with selected indices for multi-scale features
#     backbone = YoloBackbone(backbone_seq, selected_indices=[0,1,2,3,4,5,6,7,8,9,10,11])
#     input_dummy = torch.randn(1, 3, 128, 128)
#     with torch.no_grad():
#         feats = backbone(input_dummy)
#     print("Feature shapes:", [f.shape for f in feats])
#     in_channels_list = [f.shape[1] for f in feats]

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     keypoint_net = EnhancedYoloKeypointNet(backbone, in_channels_list)
#     keypoint_net = keypoint_net.to(device).half()

#     x = torch.randn(2, 3, 128, 128).to(device).half()
#     out = keypoint_net(x)
#     print(out.shape)
#     out = soft_argmax(out)
#     print("Final output shape:", out.shape)
