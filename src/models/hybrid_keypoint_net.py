import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from timm import create_model
from ..utils.model_utils import *

class PatchViTEncoder(nn.Module):
    def __init__(self, in_channels, fuse_channels, img_size=128, patch_size=16, vit_type='vit_base_patch16_224'):
        super().__init__()
        self.pre_conv = nn.Conv2d(fuse_channels, 3, kernel_size=1)
        self.vit = create_model(vit_type, pretrained=True, img_size=(img_size, img_size))
        self.vit.head = nn.Identity()
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x):
        x = self.pre_conv(x)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        vit_tokens = self.vit.forward_features(x)  # (B, N, D)
        patch_tokens = vit_tokens[:, 1:, :]         # Drop CLS token
        # Restore to spatial grid: (B, H_p, W_p, D) -> (B, D, H_p, W_p)
        H_p = W_p = self.img_size // self.patch_size
        patch_tokens = patch_tokens.view(-1, H_p, W_p, patch_tokens.shape[-1]).permute(0, 3, 1, 2)
        return patch_tokens

class SingleHeatmapDecoder(nn.Module):
    def __init__(self, token_dim, input_grid=8, output_shape=128):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(token_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(16, 1, 1)
        self.output_shape = output_shape

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = F.interpolate(x, (self.output_shape, self.output_shape))
        x = self.final(x)
        return x # (B, 1, H, W)


class HybridKeypointNet(nn.Module):
    def __init__(self, backbone, in_channels_list, num_keypoints=4, vit_img_size=128, vit_patch_size=16, output_shape=128):
        super().__init__()
        self.backbone = backbone
        self.fusion = MultiScaleFusion(in_channels_list, 128)  # Provide your MultiScaleFusion
        self.encoder = PatchViTEncoder(in_channels_list, 128, img_size=vit_img_size, patch_size=vit_patch_size)
        self.input_grid = vit_img_size // vit_patch_size
        token_dim = 768  # For ViT-B
        self.decoder = SingleHeatmapDecoder(token_dim, self.input_grid, output_shape)
        self.output_shape = output_shape

    def forward(self, x):
        features = self.backbone(x)  # list of features
        fused_features = self.fusion(features)  # for ViT
        encoded = self.encoder(fused_features)
        heatmap = self.decoder(encoded)  # (B, 1, H, W)
        softmaxed_heatmap = spatial_softmax(heatmap)
        return softmaxed_heatmap



# Example usage (commented out)
if __name__ == "__main__":
    yolo_model = YOLO('yolov8l.pt')
    backbone_seq = yolo_model.model.model[:10]
    backbone = YoloBackbone(backbone_seq, selected_indices=[0,1,2,3,4,5,6,7,8,9])
    input_dummy = torch.randn(1, 3, 128, 128)
    feats = backbone(input_dummy)
    in_channels_list = [f.shape[1] for f in feats]
    net = HybridKeypointNet(backbone, in_channels_list)
    x = torch.randn(2, 3, 128, 128)
    out = net(x)
    print('Heatmaps shape:', out.shape)
    kp = soft_argmax(out)
    print("keypoint shape:", kp.shape)
