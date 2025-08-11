import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import *

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, features=[512, 256, 128, 64]):
        super().__init__()
        self.layers = nn.ModuleList()
        prev = in_channels
        for feat in features:
            self.layers.append(DoubleConv(prev, feat))
            prev = feat
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in features])
    def forward(self, x):
        skips = []
        for conv, pool in zip(self.layers, self.pools):
            x = conv(x)
            skips.append(x)
            x = pool(x)
        return x, skips

class UNetDecoder(nn.Module):
    def __init__(self, features=[64, 128, 256, 512], out_channels=1):
        super().__init__()
        rev_feats = features[::-1]
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # First up block: input has 2x last feat channels (from bottleneck)
        self.ups.append(nn.ConvTranspose2d(rev_feats[0]*2, rev_feats[0], 2, stride=2))
        self.decoders.append(DoubleConv(rev_feats[0]*2, rev_feats[0]))
        # Subsequent blocks
        for idx in range(1, len(rev_feats)):
            self.ups.append(nn.ConvTranspose2d(rev_feats[idx-1], rev_feats[idx], 2, stride=2))
            self.decoders.append(DoubleConv(rev_feats[idx]*2, rev_feats[idx]))
        self.final_conv = nn.Conv2d(rev_feats[-1], out_channels, 1)
    def forward(self, x, skips):
        for idx in range(len(self.ups)):
            x = self.ups[idx](x)
            skip = skips[-(idx+1)]
            # Pad if needed to handle rounding issues
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = self.decoders[idx](x)
        return self.final_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]):
        super().__init__()
        self.encoder = UNetEncoder(in_channels, features)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.decoder = UNetDecoder(features, out_channels)
    def forward(self, x):
        x_in = x
        enc_out, skips = self.encoder(x_in)
        bottleneck = self.bottleneck(enc_out)
        seg = self.decoder(bottleneck, skips)  # (B, out_channels, H, W)
        seg = torch.squeeze(seg, dim=1)
        return seg

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    keypoint_net = UNet(in_channels=3, out_channels=4)
    keypoint_net = keypoint_net.to(device)

    x = torch.randn(2, 3, 128, 128).to(device)
    out = keypoint_net(x)
    print(out.shape)
    out = soft_argmax(out)
    print("Final output shape:", out.shape)