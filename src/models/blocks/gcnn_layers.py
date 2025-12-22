from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gcnn_groups import CyclicGroupC4


def _rot90_kernel(weight: torch.Tensor, k: int) -> torch.Tensor:
    """
    Rotate a conv kernel by 90° increments.

    weight: (C_out, C_in, K, K)
    k: number of 90° CCW rotations
    """

    return torch.rot90(weight, k=k, dims=(-2, -1))


class LiftingConvC4(nn.Module):
    """
    Lifting convolution: R^2 -> G, for C4.

    Input:
      x: (B, C_in, H, W)
    Output:
      y: (B, C_out, |G|=4, H', W')

    Implementation follows the daily report's key trick:
    - generate transformed filters for each group element (apply h^{-1} / left regular representation)
    - fold group dimension into output channels so we can reuse PyTorch's optimized conv2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
        group: Optional[CyclicGroupC4] = None,
    ) -> None:
        super().__init__()
        self.group = group or CyclicGroupC4()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(kernel_size // 2) if padding is None else int(padding)

        w = torch.empty(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.weight = nn.Parameter(w)
        self.bias: Optional[nn.Parameter]
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build transformed kernels for each h in C4.
        # The report applies h^{-1}; with C4 indices, inverse(i) = -i mod 4.
        kernels = []
        for h in self.group.elements():
            h_inv = self.group.inverse(h)
            # rotate the *kernel* by h^{-1}; for C4 this is rot90 by k=h_inv
            w_h = _rot90_kernel(self.weight, k=h_inv)
            kernels.append(w_h)

        # Stack => (|G|, C_out, C_in, K, K) then fold => (C_out*|G|, C_in, K, K)
        k_stack = torch.stack(kernels, dim=0)
        k_folded = k_stack.reshape(-1, self.in_channels, self.kernel_size, self.kernel_size)

        # Apply conv; do NOT pass bias here (or it would be added |G| times).
        y = F.conv2d(x, k_folded, bias=None, stride=self.stride, padding=self.padding)

        # Reshape back to (B, C_out, |G|, H', W')
        b, _, h_out, w_out = y.shape
        y = y.view(b, len(self.group.elements()), self.out_channels, h_out, w_out).permute(0, 2, 1, 3, 4).contiguous()

        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)
        return y


class GroupConvC4(nn.Module):
    """
    Group convolution on G = R^2 ⋊ C4: G -> G.

    Input:
      x: (B, C_in, |G|=4, H, W)
    Output:
      y: (B, C_out, |G|=4, H', W')

    Discrete implementation for C4 (no trilinear sampling needed):
    For each output group element g (index i) and each input group element h (index j),
    we use the relative group element r = g^{-1}h = (j - i) mod 4 and rotate the kernel
    spatially by g^{-1} (i.e., -i).

    This matches the daily report's "twist-shift" idea and keeps the group index disentangled.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = True,
        group: Optional[CyclicGroupC4] = None,
    ) -> None:
        super().__init__()
        self.group = group or CyclicGroupC4()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(kernel_size // 2) if padding is None else int(padding)

        # weight: (C_out, C_in, |G|, K, K) where |G| indexes the relative group element r
        w = torch.empty(self.out_channels, self.in_channels, len(self.group.elements()), self.kernel_size, self.kernel_size)
        self.weight = nn.Parameter(w)
        self.bias: Optional[nn.Parameter]
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * len(self.group.elements()) * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"GroupConvC4 expects x with shape (B,C,|G|,H,W); got {tuple(x.shape)}")
        if x.shape[2] != len(self.group.elements()):
            raise ValueError(f"Expected group dim=4 for C4, got {x.shape[2]}")

        b, cin, gdim, h, w = x.shape
        x_folded = x.view(b, cin * gdim, h, w)

        # Build one folded kernel per output group element i, then run conv2d.
        ys = []
        for i in self.group.elements():
            i_inv = self.group.inverse(i)  # -i
            kernel_blocks = []
            for j in self.group.elements():
                # relative = g^{-1}h = (-i) + j = (j - i) mod 4
                rel = self.group.product(i_inv, j)
                w_rel = self.weight[:, :, rel, :, :]  # (C_out, C_in, K, K)
                w_rot = _rot90_kernel(w_rel, k=i_inv)
                kernel_blocks.append(w_rot)

            # Concatenate along input-channel axis: (C_out, C_in*|G|, K, K)
            k_i = torch.cat(kernel_blocks, dim=1)
            y_i = F.conv2d(x_folded, k_i, bias=None, stride=self.stride, padding=self.padding)
            ys.append(y_i)

        # Stack along group dim => (B, |G|, C_out, H', W') then permute
        y = torch.stack(ys, dim=1).permute(0, 2, 1, 3, 4).contiguous()
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)
        return y


class GroupPooling(nn.Module):
    """
    Project from group space to an invariant plane feature by pooling over group dimension.

    Input:  (B, C, |G|, H, W)
    Output: (B, C, H, W)

    This matches the daily report's projection step (avg over group + space).
    """

    def __init__(self, mode: str = "mean") -> None:
        super().__init__()
        mode = str(mode).lower()
        if mode not in ("mean", "max"):
            raise ValueError("GroupPooling mode must be 'mean' or 'max'")
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"GroupPooling expects x with shape (B,C,|G|,H,W); got {tuple(x.shape)}")
        if self.mode == "mean":
            return x.mean(dim=2)
        return x.max(dim=2).values


