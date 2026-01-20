from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Optional, Tuple


def _manual_affine_grid(theta: torch.Tensor, size: Tuple[int, int, int, int], align_corners: bool) -> torch.Tensor:
    """
    Manual implementation of affine_grid to avoid ONNX/TensorRT issues with the AffineGrid operator.
    """
    N, C, H, W = size
    device = theta.device
    dtype = theta.dtype

    if align_corners:
        y_grid = torch.linspace(-1, 1, steps=H, device=device, dtype=dtype)
        x_grid = torch.linspace(-1, 1, steps=W, device=device, dtype=dtype)
    else:
        # Standard unaligned grid: centers of pixels
        y_grid = torch.linspace(-1 + 1/H, 1 - 1/H, steps=H, device=device, dtype=dtype)
        x_grid = torch.linspace(-1 + 1/W, 1 - 1/W, steps=W, device=device, dtype=dtype)
    
    # Meshgrid (H, W) - indexing='ij' gives y (vertical) varies first, x (horizontal) second
    grid_y, grid_x = torch.meshgrid(y_grid, x_grid, indexing='ij')
    
    # Stack to (H, W, 3) -> (x, y, 1)
    ones = torch.ones_like(grid_x)
    base_grid = torch.stack([grid_x, grid_y, ones], dim=-1)
    
    # Flatten: (H*W, 3)
    grid_flat = base_grid.reshape(-1, 3)
    
    # Transpose for matrix multiplication: (3, H*W)
    grid_flat_t = grid_flat.permute(1, 0)
    
    # theta is (N, 2, 3)
    # result: (N, 2, HW) = theta @ grid_flat_t_expanded
    grid_trans = torch.matmul(theta, grid_flat_t.unsqueeze(0))
    
    # Reshape to (N, H, W, 2)
    # (N, 2, HW) -> (N, HW, 2) -> (N, H, W, 2)
    grid_trans = grid_trans.permute(0, 2, 1).reshape(N, H, W, 2)
    
    return grid_trans


def _rotate_kernels_grid_sample(
    weight: torch.Tensor,  # (C_out, C_in, K, K)
    theta: torch.Tensor,  # (A,) in radians
    *,
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Rotate conv kernels by arbitrary angles using grid_sample.

    Returns:
      rotated: (A, C_out, C_in, K, K)

    Notes:
    - This is the core "transformed grid + interpolation sampling" idea from the docs.
    - We rotate kernels by applying the inverse rotation to the sampling grid.
    """

    if theta.ndim != 1:
        raise ValueError(f"theta must be 1D (A,), got {tuple(theta.shape)}")

    cout, cin, kH, kW = weight.shape
    if kH != kW:
        raise ValueError("Only square kernels are supported for rotation.")

    A = theta.numel()
    device = weight.device
    dtype = weight.dtype

    # Flatten kernel bank into a batch so we can grid_sample in one call.
    w = weight.view(cout * cin, 1, kH, kW)  # (N,1,K,K)

    # NOTE: We build the sampling grid in fp32 for numerical stability and wider backend support.
    # Some CUDA/cuDNN combinations are picky about fp16 + grid_sample.
    cos_t = torch.cos(theta).to(device=device, dtype=torch.float32)
    sin_t = torch.sin(theta).to(device=device, dtype=torch.float32)

    # affine_grid expects transform mapping output grid -> input sampling points.
    # For rotating the *kernel* by theta, we sample w at R^{-theta} coordinates.
    # That corresponds to using rotation matrix for -theta.
    zeros = torch.zeros_like(cos_t)
    ones = torch.ones_like(cos_t)

    # (A,2,3)
    affine = torch.stack(
        [
            torch.stack([cos_t, sin_t, zeros], dim=-1),
            torch.stack([-sin_t, cos_t, zeros], dim=-1),
        ],
        dim=1,
    )

    # Create sampling grid: (A, K, K, 2)
    # Create sampling grid: (A, K, K, 2)
    # Create sampling grid: (A, K, K, 2)
    # F.affine_grid can cause issues with TensorRT ONNX parser (UNSUPPORTED_NODE), so we use manual implementation
    grid = _manual_affine_grid(affine, size=(A, 1, kH, kW), align_corners=align_corners)

    # Expand kernel batch across angles: (A*N,1,K,K)
    # `expand()` can produce non-contiguous views; grid_sample on CUDA may fail with such tensors.
    w_rep = (
        w.unsqueeze(0)
        .expand(A, -1, -1, -1, -1)
        .reshape(A * cout * cin, 1, kH, kW)
        .contiguous()
    )
    grid_rep = (
        grid.unsqueeze(1)
        .expand(A, cout * cin, kH, kW, 2)
        .reshape(A * cout * cin, kH, kW, 2)
        .contiguous()
    )

    # Run grid_sample in fp32 then cast back to the original dtype.
    #
    # IMPORTANT: some CUDA/cuDNN combinations throw:
    #   CUDNN_STATUS_NOT_SUPPORTED (often blamed on non-contiguous inputs)
    # even when tensors look fine, especially under AMP/autocast.
    # We add a robust fallback: re-run with cuDNN disabled just for this call.
    w_in = w_rep.to(dtype=torch.float32).contiguous()
    grid_in = grid_rep.to(dtype=torch.float32).contiguous()
    try:
        w_rot = F.grid_sample(
            w_in,
            grid_in,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=align_corners,
        )  # (A*N,1,K,K)
    except RuntimeError as e:
        if "CUDNN_STATUS_NOT_SUPPORTED" not in str(e):
            raise
        # Fallback: disable cuDNN for grid_sample
        with torch.backends.cudnn.flags(enabled=False):
            w_rot = F.grid_sample(
                w_in,
                grid_in,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=align_corners,
            )

    w_rot = w_rot.view(A, cout, cin, kH, kW).to(dtype=dtype)
    return w_rot


class LiftingConvSO2(nn.Module):
    """
    Lifting convolution: R^2 -> (R^2 x H) where H is a sampled SO(2) angle set.

    Input:
      x: (B, C_in, H, W)
    Output:
      y: (B, C_out, A, H', W')   where A = num_angles

    Implementation:
    - choose A angles in [0, 2pi)
    - for each angle theta, rotate kernel by theta^{-1} (equivalently -theta)
    - fold angle dimension into output channels and run a single conv2d

    This gives an *approximation* to continuous SO(2) lifting. Increasing A improves fidelity
    but increases compute linearly.
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
        num_angles: int = 16,
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(kernel_size // 2) if padding is None else int(padding)
        self.num_angles = int(num_angles)
        if self.num_angles <= 0:
            raise ValueError("num_angles must be > 0")
        self.align_corners = bool(align_corners)

        w = torch.empty(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.weight = nn.Parameter(w)
        self.bias: Optional[nn.Parameter]
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.bias = None

        self.reset_parameters()

        # Fixed angle grid (buffer). You can later make this learnable or sample stochastically.
        theta = torch.linspace(0.0, 2.0 * math.pi, steps=self.num_angles + 1)[:-1]
        self.register_buffer("theta", theta, persistent=False)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rotate kernels by inverse angles (-theta) for left-regular lifting L_theta f(x)=f(R_-theta x)
        w_rot = _rotate_kernels_grid_sample(
            self.weight,
            theta=(-self.theta).to(device=x.device),
            align_corners=self.align_corners,
        )  # (A, C_out, C_in, K, K)

        # Fold angles into output channels: (C_out*A, C_in, K, K)
        A = w_rot.shape[0]
        k_folded = w_rot.permute(1, 0, 2, 3, 4).contiguous().view(self.out_channels * A, self.in_channels, self.kernel_size, self.kernel_size)

        y = F.conv2d(x, k_folded, bias=None, stride=self.stride, padding=self.padding)
        b, _, h_out, w_out = y.shape
        y = y.view(b, self.out_channels, A, h_out, w_out)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)
        return y


class GroupConvSO2(nn.Module):
    """
    Group convolution on G = R^2 ⋊ SO(2) (approximated by A sampled angles): G -> G.

    We approximate SO(2) with a cyclic group C_A where angles are uniformly sampled.

    Input:
      x: (B, C_in, A, H, W)
    Output:
      y: (B, C_out, A, H', W')

    Implementation mirrors the discrete C4 group conv in spirit:
    - Relative group index r = g^{-1} h becomes (j - i) mod A over sampled angles
    - Spatial action uses rotation by g^{-1} (i.e., -theta_i) implemented via kernel rotation with grid_sample

    Complexity:
    - This implementation is O(A^2) per layer (loops over output angles i and input angles j).
      For A in [8, 16] it is usually manageable for head/decoder usage.
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
        num_angles: int = 16,
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(kernel_size // 2) if padding is None else int(padding)
        self.num_angles = int(num_angles)
        if self.num_angles <= 0:
            raise ValueError("num_angles must be > 0")
        self.align_corners = bool(align_corners)

        # weight: (C_out, C_in, A, K, K) where A indexes the relative angle bin
        w = torch.empty(self.out_channels, self.in_channels, self.num_angles, self.kernel_size, self.kernel_size)
        self.weight = nn.Parameter(w)
        self.bias: Optional[nn.Parameter]
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.bias = None

        self.reset_parameters()

        theta = torch.linspace(0.0, 2.0 * math.pi, steps=self.num_angles + 1)[:-1]
        self.register_buffer("theta", theta, persistent=False)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.num_angles * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"GroupConvSO2 expects x with shape (B,C,A,H,W); got {tuple(x.shape)}")
        if x.shape[2] != self.num_angles:
            raise ValueError(f"Expected angle dim={self.num_angles}, got {x.shape[2]}")

        b, cin, A, h, w = x.shape
        x_folded = x.view(b, cin * A, h, w)

        ys = []
        for i in range(A):
            # Build folded kernel for output angle i:
            # Concatenate blocks over input angle j in order j=0..A-1.
            blocks = []
            for j in range(A):
                rel = (j - i) % A
                blocks.append(self.weight[:, :, rel, :, :])  # (C_out, C_in, K, K)
            k_i = torch.cat(blocks, dim=1)  # (C_out, C_in*A, K, K)

            # Apply spatial action by g^{-1}: rotate kernels by -theta_i
            rot_angle = (-self.theta[i]).to(device=x.device)
            k_i_rot = _rotate_kernels_grid_sample(
                k_i,
                theta=rot_angle.view(1),
                align_corners=self.align_corners,
            )[0]  # (C_out, C_in*A, K, K)

            y_i = F.conv2d(x_folded, k_i_rot, bias=None, stride=self.stride, padding=self.padding)
            ys.append(y_i)

        y = torch.stack(ys, dim=1).permute(0, 2, 1, 3, 4).contiguous()  # (B, C_out, A, H', W')
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)
        return y


