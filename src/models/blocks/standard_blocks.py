from __future__ import annotations

import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    name = str(name).lower()
    if name in ("relu",):
        return nn.ReLU(inplace=True)
    if name in ("gelu",):
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name in ("identity", "none", ""):
        return nn.Identity()
    raise ValueError(f"Unknown activation '{name}'")


class StandardConvBlock(nn.Module):
    """
    Small Conv2d(+norm+activation) block used by the factory.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool | None = None,
        norm: str = "bn",
        act: str = "relu",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        norm = str(norm).lower()
        use_norm = norm not in ("", "none", "identity")

        if bias is None:
            bias = not use_norm

        layers: list[nn.Module] = []
        layers.append(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )

        if use_norm:
            if norm in ("bn", "batchnorm", "batch_norm"):
                layers.append(nn.BatchNorm2d(out_ch))
            elif norm in ("gn", "groupnorm", "group_norm"):
                num_groups = max(1, min(32, out_ch))
                layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_ch))
            else:
                raise ValueError(f"Unknown norm '{norm}'")

        layers.append(_make_activation(act))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
