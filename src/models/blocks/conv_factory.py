from __future__ import annotations

from typing import Optional

import torch.nn as nn

from .standard_blocks import StandardConvBlock
from .types import ConvBlockKind
from .gcnn_groups import CyclicGroupC4
from .gcnn_layers import LiftingConvC4, GroupConvC4


def make_conv_block(
    kind: ConvBlockKind,
    in_ch: int,
    out_ch: int,
    *,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int | None = None,
    norm: str = "bn",
    act: str = "relu",
    # GCNN-specific knobs (kept optional for easy integration later)
    gcnn_group: str = "C4",
) -> nn.Module:
    """
    Pluggable block factory.

    This reuses the daily report's core implementation trick:
    fold group dimension into channels so we can keep using `torch.nn.functional.conv2d`.

    Notes:
    - `gcnn_lifting_c4` maps (B,C,H,W) -> (B,C_out,4,H',W')   (not a drop-in for 2D conv)
    - `gcnn_group_c4`   maps (B,C,4,H,W) -> (B,C_out,4,H',W')
    - `standard` is a standard Conv2d block mapping (B,C,H,W)->(B,C_out,H',W')
    """

    if kind == "standard":
        return StandardConvBlock(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            act=act,
        )

    if kind == "gcnn_lifting_c4":
        if str(gcnn_group).upper() != "C4":
            raise NotImplementedError("Only gcnn_group='C4' is implemented right now.")
        return LiftingConvC4(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            group=CyclicGroupC4(),
        )

    if kind == "gcnn_group_c4":
        if str(gcnn_group).upper() != "C4":
            raise NotImplementedError("Only gcnn_group='C4' is implemented right now.")
        return GroupConvC4(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            group=CyclicGroupC4(),
        )

    raise ValueError(f"Unknown conv block kind '{kind}'")
