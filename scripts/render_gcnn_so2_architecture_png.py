#!/usr/bin/env python3
"""
Render a PNG visualization of the GCNN SO(2) (sampled) architecture used in this repo.

Output:
  docs/images/gcnn_so2_architecture.png
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float
    text: str


def _draw_box(ax, b: Box, *, fc: str = "#F7F9FC", ec: str = "#1F2937") -> None:
    from matplotlib.patches import FancyBboxPatch

    rect = FancyBboxPatch(
        (b.x, b.y),
        b.w,
        b.h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
        mutation_aspect=1.0,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(rect)
    ax.text(
        b.x + b.w / 2.0,
        b.y + b.h / 2.0,
        b.text,
        ha="center",
        va="center",
        fontsize=11,
        family="DejaVu Sans",
        transform=ax.transAxes,
        zorder=3,
        color="#111827",
        wrap=True,
    )


def _arrow(ax, p0: Tuple[float, float], p1: Tuple[float, float], *, color: str = "#111827") -> None:
    from matplotlib.patches import FancyArrowPatch

    arr = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.6,
        color=color,
        transform=ax.transAxes,
        zorder=4,
    )
    ax.add_patch(arr)


def _title(ax, s: str) -> None:
    ax.text(
        0.0,
        1.02,
        s,
        ha="left",
        va="bottom",
        fontsize=14,
        weight="bold",
        transform=ax.transAxes,
        color="#111827",
    )


def _setup_ax(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def render(out_path: str) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    fig.patch.set_facecolor("white")

    # -------------------------
    # Panel 1: Macro pipeline
    # -------------------------
    ax = axes[0]
    _setup_ax(ax)
    _title(ax, "GCNN SO(2) decoder path (ClipHeatmapHead)")

    boxes = [
        Box(0.03, 0.38, 0.18, 0.26, "CLIP feat_2d\n(B, D, h, w)"),
        Box(0.26, 0.38, 0.20, 0.26, "LiftingConvSO2\n(B, hidden, A, h', w')"),
        Box(0.50, 0.38, 0.18, 0.26, "Resize (preserve A)\n→ (B, hidden, A, out, out)"),
        Box(0.72, 0.38, 0.20, 0.26, "GroupConvSO2 × L\n(B, *, A, out, out)"),
        Box(0.28, 0.05, 0.20, 0.22, "GroupPooling(mean over A)\n(B, 64, out, out)"),
        Box(0.52, 0.05, 0.16, 0.22, "1×1 conv\n(B, 1, out, out)"),
        Box(0.72, 0.05, 0.22, 0.22, "sigmoid\nheatmap\n(B, 1, out, out)"),
    ]
    for b in boxes:
        _draw_box(ax, b)

    _arrow(ax, (0.21, 0.51), (0.26, 0.51))
    _arrow(ax, (0.46, 0.51), (0.50, 0.51))
    _arrow(ax, (0.68, 0.51), (0.72, 0.51))
    _arrow(ax, (0.82, 0.38), (0.40, 0.27))  # down-left
    _arrow(ax, (0.48, 0.16), (0.52, 0.16))
    _arrow(ax, (0.68, 0.16), (0.72, 0.16))

    # -------------------------
    # Panel 2: LiftingConvSO2
    # -------------------------
    ax = axes[1]
    _setup_ax(ax)
    _title(ax, "Inside LiftingConvSO2: rotate kernels → fold angle → one-shot conv2d")

    b_x = Box(0.03, 0.38, 0.20, 0.26, "x\n(B, C_in, H, W)")
    b_w = Box(0.28, 0.38, 0.22, 0.26, "base W\n(C_out, C_in, K, K)")
    b_rot = Box(0.54, 0.38, 0.22, 0.26, "rotate by -θ_k\n(grid_sample)\n(A, C_out, C_in, K, K)")
    b_fold = Box(0.54, 0.05, 0.22, 0.22, "fold A into C_out\n(C_out·A, C_in, K, K)")
    b_conv = Box(0.28, 0.05, 0.22, 0.22, "F.conv2d\n(B, C_out·A, H', W')")
    b_out = Box(0.80, 0.22, 0.18, 0.30, "reshape\n(B, C_out, A, H', W')")

    for b in [b_x, b_w, b_rot, b_fold, b_conv, b_out]:
        _draw_box(ax, b)

    _arrow(ax, (0.23, 0.51), (0.28, 0.51))  # x -> W (conceptual)
    _arrow(ax, (0.50, 0.51), (0.54, 0.51))  # W -> rot
    _arrow(ax, (0.65, 0.38), (0.65, 0.27))  # rot -> fold (down)
    _arrow(ax, (0.54, 0.16), (0.50, 0.16))  # fold -> conv (left)
    _arrow(ax, (0.23, 0.38), (0.32, 0.27))  # x -> conv (down-right)
    _arrow(ax, (0.50, 0.16), (0.80, 0.37))  # conv -> out (up-right)

    # -------------------------
    # Panel 3: GroupConvSO2
    # -------------------------
    ax = axes[2]
    _setup_ax(ax)
    _title(ax, "Inside GroupConvSO2 (G→G): twist-shift rel=(j−i) mod A + rotate by output angle")

    b_in = Box(0.03, 0.38, 0.22, 0.26, "x\n(B, C_in, A, H, W)")
    b_loop_i = Box(0.30, 0.62, 0.22, 0.18, "for each output angle i")
    b_rot_i = Box(0.56, 0.62, 0.22, 0.18, "rotate kernels by -θ_i\n(grid_sample)")
    b_loop_j = Box(0.30, 0.38, 0.22, 0.18, "for each input angle j\nrel=(j−i) mod A")
    b_select = Box(0.56, 0.38, 0.22, 0.18, "select W[:,:,rel]\nfrom W:(C_out,C_in,A,K,K)")
    b_convij = Box(0.30, 0.12, 0.22, 0.18, "conv2d(x[:,:,j], W_i_rel)")
    b_sum = Box(0.56, 0.12, 0.22, 0.18, "sum over j\n→ y[:,:,i]")
    b_stack = Box(0.82, 0.22, 0.16, 0.30, "stack over i\n(B, C_out, A, H', W')")

    for b in [b_in, b_loop_i, b_rot_i, b_loop_j, b_select, b_convij, b_sum, b_stack]:
        _draw_box(ax, b)

    _arrow(ax, (0.25, 0.51), (0.30, 0.71))  # in -> loop i
    _arrow(ax, (0.52, 0.71), (0.56, 0.71))  # loop i -> rot i
    _arrow(ax, (0.41, 0.62), (0.41, 0.56))  # loop i down to loop j
    _arrow(ax, (0.52, 0.47), (0.56, 0.47))  # loop j -> select
    _arrow(ax, (0.67, 0.38), (0.67, 0.30))  # select down to convij area
    _arrow(ax, (0.25, 0.38), (0.30, 0.21))  # in -> convij
    _arrow(ax, (0.52, 0.21), (0.56, 0.21))  # convij -> sum
    _arrow(ax, (0.78, 0.21), (0.82, 0.37))  # sum -> stack

    # Global footer
    fig.text(
        0.01,
        0.01,
        "Legend: A = number of sampled SO(2) angles (bins). L = number of GroupConvSO2 layers. out = heatmap spatial size.",
        fontsize=10,
        color="#374151",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(repo_root, "docs", "images", "gcnn_so2_architecture.png")
    p = render(out_path)
    print(p)


if __name__ == "__main__":
    main()




