#!/usr/bin/env python3
"""
Run a traced TorchScript .pt heatmap model on a single image (560x560 input).

Usage:
  python scripts/run_traced_pt_on_image.py --model path/to/model_traced.pt --image path/to/image.jpg
  python scripts/run_traced_pt_on_image.py --image my.png --out result.png
"""

import os
import sys
import argparse
from typing import Tuple

import cv2
import numpy as np
import torch

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Default model path (fitted_sheet_inverse traced)
DEFAULT_MODEL = os.path.join(
    REPO_ROOT,
    "models/meta_clip_style_fitted_sheet_inverse_post_original/meta_clip_style_fitted_sheet_inverse_post_original_traced.pt",
)
IMAGE_SIZE = 560


def load_image_tensor(image_path: str, size: int = IMAGE_SIZE) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load image, resize to size x size, return (1, 3, H, W) tensor in [0, 1] and (orig_w, orig_h)."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_size = (img_rgb.shape[1], img_rgb.shape[0])
    img_resized = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    # Same as inference_demo: no extra normalization, 0-1 range
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    return tensor, orig_size


def main():
    parser = argparse.ArgumentParser(description="Run traced .pt heatmap model on an image (560x560).")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL, help="Path to *_traced.pt")
    parser.add_argument("--image", "-i", type=str, required=True, help="Input image path")
    parser.add_argument("--out", "-o", type=str, default=None, help="Output visualization path (default: inference_results/traced_<basename>.png)")
    parser.add_argument("--no-viz", action="store_true", help="Only print keypoints, do not save visualization")
    parser.add_argument("--threshold", type=float, default=0.1, help="Heatmap peak threshold")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        print(f"Error: model not found: {model_path}")
        sys.exit(1)
    if not os.path.isfile(args.image):
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    # Load traced model
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    # Preprocess image to 560x560
    image_tensor, orig_size = load_image_tensor(args.image, size=IMAGE_SIZE)
    image_tensor = image_tensor.to(device)

    # Forward
    with torch.no_grad():
        heatmap = model(image_tensor)

    # heatmap: (1, 1, 560, 560)
    pred_heatmap = heatmap[0, 0].cpu().numpy()

    # Keypoints from heatmap (reuse shared helpers if available)
    try:
        from shared.functions import thresholded_locations, combine_nearby_peaks
        pred_peaks = thresholded_locations(pred_heatmap, threshold=args.threshold)
        combined_peaks = combine_nearby_peaks(pred_peaks, distance_threshold=10)
    except ImportError:
        # Minimal peak finding if shared.functions not available
        pred_peaks = np.stack(np.where(pred_heatmap >= args.threshold), axis=1)  # (N, 2) row,col
        combined_peaks = pred_peaks  # no combine_nearby_peaks

    scale_x = orig_size[0] / pred_heatmap.shape[1]
    scale_y = orig_size[1] / pred_heatmap.shape[0]
    keypoints = [(int(p[1] * scale_x), int(p[0] * scale_y)) for p in combined_peaks]

    print(f"Input: {args.image} -> tensor {image_tensor.shape}")
    print(f"Heatmap shape: {pred_heatmap.shape}")
    print(f"Keypoints found: {len(keypoints)}")
    for i, (x, y) in enumerate(keypoints):
        print(f"  {i+1}: ({x}, {y})")

    if args.no_viz:
        return

    # Visualize: original, heatmap, overlay with keypoints
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("Install matplotlib and Pillow to save visualization (or use --no-viz)")
        return

    image = Image.open(args.image).convert("RGB")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(pred_heatmap, cmap="hot", alpha=0.8)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(image)
    heatmap_resized = cv2.resize(pred_heatmap, orig_size, interpolation=cv2.INTER_CUBIC)
    axes[2].imshow(
        cv2.resize(pred_heatmap, orig_size, interpolation=cv2.INTER_CUBIC),
        cmap="hot",
        alpha=0.5,
    )
    if keypoints:
        xs, ys = zip(*keypoints)
        axes[2].scatter(xs, ys, c="cyan", s=80, marker="x", linewidths=2)
    axes[2].set_title(f"Keypoints ({len(keypoints)})")
    axes[2].axis("off")

    out_path = args.out
    if out_path is None:
        os.makedirs(os.path.join(REPO_ROOT, "inference_results"), exist_ok=True)
        out_path = os.path.join(
            REPO_ROOT,
            "inference_results",
            "traced_" + os.path.basename(args.image).rsplit(".", 1)[0] + ".png",
        )
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
