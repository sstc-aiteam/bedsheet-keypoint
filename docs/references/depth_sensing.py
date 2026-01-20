from __future__ import annotations

import argparse
from pathlib import Path
import time
import sys
import threading
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"images_dir does not exist: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"images_dir is not a directory: {images_dir}")
    files = [p for p in sorted(images_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir} (supported: {sorted(IMAGE_SUFFIXES)})")
    return files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run UniDepthV2 on all images in a directory.")
    p.add_argument("--images_dir", type=Path, default=Path("images"))
    p.add_argument("--output_dir", type=Path, default=Path("outputs"))
    p.add_argument("--model_id", type=str, default="lpiccinelli/unidepth-v2-vitl14")
    p.add_argument("--device", type=str, default=None, help='Force device: "cuda" or "cpu" (default: auto).')
    p.add_argument("--max_images", type=int, default=0, help="Process at most N images (0 = all).")
    p.add_argument("--cache_dir", type=Path, default=None, help="Optional Hugging Face cache dir.")
    p.add_argument("--local_files_only", action="store_true", help="Do not download; only use cached files.")
    p.add_argument(
        "--interpolation_mode",
        type=str,
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "area"],
        help="UniDepthV2 interpolation mode.",
    )
    return p.parse_args()


@contextmanager
def _loading_spinner(label: str):
    """
    Minimal progress indicator for long model loads (e.g., Hugging Face from_pretrained).
    Shows a spinner + elapsed seconds until the context exits.
    """
    stop = threading.Event()
    start = time.time()

    def _run() -> None:
        frames = "|/-\\"
        i = 0
        while not stop.is_set():
            elapsed = int(time.time() - start)
            msg = f"\r{label} {frames[i % len(frames)]}  elapsed={elapsed:>4}s"
            sys.stdout.write(msg)
            sys.stdout.flush()
            i += 1
            stop.wait(0.1)
        # clear line
        sys.stdout.write("\r" + (" " * 80) + "\r")
        sys.stdout.flush()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=1.0)


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    # UniDepth import (needs your activated env).
    from unidepth.models import UniDepthV2
    from unidepth.utils import colorize

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {args.model_id}")
    print("Note: first run may take a few minutes to download weights from Hugging Face.")
    t0 = time.time()
    with _loading_spinner("Loading UniDepthV2 (from_pretrained)"):
        model = UniDepthV2.from_pretrained(
            args.model_id,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            local_files_only=bool(args.local_files_only),
        )
    model.interpolation_mode = args.interpolation_mode
    model = model.to(device).eval()
    print(f"Model loaded in {time.time() - t0:.1f}s on device={device.type}")

    images = iter_images(args.images_dir)
    if args.max_images and args.max_images > 0:
        images = images[: args.max_images]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        print(f"Running: {img_path.name}")
        rgb = np.array(Image.open(img_path).convert("RGB"))
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # C,H,W uint8

        preds = model.infer(rgb_t.to(device))
        depth = preds["depth"].squeeze().detach().float().cpu().numpy().astype(np.float32)  # H,W meters

        out_depth = args.output_dir / f"{img_path.stem}_depth.npy"
        np.save(out_depth, depth)

        # Visualization (magma_r), saved as RGB png
        depth_vis = colorize(depth, vmin=0.01, vmax=10.0, cmap="magma_r")
        out_png = args.output_dir / f"{img_path.stem}_depth.png"
        Image.fromarray(depth_vis).save(out_png)

        print(f"[OK] {img_path.name} -> {out_depth.name}, {out_png.name}")


if __name__ == "__main__":
    main()


