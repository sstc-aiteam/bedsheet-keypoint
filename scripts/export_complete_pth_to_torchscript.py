#!/usr/bin/env python3
"""
Export complete_model.pth checkpoints to TorchScript (.pt) format.

Finds all complete_model.pth under models/ (or a given path), loads each with the
matching config (training_config.json or config.json), and exports a traceable
vision-only graph to <dirname>_traced.pt. The text-prior branch is disabled for
export because it uses the tokenizer and is not TorchScript-traceable.

Usage:
  python scripts/export_complete_pth_to_torchscript.py
  python scripts/export_complete_pth_to_torchscript.py --models-dir models
  python scripts/export_complete_pth_to_torchscript.py --pth path/to/complete_model.pth --output path/to/out.pt

Requires the same env as training (e.g. transformers, peft, timm) so the CLIP model can be built.
Export traces on CUDA when available so the saved .pt runs on GPU without device mismatch.
"""

import os
import sys
import json
import argparse
from typing import List, Tuple

# Project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch


# Default head decoder used by training scripts when not in config
DEFAULT_HEAD_DECODER = {
    "kind": "gcnn",
    "mode": "so2",
    "hidden": 32,
    "so2_num_angles": 8,
    "so2_num_gconvs": 2,
}


def find_complete_pth_roots(models_dir: str) -> List[Tuple[str, str]]:
    """Return list of (model_root_dir, complete_model.pth path)."""
    results = []
    models_dir = os.path.abspath(models_dir)
    if not os.path.isdir(models_dir):
        return results
    for root, _dirs, files in os.walk(models_dir):
        if "complete_model.pth" in files:
            pth_path = os.path.join(root, "complete_model.pth")
            results.append((root, pth_path))
    return results


def load_config_for_checkpoint(model_root: str) -> dict:
    """Load config from training_config.json or config.json in model_root."""
    for name in ("training_config.json", "config.json"):
        path = os.path.join(model_root, name)
        if os.path.isfile(path):
            with open(path, "r") as f:
                return json.load(f)
    return {}


def build_model_kwargs(config: dict) -> dict:
    """Build kwargs for create_clip_heatmap_model from saved config."""
    kwargs = {
        "model_name": config.get("model_name", "facebook/metaclip-b16-fullcc2.5b"),
        "image_size": int(config.get("image_size", 560)),
        "use_lora": config.get("use_lora", True),
        "lora_r": int(config.get("lora_r", 16)),
        "lora_alpha": int(config.get("lora_alpha", 32)),
        "lora_dropout": float(config.get("lora_dropout", 0.05)),
        "use_text_prior": config.get("use_text_prior", True),
        "prior_prompts": config.get("prior_prompts"),
        "negative_prompts": config.get("negative_prompts"),
        "prior_weight": float(config.get("prior_weight", 0.5)),
    }
    head_decoder = config.get("head_decoder", DEFAULT_HEAD_DECODER)
    if head_decoder is not None:
        kwargs["head_decoder"] = head_decoder
    else:
        kwargs["head_use_gcnn"] = config.get("head_use_gcnn", True)
        kwargs["head_gcnn_hidden"] = int(config.get("head_gcnn_hidden", 32))
        kwargs["head_gcnn_mode"] = config.get("head_gcnn_mode", "so2")
        kwargs["head_so2_num_angles"] = int(config.get("head_so2_num_angles", 8))
        kwargs["head_so2_num_gconvs"] = int(config.get("head_so2_num_gconvs", 2))
    return kwargs


def export_one(
    pth_path: str,
    output_path: str,
    model_root: str | None = None,
    image_size: int | None = None,
    strict_load: bool = True,
) -> bool:
    """
    Load one complete_model.pth and export to TorchScript .pt.

    If model_root is None, it is the directory containing pth_path.
    If image_size is None, it is taken from config.
    """
    from src.models.clip_heatmap_model import create_clip_heatmap_model

    if model_root is None:
        model_root = os.path.dirname(pth_path)
    config = load_config_for_checkpoint(model_root)
    kwargs = build_model_kwargs(config)
    img_size = image_size if image_size is not None else kwargs["image_size"]

    # Build with use_text_prior=False so the traced graph is vision-only (no tokenizer).
    kwargs["use_text_prior"] = False
    model = create_clip_heatmap_model(**kwargs)
    state = torch.load(pth_path, map_location="cpu")
    load_ok = model.load_state_dict(state, strict=strict_load)
    if not strict_load and load_ok is not None:
        if load_ok.missing_keys or load_ok.unexpected_keys:
            print(f"  Note: missing_keys={len(load_ok.missing_keys)}, unexpected_keys={len(load_ok.unexpected_keys)}")
    model.eval()

    # Trace on CUDA if available so the saved .pt runs on CUDA without device mismatch.
    # (SO2 grid_sample uses tensors that get device from the trace; tracing on CPU bakes CPU.)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    example = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(model, example, check_trace=True)
    traced.save(output_path)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export complete_model.pth to TorchScript .pt (vision-only trace)."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=os.path.join(REPO_ROOT, "models"),
        help="Directory to search for complete_model.pth",
    )
    parser.add_argument(
        "--pth",
        type=str,
        default=None,
        help="Single .pth file to convert (overrides --models-dir search)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pt path (only with --pth)",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Load state_dict with strict=False",
    )
    args = parser.parse_args()

    if args.pth is not None:
        pth_path = os.path.abspath(args.pth)
        if not os.path.isfile(pth_path):
            print(f"Error: not a file: {pth_path}")
            sys.exit(1)
        out = args.output
        if not out:
            out = pth_path.replace(".pth", "_traced.pt")
            if out == pth_path:
                out = pth_path + "_traced.pt"
        print(f"Exporting: {pth_path} -> {out}")
        try:
            export_one(pth_path, out, strict_load=not args.no_strict)
            print(f"  Saved: {out}")
        except Exception as e:
            print(f"  Failed: {e}")
            raise
        return

    pairs = find_complete_pth_roots(args.models_dir)
    if not pairs:
        print(f"No complete_model.pth found under {args.models_dir}")
        return
    print(f"Found {len(pairs)} checkpoint(s). Exporting (vision-only TorchScript)...")
    for model_root, pth_path in pairs:
        out_name = os.path.basename(model_root.rstrip(os.sep)) + "_traced.pt"
        out_path = os.path.join(model_root, out_name)
        print(f"  {pth_path} -> {out_path}")
        try:
            export_one(pth_path, out_path, model_root=model_root, strict_load=not args.no_strict)
            print(f"    Saved: {out_path}")
        except Exception as e:
            print(f"    Failed: {e}")


if __name__ == "__main__":
    main()
