#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.fitted_sheet_metaclip_dataset import FittedSheetMetaCLIPDataset
from src.models.metaclip_image_classifier import MetaCLIPClassifierConfig, MetaCLIPImageClassifier
from src.utils.yolo_segmenter import resolve_yolo_weights, YoloFittedSheetSegmenter


DEFAULT_YOLO_WEIGHTS = "models/yolo_finetuned/sheet_without_plastic.v13i.yolov11/runs/segment/train/weights/best.pt"
DEFAULT_YOLO_FALLBACKS = (
    "models/yolo_finetuned/sheet_without_plastic.v11i.yolov11/runs/segment/train/weights/best.pt",
)


@dataclass(frozen=True)
class TrainConfig:
    class0_dir: str
    class1_dir: str
    class2_dir: str
    yolo_weights: str
    imgsz: int
    seg_conf: float
    crop_size: int
    cache_dir: str
    out_dir: str
    model_name: str
    freeze_vision: bool
    hidden_dim: int
    dropout: float
    use_lora: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    val_ratio: float
    seed: int
    num_workers: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return float(correct) / float(max(1, total))


def _count_params(module: nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = 0
    trainable = 0
    for p in module.parameters():
        n = int(p.numel())
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def _print_trainable_summary(model: MetaCLIPImageClassifier) -> None:
    total, trainable = _count_params(model)
    print(f"🔧 Params: trainable={trainable:,} / total={total:,} ({(100.0*trainable/max(1,total)):.2f}%)")

    # Breakdown: vision / head / (optional) text / (optional) LoRA-only
    v_total, v_train = _count_params(model.vision)
    h_total, h_train = _count_params(model.head)
    print(f"  - vision: trainable={v_train:,} / total={v_total:,}")
    print(f"  - head:   trainable={h_train:,} / total={h_total:,}")

    if hasattr(model.clip, "text_model"):
        t_total, t_train = _count_params(model.clip.text_model)
        print(f"  - text:   trainable={t_train:,} / total={t_total:,}")

    # LoRA params (by name)
    lora_total = 0
    lora_train = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            n = int(p.numel())
            lora_total += n
            if p.requires_grad:
                lora_train += n
    if lora_total > 0:
        print(f"  - lora:   trainable={lora_train:,} / total={lora_total:,}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train fitted-sheet 3-class classifier (YOLO crop -> MetaCLIP vision -> MLP head)")
    p.add_argument("--class0_dir", default="image_data/床包圖片")
    p.add_argument("--class1_dir", default="image_data/床包圖片2")
    p.add_argument("--class2_dir", default="image_data/床包圖片3")

    p.add_argument("--yolo_weights", default=DEFAULT_YOLO_WEIGHTS)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--seg_conf", type=float, default=0.25)
    p.add_argument("--crop_size", type=int, default=256, help="Input size before MetaCLIP (model will auto-fix to patch-aligned size if needed)")
    p.add_argument("--cache_dir", default="processed_data/fitted_sheet_metaclip_cache")

    p.add_argument("--out_dir", default="models/fitted_sheet_metaclip_cls")
    p.add_argument("--model_name", default="facebook/metaclip-b16-fullcc2.5b")
    # Default: freeze vision backbone. Use --no_freeze_vision to full-finetune vision.
    p.add_argument(
        "--freeze_vision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze MetaCLIP vision encoder (default: enabled). Use --no_freeze_vision to disable.",
    )
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)

    # Default LoRA ON (as requested). Disable via --no_use_lora.
    p.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LoRA on MetaCLIP vision encoder (default: enabled). Use --no_use_lora to disable.",
    )
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    cfg = TrainConfig(
        class0_dir=args.class0_dir,
        class1_dir=args.class1_dir,
        class2_dir=args.class2_dir,
        yolo_weights=str(args.yolo_weights),
        imgsz=int(args.imgsz),
        seg_conf=float(args.seg_conf),
        crop_size=int(args.crop_size),
        cache_dir=str(args.cache_dir),
        out_dir=str(args.out_dir),
        model_name=str(args.model_name),
        freeze_vision=bool(args.freeze_vision),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        use_lora=bool(args.use_lora),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        num_workers=int(args.num_workers),
    )
    with open(os.path.join(cfg.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.num_workers > 0 and torch.cuda.is_available():
        print(
            "⚠️  Note: DataLoader workers + Ultralytics CUDA can crash on Linux (fork). "
            "This training pipeline forces YOLO to run on CPU inside workers for cache-miss samples. "
            "For fastest setup: run with --num_workers 0 once to fill cache, then rerun with >0."
        )

    yolo_weights = resolve_yolo_weights(cfg.yolo_weights, DEFAULT_YOLO_FALLBACKS)
    segmenter = YoloFittedSheetSegmenter(
        yolo_weights,
        allowed_class_ids=(1,),
        conf=cfg.seg_conf,
        imgsz=cfg.imgsz,
    )

    class_dirs: Dict[int, str] = {0: cfg.class0_dir, 1: cfg.class1_dir, 2: cfg.class2_dir}
    ds = FittedSheetMetaCLIPDataset(
        class_dirs=class_dirs,
        segmenter=segmenter,
        out_size=cfg.crop_size,
        cache_dir=cfg.cache_dir,
    )

    n = len(ds)
    idx = list(range(n))
    random.shuffle(idx)
    n_val = int(round(n * float(cfg.val_ratio)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model_cfg = MetaCLIPClassifierConfig(
        model_name=cfg.model_name,
        num_classes=3,
        freeze_vision=cfg.freeze_vision,
        dropout=cfg.dropout,
        hidden_dim=cfg.hidden_dim,
        use_lora=cfg.use_lora,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
    )
    model = MetaCLIPImageClassifier(model_cfg).to(device)
    _print_trainable_summary(model)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best = -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        total = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += float(loss.item()) * int(y.numel())
            total += int(y.numel())

        train_loss = running / float(max(1, total))
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d}/{cfg.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")

        ckpt = {"model": model.state_dict(), "model_cfg": asdict(model_cfg), "train_cfg": asdict(cfg)}
        torch.save(ckpt, os.path.join(cfg.out_dir, "last.pth"))
        if val_acc > best:
            best = val_acc
            ckpt["best_acc"] = float(best)
            torch.save(ckpt, os.path.join(cfg.out_dir, "best.pth"))

    print(f"✅ Done. best_val_acc={best:.3f}  saved_to={cfg.out_dir}")


if __name__ == "__main__":
    main()


