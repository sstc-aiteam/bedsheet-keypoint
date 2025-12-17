#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.fitted_sheet_segmented_dataset import FittedSheetSegmentedDataset
from src.models.fitted_sheet_cnn_classifier import FittedSheetCNNClassifier
from src.utils.yolo_segmenter import YoloFittedSheetSegmenter


DEFAULT_YOLO_WEIGHTS = "models/yolo_finetuned/sheet_without_plastic.v13i.yolov11/runs/segment/train/weights/best.pt"


@dataclass(frozen=True)
class TrainConfig:
    class0_dir: str
    class1_dir: str
    class2_dir: str
    yolo_weights: str
    cache_dir: str
    out_dir: str
    imgsz: int
    seg_conf: float
    crop_size: int
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


def main() -> None:
    p = argparse.ArgumentParser(description="Train 3-class fitted-sheet classifier using YOLO-segmented crops")
    p.add_argument("--class0_dir", default="image_data/床包圖片", help="Folder for class 0 images")
    p.add_argument("--class1_dir", default="image_data/床包圖片2", help="Folder for class 1 images")
    p.add_argument("--class2_dir", default="image_data/床包圖片3", help="Folder for class 2 images")
    p.add_argument("--yolo_weights", default=DEFAULT_YOLO_WEIGHTS, help="YOLO segmentation weights (.pt)")
    p.add_argument("--cache_dir", default="processed_data/fitted_sheet_cls_cache", help="Cache segmented crops here")
    p.add_argument("--out_dir", default="models/fitted_sheet_cls_train", help="Output dir for checkpoints/logs")

    p.add_argument("--imgsz", type=int, default=640, help="YOLO imgsz")
    p.add_argument("--seg_conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--crop_size", type=int, default=224, help="CNN input crop size")

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    cfg = TrainConfig(
        class0_dir=args.class0_dir,
        class1_dir=args.class1_dir,
        class2_dir=args.class2_dir,
        yolo_weights=args.yolo_weights,
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        imgsz=args.imgsz,
        seg_conf=args.seg_conf,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    with open(os.path.join(args.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    segmenter = YoloFittedSheetSegmenter(
        args.yolo_weights,
        allowed_class_ids=(1,),
        conf=args.seg_conf,
        imgsz=args.imgsz,
    )

    class_dirs: Dict[int, str] = {
        0: args.class0_dir,
        1: args.class1_dir,
        2: args.class2_dir,
    }
    ds = FittedSheetSegmentedDataset(
        class_dirs=class_dirs,
        segmenter=segmenter,
        out_size=args.crop_size,
        cache_dir=args.cache_dir,
    )

    n = len(ds)
    indices = list(range(n))
    random.shuffle(indices)
    n_val = int(round(n * float(args.val_ratio)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(ds, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = FittedSheetCNNClassifier(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    best_acc = -1.0
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

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

            running_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())
            correct += int((logits.argmax(dim=1) == y).sum().item())

        train_loss = running_loss / float(max(1, total))
        train_acc = float(correct) / float(max(1, total))
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d}/{args.epochs}  loss={train_loss:.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        ckpt_last = os.path.join(args.out_dir, "last.pth")
        torch.save({"model": model.state_dict()}, ckpt_last)

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_best = os.path.join(args.out_dir, "best.pth")
            torch.save({"model": model.state_dict(), "best_acc": best_acc}, ckpt_best)

    with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "0": os.path.basename(args.class0_dir.rstrip("/")),
                "1": os.path.basename(args.class1_dir.rstrip("/")),
                "2": os.path.basename(args.class2_dir.rstrip("/")),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"✅ Done. best_val_acc={best_acc:.3f}  outputs={args.out_dir}")


if __name__ == "__main__":
    main()



