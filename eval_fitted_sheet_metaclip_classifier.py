#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.fitted_sheet_metaclip_dataset import FittedSheetMetaCLIPDataset
from src.models.metaclip_image_classifier import MetaCLIPClassifierConfig, MetaCLIPImageClassifier
from src.utils.yolo_segmenter import resolve_yolo_weights, YoloFittedSheetSegmenter


DEFAULT_YOLO_FALLBACKS = (
    "models/yolo_finetuned/sheet_without_plastic.v11i.yolov11/runs/segment/train/weights/best.pt",
    "models/yolo_finetuned/sheet_without_plastic.v13i.yolov11/runs/segment/train/weights/best.pt",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[int(t), int(p)] += 1
    return cm


def _per_class_metrics(cm: np.ndarray) -> Dict[str, List[float]]:
    # rows=true, cols=pred
    num_classes = int(cm.shape[0])
    precision: List[float] = []
    recall: List[float] = []
    f1: List[float] = []
    support: List[float] = []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        prec = tp / max(1.0, tp + fp)
        rec = tp / max(1.0, tp + fn)
        f = 2.0 * prec * rec / max(1e-12, (prec + rec))
        precision.append(prec)
        recall.append(rec)
        f1.append(f)
        support.append(float(cm[c, :].sum()))
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    correct_flags: List[int] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y_t = torch.as_tensor(y, device=device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        y_true.extend([int(v) for v in y_t.detach().cpu().tolist()])
        y_pred.extend([int(v) for v in pred.detach().cpu().tolist()])
        correct_flags.extend([int(a == b) for a, b in zip(pred.detach().cpu().tolist(), y_t.detach().cpu().tolist())])
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64), correct_flags


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate fitted-sheet MetaCLIP classifier checkpoint")
    p.add_argument("--checkpoint", required=True, help="Path to best.pth/last.pth")
    p.add_argument("--output_json", default=None, help="Optional: write metrics JSON to this path")

    # Data paths (default to checkpoint train_cfg if available)
    p.add_argument("--class0_dir", default=None)
    p.add_argument("--class1_dir", default=None)
    p.add_argument("--class2_dir", default=None)

    # YOLO settings (default to checkpoint train_cfg if available)
    p.add_argument("--yolo_weights", default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--seg_conf", type=float, default=None)

    p.add_argument("--image_size", type=int, default=None, help="Resize size for dataset (defaults to train_cfg crop_size)")
    p.add_argument("--cache_dir", default=None, help="Cache dir (defaults to train_cfg cache_dir, else none)")

    # Evaluation split
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0, help="Keep 0 unless cache is filled (YOLO in workers may be slow)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError("Expected checkpoint dict with key 'model'")

    train_cfg = ckpt.get("train_cfg", {}) if isinstance(ckpt, dict) else {}
    model_cfg_raw = ckpt.get("model_cfg", {}) if isinstance(ckpt, dict) else {}

    # If the user explicitly passes any class dir, do NOT backfill from checkpoint/defaults.
    # This prevents surprises like "I only set class2_dir but it still evaluated class0/1".
    user_overrode_dirs = any(
        x is not None for x in (args.class0_dir, args.class1_dir, args.class2_dir)
    )
    if user_overrode_dirs:
        class0_dir = args.class0_dir
        class1_dir = args.class1_dir
        class2_dir = args.class2_dir
    else:
        class0_dir = train_cfg.get("class0_dir") or "image_data/床包圖片"
        class1_dir = train_cfg.get("class1_dir") or "image_data/床包圖片2"
        class2_dir = train_cfg.get("class2_dir") or "image_data/床包圖片3"

    yolo_weights = args.yolo_weights or train_cfg.get("yolo_weights") or ""
    yolo_weights = resolve_yolo_weights(str(yolo_weights), DEFAULT_YOLO_FALLBACKS)
    imgsz = int(args.imgsz) if args.imgsz is not None else int(train_cfg.get("imgsz", 640))
    seg_conf = float(args.seg_conf) if args.seg_conf is not None else float(train_cfg.get("seg_conf", 0.25))
    image_size = int(args.image_size) if args.image_size is not None else int(train_cfg.get("crop_size", 256))
    cache_dir = args.cache_dir if args.cache_dir is not None else train_cfg.get("cache_dir", None)

    # Recreate model
    model_cfg = MetaCLIPClassifierConfig(
        model_name=str(model_cfg_raw.get("model_name", train_cfg.get("model_name", "facebook/metaclip-2-worldwide-l14"))),
        num_classes=int(model_cfg_raw.get("num_classes", 3)),
        freeze_vision=bool(model_cfg_raw.get("freeze_vision", False)),
        dropout=float(model_cfg_raw.get("dropout", 0.2)),
        hidden_dim=int(model_cfg_raw.get("hidden_dim", 256)),
        use_lora=bool(model_cfg_raw.get("use_lora", True)),
        lora_r=int(model_cfg_raw.get("lora_r", 16)),
        lora_alpha=int(model_cfg_raw.get("lora_alpha", 32)),
        lora_dropout=float(model_cfg_raw.get("lora_dropout", 0.1)),
    )
    model = MetaCLIPImageClassifier(model_cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    seg = YoloFittedSheetSegmenter(
        yolo_weights,
        allowed_class_ids=(1,),
        conf=seg_conf,
        imgsz=imgsz,
    )
    class_dirs: Dict[int, str] = {}
    if class0_dir:
        class_dirs[0] = str(class0_dir)
    if class1_dir:
        class_dirs[1] = str(class1_dir)
    if class2_dir:
        class_dirs[2] = str(class2_dir)
    if not class_dirs:
        raise ValueError("No class directories provided. Pass at least one of --class0_dir/--class1_dir/--class2_dir.")
    ds = FittedSheetMetaCLIPDataset(
        class_dirs=class_dirs,
        segmenter=seg,
        out_size=image_size,
        cache_dir=cache_dir,
    )

    indices = list(range(len(ds)))
    subset = ds

    loader = DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    y_true, y_pred, _ = _predict(model, loader, device)
    num_classes = int(model_cfg.num_classes)
    cm = _confusion_matrix(y_true, y_pred, num_classes=num_classes)
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    per = _per_class_metrics(cm)

    print(f"✅ Accuracy: {acc:.4f}  (n={int(y_true.size)})")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    for c in range(num_classes):
        print(
            f"class {c}: precision={per['precision'][c]:.3f}  recall={per['recall'][c]:.3f}  f1={per['f1'][c]:.3f}  support={int(per['support'][c])}"
        )

    out = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "per_class": per,
        "model_cfg": asdict(model_cfg),
        "data": {
            "class_dirs": class_dirs,
            "yolo_weights": yolo_weights,
            "imgsz": imgsz,
            "seg_conf": seg_conf,
            "image_size": image_size,
            "cache_dir": cache_dir,
            "seed": args.seed,
        },
    }
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()


