#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.models.fitted_sheet_cnn_classifier import FittedSheetCNNClassifier
from src.utils.yolo_segmenter import YoloFittedSheetSegmenter, crop_masked_square_rgb


DEFAULT_YOLO_WEIGHTS = "models/yolo_finetuned/sheet_without_plastic.v11i.yolov11/runs/segment/train/weights/best.pt"


def _read_image_rgb(path: str) -> np.ndarray:
    if path.lower().endswith(".heic"):
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
            with Image.open(path) as img:
                return np.array(img.convert("RGB"))
        except ImportError:
            raise ImportError("pillow-heif is required to read HEIC images. Install with: pip install pillow-heif")

    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _iter_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".heic")
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]


def main() -> None:
    p = argparse.ArgumentParser(description="Segment fitted sheet via YOLO, then classify into 3 fitted-sheet classes")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to one image")
    group.add_argument("--folder", help="Folder of images")

    p.add_argument("--checkpoint", required=True, help="Path to trained classifier checkpoint (best.pth)")
    p.add_argument("--labels_json", default=None, help="Optional labels.json produced by training")

    p.add_argument("--yolo_weights", default=DEFAULT_YOLO_WEIGHTS)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--seg_conf", type=float, default=0.25)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--output_dir", default="inference_results/fitted_sheet_cls_infer")
    p.add_argument("--save_crop", action="store_true", help="Also save the segmented crop PNG")
    p.add_argument("--save_vis", action="store_true", help="Also save a visualization image (bbox + predicted label)")
    p.add_argument("--output_json", default=None, help="Optional: write per-image results JSON to this path")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    label_map: Dict[int, str] = {0: "class0", 1: "class1", 2: "class2"}
    if args.labels_json and os.path.exists(args.labels_json):
        with open(args.labels_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        label_map = {int(k): str(v) for k, v in raw.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FittedSheetCNNClassifier(num_classes=3).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    seg = YoloFittedSheetSegmenter(
        args.yolo_weights,
        allowed_class_ids=(1,),
        conf=args.seg_conf,
        imgsz=args.imgsz,
    )

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    paths: List[str]
    if args.image:
        paths = [args.image]
    else:
        paths = _iter_images(args.folder)

    results = []
    for path in paths:
        img_rgb = _read_image_rgb(path)
        seg_res = seg.segment_largest(img_rgb)
        if seg_res is None:
            crop_rgb = cv2.resize(img_rgb, (args.crop_size, args.crop_size), interpolation=cv2.INTER_LINEAR)
            bbox = None
        else:
            crop_rgb = crop_masked_square_rgb(
                img_rgb,
                seg_res.mask01,
                seg_res.bbox_xyxy,
                pad_ratio=0.08,
                out_size=args.crop_size,
            )
            bbox = seg_res.bbox_xyxy

        x = tfm(Image.fromarray(crop_rgb)).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0]
            pred = int(prob.argmax().item())
            conf = float(prob[pred].item())

        base = os.path.splitext(os.path.basename(path))[0]
        out_txt = os.path.join(args.output_dir, f"{base}.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"pred={pred} label={label_map.get(pred, str(pred))} conf={conf:.4f}\n")
            if bbox is not None:
                f.write(f"bbox_xyxy={bbox}\n")

        if args.save_crop:
            out_crop = os.path.join(args.output_dir, f"{base}.crop.png")
            cv2.imwrite(out_crop, cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR))

        if args.save_vis:
            vis = img_rgb.copy()
            label = label_map.get(pred, str(pred))
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                anchor = (int(x1), max(0, int(y1) - 6))
            else:
                anchor = (8, 24)
            cv2.putText(
                vis,
                f"{label} ({conf:.3f})",
                anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            out_vis = os.path.join(args.output_dir, f"{base}.vis.png")
            cv2.imwrite(out_vis, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        results.append(
            {
                "path": path,
                "pred": pred,
                "label": label_map.get(pred, str(pred)),
                "conf": conf,
                "bbox_xyxy": bbox,
            }
        )
        print(f"✅ {path} -> pred={pred} ({label_map.get(pred,'?')}) conf={conf:.3f}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
