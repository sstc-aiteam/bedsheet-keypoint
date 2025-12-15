from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class SegmentationResult:
    """Segmentation result in original image coordinates."""

    mask01: np.ndarray  # HxW uint8 {0,1}
    bbox_xyxy: Tuple[int, int, int, int]  # (x1,y1,x2,y2) in pixel coords


def resolve_yolo_weights(preferred_path: str, fallbacks: Sequence[str]) -> str:
    """
    Resolve YOLO weights path.

    - Uses preferred_path if it exists
    - Otherwise tries fallbacks in order
    """
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path
    for p in fallbacks:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "YOLO weights not found. Tried:\n"
        + "\n".join([preferred_path, *fallbacks])
    )


class YoloFittedSheetSegmenter:
    """
    Thin wrapper around Ultralytics YOLO segmentation to extract the fitted-sheet mask.

    Notes:
    - We keep inputs as RGB numpy arrays; this matches existing repo usage in `inference_demo_simple.py`.
    - We always return mask/bbox in original image coordinates.
    """

    def __init__(
        self,
        weights_path: str,
        *,
        allowed_class_ids: Iterable[int] = (1,),
        conf: float = 0.25,
        imgsz: int = 640,
    ) -> None:
        self.yolo = YOLO(weights_path)
        self.allowed_class_ids = set(int(x) for x in allowed_class_ids)
        self.conf = float(conf)
        self.imgsz = int(imgsz)

    def segment_largest(self, image_rgb: np.ndarray) -> Optional[SegmentationResult]:
        """
        Segment the largest allowed-class instance and return its mask and bbox.
        Returns None if no instance found.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[-1] != 3:
            raise ValueError(f"Expected image_rgb HxWx3, got {image_rgb.shape}")
        img_h, img_w = image_rgb.shape[:2]

        results = self.yolo(image_rgb, conf=self.conf, imgsz=self.imgsz, verbose=False)
        if not results:
            return None
        r0 = results[0]
        if r0.masks is None or r0.boxes is None:
            return None

        masks = r0.masks.data.cpu().numpy()  # (N,Hm,Wm) float
        classes = r0.boxes.cls.cpu().numpy().astype(int)  # (N,)

        best_mask01: Optional[np.ndarray] = None
        best_area = -1
        for i in range(masks.shape[0]):
            if int(classes[i]) not in self.allowed_class_ids:
                continue
            m = (masks[i] > 0.5).astype(np.uint8)
            if m.shape[0] != img_h or m.shape[1] != img_w:
                m = cv2.resize(m, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            area = int(m.sum())
            if area > best_area:
                best_area = area
                best_mask01 = m

        if best_mask01 is None:
            return None

        ys, xs = np.where(best_mask01 > 0)
        if xs.size == 0:
            return None
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
        return SegmentationResult(mask01=best_mask01, bbox_xyxy=(x1, y1, x2, y2))


def crop_masked_square_rgb(
    image_rgb: np.ndarray,
    mask01: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    *,
    pad_ratio: float = 0.08,
    out_size: int = 224,
) -> np.ndarray:
    """
    Create a masked crop and resize to a square output.

    - Background (outside mask) is set to 0
    - Crop uses bbox expanded by pad_ratio
    - Crop is padded to square then resized to out_size x out_size

    Returns: RGB uint8 array (out_size, out_size, 3)
    """
    img_h, img_w = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pad = int(max(w, h) * float(pad_ratio))
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(img_w, x2 + pad)
    y2p = min(img_h, y2 + pad)

    crop_rgb = image_rgb[y1p:y2p, x1p:x2p].copy()
    crop_mask01 = mask01[y1p:y2p, x1p:x2p]
    crop_rgb[crop_mask01 == 0] = 0

    ch, cw = crop_rgb.shape[:2]
    side = max(ch, cw)
    top = (side - ch) // 2
    bottom = side - ch - top
    left = (side - cw) // 2
    right = side - cw - left

    crop_rgb = cv2.copyMakeBorder(
        crop_rgb, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    crop_rgb = cv2.resize(crop_rgb, (int(out_size), int(out_size)), interpolation=cv2.INTER_LINEAR)
    return crop_rgb



