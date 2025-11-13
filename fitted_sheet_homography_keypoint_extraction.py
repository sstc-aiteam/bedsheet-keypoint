#!/usr/bin/env python3
"""
Ordered-Contour Quadrilateral Fitted Sheet Corner Extraction

We operate directly on the segmented contour (no convex hull). After segmenting
and cleaning the fitted sheet mask, we downsample the contour to a manageable
number of points, then examine ordered quadruples (i < j < k < l) so every
candidate corner lies on the actual boundary. Each candidate quad is rasterized
and scored by its IoU overlap with the segmented mask, and the highest-IoU quad
is mapped via homography to a canonical square before overlaying the corners on
the segmented image for inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

ALLOWED_CLASSES = (1, 3)
TARGET_SIZE = 256
MAX_CONTOUR_POINTS = 200
MAX_CANDIDATE_POINTS = 20
UNIFORM_SAMPLE_POINTS = 40


def list_images(path: str, extensions: Sequence[str]) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    if p.is_file():
        if p.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported file extension: {p.suffix}")
        return [str(p)]
    matches: List[str] = []
    for ext in extensions:
        matches.extend(sorted(str(fp) for fp in p.rglob(f"*{ext}")))
    if not matches:
        raise ValueError(f"No images with extensions {extensions} found under {path}")
    return matches


def refine_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def extract_primary_contour(mask: np.ndarray, min_area_ratio: float = 0.005) -> Optional[np.ndarray]:
    h, w = mask.shape[:2]
    min_area = min_area_ratio * h * w
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            return contour
    return None


def downsample_contour(contour: np.ndarray, max_points: int) -> np.ndarray:
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step]


def farthest_point_indices(points: np.ndarray, max_candidates: int) -> np.ndarray:
    n = len(points)
    if n <= max_candidates:
        return np.arange(n, dtype=int)

    selected = [0]
    distances = np.linalg.norm(points - points[0], axis=1)

    for _ in range(1, max_candidates):
        idx = int(np.argmax(distances))
        if idx in selected:
            break
        selected.append(idx)
        new_dist = np.linalg.norm(points - points[idx], axis=1)
        distances = np.minimum(distances, new_dist)

    if len(selected) < 4:
        uniform = np.linspace(0, n - 1, 4, dtype=int).tolist()
        selected.extend(uniform)

    selected = sorted(set(selected))[:max_candidates]
    return np.array(selected, dtype=int)


def uniform_sample_indices(length: int, count: int) -> np.ndarray:
    count = min(length, max(4, count))
    if count <= 0:
        return np.arange(0)
    indices = np.linspace(0, length - 1, count, dtype=int)
    return np.unique(indices)


def signed_polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(points: np.ndarray) -> float:
    return abs(signed_polygon_area(points))


def ensure_clockwise(points: np.ndarray) -> np.ndarray:
    if signed_polygon_area(points) < 0:
        return points[::-1]
    return points


def is_convex_polygon(points: np.ndarray) -> bool:
    if len(points) < 3:
        return False
    cross_sign = 0
    n = len(points)
    for i in range(n):
        a = points[(i + 1) % n] - points[i]
        b = points[(i + 2) % n] - points[(i + 1) % n]
        cross = np.cross(a, b)
        if cross != 0:
            if cross_sign == 0:
                cross_sign = np.sign(cross)
            elif np.sign(cross) != cross_sign:
                return False
    return True


def clip_polygon_with_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def compute_intersection(p1, p2, a, b):
        s1 = p2 - p1
        s2 = b - a
        denom = s1[0] * s2[1] - s2[0] * s1[1]
        if abs(denom) < 1e-9:
            return p2
        t = ((a[0] - p1[0]) * s2[1] - (a[1] - p1[1]) * s2[0]) / denom
        return p1 + t * s1

    output = subject.copy()
    for i in range(len(clip)):
        input_list = output.copy()
        if len(input_list) == 0:
            break
        output = []
        A = clip[i]
        B = clip[(i + 1) % len(clip)]
        for j in range(len(input_list)):
            P = input_list[j]
            Q = input_list[(j + 1) % len(input_list)]
            if inside(Q, A, B):
                if not inside(P, A, B):
                    output.append(compute_intersection(P, Q, A, B))
                output.append(Q)
            elif inside(P, A, B):
                output.append(compute_intersection(P, Q, A, B))
        output = np.array(output, dtype=np.float32)
    return output


def quad_iou(candidate: np.ndarray, subject_polygon: np.ndarray, subject_area: float) -> Tuple[float, Optional[np.ndarray]]:
    if len(candidate) != 4:
        return -1.0, None
    candidate = ensure_clockwise(candidate)

    # Reject degenerate quads where any two vertices coincide or are extremely close.
    diffs = np.linalg.norm(np.diff(np.vstack([candidate, candidate[0]]), axis=0), axis=1)
    if np.any(diffs < 1.0):
        return -1.0, None

    if polygon_area(candidate) < 1.0 or not is_convex_polygon(candidate):
        return -1.0, None

    clipped = clip_polygon_with_convex(subject_polygon, candidate)
    if len(clipped) < 3:
        return -1.0, None

    inter_area = polygon_area(clipped)
    quad_area = polygon_area(candidate)
    union = subject_area + quad_area - inter_area
    if union <= 0:
        return -1.0, None
    iou = inter_area / union
    return iou, candidate


def search_best_quad(points: np.ndarray, subject_polygon: np.ndarray, subject_area: float) -> Optional[np.ndarray]:
    n = len(points)
    if n < 4:
        return None

    best_score = -1.0
    best_quad: Optional[np.ndarray] = None

    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            for k in range(j + 1, n - 1):
                for l in range(k + 1, n):
                    candidate = np.array([points[i], points[j], points[k], points[l]], dtype=np.float32)
                    score, quad = quad_iou(candidate, subject_polygon, subject_area)
                    if score > best_score and quad is not None:
                        best_score = score
                        best_quad = quad
                    if best_score >= 0.99:
                        return best_quad

    return best_quad


def ordered_quadrilateral_from_contour(contour: np.ndarray) -> Optional[np.ndarray]:
    subject_area = cv2.contourArea(contour)
    if subject_area < 10:
        return None

    points_full = downsample_contour(contour, MAX_CONTOUR_POINTS)
    if len(points_full) < 4:
        return None
    subject_polygon = ensure_clockwise(points_full)

    candidate_sets = []
    far_idx = farthest_point_indices(points_full, MAX_CANDIDATE_POINTS)
    candidate_sets.append(points_full[far_idx])

    uniform_idx = uniform_sample_indices(len(points_full), UNIFORM_SAMPLE_POINTS)
    candidate_sets.append(points_full[uniform_idx])

    if len(points_full) <= UNIFORM_SAMPLE_POINTS:
        candidate_sets.append(points_full)

    for candidates in candidate_sets:
        quad = search_best_quad(candidates, subject_polygon, subject_area)
        if quad is not None:
            return quad

    return None


def order_points_clockwise(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def segment_fitted_sheet(image_rgb: np.ndarray, model: YOLO, allowed_classes: Sequence[int]) -> Optional[np.ndarray]:
    results = model(image_rgb, task="segment", verbose=False)
    if not results:
        return None
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    first = results[0]
    if first.masks is None or first.boxes is None:
        return None
    masks = first.masks.data.cpu().numpy()
    classes = first.boxes.cls.cpu().numpy()
    for mask_data, cls in zip(masks, classes):
        if int(cls) not in allowed_classes:
            continue
        resized = cv2.resize(mask_data, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        binary = (resized > 0.5).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, binary)
    return mask if np.any(mask) else None


def process_image(
    image_path: str,
    model: YOLO,
    allowed_classes: Sequence[int],
    target_size: int,
    save_overlay: bool,
    overlay_dir: Path,
) -> Optional[np.ndarray]:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"[WARN] Unable to read {image_path}")
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = segment_fitted_sheet(image_rgb, model, allowed_classes)
    if mask is None:
        print(f"[WARN] Segmentation failed for {image_path}")
        return None

    mask_refined = refine_mask(mask)
    contour = extract_primary_contour(mask_refined)
    if contour is None:
        print(f"[WARN] No contour for {image_path}")
        return None

    quad = ordered_quadrilateral_from_contour(contour)
    if quad is None or quad.shape != (4, 2):
        print(f"[WARN] Could not infer quadrilateral for {image_path}")
        return None

    quad_original = quad.copy()

    if save_overlay:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        segmented = cv2.bitwise_and(image_bgr, image_bgr, mask=mask_refined)
        overlay = segmented.copy()
        poly = np.round(quad_original).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [poly], True, (0, 255, 0), 3)
        for idx, pt in enumerate(quad_original):
            px = tuple(int(round(v)) for v in pt)
            cv2.circle(overlay, px, 6, (0, 0, 255), -1)
            cv2.putText(overlay, str(idx), (px[0] + 4, px[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        out_path = overlay_dir / f"{Path(image_path).stem}_corners.png"
        cv2.imwrite(str(out_path), overlay)

    return quad_original


def main() -> None:
    parser = argparse.ArgumentParser(description="Ordered contour quadrilateral fitted sheet corner extraction")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument(
        "--yolo-model",
        default="models/yolo_finetuned/sheet_without_plastic.v10i.yolov11/runs/segment/train/weights/best.pt",
        help="Path to YOLO segmentation weights",
    )
    parser.add_argument("--save-overlay", action="store_true", help="Save overlay visualizations")
    parser.add_argument("--overlay-dir", default="results/ordered_quad_overlays", help="Output directory for overlays")
    args = parser.parse_args()

    images = list_images(args.input, [".png", ".jpg", ".jpeg"])
    model = YOLO(args.yolo_model)

    for image_path in images:
        corners = process_image(
            image_path,
            model,
            allowed_classes=ALLOWED_CLASSES,
            target_size=TARGET_SIZE,
            save_overlay=args.save_overlay,
            overlay_dir=Path(args.overlay_dir),
        )
        if corners is None:
            continue
        tl = corners[0]
        br = corners[2]
        print(f"{image_path}: TL=({tl[0]:.1f}, {tl[1]:.1f}), BR=({br[0]:.1f}, {br[1]:.1f})")


if __name__ == "__main__":
    main()
