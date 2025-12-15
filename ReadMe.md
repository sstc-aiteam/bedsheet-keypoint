# Bedsheet Keypoint Detection (Meta CLIP + YOLO)

Lightweight pipelines for detecting corner keypoints on bedsheets, mattresses, and fitted sheets using Meta CLIP heatmap models, LoRA fine‑tuning, and YOLO segmentation masks. Includes training, evaluation, TensorRT export, and a simple inference demo that can batch a folder and overlay ground truth from VIA JSON.

## Highlights
- Meta CLIP B16 heatmap model with LoRA adapters (fast to fine‑tune, ~3M trainable params).
- Scene presets: bedsheet, mattress, fitted_sheet_inverse (back side).
- YOLO‑based segmentation masks integrated in both training and inference.
- Simple inference script: single image or folder, optional GT overlay from `via_proj`.
- TensorRT conversion + demo for fast deployment.
- Synthetic cloth data generation (Warp + Blender) and GUI annotators.

## Repo Layout
- `src/` models & utils (`clip_heatmap_model.py`, `keypoint_metrics.py`, augmentation, TensorRT helpers).
- `shared/` common helpers (thresholding, VIA I/O, resize).
- `meta_clip_style_*_training.py` scene‑specific training pipelines.
- `train_fitted_sheet_classifier.py` / `classify_fitted_sheet.py` YOLO 分割後的床包三分類（CNN）訓練與推論。
- `inference_demo_simple.py` minimal inference/visualization runner.
- `cloth_data_gen/` synthetic data pipeline (Warp, Blender).
- `scripts/` tools (annotators, segmentation tests).
- `via_proj/` VIA JSON annotations (bedsheet, mattress, fitted_sheet_inverse).
- `models/` saved checkpoints; `image_data/` sample real images.

## Setup
```bash
git clone <repo>
cd bedsheet-keypoint
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Inference (simple demo)
- Single image:
```bash
python inference_demo_simple.py --model mattress --image path/to/img.jpg
```
- Folder:
```bash
python inference_demo_simple.py --model bedsheet --folder path/to/dir
```
Options:
- Outputs saved under `inference_results/<model_type>/`.

## 床包三分類（YOLO 分割 -> CNN 分類）
此流程會先用「原本的 YOLO segmentation 模型」擷取床包（`class=1`），再將分割後的遮罩區域裁切成方形輸入 CNN，分類成三種床包類別：
- Class 0：`image_data/床包圖片`
- Class 1：`image_data/床包圖片2`
- Class 2：`image_data/床包圖片3`

### 訓練
```bash
python -u train_fitted_sheet_classifier.py \
  --class0_dir image_data/床包圖片 \
  --class1_dir image_data/床包圖片2 \
  --class2_dir image_data/床包圖片3 \
  --cache_dir processed_data/fitted_sheet_cls_cache \
  --out_dir inference_results/fitted_sheet_cls_train
```
- **cache**：建議開啟 `--cache_dir`，會把 YOLO 分割後的 crop 存成 PNG，之後訓練 epoch 不需要重跑 YOLO（速度更快，也更適合開 `--num_workers 2~8`）。
- **num_workers**：如果沒有 cache（每次都要跑 YOLO），建議 `--num_workers 0` 避免多進程重複載入 YOLO 造成效能/記憶體問題。

訓練輸出：
- `best.pth` / `last.pth`
- `labels.json`（類別名稱）
- `train_config.json`

### 推論（單張/資料夾）
```bash
python -u classify_fitted_sheet.py \
  --image image_data/fitted_sheet1/IMG_2066.jpg \
  --checkpoint inference_results/fitted_sheet_cls_train/best.pth \
  --labels_json inference_results/fitted_sheet_cls_train/labels.json \
  --output_dir inference_results/fitted_sheet_cls_infer \
  --save_crop --save_vis \
  --output_json inference_results/fitted_sheet_cls_infer/results.json
```
- `--folder <dir>` 可改成資料夾推論
- `--save_crop` 會輸出分割後 crop 圖
- `--save_vis` 會輸出 bbox + 預測 label/conf 的視覺化圖

## Training (scene examples)
```bash
# Cloth (bedsheet-like)
python meta_clip_style_cloth_training.py

# Mattress corners
python meta_clip_style_mattress_training.py

# Fitted sheet (inside-out/back)
python meta_clip_style_fitted_sheet_inverse_training.py

# Fitted sheet (front)
python meta_clip_style_fitted_sheet_training.py
```
Each script contains a `DEFAULT_CONFIG` you can edit for data paths, LoRA settings, augmentation, text priors, and YOLO mask classes.

## Data & Annotations
- Images live in `image_data/` (mattress1/2, fitted_sheet2, etc.).
- Ground truth keypoints use VIA JSON in `via_proj/<scene>/`.
- Keypoint format: list of `[x, y]` points; heatmaps are 256×256 during training.

## TensorRT
```bash
python tensorrt_convert.py --model_type bedsheet --model_path models/meta_clip_style_bedsheet_post_pretrained
python tensorrt_demo.py --pytorch_model models/meta_clip_style_bedsheet_post_pretrained --tensorrt_model models/bedsheet_model.trt
```

## Tools
- `scripts/keypoint_annotator.py` GUI for VIA point labels.
- `scripts/test_segmentations.py` quick YOLO mask check.
- `realsense/` capture and calibration helpers.
- `cloth_data_gen/` Warp + Blender synthetic generator.

## Notes
- Input images are resized to 256×256; YOLO masks are applied in that space to match training.
- Thresholding for peaks: `thresholded_locations` with 0.3, peaks merged within 10 px.
- Inference demo draws predictions (cyan X) and optional GT (green circles).

## License
Add your license here.***
