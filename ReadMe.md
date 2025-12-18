# Bedsheet Keypoint Detection (Meta CLIP + YOLO)

Lightweight pipelines for detecting corner keypoints on bedsheets, mattresses, and fitted sheets using Meta CLIP heatmap models, LoRA fine‑tuning, and YOLO segmentation masks. Includes training, evaluation, TensorRT export, and a simple inference demo that can batch a folder and overlay ground truth from VIA JSON.

## Highlights
- Meta CLIP B16 heatmap model with LoRA adapters (fast to fine‑tune, ~3M trainable params).
- Upgraded to **MetaCLIP2 L/14** by default (`facebook/metaclip-2-worldwide-l14`) for CLIP-style heatmap models.
- Scene presets: bedsheet, mattress, fitted_sheet_inverse (back side).
- YOLO‑based segmentation masks integrated in both training and inference.
- Simple inference script: single image or folder, optional GT overlay from `via_proj`.
- TensorRT conversion + demo for fast deployment.
- Synthetic cloth data generation (Warp + Blender) and GUI annotators.

## Repo Layout
- `src/` models & utils (`clip_heatmap_model.py`, `keypoint_metrics.py`, augmentation, TensorRT helpers).
- `shared/` common helpers (thresholding, VIA I/O, resize).
- `meta_clip_style_*_training.py` scene‑specific training pipelines.
- `train_fitted_sheet_metaclip_classifier.py` / `eval_fitted_sheet_metaclip_classifier.py` YOLO 分割後的床包三分類（MetaCLIP + LoRA）訓練與評估。
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

## MetaCLIP2（L/14）使用說明
本專案的 CLIP heatmap 模型目前預設使用 MetaCLIP2：
- **模型**：`facebook/metaclip-2-worldwide-l14`
- **載入方式**：在 `src/models/clip_heatmap_model.py` 使用 `AutoModel.from_pretrained(...)`（MetaCLIP2 checkpoint 不一定能用 `CLIPModel.from_pretrained` 正常反序列化）

### 重要：輸入尺寸必須可被 patch size 整除
MetaCLIP2 L/14 的 vision patch size = **14**，因此 `image_size`（以及實際輸入 tensor 的 H/W）必須滿足：
\(H \bmod 14 = 0\) 且 \(W \bmod 14 = 0\)

常見建議：
- **可用**：224、238、252、280、294、308、560…（都可被 14 整除）
- **不可用**：300×300（300 無法被 14 整除）

程式也會在 forward 內做檢查並拋出 `ValueError`，避免「跑到一半才 shape mismatch」。

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

## 床包三分類（YOLO mask -> MetaCLIP + LoRA）
此流程會先用「原本的 YOLO segmentation 模型」擷取床包（`class=1`），再把 **mask 後的床包影像**輸入 MetaCLIP 的 vision encoder，並用 **LoRA（rank=16）+ 小型分類 head** 做三分類：
- Class 0：`image_data/床包圖片`
- Class 1：`image_data/床包圖片2`
- Class 2：`image_data/床包圖片3`

### 訓練
```bash
python -u train_fitted_sheet_metaclip_classifier.py \
  --class0_dir image_data/床包圖片 \
  --class1_dir image_data/床包圖片2 \
  --class2_dir image_data/床包圖片3 \
  --cache_dir processed_data/fitted_sheet_metaclip_cache \
  --out_dir models/fitted_sheet_metaclip_cls
```
- **預設模型**：`--model_name facebook/metaclip-b16-fullcc2.5b`（可自行改成其他 MetaCLIP/MetaCLIP2 checkpoint）
- **LoRA**：預設 **啟用**（rank=16），可用 `--no_use_lora` 關閉
- **freeze vision**：預設 **啟用**（只訓練 LoRA + head），要全量微調 vision 用 `--no_freeze_vision`
- **image size**：預設 `--crop_size 256`（模型內部會自動調整成符合 patch size 的尺寸，避免 shape mismatch）
- **cache**：建議開啟 `--cache_dir`，會把「resize + mask」後的影像存成 PNG，之後訓練 epoch 不需要重跑 YOLO。
- **num_workers / CUDA**：Linux 下若 DataLoader worker 內嘗試用 CUDA 跑 YOLO 會爆（fork issue）。本 pipeline 會在 worker 內強制 YOLO 用 CPU 來避免 crash；如果要最快：
  - 先跑一次 `--num_workers 0` 讓 cache 填滿
  - 再用 `--num_workers 2~8` 讀 cache 訓練

訓練輸出：
- `best.pth` / `last.pth`
- `train_config.json`

### 評估（accuracy / confusion matrix / per-class metrics）
```bash
python -u eval_fitted_sheet_metaclip_classifier.py \
  --checkpoint models/fitted_sheet_metaclip_cls/best.pth \
  --class2_dir image_data/床包圖片3
```

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
- Input images are resized to `image_size × image_size`（由各 training script 的 `DEFAULT_CONFIG` 決定；MetaCLIP2 L/14 常見為 560 或 224）。
- YOLO masks 會被對齊/縮放到相同尺寸後再套用，避免 mask 與原圖解析度不一致造成 indexing 錯誤。
- Thresholding for peaks: `thresholded_locations` with 0.3, peaks merged within 10 px.
- Inference demo draws predictions (cyan X) and optional GT (green circles).
- `models/` 目錄在本 repo 預設 **gitignored**，訓練 checkpoint 放這裡是正常的（避免把大檔提交到 git）。

## License
Add your license here.***
