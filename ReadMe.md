# 床單摺疊機器人 - 關鍵點檢測

一個使用深度學習和電腦視覺技術的床單摺疊機器人關鍵點檢測系統。

## 🚀 功能特色

- **混合關鍵點檢測模型**：YOLO + Vision Transformer 架構
- **兩階段訓練流程**：預訓練 + 後訓練優化
- **優化訓練流程**：使用 `torch.compile()`、早停機制和 mixup 增強
- **即時推理**：針對即時床單關鍵點檢測優化
- **全面數據處理**：YOLO 分割 + 關鍵點註解流程
- **合成數據生成**：使用 Blender 生成訓練數據

## 📁 專案結構

```
bedSheetFoldingRobot/
├── src/
│   ├── models/                 # 模型架構
│   │   ├── hybrid_keypoint_net.py
│   │   └── efficient_keypoint_net.py
│   ├── utils/                  # 工具函數
│   │   ├── model_utils.py      # YOLO 骨幹和模型工具
│   │   └── tensorrt_utils.py   # TensorRT 轉換工具（未來）
│   └── training/               # 訓練流程
├── cloth_data_gen/             # 合成數據生成
│   ├── cloth_dataset_gen.py    # Blender 數據生成腳本
│   └── output/                 # 生成的數據輸出目錄
├── realsense/                  # RealSense 工具集
├── shared/                     # 共享函數和工具
├── models/                     # 訓練模型和 YOLO 權重
├── data/                       # 數據集和註解
├── results/                    # 訓練結果和視覺化
├── keypoint_detection_model_training.py  # 第一階段：預訓練腳本
├── post_keypoint_detection_model_training.py  # 第二階段：後訓練腳本
├── convert_to_tensorrt.py      # TensorRT 轉換腳本（未來）
└── test_tensorrt_inference.py  # 效能基準測試（未來）
```

## 🛠️ 安裝

1. **複製儲存庫：**
```bash
git clone <repository-url>
cd bedSheetFoldingRobot
```

2. **安裝依賴：**
```bash
pip install -r requirements.txt
```

3. **安裝 Blender（用於數據生成）：**
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install blender

# 或從官網下載：https://www.blender.org/download/
```

## 🎯 快速開始

### 步驟 0：生成訓練數據（可選）

如果您需要生成合成訓練數據，可以使用 Blender 腳本：

```bash
# 進入數據生成目錄
cd cloth_data_gen

# 使用 Blender 執行數據生成腳本
blender --background --python cloth_dataset_gen.py
```

#### 數據生成配置
腳本會生成以下內容：
- **3000 個樣本**（可調整 `n_samples` 參數）
- **128x128 像素圖像**
- **4 個角點關鍵點**
- **隨機變形和顏色**
- **輸出目錄**：`output/images/` 和 `output/keypoints/`

#### 自訂生成參數
編輯 `cloth_dataset_gen.py` 中的參數：
```python
n_samples = 3000          # 生成樣本數量
length_range = (3.5, 5.0) # 布料長度範圍
width_range = (3.5, 5.0)  # 布料寬度範圍
res = 40                  # 網格細分數
```

### 第一階段：預訓練

首先，您需要使用原始訓練腳本預訓練模型：

```bash
# 第一階段：預訓練模型
python keypoint_detection_model_training.py
```

這將會：
- 在您的數據集上從頭開始訓練模型
- 將預訓練模型儲存到 `models/keypoint_model_vit.pth`
- 建立基準效能

### 第二階段：後訓練優化

預訓練後，使用後訓練腳本進行優化：

```bash
# 第二階段：後訓練優化
python post_keypoint_detection_model_training.py config_quantization_fixed.json
```

**配置選項：**
- `num_epochs`：訓練輪數（預設：50）
- `batch_size`：批次大小（預設：16）
- `learning_rate`：學習率（預設：0.001）
- `early_stopping_patience`：早停耐心值（預設：10）
- `use_mixup`：啟用 mixup 增強（預設：true）

## 🏗️ 模型架構

### 混合關鍵點網路
- **骨幹**：YOLO11L-pose（前12層）
- **頭部**：用於關鍵點檢測的 Vision Transformer
- **輸出**：基於熱力圖的關鍵點預測
- **輸入**：128x128 RGB 圖像
- **參數**：約100M 參數

### 主要特色
- **torch.compile()**：使用 PyTorch 2.0 編譯優化訓練
- **早停機制**：基於驗證損失平台的自動訓練終止
- **Mixup 增強**：使用 mixup 數據增強改善泛化能力
- **最佳模型儲存**：基於驗證損失自動儲存最佳模型

## 📊 效能

### 訓練優化
- **torch.compile()**：約20-30% 更快的訓練
- **混合精度**：FP16 訓練以節省記憶體
- **梯度裁剪**：穩定的梯度裁剪訓練
- **學習率調度**：自適應學習率調度

## 🔧 配置

### 訓練配置（`config_quantization_fixed.json`）
```json
{
    "model_name": "HybridKeypointNet",
    "model_save_path": "models/keypoint_model_vit_post",
    "pretrained_model_path": "models/keypoint_model_vit.pth",
    "yolo_model_path": "models/yolo_finetuned/best.pt",
    "keypoints_data_src": "via_proj/via_project_22Aug2025_16h07m06s.json",
    "image_path": "RGB-images-jpg/",
    "allowed_classes": [1],
    "batch_size": 16,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "use_augmentation": true,
    "use_mixup": true,
    "early_stopping_patience": 10
}
```

## 📈 訓練流程

### 數據準備
1. **真實數據**：使用 VIA 工具標註的床單圖像和關鍵點
2. **合成數據**：使用 Blender 生成的變形布料數據
3. **數據增強**：旋轉、翻轉、顏色變化等

### 第一階段：預訓練
1. **數據載入**：載入圖像和關鍵點註解
2. **YOLO 分割**：使用微調的 YOLO 提取床單遮罩
3. **模型訓練**：使用基本優化從頭開始訓練
4. **模型儲存**：儲存預訓練模型供後訓練使用

### 第二階段：後訓練
1. **載入預訓練模型**：從第一階段結果載入
2. **進階增強**：應用旋轉、翻轉和 mixup
3. **優化訓練**：使用 torch.compile() 和早停機制訓練
4. **評估**：在測試集上評估並視覺化結果

## 🚀 部署

### 生產部署
1. **完成第一階段**：預訓練模型
2. **完成第二階段**：後訓練優化
3. **部署模型**：使用最終優化模型進行推理

### 即時推理
```python
import torch
from src.models import HybridKeypointNet

# 載入訓練模型
model = HybridKeypointNet(...)
model.load_state_dict(torch.load("models/keypoint_model_vit_post.pth"))
model.eval()

# 執行推理
with torch.no_grad():
    output = model(input_tensor)
```

## 📝 使用範例

### 完整訓練工作流程
```bash
# 步驟0：生成合成數據（可選）
cd cloth_data_gen
blender --background --python cloth_dataset_gen.py

# 步驟1：預訓練
python keypoint_detection_model_training.py

# 步驟2：後訓練優化
python post_keypoint_detection_model_training.py config_quantization_fixed.json
```

### 自訂配置
```python
# 根據需求修改 config_quantization_fixed.json
{
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.0005,
    "early_stopping_patience": 20
}
```

## 🔮 未來改進

### 計劃功能
- **TensorRT 優化**：使用 TensorRT 轉換實現2-5倍更快的推理
- **量化支援**：INT8 量化用於邊緣部署
- **模型匯出**：ONNX 和 TorchScript 匯出功能
- **進階增強**：更複雜的數據增強策略
- **主動學習**：不確定性採樣用於高效訓練

### TensorRT 整合（未來）
```bash
# 轉換為 TensorRT 以優化推理
python convert_to_tensorrt.py \
    --model_path models/keypoint_model_vit_post.pth \
    --precision fp16 \
    --test_inference

# 效能基準測試
python test_tensorrt_inference.py \
    --pytorch_model models/keypoint_model_vit_post.pth \
    --tensorrt_model models/keypoint_model_vit_post.trt
```

## 📄 授權

本專案採用 MIT 授權條款 - 詳見 LICENSE 檔案。

## 🙏 致謝

- YOLO 架構由 Ultralytics 提供
- Vision Transformer 由 Google Research 提供
- PyTorch 由 Facebook Research 提供
