 i# 床單摺疊機器人 - 關鍵點檢測系統

一個使用深度學習和電腦視覺技術的床單摺疊機器人關鍵點檢測系統，整合了先進的 Meta CLIP 模型和合成數據生成技術。

## 🚀 功能特色

- **Meta CLIP 關鍵點檢測**：基於 Meta CLIP 的視覺-語言模型，具備強大的表示能力
- **公平模型比較**：確保原始模型與預訓練模型具有相同的可訓練參數數量
- **合成數據生成**：使用 Warp-lang 和 Blender 生成高品質的床單訓練數據
- **兩階段訓練流程**：預訓練 + 後訓練優化，最大化模型性能
- **參數高效微調**：使用 PEFT 和 LoRA 技術，節省 95% 記憶體
- **即時推理優化**：支援 TensorRT 轉換，實現 2-5 倍推理加速
- **全面數據處理**：YOLO 分割 + 關鍵點註解流程

## 📁 專案結構

```
bedsheet-keypoint/
├── src/                              # 核心模型架構
│   ├── models/
│   │   ├── clip_heatmap_model.py     # Meta CLIP 熱力圖模型
│   │   ├── clip_heatmap_model_v2.py  # Meta CLIP 模型 v2
│   │   └── hybrid_keypoint_net.py    # 混合關鍵點網路
│   ├── utils/                        # 工具函數
│   └── training/                     # 訓練流程
├── cloth_data_gen/                   # 合成數據生成
│   ├── warp_sim.py                   # Warp-lang 布料模擬
│   ├── render.py                     # Blender 渲染引擎
│   ├── keypoint_tracker.py           # 關鍵點追蹤
│   ├── batch_pipeline.py             # 批次生成管道
│   ├── bedsheet_dataset_3000/        # 3000 張床單數據集
│   │   ├── imgs/                     # 床單圖像
│   │   ├── keypoints/                # 關鍵點註解
│   │   └── visualizations/           # 視覺化結果
│   └── README_zh_TW.md              # 數據生成說明文件
├── models/                           # 訓練模型和權重
│   ├── meta_clip_style_cloth/        # Meta CLIP 預訓練模型
│   ├── meta_clip_style_bedsheet_post_original/    # 原始模型後訓練
│   ├── meta_clip_style_bedsheet_post_pretrained/  # 預訓練模型後訓練
│   └── yolo_finetuned/               # YOLO 微調模型
├── results_meta_clip_bedsheet_post_original/      # 原始模型結果
├── results_meta_clip_bedsheet_post_pretrained/    # 預訓練模型結果
├── meta_clip_style_cloth_training.py              # Meta CLIP 預訓練腳本
├── post_meta_clip_style_training.py               # Meta CLIP 後訓練腳本
├── compare_meta_clip_models.py                    # 模型比較工具
├── keypoint_detection_model_training.py           # 混合網路預訓練
├── post_keypoint_detection_model_training.py      # 混合網路後訓練
├── tensorrt_demo.py                               # TensorRT 演示
└── requirements.txt                               # 依賴套件列表
```

## 🛠️ 安裝

### 1. 環境設置

```bash
# 複製儲存庫
git clone <repository-url>
cd bedsheet-keypoint

# 創建虛擬環境
python -m venv pytorch_env
source pytorch_env/bin/activate  # Linux/Mac
# 或 pytorch_env\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt
```

### 2. 安裝 Blender（用於數據生成）

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install blender

# 或從官網下載：https://www.blender.org/download/
```

### 3. 安裝 Warp-lang（用於布料模擬）

```bash
pip install warp-lang
```

## 🎯 快速開始

### 步驟 1：生成合成訓練數據

使用 Warp-lang 和 Blender 生成高品質的床單數據：

```bash
# 進入數據生成目錄
cd cloth_data_gen

# 生成 3000 張床單圖像（已預生成）
# 如需重新生成，執行：
python batch_pipeline.py --n_images 3000
```

#### 數據生成特色
- **Warp-lang 物理模擬**：真實的布料物理行為
- **多方向風力**：模擬自然風力對床單的影響
- **自適應相機**：確保床單始終在畫面中央
- **關鍵點追蹤**：精確的 3D 到 2D 座標轉換
- **視覺化驗證**：自動生成關鍵點視覺化圖像

### 步驟 2：Meta CLIP 預訓練

使用合成數據進行 Meta CLIP 模型的預訓練：

```bash
# Meta CLIP 預訓練
python meta_clip_style_cloth_training.py
```

#### 預訓練配置
```python
# 主要配置參數
config = {
    'data_dir': 'cloth_data_gen/bedsheet_dataset_3000',  # 合成數據目錄
    'image_size': 256,                                   # 輸入圖像尺寸
    'max_samples': None,                                 # 使用所有 3000 個樣本
    'num_epochs': 20,                                    # 訓練輪數
    'batch_size': 8,                                     # 批次大小
    'learning_rate': 3e-4,                               # 學習率
    'use_lora': True,                                    # 啟用 LoRA
    'lora_r': 16,                                        # LoRA 秩
    'lora_alpha': 32,                                    # LoRA 縮放參數
    'lora_dropout': 0.05,                                # LoRA dropout
    'use_text_prior': True,                              # 啟用文字先驗
    'prior_weight': 0.3                                  # 先驗權重
}
```

### 步驟 3：Meta CLIP 後訓練

在真實床單數據上進行後訓練微調：

```bash
# 比較兩個模型以確保公平比較
python post_meta_clip_style_training.py --compare

# 訓練原始 Meta CLIP 模型
python post_meta_clip_style_training.py --use_original --epochs 20

# 訓練預訓練 Meta CLIP 模型
python post_meta_clip_style_training.py --epochs 20
```

#### 公平比較特色
- **參數數量匹配**：確保兩個模型具有相同的可訓練參數（2,900,609 個）
- **智能 LoRA 處理**：自動處理 PEFT 載入失敗情況
- **詳細比較報告**：提供完整的參數比較和前向傳播測試
- **命令列介面**：支援多種訓練配置選項

## 🏗️ 模型架構

### 1. YOLO + Vision Transformer 混合模型（傳統方法）

本專案最初使用 YOLO + ViT 混合架構進行關鍵點檢測，這是一個兩階段訓練的深度學習模型。

#### 架構組成
- **骨幹網路**：YOLO11L-pose（前12層）作為特徵提取器
- **融合層**：多尺度特徵融合（MultiScaleFusion）
- **編碼器**：Vision Transformer (ViT-B) 進行特徵編碼
- **解碼器**：上採樣解碼器生成熱力圖
- **輸出**：基於熱力圖的關鍵點預測

#### 技術特色
- **增強 YOLO 骨幹**：使用完整的 YOLO 架構（骨幹 + 頸部）
- **跳躍連接**：適當的跳躍連接處理
- **多尺度特徵金字塔**：FPN 特徵的戰略選擇
- **空間軟最大化**：用於關鍵點定位的空間軟最大化
- **高斯模糊**：智能高斯模糊用於熱力圖平滑

#### 模型參數
- **總參數**：約 100M 參數
- **輸入尺寸**：128x128 RGB 圖像
- **輸出格式**：128x128 熱力圖
- **關鍵點數量**：4 個角點

#### 訓練流程
```python
# 第一階段：預訓練
python keypoint_detection_model_training.py

# 第二階段：後訓練優化
python post_keypoint_detection_model_training.py config_quantization_fixed.json
```

#### 配置選項
```json
{
    "model_name": "HybridKeypointNet",
    "pretrained_model_path": "models/keypoint_model_vit.pth",
    "yolo_model_path": "models/yolo_finetuned/best.pt",
    "batch_size": 16,
    "learning_rate": 0.001,
    "num_epochs": 50,
    "use_mixup": true,
    "early_stopping_patience": 10
}
```

### 2. Meta CLIP 關鍵點檢測模型（先進方法）

- **基礎模型**：Meta CLIP B16 (facebook/metaclip-b16-fullcc2.5b)
- **參數效率**：使用 LoRA 微調，僅訓練 1.9% 的參數
- **輸出格式**：基於熱力圖的關鍵點預測
- **輸入尺寸**：256x256 RGB 圖像
- **總參數**：152.5M 參數
- **可訓練參數**：2.9M 參數（LoRA 適配器）

### 主要特色

- **視覺-語言融合**：結合 CLIP 的視覺編碼器和文字先驗
- **智能文字先驗**：使用正面和負面提示區分真正的角落和邊緣
- **參數高效微調**：使用 PEFT 和 LoRA 技術
- **公平模型比較**：確保原始模型與預訓練模型具有相同參數數量
- **自動參數匹配**：智能處理 PEFT 結構造成的參數名稱差異

### LoRA 配置

```python
# LoRA 配置
lora_config = {
    'r': 16,                    # 低秩適應的秩
    'lora_alpha': 32,           # LoRA 縮放參數
    'target_modules': [         # 目標模組
        'q_proj', 'v_proj',     # 注意力查詢和值投影
        'k_proj', 'out_proj'    # 注意力鍵和輸出投影
    ],
    'lora_dropout': 0.05,       # LoRA dropout
    'bias': 'none',             # 偏置設定
    'task_type': 'FEATURE_EXTRACTION'
}
```

## 📊 性能比較

### 模型比較結果

| 模型類型 | 可訓練參數 | 預訓練數據 | 後訓練數據 | 參數效率 | 推理速度 |
|----------|------------|------------|------------|----------|----------|
| 原始 Meta CLIP | 2,900,609 | 無 | 真實床單 | 標準 | 中等 |
| 預訓練 Meta CLIP | 2,900,609 | 合成床單 | 真實床單 | 高效 (LoRA) | 快速 |

### 訓練優化

- **torch.compile()**：約 20-30% 更快的訓練
- **混合精度**：FP16 訓練以節省記憶體
- **梯度裁剪**：穩定的梯度裁剪訓練
- **學習率調度**：自適應學習率調度
- **早停機制**：基於驗證損失平台的自動訓練終止

## 🔧 配置選項

### Meta CLIP 後訓練配置

```python
DEFAULT_CONFIG = {
    # 模型配置
    "model_name": "facebook/metaclip-b16-fullcc2.5b",
    "pretrained_model_path": "models/meta_clip_style_cloth",
    "use_original_metaclip": False,  # 使用預訓練模型
    "ensure_equal_params": True,     # 確保參數數量匹配
    
    # 數據配置
    "keypoints_data_srcs": ["via_proj/bedsheets"],
    "image_paths": ["image_data/RGB-images", "image_data/RGB-images2"],
    "yolo_model_path": "models/yolo_finetuned/best.pt",
    "allowed_classes": [1],  # 床單類別
    "image_size": 256,
    
    # 訓練配置
    "batch_size": 4,
    "num_epochs": 20,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "use_fp16": True,
    "use_lora": True,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
}
```

### 命令列選項

```bash
# 比較兩個模型
python post_meta_clip_style_training.py --compare

# 使用原始 Meta CLIP 模型
python post_meta_clip_style_training.py --use_original --epochs 20

# 使用預訓練 Meta CLIP 模型
python post_meta_clip_style_training.py --epochs 20 --lr 1e-4

# 禁用參數匹配（舊行為）
python post_meta_clip_style_training.py --disable_equal_params
```

## 📈 訓練流程

### 第一階段：合成數據預訓練

1. **數據生成**：使用 Warp-lang 和 Blender 生成 3000 張床單圖像
2. **物理模擬**：真實的布料物理行為和風力效果
3. **關鍵點追蹤**：精確的 3D 到 2D 座標轉換
4. **模型預訓練**：在合成數據上訓練 Meta CLIP 模型
5. **模型儲存**：儲存預訓練模型供後訓練使用

### 第二階段：真實數據後訓練

1. **公平比較設置**：確保原始模型與預訓練模型參數數量相同
2. **預訓練模型載入**：從第一階段載入已訓練的 Meta CLIP 模型
3. **LoRA 微調**：僅微調 LoRA 適配器，保持原始 CLIP 權重
4. **性能評估**：在測試集上評估並比較兩個模型的性能
5. **結果視覺化**：生成關鍵點檢測結果的視覺化圖像

## 🚀 部署

### 生產部署

```bash
# 完成預訓練
python meta_clip_style_cloth_training.py

# 完成後訓練
python post_meta_clip_style_training.py --epochs 20

# 部署模型進行推理
python tensorrt_demo.py
```

### 即時推理

```python
import torch
from src.models import ClipHeatmapModel

# 載入訓練模型
model = ClipHeatmapModel(...)
model.load_state_dict(torch.load("models/meta_clip_style_bedsheet_post/head.pth"))
model.eval()

# 執行推理
with torch.no_grad():
    output = model(input_tensor)
```

## 📝 使用範例

### 完整 Meta CLIP 訓練流程

```bash
# 步驟 1：生成合成數據（已預生成 3000 張）
cd cloth_data_gen
# python batch_pipeline.py --n_images 3000  # 如需重新生成

# 步驟 2：Meta CLIP 預訓練
python meta_clip_style_cloth_training.py

# 步驟 3：比較模型參數
python post_meta_clip_style_training.py --compare

# 步驟 4：後訓練原始模型
python post_meta_clip_style_training.py --use_original --epochs 20

# 步驟 5：後訓練預訓練模型
python post_meta_clip_style_training.py --epochs 20
```

### 自訂配置

```python
# 修改 post_meta_clip_style_training.py 中的配置
DEFAULT_CONFIG = {
    'use_original_metaclip': False,    # 使用預訓練模型
    'ensure_equal_params': True,       # 確保參數匹配
    'num_epochs': 30,                  # 增加訓練輪數
    'batch_size': 8,                   # 調整批次大小
    'learning_rate': 1e-4,             # 調整學習率
    'lora_r': 32,                      # 增加 LoRA 秩
    'lora_alpha': 64,                  # 調整縮放參數
}
```

## 🔮 未來改進

### 計劃功能

- **TensorRT 優化**：使用 TensorRT 轉換實現 2-5 倍更快的推理
- **量化支援**：INT8 量化用於邊緣部署
- **模型匯出**：ONNX 和 TorchScript 匯出功能
- **進階增強**：更複雜的數據增強策略
- **主動學習**：不確定性採樣用於高效訓練
- **多尺度檢測**：支援不同尺寸的床單檢測

### TensorRT 整合

```bash
# 轉換為 TensorRT 以優化推理
python convert_to_tensorrt.py \
    --model_path models/meta_clip_style_bedsheet_post \
    --precision fp16 \
    --test_inference

# 效能基準測試
python test_tensorrt_inference.py \
    --pytorch_model models/meta_clip_style_bedsheet_post \
    --tensorrt_model models/meta_clip_style_bedsheet_post.trt
```

## 🙏 致謝

- Meta CLIP 架構由 Meta AI 提供
- Warp-lang 由 NVIDIA 提供
- Blender 由 Blender Foundation 提供
- PyTorch 由 Facebook Research 提供
- PEFT 和 LoRA 技術由 Hugging Face 提供

## 📞 支援

如有問題或建議，請提交 Issue 或聯繫開發團隊。
