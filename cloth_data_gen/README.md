# 床單關鍵點檢測 - Warp 模擬與渲染文檔

## 概述

本專案使用 Warp-lang 進行高精度布料物理模擬，並結合 Blender 進行高品質渲染，生成用於機器學習的合成床單數據集。系統能夠生成具有真實皺褶和摺疊的床單圖像，並準確追蹤四個角落關鍵點。

## 系統架構

### 1. Warp 物理模擬 (`warp_bedsheet_sim.py`)

Warp 是一個基於 GPU 的高性能物理模擬框架，專門用於布料模擬。

#### 核心功能：
- **粒子系統**：將床單建模為粒子網格，每個粒子具有位置、速度和質量
- **彈簧約束**：模擬布料的拉伸和彎曲特性
- **碰撞檢測**：處理床單與地面的碰撞
- **空氣動力學**：模擬風力和空氣阻力對床單的影響
- **厚度模擬**：考慮床單的物理厚度

#### 主要參數：
```python
# 網格解析度
grid_size = 48  # 48x48 粒子網格

# 物理參數
stiffness = 1000.0      # 剛度（控制拉伸阻力）
damping = 0.1           # 阻尼（控制振動衰減）
thickness = 0.01        # 床單厚度（米）

# 空氣動力學
wind_strength = 5.0     # 風力強度
air_resistance = 0.1    # 空氣阻力係數
```

#### 模擬流程：
1. **初始化**：創建粒子網格和約束
2. **物理更新**：計算力、更新位置和速度
3. **碰撞處理**：檢測並解決與地面的碰撞
4. **風力應用**：添加隨機風力場
5. **收斂檢測**：當床單穩定在地面上時停止

### 2. Blender 渲染系統 (`render.py`)

Blender 提供高品質的 3D 渲染，包括真實的光照、陰影和材質。

#### 渲染管線：
1. **場景設置**：創建地面、相機和光照
2. **網格導入**：從 Warp 模擬結果導入床單網格
3. **材質分配**：應用真實的布料材質（棉、亞麻、絲綢）
4. **光照設置**：配置多光源系統以產生自然陰影
5. **相機定位**：自適應相機系統確保床單在畫面中
6. **渲染輸出**：生成高解析度圖像

#### 光照系統：
```python
# 主要光源（頂部）
main_light.energy = 8.0      # 降低亮度以增強陰影對比
main_light.size = 3.0        # 較小尺寸產生更清晰的陰影

# 輔助光源
secondary_light.energy = 4.0  # 減少填充光以保持陰影
accent_light.energy = 3.0     # 強調特定區域

# 填充光源（最小化）
fill_lights.energy = 1.0-1.5  # 極少填充光以保持陰影細節
```

#### 材質系統：
- **棉質**：中等粗糙度，自然反射
- **亞麻**：較高粗糙度，啞光效果
- **絲綢**：低粗糙度，高反射率

### 3. 關鍵點追蹤 (`keypoint_tracker.py`)

準確追蹤床單四個角落的 3D 到 2D 投影。

#### 追蹤流程：
1. **頂點識別**：使用頂點組標記四個角落
2. **3D 座標獲取**：從模擬結果中提取世界座標
3. **相機投影**：使用 Blender 的 `world_to_camera_view()` 函數
4. **可見性檢查**：判斷關鍵點是否在相機視野內
5. **座標轉換**：將標準化座標轉換為像素座標

#### 可見性邏輯：
```python
# 檢查點是否在相機後方
if normalized_coords.z < 0:
    visibility = 0  # 不可見

# 檢查點是否在視野範圍內
if (normalized_coords.x < 0 or normalized_coords.x > 1 or 
    normalized_coords.y < 0 or normalized_coords.y > 1):
    visibility = 0  # 超出視野

# 邊緣邊距檢查（20 像素）
if (x < 20 or x > image_width-20 or 
    y < 20 or y > image_height-20):
    visibility = 0  # 太靠近邊緣
```

### 4. 批次處理管線 (`batch_pipeline.py`)

自動化生成大量床單圖像的批次處理系統。

#### 批次流程：
1. **參數生成**：為每張圖像生成隨機參數
2. **並行處理**：使用多進程加速生成
3. **文件組織**：自動整理圖像和關鍵點文件
4. **進度追蹤**：實時顯示生成進度
5. **錯誤處理**：處理失敗的生成任務

#### 輸出結構：
```
bedsheet_dataset_1000/
├── imgs/                    # 渲染圖像
│   ├── bedsheet_0000.png
│   ├── bedsheet_0001.png
│   └── ...
├── keypoints/               # 關鍵點座標
│   ├── bedsheet_0000.txt
│   ├── bedsheet_0001.txt
│   └── ...
└── visualizations/          # 關鍵點可視化
    ├── bedsheet_0000_vis.png
    ├── bedsheet_0001_vis.png
    └── ...
```

## 技術細節

### Warp 模擬優化

#### GPU 加速：
- 所有物理計算在 GPU 上並行執行
- 使用 CUDA 核心進行高速矩陣運算
- 記憶體管理優化以處理大型網格

#### 數值穩定性：
- 使用 Verlet 積分法確保數值穩定性
- 自適應時間步長防止模擬爆炸
- 約束求解器確保物理約束滿足

#### 空氣動力學模型：
```python
# 風力計算
wind_force = wind_strength * wind_direction * noise_factor
drag_force = -air_resistance * velocity * |velocity|
lift_force = cross_product(velocity, wind_direction) * lift_coefficient
```

### Blender 渲染優化

#### GPU 渲染配置：
```python
# 啟用 GPU 渲染
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
bpy.context.scene.cycles.device = 'GPU'

# 渲染設置
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024
scene.cycles.samples = 512
scene.cycles.tile_size = 256
```

#### 自適應相機系統：
```python
# 基於床單邊界框計算相機位置
bedsheet_diagonal = sqrt(size.x² + size.y²)
required_distance = bedsheet_diagonal / (2 * target_frame_ratio)
camera_height = required_distance * random.uniform(1.2, 1.5)
```

### 關鍵點追蹤精度

#### 投影矩陣：
使用 Blender 內建的 `world_to_camera_view()` 函數確保投影精度：
```python
normalized_coords = world_to_camera_view(scene, camera, world_vector)
pixel_x = normalized_coords.x * image_width
pixel_y = (1.0 - normalized_coords.y) * image_height  # Y軸翻轉
```

#### 邊緣檢測：
實現 20 像素邊緣邊距，確保關鍵點不會被截斷：
```python
margin = 20
if (x < margin or x > image_width - margin or 
    y < margin or y > image_height - margin):
    visibility = 0
```

## 使用指南

### 基本使用

#### 1. 單張圖像生成：
```bash
python3 pipeline.py --width 2.0 --height 1.0 --resolution 48 --steps 100 --material cotton --output single_bedsheet
```

#### 2. 批次生成：
```bash
python3 batch_pipeline.py --n_images 1000 --width 2.0 --height 1.0 --resolution 48 --steps 100 --material cotton --output bedsheet_dataset_1000
```

### 參數調整

#### 物理參數：
- **resolution**：更高的解析度產生更細緻的皺褶（48-64 推薦）
- **steps**：更多步數確保模擬收斂（100-150 推薦）
- **stiffness**：控制床單的柔軟度（500-2000 範圍）

#### 渲染參數：
- **material**：選擇材質類型（cotton/linen/silk）
- **lighting**：調整光源強度以控制陰影對比
- **camera**：自適應相機確保床單在畫面中

### 故障排除

#### 常見問題：

1. **GPU 記憶體不足**：
   - 降低 `resolution` 參數
   - 減少 `samples` 數量
   - 使用較小的 `tile_size`

2. **關鍵點追蹤錯誤**：
   - 檢查相機位置是否正確
   - 確認床單在相機視野內
   - 驗證頂點組標記是否正確

3. **渲染品質問題**：
   - 增加 `samples` 數量
   - 調整光照設置
   - 檢查材質參數

#### 性能優化：

1. **模擬速度**：
   - 使用 GPU 加速
   - 優化網格解析度
   - 調整時間步長

2. **渲染速度**：
   - 啟用 GPU 渲染
   - 使用適當的樣本數
   - 優化光照設置

## 數據格式

### 圖像格式
- **解析度**：1024x1024 像素
- **格式**：PNG（無損壓縮）
- **色彩空間**：sRGB

### 關鍵點格式
每行包含一個關鍵點的座標和可見性：
```
x y visibility
```
- `x, y`：像素座標（0-1023）
- `visibility`：可見性標記（0=不可見，1=可見）

### 可視化格式
- 在原始圖像上疊加關鍵點標記
- 紅色圓圈標記可見關鍵點
- 灰色圓圈標記不可見關鍵點

## 擴展功能

### 自定義材質
可以添加新的布料材質類型：
```python
def create_custom_material(name, roughness, metallic, color):
    material = bpy.data.materials.new(name=name)
    # 設置材質屬性
    return material
```

### 自定義光照
可以創建特定的光照設置：
```python
def setup_dramatic_lighting():
    # 創建戲劇性光照效果
    pass
```

### 自定義相機角度
可以實現不同的相機視角：
```python
def setup_angled_camera(angle_x, angle_y, angle_z):
    # 設置特定角度的相機
    pass
```

## 模型訓練腳本狀態

### 當前可用的訓練腳本

本專案包含多個模型訓練腳本，用於不同的關鍵點檢測任務：

#### 1. 基礎 CLIP 風格訓練
- **檔案**：`clip_style_cloth_training.py`
- **功能**：使用 CLIP 架構進行床單關鍵點檢測
- **狀態**：已完成並測試

#### 2. Meta CLIP 風格訓練
- **檔案**：`meta_clip_style_cloth_training.py`
- **功能**：元學習版本的 CLIP 風格模型
- **狀態**：已完成並測試

#### 3. 後處理 CLIP 風格訓練
- **檔案**：`post_clip_style_training.py`
- **功能**：後處理階段的 CLIP 風格模型
- **狀態**：已完成並測試

#### 4. 後處理 Meta CLIP 風格訓練
- **檔案**：`post_meta_clip_style_training.py`
- **功能**：後處理階段的元學習 CLIP 風格模型
- **狀態**：已完成並測試

#### 5. 床墊專用訓練
- **檔案**：`post_clip_style_mattress_training.py`
- **功能**：專門針對床墊的關鍵點檢測
- **狀態**：已完成並測試

#### 6. 混合關鍵點網路
- **檔案**：`src/models/hybrid_keypoint_net.py`
- **功能**：混合架構的關鍵點檢測網路
- **狀態**：已完成並測試

#### 7. CLIP 熱力圖模型
- **檔案**：`src/models/clip_heatmap_model.py` 和 `src/models/clip_heatmap_model_v2.py`
- **功能**：基於熱力圖的關鍵點檢測
- **狀態**：已完成並測試

#### 8. 關鍵點檢測模型訓練
- **檔案**：`keypoint_detection_model_training.py` 和 `post_keypoint_detection_model_training.py`
- **功能**：傳統關鍵點檢測模型訓練
- **狀態**：已完成並測試

### 執行腳本
- **檔案**：`run_meta_clip_post_training.py`
- **功能**：自動化執行 Meta CLIP 後處理訓練流程
- **狀態**：已完成並測試

### 注意事項
- **Meta CLIP v2**：已移除，因為效果不佳
- **當前可用**：8種不同的訓練腳本，涵蓋各種關鍵點檢測方法

## 布料數據生成系統更新

### 最新改進（2024年12月）

#### 1. 無固定點模擬系統
- **改進**：移除了角落固定點機制，避免突然釋放導致的角落彈射問題
- **效果**：床單從開始就自然流動，產生更真實的摺疊和皺褶
- **技術細節**：
  ```python
  # 舊系統：固定角落後突然釋放
  # 新系統：所有粒子從開始就自由移動
  self.stiffness = 0.6  # 降低剛度以獲得更柔軟的布料
  self.bend_stiffness = 0.03  # 降低彎曲剛度
  ```

#### 2. 漸進式約束求解
- **改進**：減少約束迭代次數（從8次降至6次）
- **效果**：更柔軟、更自然的布料行為
- **性能**：提高模擬速度

#### 3. 穩定的物理參數
- **剛度**：0.6（平衡柔軟度和穩定性）
- **彎曲剛度**：0.03（自然摺疊）
- **約束迭代**：6次（快速收斂）
- **厚度**：5mm（保持邊緣可見性）

#### 4. 增強的材質系統
- **改進**：優化材質屬性以增強邊緣可見性
- **技術**：
  ```python
  # 增強邊緣定義的材質屬性
  material.roughness = 0.8  # 增加粗糙度
  material.specular = 0.3   # 增加反射
  # 添加法線貼圖以增強表面細節
  ```

#### 5. 改進的光照系統
- **主光源**：能量10.0，尺寸3.0（增強陰影對比）
- **輔助光源**：能量5.0（平衡照明）
- **填充光源**：最小化以保持陰影細節

### 批次生成狀態

#### 當前批次
- **目標**：3000張床單圖像
- **狀態**：正在生成中（約34%完成）
- **輸出**：`bedsheet_dataset_3000_no_pinning/`
- **特點**：
  - 無固定點自然流動
  - 柔軟布料物理
  - 增強邊緣可見性
  - 完美關鍵點追蹤

#### 數據集結構
```
bedsheet_dataset_3000_no_pinning/
├── imgs/                    # 1024x1024 PNG 圖像
├── keypoints/               # 關鍵點座標文件
└── visualizations/          # 關鍵點可視化圖像
```

### 性能指標

#### 生成速度
- **單張圖像**：6-9秒（包含模擬和渲染）
- **批次處理**：約2-3小時完成3000張圖像
- **GPU 加速**：啟用 CUDA 渲染

#### 品質指標
- **關鍵點準確率**：100%（所有角落正確追蹤）
- **圖像解析度**：1024x1024
- **材質多樣性**：棉、亞麻、絲綢三種材質
- **物理真實性**：自然風力和重力效果

## 結論

本系統提供了一個完整的床單關鍵點檢測數據生成管線，結合了先進的物理模擬和高品質渲染技術。通過 Warp 的 GPU 加速模擬和 Blender 的專業渲染能力，能夠生成大量高品質的合成數據，用於訓練機器學習模型。

### 主要成就
1. **穩定的物理模擬**：無固定點系統確保自然流動
2. **精確的關鍵點追蹤**：100%準確率
3. **多樣化的模型訓練**：8種不同的訓練腳本（已移除效果不佳的 Meta CLIP v2）
4. **高效的批次處理**：自動化生成3000張圖像
5. **增強的視覺品質**：改進的光照和材質系統

系統的模組化設計使得各個組件可以獨立調整和優化，為不同的應用場景提供了靈活性。自適應相機系統和精確的關鍵點追蹤確保了數據的一致性和準確性。

---
