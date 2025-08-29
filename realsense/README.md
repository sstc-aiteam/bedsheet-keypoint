# RealSense 工具集

這個資料夾包含用於 RealSense 深度感測器的各種工具和腳本。

## 📁 檔案說明

### 主要工具
- **`simple_xyz_capture.py`** - 簡單的 Tkinter GUI 座標擷取器（支援自訂像素座標）
- **`realsense_test.py`** - RealSense 相機測試腳本
- **`convert_depth_to_scale.py`** - 深度比例轉換工具
- **`masked_image_to_polygon_mask.py`** - 遮罩影像轉多邊形工具
- **`software_calibrate.py`** - 軟體校準工具

### 資料檔案
- **`realsense_data/`** - RealSense 資料目錄
- **`realsense_data2/`** - 額外 RealSense 資料目錄
- **`realsense_data.zip`** - RealSense 資料壓縮檔
- **`test_bin-20250818T085952Z-1-001.zip`** - 測試二進位檔案

### 文件
- **`README.md`** - 本說明文件

## 🚀 功能特色

### Simple XYZ Capture (`simple_xyz_capture.py`)
- **簡單的 Tkinter GUI 介面**
- **即時相機預覽**
- **自訂像素座標輸入** - 使用者可指定任意像素位置
- **快速預設按鈕** - 中心點、四角等常用位置
- **即時視覺回饋** - 十字準星顯示目標像素
- **自動深度比例轉換為公尺**
- **JSON 格式儲存**
- **多點座標記錄**

### RealSense Test (`realsense_test.py`)
- **RealSense 相機連接測試**
- **深度和彩色串流測試**
- **基本功能驗證**

### Depth Scale Conversion (`convert_depth_to_scale.py`)
- **深度比例轉換工具**
- **單位轉換功能**

### Mask to Polygon (`masked_image_to_polygon_mask.py`)
- **遮罩影像處理**
- **多邊形轉換功能**

## 🛠️ 安裝需求

```bash
pip install pyrealsense2 numpy opencv-python pillow
```

## 📖 使用方法

### Simple XYZ Capture

```bash
cd realsense
python simple_xyz_capture.py
```

#### 操作步驟
1. **啟動程式**：執行腳本後會自動連接 RealSense 相機
2. **查看預覽**：相機畫面會顯示在 GUI 視窗中，包含十字準星
3. **輸入像素座標**：
   - 在 X 和 Y 輸入欄位中輸入像素座標（X: 0-639, Y: 0-479）
   - 或使用快速預設按鈕（中心、四角等）
4. **擷取座標**：點擊 "Capture Coordinates" 按鈕擷取指定像素的 3D 座標
5. **查看結果**：擷取的座標會顯示在下方的文字區域
6. **重複擷取**：可以擷取多個不同像素位置的座標
7. **儲存檔案**：點擊 "Save Coordinates" 將所有座標儲存為 JSON 檔案

#### 快速預設按鈕
- **Center** (320, 240) - 影像中心點
- **Top-Left** (50, 50) - 左上角
- **Top-Right** (590, 50) - 右上角
- **Bottom-Left** (50, 430) - 左下角
- **Bottom-Right** (590, 430) - 右下角

### RealSense Test

```bash
python realsense_test.py
```

### Depth Scale Conversion

```bash
python convert_depth_to_scale.py
```

### Mask to Polygon

```bash
python masked_image_to_polygon_mask.py
```

## 🎮 GUI 介面 (Simple XYZ Capture)

### 主要元件
- **相機預覽區域**：顯示即時相機畫面，包含十字準星標示目標像素
- **像素座標輸入區域**：X 和 Y 座標輸入欄位
- **快速預設按鈕**：常用像素位置的快速設定
- **擷取按鈕**：擷取指定像素位置的 3D 座標
- **清除按鈕**：清除所有已擷取的座標
- **儲存按鈕**：將所有擷取的座標儲存為 JSON 檔案
- **狀態列**：顯示程式狀態和錯誤訊息
- **座標顯示區域**：顯示所有擷取的座標資訊

### 座標資訊
每次擷取會記錄以下資訊：
- **時間戳記**：擷取時間
- **像素座標**：影像中的像素位置 (X, Y)
- **3D 座標**：X, Y, Z 空間座標（公尺）
- **深度值**：原始深度值（公尺）

### 視覺回饋
- **藍色十字準星**：顯示當前選擇的像素位置
- **座標文字**：在預覽畫面上顯示像素座標
- **即時更新**：座標輸入時十字準星會即時移動

## 📊 座標系統

### 座標軸定義
- **X 軸** - 水平方向（左右）
- **Y 軸** - 垂直方向（上下）
- **Z 軸** - 深度方向（前後）

### 像素座標範圍
- **X 像素**：0-639（640 像素寬度）
- **Y 像素**：0-479（480 像素高度）

### 深度比例
腳本會自動檢測並應用正確的深度比例：
```python
depth_scale = depth_sensor.get_depth_scale()
```

### 座標轉換
使用 RealSense 的內建函數進行像素到 3D 座標的轉換：
```python
point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
```

## 💾 JSON 輸出格式

### 檔案結構
```json
{
  "capture_info": {
    "total_points": 3,
    "capture_date": "2024-01-15T10:30:45.123456",
    "depth_scale": 0.001,
    "camera_resolution": "640x480"
  },
  "coordinates": [
    {
      "timestamp": "2024-01-15 10:30:45",
      "x": 0.123,
      "y": -0.456,
      "z": 0.789,
      "pixel_x": 320,
      "pixel_y": 240,
      "depth_meters": 0.789
    }
  ]
}
```

### 欄位說明
- **timestamp**：擷取時間
- **x, y, z**：3D 座標（公尺）
- **pixel_x, pixel_y**：影像像素座標
- **depth_meters**：深度值（公尺）

## ⚠️ 注意事項

1. **硬體需求**：需要 RealSense 深度相機（D415, D435, L515 等）
2. **驅動程式**：確保已安裝 Intel RealSense SDK 2.0
3. **USB 連接**：使用 USB 3.0 連接以獲得最佳效能
4. **環境光線**：避免強光直射以獲得準確的深度資料
5. **像素座標範圍**：X 必須在 0-639 範圍內，Y 必須在 0-479 範圍內
6. **深度有效性**：某些像素可能沒有有效的深度資料（顯示為 0）

## 🐛 故障排除

### 常見問題

**Q: 無法偵測到 RealSense 相機**
A: 檢查 USB 連接和驅動程式安裝

**Q: 深度資料不準確**
A: 確保環境光線適中，避免反光表面

**Q: 座標為 (0, 0, 0)**
A: 表示該點沒有有效的深度資料，可能是距離太遠或表面反光

**Q: GUI 無法顯示**
A: 確保已安裝 tkinter 和 PIL 套件

**Q: 像素座標輸入錯誤**
A: 確保 X 座標在 0-639 範圍內，Y 座標在 0-479 範圍內

### 錯誤訊息
- `No RealSense device detected` - 未偵測到 RealSense 裝置
- `Failed to start pipeline` - 無法啟動串流管道
- `No valid depth at pixel (x, y)` - 指定像素沒有有效深度資料
- `Invalid pixel coordinates` - 像素座標超出有效範圍

## 🔧 自訂設定

### 修改相機解析度
如需修改相機解析度，請編輯以下程式碼：

```python
# 在 start_realsense 方法中
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
```

### 修改預設按鈕位置
如需修改快速預設按鈕的位置，請編輯以下程式碼：

```python
# 在 setup_gui 方法中
ttk.Button(preset_frame, text="Center", 
          command=lambda: self.set_pixel_coords(320, 240))
```
