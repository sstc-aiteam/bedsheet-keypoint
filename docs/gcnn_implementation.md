# GCNN / Steerable Layers 實作文件（C4 + SO(2)/E(2) 取樣版）

本文件完整說明本 repo 目前「群論/可操縱（steerable）」層的 **原生 PyTorch** 實作與使用方式，包含：
- **離散 C4 GCNN**（0/90/180/270 度）
- **SO(2) / 任意角度旋轉（取樣近似）**：用 `affine_grid + grid_sample` 旋轉 kernel
- **SO(2) group conv 疊層**：在角度維度上做 twist-shift（G→G）而不是一 lifting 就立刻 pooling

概念參考：
- `docs/steerable_cnn_math.md` / `docs/steerable_cnn_from_scratch.py`（E(2)/SO(2) 理解、kernel 旋轉與近似）

---

## 1. 一句話總結

我們把「旋轉對稱性」寫進 head/decoder：
- **C4**：對 90 度倍數旋轉是「精準」等變（不需要插值）
- **SO(2) 取樣**：用 A 個角度 bin 逼近任意角度旋轉（需要插值，A 越大越接近）

---

## 2. 檔案位置（你要看的重點）

### 2.1 C4（離散）GCNN

- `src/models/blocks/gcnn_groups.py`
  - `CyclicGroupC4`
- `src/models/blocks/gcnn_layers.py`
  - `LiftingConvC4`（R² → G）
  - `GroupConvC4`（G → G）
  - `GroupPooling`（G → R²）

### 2.2 SO(2)（取樣近似）Steerable / Group layers

- `src/models/blocks/so2_layers.py`
  - `_rotate_kernels_grid_sample`：任意角度旋轉 kernel（`affine_grid` + `grid_sample`）
  - `LiftingConvSO2`：`(B,C,H,W) -> (B,C,A,H',W')`
  - `GroupConvSO2`：`(B,C,A,H,W) -> (B,C,A,H',W')`（twist-shift）

> 註：SO(2) 這裡是「用 A 個角度近似」的版本，本質上等同把 SO(2) 以 C\_A 逼近。

可視化架構圖（含張量形狀、twist-shift 與 kernel rotation 流程）：`docs/gcnn_so2_architecture.md`

### 2.3 已整合到 CLIP heatmap head

- `src/models/clip_heatmap_model.py`
  - `ClipHeatmapHead(..., use_gcnn=True, gcnn_mode="c4"|"so2", ...)`
  - `ClipHeatmapModel(..., head_* 參數 ...)`
  - `create_clip_heatmap_model(..., head_* 參數 ...)`

---

## 3. 張量形狀總覽（最重要）

記號：
- **B**：batch
- **C**：channels
- **H,W**：空間
- **A**：角度樣本數（SO2 的 angle bins）

### 3.1 C4

- Lifting：`(B, C_in, H, W) -> (B, C_out, 4, H', W')`
- GroupConv：`(B, C_in, 4, H, W) -> (B, C_out, 4, H', W')`
- Pool：`(B, C, 4, H, W) -> (B, C, H, W)`

### 3.2 SO(2)（取樣近似）

- Lifting：`(B, C_in, H, W) -> (B, C_out, A, H', W')`
- GroupConv：`(B, C_in, A, H, W) -> (B, C_out, A, H', W')`
- Pool：`(B, C, A, H, W) -> (B, C, H, W)`

---

## 4. C4：我們怎麼做（純離散，無插值）

### 4.1 LiftingConvC4（R² → G）

核心想法：
- 對每個群元素 `h`，用 `h^{-1}` 去旋轉 kernel（left-regular）
- 使用 `torch.rot90` 產生 4 組 kernel
- 把 group 維度摺進 output channels，用一次 `F.conv2d` 算完，再 reshape 回來

### 4.2 GroupConvC4（G → G）

核心想法（twist-shift / left-regular）：
- 對 output group index `i`、input group index `j`：
  - 相對元素 `rel = (j - i) mod 4`（對應 \(g^{-1}h\)）
  - 取 `weight[:,:,rel]`
  - 再做空間旋轉（由 \(g^{-1}\) 決定）

---

## 5. SO(2)/E(2) 取樣版：我們怎麼做（任意角度）

### 5.1 為什麼需要插值？

任意角度旋轉不是 90 度倍數，所以不能用 `rot90`。
我們採用「變換 grid + 插值採樣」策略：
- `affine_grid` 產生旋轉後的座標網格
- `grid_sample` 在這個網格上對 kernel 做雙線性取樣（bilinear）

### 5.2 LiftingConvSO2（R² → R²×H）

做法：
- 固定 A 個角度：\(\theta_k = 2\pi k / A\)
- 對每個角度，用 `-theta` 旋轉 kernel（left-regular）
- 把角度維度摺入 output channel，用一次 `F.conv2d` 計算，再 reshape 回 `(B, C_out, A, H', W')`

### 5.3 GroupConvSO2（R²×H → R²×H）

這一層讓我們能在 pooling 前「保留角度維度」做多層 G→G：
- 相對角度 index：`rel = (j - i) mod A`
- 空間旋轉由 output angle `i` 的 \(g^{-1}\) 決定：旋轉 kernel \(-\theta_i\)

> 注意：目前 `GroupConvSO2` 是 **O(A²)**（以 head/decoder 為主），A=8/16 通常可用；A 再往上會變慢很多。

### 5.4 CUDA/cuDNN 注意事項（很實務）

`grid_sample` 在某些 CUDA/cuDNN + AMP 組合下會觸發：
`CUDNN_STATUS_NOT_SUPPORTED`

我們在 `_rotate_kernels_grid_sample` 做了強健處理：
- 強制 input/grid **contiguous**
- `grid_sample` 使用 **fp32**
- 若仍觸發 cuDNN error，會對該次呼叫**暫時關閉 cuDNN**再重試

---

## 6. ClipHeatmapHead：怎麼啟用（C4 / SO2）

你主要會用到 `create_clip_heatmap_model(...)` 的這些參數：
- `head_use_gcnn`: bool
- `head_gcnn_mode`: `"c4"` 或 `"so2"`
- `head_gcnn_hidden`: int（例如 32/64/128）
- `head_so2_num_angles`: A（例如 8/16/32）
- `head_so2_num_gconvs`: SO2 group conv 層數（例如 1~4）

### 6.1 SO2 head 的目前流程

對 `feat_2d: (B, D, h, w)`：
- `LiftingConvSO2`: `(B, hidden, A, h, w)`
- resize（保留 A）
- `GroupConvSO2` × `head_so2_num_gconvs`
- `GroupPooling(mean)`：`(B, 64, out_size, out_size)`
- `1x1 conv -> sigmoid`：`(B, 1, out_size, out_size)`

---

## 7. 什麼時候不是加越多越好？

### 7.1 SO2 不一定總是贏

- 資料/標籤不符合「旋轉對稱」時（例如語意上有固定方向）
- A 太小（角度 bin 太少）時近似誤差大，插值造成 blur/aliasing
- A 太大或 gconvs 太多時，算力/顯存成本會上升很快

### 7.2 Decoder vs “更低層”

一般而言把旋轉等變性放更早更有力；但在本 repo 的現實是：
- backbone（CLIP ViT）不是等變設計，直接改 backbone 成本很高
- 因此先在 head/decoder 做「對稱性補正」是合理工程折衷

---

## 8. 建議的超參數起手式

- **快速試驗**：`mode=so2`, `A=8`, `gconvs=1~2`, `hidden=32`
- **更穩/更準**：`A=16`, `gconvs=2~4`, `hidden=64`
- 若訓練變慢或顯存吃緊：先降 `A` 或 `gconvs`，再調 `hidden`



