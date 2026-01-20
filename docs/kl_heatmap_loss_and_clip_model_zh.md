# KL Heatmap Divergence Loss 與 CLIP Heatmap Model 架構詳解

本文檔詳細解釋了 `kl-heatmap-divergence-loss` 的數學原理與實現，以及當前 `ClipHeatmapModel` 的模型架構設計。

## 1. KL Heatmap Divergence Loss (KL 熱圖散度損失)

### 1.1 概念與背景
傳統的關鍵點檢測常用均方誤差 (MSE) 作為損失函數，要求模型回歸出的熱圖與真值高斯熱圖在像素級別上完全一致。然而，MSE 對極值點非常敏感，且難以處理標註不確定性或「長尾」分佈。

`kl-heatmap-divergence-loss` 將關鍵點預測問題視為**概率分佈匹配 (Probability Distribution Matching)** 問題。我們希望模型預測的空間概率分佈 $P$ 盡可能接近真實的空間概率分佈 $Q$。Kullback-Leibler (KL) 散度正是衡量兩個概率分佈差異的指標。

### 1.2 數學原理
給定輸入圖像的真值熱圖 $H_{gt}$ 和預測熱圖 $H_{pred}$ (形狀均為 $H \times W$)，我們首先將它們歸一化為概率分佈：

$$
Q(i,j) = \frac{H_{gt}(i,j)}{\sum_{x,y} H_{gt}(x,y) + \epsilon}
$$
$$
P(i,j) = \frac{H_{pred}(i,j)}{\sum_{x,y} H_{pred}(x,y) + \epsilon}
$$

其中 $(i,j)$ 是空間坐標，$\epsilon$ 是防止除以零的極小值。

KL 散度 $D_{KL}(Q || P)$ 定義為：

$$
D_{KL}(Q || P) = \sum_{i,j} Q(i,j) \log \frac{Q(i,j)}{P(i,j)} = \sum_{i,j} Q(i,j) \log Q(i,j) - \sum_{i,j} Q(i,j) \log P(i,j)
$$

由於真值分佈 $Q$ 是固定的，$\sum Q \log Q$ (負熵) 是一個常數。因此，最小化 KL 散度等價於最小化交叉熵 (Cross Entropy) 或最大化對數似然：

$$
\mathcal{L} = - \sum_{i,j} Q(i,j) \log P(i,j)
$$

### 1.3 代碼實現細節
在 `src/utils/model_utils.py` 中的 `kl_heatmap_loss` 函數實現了這一邏輯：

1.  **歸一化 (Normalization)**:
    代碼分別對 `pred_hm` 和 `gt_hm` 進行空間維度 (dim=2,3) 的求和歸一化，確保它們總和為 1 (代表概率)。
    ```python
    pred_probs = pred_probs / pred_sum.clamp(min=eps)
    gt_probs = gt_probs / gt_sum.clamp(min=eps)
    ```

2.  **KL 散度計算**:
    使用 PyTorch 的 `F.kl_div` 函數。注意 `F.kl_div` 接受的輸入是**對數概率 (log-probabilities)** 和 **目標概率 (target probabilities)**。
    ```python
    log_pred = pred_probs.log()
    kl_div = F.kl_div(log_pred, gt_probs, reduction='none').sum(dim=(2, 3))
    ```
    這計算了 $Q \cdot (\log Q - \log P)$ 形式的散度（`reduction='batchmean'` 或手動求和時）。

3.  **空真值處理 (Empty GT Handling)**:
    如果某些樣本沒有關鍵點 (真值熱圖全為 0)，歸一化會出問題且不應計算損失。代碼通過檢測 `gt_sum < eps` 來生成掩碼 `gt_zero_mask`，並將這些樣本的損失置為 0。
    ```python
    kl_div = kl_div.masked_fill(gt_zero_mask, 0.)
    ```

### 1.4 優點
*   **關注結構而非數值**: 相比 MSE，KL Loss 更關注分佈的形狀和峰值位置，對背景噪聲不那麼敏感。
*   **概率解釋**: 輸出的熱圖可以直接解釋為關鍵點在該位置出現的概率。

---

## 2. CLIP Heatmap Model 架構

`ClipHeatmapModel` 是一個結合了 **CLIP (Contrastive Language-Image Pre-Training)** 強大語義理解能力的關鍵點檢測模型。它不僅利用圖形特徵，還利用文本提示 (Text Prompts) 來輔助定位特定物體（如布料角點）。

### 2.1 整體架構
模型由三個主要部分組成：
1.  **Vision Backbone (視覺骨幹)**: 基於 CLIP 的 Vision Transformer (ViT)。
2.  **Text Prior Gating (文本先驗門控)**: 利用 CLIP 文本編碼器注入語義信息。
3.  **Heatmap Head (熱圖解碼頭)**: 將特徵解碼為空間熱圖。

### 2.2 核心組件詳解

#### A. Vision Backbone與 LoRA 微調
*   **CLIP ViT**: 模型使用預訓練的 CLIP (如 `openai/clip-vit-base-patch16`) 視覺編碼器。它將圖像分割成 Patch (如 16x16)，並提取每個 Patch 的高維特徵。
*   **LoRA (Low-Rank Adaptation)**: 為了適應特定領域（如床單/布料）而不破壞 CLIP 預訓練的知識，模型使用 LoRA 技術。
    *   在 Transformer 的 `q_proj`, `v_proj` 等層旁路添加低秩矩陣。
    *   訓練時只更新 LoRA 參數，大大減少了參數量並防止過擬合。

#### B. Text Prior Gating (文本先驗門控)
這是該模型的獨特之處。它利用文本描述來引導模型關注圖像中的特定區域。

1.  **提示詞 (Prompts)**:
    *   **正向提示 (Prior Prompts)**: 描述目標關鍵點，例如 "a photo of a cloth corner", "fabric corner point"。
    *   **負向提示 (Negative Prompts)**: 描述需抑制的干擾項，例如 "cloth fold line", "wrinkle"。

2.  **相似度計算**:
    *   提取圖像每個 Patch 的特徵向量。
    *   提取文本提示詞的特徵向量 (通過 CLIP Text Encoder)。
    *   計算 Patch 特徵與 正向/負向 文本特徵的餘弦相似度。

3.  **門控機制 (Gating)**:
    *   生成一個空間注意力圖 (Attention Map)，其中與"角點"相似度高、與"皺褶"相似度低的區域數值較高。
    *   公式概念：$Gate = PosSim - \alpha \times NegSim$。
    *   將此注意力圖乘到視覺特徵上：$Feature_{new} = Feature_{old} \times (1 + w \times Gate)$。
    *   這使得進入解碼頭的特徵已經被"高亮"了潛在的角點區域。

#### C. Heatmap Head (解碼頭)
模型支持兩種解碼頭：

1.  **Standard Decoder (標準頭)**:
    *   簡單的 $1 \times 1$ 卷積降維 -> 雙線性插值上採樣 -> $3 \times 3$ 卷積層 -> $1 \times 1$ 輸出。
    *   適用於一般場景。

2.  **GCNN Decoder (幾何/群卷積頭)** (*預設/推薦*):
    *   針對布料這種無固定方向的物體，使用 **Steerable CNN (可控卷積)** 或 **Group CNN**。
    *   **Lifting Convolution**: 將 2D 特徵提升到群空間 (如旋轉群 $C_4$ 或 $SO(2)$)。
    *   **Group Convolutions**: 在群空間進行卷積，保證旋轉等變性 (Rotation Equivariance)。即如果圖像旋轉，特徵圖也嚴格對應旋轉，不會產生特徵變形。
    *   **Group Pooling**: 最後將特徵投影回 2D 空間。
    *   這對於檢測任意方向的布料角點非常有效。

### 2.3 數據流
1.  **輸入**: $B \times 3 \times H \times W$ 圖像。
2.  **Backbone**: 經過 CLIP ViT (帶 LoRA) $\rightarrow$ 輸出 Patch Tokens序列。
3.  **Reshape**: Patch Tokens 重組為 $B \times D \times h \times w$ 特徵圖。
4.  **Gating**: 計算文本相似度圖，加權特徵圖。
5.  **Head**: GCNN 或 CNN 解碼 $\rightarrow$ $B \times 1 \times H \times W$ 熱圖。
6.  **Loss**: 與真值熱圖計算 KL Divergence。
