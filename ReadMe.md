## 自動產生被單訓練資料

`cd cloth_data_gen`

`blender --background --python cloth_dataset_gen.py`

## 訓練模型

執行 notebook: keypoint_detection_model_training.ipynb

## 產生切割圖片

預先在系統上安裝 facebook-research:sam2

執行 notebook: yolo_sam2_segmentation.ipynb

## 關鍵點偵測

執行 bedsheet_keypoint_detection.ipynb