import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# 配置路徑
checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg  = "configs/sam2.1/sam2.1_hiera_l.yaml"
# -*- coding: utf-8 -*-

import os
import cv2

from ultralytics import YOLO

# 載入分割模型
model = YOLO("yolov8m-seg.pt")

predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def get_boundary_points_and_center(mask):
    """獲取物體的上下左右四個邊界點和重心"""
    coords = np.column_stack(np.where(mask > 0))
    
    if len(coords) == 0:
        return None, None
    
    # 獲取邊界點
    top_point = coords[np.argmin(coords[:, 0])]      # 最上方的點
    bottom_point = coords[np.argmax(coords[:, 0])]   # 最下方的點
    left_point = coords[np.argmin(coords[:, 1])]     # 最左邊的點
    right_point = coords[np.argmax(coords[:, 1])]    # 最右邊的點
    
    # 轉換為 (x, y) 格式
    boundary_points = {
        'top': (top_point[1], top_point[0]),
        'bottom': (bottom_point[1], bottom_point[0]),
        'left': (left_point[1], left_point[0]),
        'right': (right_point[1], right_point[0])
    }
    
    # 計算重心 (使用OpenCV moments方法)
    M = cv2.moments((mask * 255).astype(np.uint8))
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = (cx, cy)
    else:
        # 備用方法：像素平均
        mean_y = np.mean(coords[:, 0])
        mean_x = np.mean(coords[:, 1])
        center = (int(mean_x), int(mean_y))
    
    return boundary_points, center

def draw_points_on_image(image, boundary_points, center, mask=None):
    """在圖像上繪制邊界點、重心和遮罩輪廓"""
    result_image = image.copy()
    
    # 繪制遮罩輪廓
    if mask is not None:
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
    
    # 繪制邊界點
    colors = {
        'top': (255, 0, 0),     # 藍色
        'bottom': (0, 255, 0),  # 綠色  
        'left': (0, 0, 255),    # 紅色
        'right': (255, 0, 255)  # 紫色
    }
    
    labels = {
        'top': 'T',
        'bottom': 'B',
        'left': 'L', 
        'right': 'R'
    }
    
    for point_name, (x, y) in boundary_points.items():
        color = colors[point_name]
        cv2.circle(result_image, (int(x), int(y)), 8, color, -1)
        cv2.circle(result_image, (int(x), int(y)), 10, (255, 255, 255), 2)
        
        cv2.putText(result_image, labels[point_name], 
                   (int(x) - 10, int(y) - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, color, 2)
    
    # 繪制重心
    cx, cy = center
    cv2.circle(result_image, (cx, cy), 12, (255, 255, 0), -1)  # 青色
    cv2.circle(result_image, (cx, cy), 15, (0, 0, 0), 2)
    
    # 繪制十字標記
    cv2.line(result_image, (cx-10, cy), (cx+10, cy), (0, 0, 0), 2)
    cv2.line(result_image, (cx, cy-10), (cx, cy+10), (0, 0, 0), 2)
    
    cv2.putText(result_image, 'CENTER', 
               (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (0, 0, 0), 2)
    
    return result_image

def extract_mask_compare(image_path):
    image_name = os.path.basename(image_path)
    # 推論圖片
    results = model(image_path, conf=0.15)

    # 如果想儲存結果圖：
    box = None
    for result in results:
        for obj in result.summary():
            if obj["name"] == "bed":
                result.save(filename= "results/" + image_name.replace('.jpg', "_output1.jpg"))
                box = obj["box"]
    if box != None:
        # find the pixel point in the bounding box where the pixel has the most common color
        input_point = np.array([[(box["x1"] + box["x2"])//2 - (box["x1"] + box["x2"])//8, (box["y1"]+box["y2"])//2 - (box["y1"]+box["y2"])//8], 
                                [(box["x1"] + box["x2"])//2 - (box["x1"] + box["x2"])//8, (box["y1"]+box["y2"])//2 + (box["y1"]+box["y2"])//8],
                                [(box["x1"] + box["x2"])//2 + (box["x1"] + box["x2"])//8, (box["y1"]+box["y2"])//2 - (box["y1"]+box["y2"])//8],
                                [(box["x1"] + box["x2"])//2 + (box["x1"] + box["x2"])//8, (box["y1"]+box["y2"])//2 + (box["y1"]+box["y2"])//8]])
        input_label = np.array([1, 1, 1, 1])
        # 載入圖像
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"無法載入圖像: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True # 會自動選擇分數最高的
        )
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        best_score = scores[best_mask_idx]
        boundary_points, center = get_boundary_points_and_center(best_mask)
        result_image = draw_points_on_image(image, boundary_points, center, best_mask)

        # save the best mask to file
        mask_filename = image_name.replace('.jpg', "_mask.jpg")
        cv2.imwrite("results/" + mask_filename, result_image)

img_files = os.listdir("bed-images")
for img_file in img_files:
    if img_file.endswith('.jpg'):
        image_path = os.path.join("bed-images", img_file)
        extract_mask_compare(image_path)
        print(f"Processed {img_file}")