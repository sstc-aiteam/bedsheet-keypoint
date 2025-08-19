# realsense app for dynamically identifying keypoints

import pyrealsense2 as rs
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time, os
import torch
from shared.functions import *

# init device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# models related
from ultralytics import YOLO

# 模型與圖片路徑
model_path = "models/yolo_finetuned/best.pt"

# 只保留這些類別 ID（根據 data.yaml 順序）
allowed_classes = [1]  # 只要床單

# 載入模型
yolo_model_finetuned = YOLO(model_path)

# 載入關鍵點模型
from models.yolo_vit import HybridKeypointNet
from models.utils import *
from ultralytics import YOLO
# yolo vit
yolo_model = YOLO('yolov8l.pt')
backbone_seq = yolo_model.model.model[:10]
backbone = YoloBackbone(backbone_seq, selected_indices=[0,1,2,3,4,5,6,7,8,9])
input_dummy = torch.randn(1, 3, 128, 128)
with torch.no_grad():
    feats = backbone(input_dummy)
in_channels_list = [f.shape[1] for f in feats]
keypoint_net = HybridKeypointNet(backbone, in_channels_list)
model = keypoint_net
model = model.to(device)
compiled_model = torch.compile(model)
# load pretrained model
compiled_model.load_state_dict(torch.load('models/keypoint_model_vit_depth.pth', map_location=device))
compiled_model.eval()
keypoint_model = compiled_model

# save path
save_image_dir = "test_bin/"

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30) # Match resolutions

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

def get_aligned_frames():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None, None

    depth_image = np.asanyarray(depth_frame.get_data())
    depth_image = depth_image * depth_scale

    color_image = np.asanyarray(color_frame.get_data())

    # Normalize the depth image for better visualization
    if np.any(depth_image):
        dmin = np.min(depth_image[np.nonzero(depth_image)])
        dmax = np.max(depth_image)
        depth_norm = ((depth_image - dmin) / (dmax - dmin) * 255).astype(np.uint8) if dmax > dmin and dmin > 0 else np.zeros_like(depth_image, dtype=np.uint8)
    else:
        depth_norm = np.zeros_like(depth_image, dtype=np.uint8)

    # Colorize the normalized depth for visualization
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    return color_image, depth_image, depth_colored

def extract_mask_compare(image):
    # 推論
    results = yolo_model_finetuned(image, task="segment")[0]

    # 原圖
    orig_img = image.copy()
    h, w = orig_img.shape[:2]

    # 空白遮罩
    mask_all = np.zeros((h, w), dtype=np.uint8)
    for r in results:
        if r.masks is None:
            continue
        masks = r.masks.data.cpu().numpy()     # [N, H_pred, W_pred]
        classes = r.boxes.cls.cpu().numpy()    # [N] 物件的類別 ID
        for m, cls_id in zip(masks, classes):
            if int(cls_id) not in allowed_classes:
                continue  # 跳過不在清單內的類別
            m = (m * 255).astype(np.uint8)
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_all = cv2.bitwise_or(mask_all, m)
    masked_image = orig_img.copy()
    masked_image[mask_all==0] = 0
    return mask_all

def detect_pipeline(color_image, depth_image):
    c_copy = color_image.copy(); d_copy = depth_image.copy()
    mask = extract_mask_compare(color_image)
    if np.sum(mask) > 0:
        c_copy[mask==0] = 0
        d_copy[mask==0] = 0
        H,W = color_image.shape[:2]
        color_image_resized = cv2.resize(c_copy, (128, 128), interpolation=cv2.INTER_AREA)
        depth_image_resized = cv2.resize(d_copy, (128, 128), interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            batch_image = torch.Tensor(np.transpose(depth_image_resized, (2, 0, 1))).unsqueeze(0).to(device)
            outputs = keypoint_model(batch_image)
            kp = outputs[0].cpu().numpy()
            kp = kp[0,:,:]
            points = thresholded_locations(kp, 0.003)
            for p in points:
                i, j = p
                cv2.circle(color_image_resized, (int(j), int(i)), 3, (255,0,0), -1)
        color_image = cv2.resize(color_image_resized, (W, H), interpolation=cv2.INTER_AREA)
        depth_image = cv2.resize(depth_image_resized, (W, H), interpolation=cv2.INTER_AREA)
    return color_image, depth_image

class RGBDepthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RGB & Depth Capture")
        self.label = tk.Label(root)
        self.label.pack()
        self.color = None
        self.depth = None
        self.depth_color = None
        self.update()

    def update(self):
        c, d, d_colored = get_aligned_frames()
        if c is not None and d_colored is not None:
            depth_image = depth_map_to_image(d)
            depth_image =  cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
            c, depth_image = detect_pipeline(c, depth_image)
            c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            self.color = c
            self.depth = d
            self.depth_color = d_colored
            stack = np.hstack((c, depth_image))
            img = Image.fromarray(stack)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.config(image=imgtk)
        self.root.after(30, self.update)

root = tk.Tk()
app = RGBDepthApp(root)

try:
    root.mainloop()
finally:
    pipeline.stop()

# import matplotlib.pyplot as plt

# image_dir = "realsense/test_bin/"
# orig_hw = None
# for f in os.listdir(image_dir):
#     if f[:6] == "color_":
#         fnumber = f[6:]; fnumber = fnumber[:-4]
#         depth_f = "depth_raw_" + fnumber + ".npy"
#         color_f = "color_" + fnumber + ".png"
#         depth_color_f = "depth_color_" + fnumber + ".png"
#         # Example usage:
#         depth_map = np.load(image_dir + depth_f)
#         # Now you can save with cv2.imwrite or display with OpenCV/Matplotlib
#         color_img = cv2.imread(image_dir+color_f)
#         depth_image = depth_map_to_image(depth_map)
#         depth_image =  cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)
#         print(color_img.shape, depth_image.shape)
#         c, depth_image = detect_pipeline(color_img, depth_image)
#         plt.imshow(c)
#         plt.show()
#         plt.imshow(depth_image)
#         plt.show()
