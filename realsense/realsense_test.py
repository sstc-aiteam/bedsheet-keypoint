import pyrealsense2 as rs
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import time

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

class RGBDepthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RGB & Depth Capture")
        self.label = tk.Label(root)
        self.label.pack()
        self.button = tk.Button(root, text="Photo Shoot", command=self.save_images)
        self.button.pack()
        self.color = None
        self.depth = None
        self.depth_color = None
        self.update()

    def update(self):
        c, d, d_colored = get_aligned_frames()
        if c is not None and d_colored is not None:
            self.color = c
            self.depth = d
            self.depth_color = d_colored
            stack = np.hstack((c, d_colored))
            img = Image.fromarray(stack)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.config(image=imgtk)
        self.root.after(30, self.update)

    def save_images(self):
        if self.color is not None and self.depth is not None and self.depth_color is not None:
            ts = int(time.time())
            cv2.imwrite(f"{save_image_dir}color_{ts}.png", self.color)
            cv2.imwrite(f"{save_image_dir}depth_color_{ts}.png", self.depth_color)
            # Save the raw depth as .npy to preserve all values
            np.save(f"{save_image_dir}depth_raw_{ts}.npy", self.depth)
            print(f"Saved: color_{ts}.png, depth_color_{ts}.png, depth_raw_{ts}.npy")

root = tk.Tk()
app = RGBDepthApp(root)

try:
    root.mainloop()
finally:
    pipeline.stop()
