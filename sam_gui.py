import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np

# Load your SAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load model config and checkpoint
checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg  = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

class SAM2GUI:
    def __init__(self, master):
        self.master = master
        self.img = None
        self.imgtk = None
        self.points = []

        # GUI elements
        self.canvas = tk.Canvas(master, bg="gray")
        self.canvas.pack(fill="both", expand=True)
        self.btn_load = tk.Button(master, text="Load Image", command=self.load_image)
        self.btn_load.pack(side="left")
        self.btn_segment = tk.Button(master, text="Segment", command=self.segment)
        self.btn_segment.pack(side="left")
        self.btn_save = tk.Button(master, text="Save", command=self.save_mask)
        self.btn_save.pack(side="left")

        self.canvas.bind("<Button-1>", self.on_click)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        img_cv = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_cv.shape[:2]

        # Set a maximum display size (e.g., 900x700)
        max_w, max_h = 900, 700

        scale = min(1.0, max_w / orig_w, max_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        # Resize if necessary
        if scale < 1.0:
            resized_img = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized_img = img_cv.copy()

        self.img = img_cv          # keep the original full-res for processing
        self.display_img = resized_img  # image for display
        self.img_scale = scale     # cache the scale for later (point mapping, etc.)

        self.tk_img = ImageTk.PhotoImage(Image.fromarray(self.display_img))
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.points = []


    def on_click(self, event):
        x, y = int(event.x / self.img_scale), int(event.y / self.img_scale)
        self.points.append([x, y])
        # Draw on the display image coordinates
        draw_x, draw_y = event.x, event.y
        self.canvas.create_oval(draw_x-3, draw_y-3, draw_x+3, draw_y+3, fill="red")


    def segment(self):
        if self.img is None or not self.points:
            return

        image = self.img.copy()
        points = np.array(self.points)
        labels = np.ones(len(points))  # single class

        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=points, point_labels=labels, multimask_output=True
        )

        best_mask = masks[np.argmax(scores)]
        mask_img = (best_mask * 255).astype(np.uint8)
        best_image = np.transpose(np.array([image[:,:,0] * best_mask, image[:,:,1] * best_mask, image[:,:,2] * best_mask]), (1, 2, 0))

        # Create RGBA overlay: blue with desired alpha
        blue_overlay = np.zeros((*mask_img.shape, 4), dtype=np.uint8)
        blue_overlay[..., 2] = 255  # Blue channel
        blue_overlay[..., 3] = (mask_img * 0.5).astype(np.uint8)  # 50% transparency

        # Convert image to RGBA
        base_rgba = np.concatenate([image, np.full((*image.shape[:2], 1), 255, dtype=np.uint8)], axis=2)

        # Alpha composite
        alpha = blue_overlay[..., 3:4].astype(float) / 255
        composite = (1 - alpha) * base_rgba[..., :3] + alpha * blue_overlay[..., :3]
        composite = composite.astype(np.uint8)

        # IMPORTANT: Resize composite to display size
        display_h, display_w = self.display_img.shape[:2]
        composite_resized = cv2.resize(composite, (display_w, display_h), interpolation=cv2.INTER_AREA)

        self.tk_img = ImageTk.PhotoImage(Image.fromarray(composite_resized))
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.current_mask = mask_img
        self.current_masked_image = best_image

    def save_mask(self):
        if hasattr(self, "current_mask"):
            fname = filedialog.asksaveasfilename(defaultextension=".png")
            cv2.imwrite(fname, self.current_masked_image)
            print("Saved mask to:", fname)

root = tk.Tk()
app = SAM2GUI(root)
root.mainloop()
