
import os
import torch
import numpy as np
from PIL import Image
import threading
from typing import Optional, Union
import cv2

class UniDepthGenerator:
    """
    Wrapper for UniDepthV2 to generate depth maps.
    Singleton-like usage recommended to avoid reloading model.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self, model_id="lpiccinelli/unidepth-v2-vitl14", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"Loading UniDepthV2 model: {model_id} on {self.device}...")
        try:
            from unidepth.models import UniDepthV2
            self.model = UniDepthV2.from_pretrained(model_id)
            self.model = self.model.to(self.device).eval()
            print("UniDepthV2 loaded successfully.")
        except ImportError:
            print("Error: unidepth library not found. Please install it.")
            raise
        except Exception as e:
            print(f"Error loading UniDepthV2: {e}")
            raise

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def clear(cls):
        """Release the model and clear the instance to free memory."""
        with cls._lock:
            if cls._instance is not None:
                if hasattr(cls._instance, 'model'):
                    del cls._instance.model
                torch.cuda.empty_cache()
                cls._instance = None
                print("UniDepthV2 model unloaded and memory cleared.")

    @torch.no_grad()
    def infer(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Infer depth from an image.
        Args:
            image: Path to image, numpy array (BGR or RGB), or PIL Image.
        Returns:
            Depth map as numpy array (H, W) in meters.
        """
        # Load image
        if isinstance(image, str):
            image_pil = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            # Assume BGR if coming from cv2, convert to RGB
            if image.ndim == 3 and image.shape[2] == 3:
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                 image_pil = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            image_pil = image.convert("RGB")
        else:
            raise ValueError("Unsupported image type")

        # Convert to tensor
        rgb_np = np.array(image_pil)
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).contiguous() # C, H, W
        rgb_tensor = rgb_tensor.to(self.device)

        # Infer
        preds = self.model.infer(rgb_tensor)
        depth = preds["depth"].squeeze().detach().float().cpu().numpy() # (H, W)
        
        return depth

    def process_and_save(self, image_path: str, output_path: str) -> Optional[np.ndarray]:
        """
        Generate depth for an image and save it to a .npy file.
        """
        try:
            depth_map = self.infer(image_path)
            np.save(output_path, depth_map.astype(np.float32))
            return depth_map
        except Exception as e:
            print(f"Failed to generate depth for {image_path}: {e}")
            return None

def generate_depth_for_directory(image_dir: str):
    """
    Helper to process an entire directory.
    """
    generator = UniDepthGenerator.get_instance()
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for fname in os.listdir(image_dir):
        if os.path.splitext(fname)[1].lower() in valid_exts:
            img_path = os.path.join(image_dir, fname)
            
            # Construct expected depth filename
            # Assumption: if file is color_TIMESTAMP.png, depth is depth_TIMESTAMP.npy
            # Or just name_depth.npy if generalized.
            # Following project convention: color_X.png -> depth_X.npy
            if fname.startswith("color_"):
                 timestamp = fname.replace("color_", "").rsplit(".", 1)[0]
                 depth_fname = f"depth_{timestamp}.npy"
            else:
                 depth_fname = f"{os.path.splitext(fname)[0]}_depth.npy"
            
            depth_path = os.path.join(image_dir, depth_fname)
            
            if not os.path.exists(depth_path):
                print(f"Generating depth for {fname}...")
                generator.process_and_save(img_path, depth_path)
            else:
                pass
                # print(f"Depth exists for {fname}, skipping.")
