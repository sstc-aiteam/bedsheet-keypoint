#!/usr/bin/env python3
"""
Simple and Reliable Data Augmentation with Lighting and Color Differences

This module provides a clean, PIL-based augmentation system that's easy to understand
and debug. Uses ImageEnhance for reliable color and lighting adjustments.
"""

import random
import numpy as np
import cv2
import torch
from PIL import Image, ImageEnhance, ImageOps
from typing import Dict, Tuple, Optional, Union
import torchvision.transforms.functional as TF


class SimpleLightingColorAugmentation:
    """
    Simple, reliable augmentation using PIL ImageEnhance.
    
    Features:
    - Brightness, contrast, saturation, hue adjustments
    - Spatial augmentations (rotation, flip, scale)
    - Keypoint coordinate preservation
    - Configurable intensity levels
    """
    
    def __init__(self, 
                 image_size: int = 256,
                 intensity: str = 'medium',  # 'light', 'medium', 'strong'
                 preserve_keypoints: bool = True):
        """
        Initialize simple augmentation.
        
        Args:
            image_size: Target image size
            intensity: Augmentation intensity level
            preserve_keypoints: Whether to preserve keypoint coordinates
        """
        self.image_size = image_size
        self.intensity = intensity
        self.preserve_keypoints = preserve_keypoints
        
        # Set intensity parameters
        self._set_intensity_params()
    
    def _set_intensity_params(self):
        """Set augmentation parameters based on intensity level."""
        if self.intensity == 'light':
            self.params = {
                'brightness': 0.1,      # ±10% brightness
                'contrast': 0.1,        # ±10% contrast  
                'saturation': 0.1,      # ±10% saturation
                'hue': 0.05,            # ±5% hue shift
                'rotation_prob': 0.3,   # 30% chance of rotation
                'rotation_range': 10,   # ±10 degrees
                'flip_prob': 0.4,       # 40% chance of flip
                'scale_prob': 0.3,      # 30% chance of scale
                'scale_range': 0.05,    # ±5% scale
                'noise_prob': 0.2,      # 20% chance of noise
                'noise_std': 0.01       # Low noise level
            }
        elif self.intensity == 'medium':
            self.params = {
                'brightness': 0.2,      # ±20% brightness
                'contrast': 0.2,        # ±20% contrast
                'saturation': 0.2,      # ±20% saturation
                'hue': 0.1,             # ±10% hue shift
                'rotation_prob': 0.5,   # 50% chance of rotation
                'rotation_range': 15,   # ±15 degrees
                'flip_prob': 0.5,       # 50% chance of flip
                'scale_prob': 0.4,      # 40% chance of scale
                'scale_range': 0.1,     # ±10% scale
                'noise_prob': 0.3,      # 30% chance of noise
                'noise_std': 0.02       # Medium noise level
            }
        else:  # strong
            self.params = {
                'brightness': 0.3,      # ±30% brightness
                'contrast': 0.3,        # ±30% contrast
                'saturation': 0.3,      # ±30% saturation
                'hue': 0.15,            # ±15% hue shift
                'rotation_prob': 0.7,   # 70% chance of rotation
                'rotation_range': 20,   # ±20 degrees
                'flip_prob': 0.6,       # 60% chance of flip
                'scale_prob': 0.5,      # 50% chance of scale
                'scale_range': 0.15,    # ±15% scale
                'noise_prob': 0.4,      # 40% chance of noise
                'noise_std': 0.03       # High noise level
            }
    
    def apply_color_augmentation(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Apply color-based augmentations using PIL ImageEnhance."""
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Random brightness
        if random.random() < 0.7:  # 70% chance
            enhancer = ImageEnhance.Brightness(image)
            factor = 1.0 + random.uniform(-self.params['brightness'], self.params['brightness'])
            image = enhancer.enhance(factor)
        
        # Random contrast
        if random.random() < 0.7:  # 70% chance
            enhancer = ImageEnhance.Contrast(image)
            factor = 1.0 + random.uniform(-self.params['contrast'], self.params['contrast'])
            image = enhancer.enhance(factor)
        
        # Random saturation
        if random.random() < 0.6:  # 60% chance
            enhancer = ImageEnhance.Color(image)
            factor = 1.0 + random.uniform(-self.params['saturation'], self.params['saturation'])
            image = enhancer.enhance(factor)
        
        # Random hue shift
        if random.random() < 0.5:  # 50% chance
            image = self._apply_hue_shift(image)
        
        return image
    
    def _apply_hue_shift(self, image: Image.Image) -> Image.Image:
        """Apply hue shift to image."""
        # Convert to HSV
        hsv = image.convert('HSV')
        hsv_array = np.array(hsv, dtype=np.uint8)
        
        # Apply hue shift
        hue_shift = int(random.uniform(-self.params['hue'], self.params['hue']) * 255)
        hsv_array[:, :, 0] = (hsv_array[:, :, 0].astype(int) + hue_shift) % 255
        
        # Convert back to RGB
        hsv_image = Image.fromarray(hsv_array, mode='HSV')
        return hsv_image.convert('RGB')
    
    def apply_spatial_augmentation(self, image: Image.Image, keypoints: Optional[np.ndarray] = None) -> Tuple[Image.Image, Optional[np.ndarray]]:
        """Apply spatial augmentations with keypoint preservation."""
        
        # Random horizontal flip
        if random.random() < self.params['flip_prob']:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if keypoints is not None:
                keypoints = np.fliplr(keypoints)
        
        # Random rotation
        if random.random() < self.params['rotation_prob']:
            angle = random.uniform(-self.params['rotation_range'], self.params['rotation_range'])
            image = image.rotate(angle, fillcolor=(0, 0, 0), expand=False)
            if keypoints is not None:
                keypoints = self._rotate_keypoints(keypoints, angle)
        
        # Random scaling
        if random.random() < self.params['scale_prob']:
            scale = 1.0 + random.uniform(-self.params['scale_range'], self.params['scale_range'])
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.LANCZOS)
            
            # Crop or pad to original size
            if scale > 1.0:
                # Crop from center
                left = (image.width - self.image_size) // 2
                top = (image.height - self.image_size) // 2
                image = image.crop((left, top, left + self.image_size, top + self.image_size))
                if keypoints is not None:
                    keypoints = self._crop_keypoints(keypoints, left, top, self.image_size, self.image_size)
            else:
                # Pad to original size
                image = self._pad_to_size(image, self.image_size, self.image_size)
                if keypoints is not None:
                    keypoints = self._pad_keypoints(keypoints, self.image_size, self.image_size)
        
        # Ensure final size is correct
        if image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        if keypoints is not None and keypoints.shape != (self.image_size, self.image_size):
            keypoints = cv2.resize(keypoints, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        return image, keypoints
    
    def apply_noise_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply noise augmentation."""
        if random.random() < self.params['noise_prob']:
            # Convert to numpy for noise addition
            img_array = np.array(image, dtype=np.float32)
            noise = np.random.normal(0, self.params['noise_std'] * 255, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
            image = Image.fromarray(img_array.astype(np.uint8))
        
        return image
    
    def _rotate_keypoints(self, keypoints: np.ndarray, angle: float) -> np.ndarray:
        """Rotate keypoints by the same angle as the image."""
        h, w = keypoints.shape
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated_keypoints = cv2.warpAffine(
            keypoints, rotation_matrix, (w, h), 
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        
        return rotated_keypoints
    
    def _crop_keypoints(self, keypoints: np.ndarray, left: int, top: int, width: int, height: int) -> np.ndarray:
        """Crop keypoints to match image crop."""
        return keypoints[top:top+height, left:left+width]
    
    def _pad_keypoints(self, keypoints: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """Pad keypoints to match image padding."""
        h, w = keypoints.shape
        pad_h = (target_height - h) // 2
        pad_w = (target_width - w) // 2
        
        padded = np.zeros((target_height, target_width), dtype=keypoints.dtype)
        padded[pad_h:pad_h+h, pad_w:pad_w+w] = keypoints
        
        return padded
    
    def _pad_to_size(self, image: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """Pad image to target size."""
        # Create new image with black background
        new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
        
        # Paste original image in center
        x = (target_width - image.width) // 2
        y = (target_height - image.height) // 2
        new_image.paste(image, (x, y))
        
        return new_image
    
    def __call__(self, img: np.ndarray, keypoints: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply comprehensive augmentation to image and keypoints.
        
        Args:
            img: Input image as numpy array (H, W, C) with values in [0, 1]
            keypoints: Optional keypoint heatmap as numpy array (H, W)
            
        Returns:
            Tuple of (augmented_image, augmented_keypoints)
        """
        # Convert to PIL Image
        if img.dtype != np.uint8:
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
        else:
            img_pil = Image.fromarray(img)
        
        # Apply color augmentations
        img_pil = self.apply_color_augmentation(img_pil)
        
        # Apply spatial augmentations
        img_pil, keypoints = self.apply_spatial_augmentation(img_pil, keypoints)
        
        # Apply noise augmentation
        img_pil = self.apply_noise_augmentation(img_pil)
        
        # Convert back to numpy
        img_aug = np.array(img_pil, dtype=np.float32) / 255.0
        
        return img_aug, keypoints
    
    def _apply_augmentation(self, img: np.ndarray, keypoints: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply comprehensive augmentation to image and keypoints.
        
        Args:
            img: Input image as numpy array (H, W, C) with values in [0, 1]
            keypoints: Optional keypoint heatmap as numpy array (H, W)
            
        Returns:
            Tuple of (augmented_image, augmented_keypoints)
        """
        # Convert to PIL Image
        if img.dtype != np.uint8:
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
        else:
            img_pil = Image.fromarray(img)
        
        # Apply color augmentations
        img_pil = self.apply_color_augmentation(img_pil)
        
        # Apply spatial augmentations
        img_pil, keypoints = self.apply_spatial_augmentation(img_pil, keypoints)
        
        # Apply noise augmentation
        img_pil = self.apply_noise_augmentation(img_pil)
        
        # Convert back to numpy
        img_aug = np.array(img_pil, dtype=np.float32) / 255.0
        
        return img_aug, keypoints
    
    
class SimpleClothAugmentation:
    """Simple augmentation for cloth keypoint detection."""
    
    def __init__(self, image_size: int = 256, intensity: str = 'medium'):
        self.image_size = image_size
        self.lighting_color_aug = SimpleLightingColorAugmentation(
            image_size=image_size,
            intensity=intensity,
            preserve_keypoints=True
        )
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply simple augmentation to a sample."""
        pixel_values = sample['pixel_values']  # (3, H, W)
        gt_heatmap = sample['gt_heatmap']  # (1, H, W)
        
        # Convert to numpy for augmentation
        img = pixel_values.permute(1, 2, 0).numpy()  # (H, W, 3)
        heatmap = gt_heatmap.squeeze(0).numpy()  # (H, W)
        
        # Apply simple augmentation
        img_aug, heatmap_aug = self.lighting_color_aug(img, heatmap)
        
        # Convert back to tensors
        pixel_values_aug = torch.from_numpy(img_aug.transpose(2, 0, 1)).float()
        gt_heatmap_aug = torch.from_numpy(heatmap_aug).unsqueeze(0).float()
        
        # Update sample
        sample['pixel_values'] = pixel_values_aug
        sample['gt_heatmap'] = gt_heatmap_aug
        
        return sample


class SimpleBedsheetAugmentation:
    """Simple augmentation for bedsheet keypoint detection."""
    
    def __init__(self, image_size: int = 256, intensity: str = 'medium'):
        self.image_size = image_size
        self.lighting_color_aug = SimpleLightingColorAugmentation(
            image_size=image_size,
            intensity=intensity,
            preserve_keypoints=True
        )
    
    def __call__(self, img_tensor: torch.Tensor, keypoints_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply simple augmentation to image and keypoints."""
        # Convert to numpy for augmentation
        img = img_tensor.permute(1, 2, 0).numpy()  # (H, W, 3)
        keypoints = keypoints_tensor.squeeze(0).numpy()  # (H, W)
        
        # Apply simple augmentation
        img_aug, keypoints_aug = self.lighting_color_aug._apply_augmentation(img, keypoints)
        
        # Convert back to tensors (make copies to avoid negative strides)
        img_tensor_aug = torch.from_numpy(img_aug.transpose(2, 0, 1).copy()).float()
        keypoints_tensor_aug = torch.from_numpy(keypoints_aug.copy()).unsqueeze(0).float()
        
        return img_tensor_aug, keypoints_tensor_aug


def create_simple_lighting_color_augmentation(image_size: int = 256, 
                                            intensity: str = 'medium',
                                            augmentation_type: str = 'cloth') -> object:
    """
    Factory function to create appropriate simple augmentation class.
    
    Args:
        image_size: Target image size
        intensity: Augmentation intensity ('light', 'medium', 'strong')
        augmentation_type: Type of augmentation ('cloth', 'bedsheet', 'mattress', 'fitted_sheet')
    
    Returns:
        Appropriate augmentation class instance
    """
    if augmentation_type == 'cloth':
        return SimpleClothAugmentation(image_size, intensity)
    elif augmentation_type in ['bedsheet', 'mattress', 'fitted_sheet']:
        return SimpleBedsheetAugmentation(image_size, intensity)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test the simple augmentation system
    print("Testing simple augmentation system...")
    
    # Create sample data
    sample_img = np.random.rand(256, 256, 3).astype(np.float32)
    sample_keypoints = np.random.rand(256, 256).astype(np.float32)
    
    # Test different intensities
    for intensity in ['light', 'medium', 'strong']:
        print(f"\nTesting {intensity} intensity:")
        
        aug = SimpleLightingColorAugmentation(intensity=intensity)
        img_aug, kp_aug = aug(sample_img, sample_keypoints)
        
        print(f"  Original image shape: {sample_img.shape}")
        print(f"  Augmented image shape: {img_aug.shape}")
        print(f"  Original keypoints shape: {sample_keypoints.shape}")
        print(f"  Augmented keypoints shape: {kp_aug.shape}")
        print(f"  Image value range: [{img_aug.min():.3f}, {img_aug.max():.3f}]")
        print(f"  Keypoints value range: [{kp_aug.min():.3f}, {kp_aug.max():.3f}]")
