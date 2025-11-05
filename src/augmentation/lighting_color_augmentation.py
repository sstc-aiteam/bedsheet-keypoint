#!/usr/bin/env python3
"""
Enhanced Data Augmentation with Lighting and Color Differences

This module provides comprehensive data augmentation techniques including:
- Lighting variations (brightness, contrast, exposure, shadows)
- Color transformations (hue, saturation, temperature, tint)
- Advanced photometric augmentations
- Spatial augmentations with proper keypoint handling
"""

import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2


class LightingColorAugmentation:
    """
    Comprehensive data augmentation with lighting and color variations.
    
    Features:
    - Advanced lighting simulation (shadows, highlights, exposure)
    - Color temperature and tint adjustments
    - Hue, saturation, and brightness variations
    - Spatial augmentations with keypoint preservation
    - Configurable intensity levels
    """
    
    def __init__(self, 
                 image_size: int = 256,
                 intensity: str = 'medium',  # 'light', 'medium', 'strong'
                 use_albumentations: bool = True,
                 preserve_keypoints: bool = True):
        """
        Initialize augmentation with configurable intensity.
        
        Args:
            image_size: Target image size for augmentation
            intensity: Augmentation intensity ('light', 'medium', 'strong')
            use_albumentations: Whether to use albumentations library for advanced augmentations
            preserve_keypoints: Whether to preserve keypoint coordinates during spatial transforms
        """
        self.image_size = image_size
        self.intensity = intensity
        self.use_albumentations = use_albumentations
        self.preserve_keypoints = preserve_keypoints
        
        # Set intensity parameters
        self._set_intensity_params()
        
        # Initialize albumentations if available
        if self.use_albumentations:
            self._setup_albumentations()
    
    def _set_intensity_params(self):
        """Set augmentation parameters based on intensity level."""
        if self.intensity == 'light':
            self.lighting_params = {
                'brightness_range': (0.9, 1.1),
                'contrast_range': (0.9, 1.1),
                'exposure_range': (0.9, 1.1),
                'shadow_strength': (0.0, 0.2),
                'highlight_strength': (0.0, 0.2),
                'color_temp_range': (0.95, 1.05),
                'tint_range': (0.95, 1.05),
                'hue_shift_range': (-5, 5),
                'saturation_range': (0.9, 1.1),
                'noise_std': (0.0, 0.01),
                'blur_prob': 0.1,
                'blur_range': (0.5, 1.0)
            }
        elif self.intensity == 'medium':
            self.lighting_params = {
                'brightness_range': (0.8, 1.2),
                'contrast_range': (0.8, 1.2),
                'exposure_range': (0.8, 1.2),
                'shadow_strength': (0.0, 0.3),
                'highlight_strength': (0.0, 0.3),
                'color_temp_range': (0.9, 1.1),
                'tint_range': (0.9, 1.1),
                'hue_shift_range': (-10, 10),
                'saturation_range': (0.8, 1.2),
                'noise_std': (0.0, 0.02),
                'blur_prob': 0.2,
                'blur_range': (0.5, 1.5)
            }
        else:  # strong
            self.lighting_params = {
                'brightness_range': (0.7, 1.3),
                'contrast_range': (0.7, 1.3),
                'exposure_range': (0.7, 1.3),
                'shadow_strength': (0.0, 0.4),
                'highlight_strength': (0.0, 0.4),
                'color_temp_range': (0.85, 1.15),
                'tint_range': (0.85, 1.15),
                'hue_shift_range': (-15, 15),
                'saturation_range': (0.7, 1.3),
                'noise_std': (0.0, 0.03),
                'blur_prob': 0.3,
                'blur_range': (0.5, 2.0)
            }
    
    def _setup_albumentations(self):
        """Setup albumentations pipeline for advanced augmentations."""
        try:
            # Define albumentations transforms
            self.albumentations_transform = A.Compose([
                # Lighting augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=self.lighting_params['brightness_range'],
                    contrast_limit=self.lighting_params['contrast_range'],
                    p=0.7
                ),
                A.RandomGamma(
                    gamma_limit=(80, 120),
                    p=0.5
                ),
                
                # Color augmentations
                A.HueSaturationValue(
                    hue_shift_limit=self.lighting_params['hue_shift_range'][1],
                    sat_shift_limit=int((self.lighting_params['saturation_range'][1] - 1) * 50),
                    val_shift_limit=int((self.lighting_params['brightness_range'][1] - 1) * 50),
                    p=0.7
                ),
                
                # Advanced color adjustments
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                
                # Noise and blur
                A.OneOf([
                    A.GaussNoise(
                        var_limit=(10.0, 50.0),
                        p=0.3
                    ),
                    A.GaussianBlur(
                        blur_limit=(1, 3),
                        p=0.3
                    ),
                    A.MotionBlur(
                        blur_limit=(3, 7),
                        p=0.3
                    )
                ], p=0.4),
                
                # Spatial augmentations
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.3),
                    A.RandomRotate90(p=0.3)
                ], p=0.6),
                
                A.Rotate(
                    limit=15,
                    p=0.5
                ),
                
                A.RandomScale(
                    scale_limit=0.1,
                    p=0.5
                ),
                
                # Advanced lighting effects
                A.OneOf([
                    A.RandomShadow(
                        shadow_roi=(0, 0.5, 1, 1),
                        num_shadows_lower=1,
                        num_shadows_upper=2,
                        shadow_dimension=5,
                        p=0.3
                    ),
                    A.RandomSunFlare(
                        flare_roi=(0, 0, 1, 0.5),
                        angle_lower=0,
                        angle_upper=1,
                        num_flare_circles_lower=6,
                        num_flare_circles_upper=10,
                        src_radius=400,
                        src_color=(255, 255, 255),
                        p=0.3
                    )
                ], p=0.3),
                
                # Resize to target size
                A.Resize(self.image_size, self.image_size),
                
                # Normalize
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        except ImportError:
            print("Albumentations not available, falling back to basic augmentations")
            self.use_albumentations = False
    
    def apply_lighting_variations(self, img: np.ndarray) -> np.ndarray:
        """Apply various lighting variations to simulate different lighting conditions."""
        img = img.copy()
        
        # Random brightness adjustment
        if random.random() < 0.7:
            brightness_factor = random.uniform(*self.lighting_params['brightness_range'])
            img = np.clip(img * brightness_factor, 0, 1)
        
        # Random contrast adjustment
        if random.random() < 0.7:
            contrast_factor = random.uniform(*self.lighting_params['contrast_range'])
            mean = img.mean()
            img = np.clip((img - mean) * contrast_factor + mean, 0, 1)
        
        # Random exposure adjustment
        if random.random() < 0.5:
            exposure_factor = random.uniform(*self.lighting_params['exposure_range'])
            img = np.clip(img ** (1.0 / exposure_factor), 0, 1)
        
        # Add shadows
        if random.random() < 0.4:
            img = self._add_shadows(img)
        
        # Add highlights
        if random.random() < 0.4:
            img = self._add_highlights(img)
        
        return img
    
    def apply_color_variations(self, img: np.ndarray) -> np.ndarray:
        """Apply various color variations."""
        img = img.copy()
        
        # Convert to HSV for color manipulation
        hsv = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
        
        # Hue shift
        if random.random() < 0.6:
            hue_shift = random.uniform(*self.lighting_params['hue_shift_range'])
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift / 360.0) % 1.0
        
        # Saturation adjustment
        if random.random() < 0.6:
            sat_factor = random.uniform(*self.lighting_params['saturation_range'])
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 1)
        
        # Value (brightness) adjustment
        if random.random() < 0.6:
            val_factor = random.uniform(*self.lighting_params['brightness_range'])
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_factor, 0, 1)
        
        # Convert back to RGB
        img = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        
        # Color temperature adjustment
        if random.random() < 0.5:
            img = self._adjust_color_temperature(img)
        
        # Tint adjustment
        if random.random() < 0.5:
            img = self._adjust_tint(img)
        
        return img
    
    def apply_advanced_photometric_augmentations(self, img: np.ndarray) -> np.ndarray:
        """Apply advanced photometric augmentations."""
        img = img.copy()
        
        # Gaussian noise
        if random.random() < 0.4:
            noise_std = random.uniform(*self.lighting_params['noise_std'])
            noise = np.random.normal(0, noise_std, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        # Gaussian blur
        if random.random() < self.lighting_params['blur_prob']:
            blur_sigma = random.uniform(*self.lighting_params['blur_range'])
            img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
        
        # Motion blur simulation
        if random.random() < 0.2:
            img = self._apply_motion_blur(img)
        
        # Jpeg compression artifacts
        if random.random() < 0.3:
            img = self._apply_jpeg_artifacts(img)
        
        return img
    
    def _add_shadows(self, img: np.ndarray) -> np.ndarray:
        """Add realistic shadows to the image."""
        h, w = img.shape[:2]
        shadow_strength = random.uniform(*self.lighting_params['shadow_strength'])
        
        # Create shadow mask
        shadow_mask = np.ones((h, w), dtype=np.float32)
        
        # Add multiple shadow sources
        num_shadows = random.randint(1, 3)
        for _ in range(num_shadows):
            # Random shadow position and size
            center_x = random.randint(0, w)
            center_y = random.randint(0, h)
            radius = random.randint(50, min(w, h) // 2)
            
            # Create circular shadow
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
            shadow_mask[mask] *= (1 - shadow_strength)
        
        # Apply shadow to all channels
        for c in range(img.shape[2]):
            img[:, :, c] *= shadow_mask
        
        return img
    
    def _add_highlights(self, img: np.ndarray) -> np.ndarray:
        """Add realistic highlights to the image."""
        h, w = img.shape[:2]
        highlight_strength = random.uniform(*self.lighting_params['highlight_strength'])
        
        # Create highlight mask
        highlight_mask = np.ones((h, w), dtype=np.float32)
        
        # Add multiple highlight sources
        num_highlights = random.randint(1, 2)
        for _ in range(num_highlights):
            # Random highlight position and size
            center_x = random.randint(0, w)
            center_y = random.randint(0, h)
            radius = random.randint(30, min(w, h) // 3)
            
            # Create circular highlight
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2
            highlight_mask[mask] += highlight_strength
        
        # Apply highlights to all channels
        for c in range(img.shape[2]):
            img[:, :, c] = np.clip(img[:, :, c] * highlight_mask, 0, 1)
        
        return img
    
    def _adjust_color_temperature(self, img: np.ndarray) -> np.ndarray:
        """Adjust color temperature of the image."""
        temp_factor = random.uniform(*self.lighting_params['color_temp_range'])
        
        # Warm/cool temperature adjustment
        if temp_factor > 1.0:  # Warmer
            img[:, :, 0] = np.clip(img[:, :, 0] * temp_factor, 0, 1)  # Red
            img[:, :, 2] = np.clip(img[:, :, 2] / temp_factor, 0, 1)  # Blue
        else:  # Cooler
            img[:, :, 0] = np.clip(img[:, :, 0] / temp_factor, 0, 1)  # Red
            img[:, :, 2] = np.clip(img[:, :, 2] * temp_factor, 0, 1)  # Blue
        
        return img
    
    def _adjust_tint(self, img: np.ndarray) -> np.ndarray:
        """Adjust tint (green/magenta balance) of the image."""
        tint_factor = random.uniform(*self.lighting_params['tint_range'])
        
        # Green/magenta tint adjustment
        if tint_factor > 1.0:  # More green
            img[:, :, 1] = np.clip(img[:, :, 1] * tint_factor, 0, 1)  # Green
        else:  # More magenta
            img[:, :, 1] = np.clip(img[:, :, 1] / tint_factor, 0, 1)  # Green
        
        return img
    
    def _apply_motion_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply motion blur to simulate camera movement."""
        h, w = img.shape[:2]
        
        # Random motion direction and strength
        angle = random.uniform(0, 360)
        length = random.randint(5, 20)
        
        # Create motion blur kernel
        kernel = np.zeros((length, length))
        kernel[length // 2, :] = 1
        
        # Rotate kernel
        M = cv2.getRotationMatrix2D((length // 2, length // 2), angle, 1)
        kernel = cv2.warpAffine(kernel, M, (length, length))
        kernel = kernel / kernel.sum()
        
        # Apply blur to each channel
        blurred = np.zeros_like(img)
        for c in range(img.shape[2]):
            blurred[:, :, c] = cv2.filter2D(img[:, :, c], -1, kernel)
        
        return blurred
    
    def _apply_jpeg_artifacts(self, img: np.ndarray) -> np.ndarray:
        """Apply JPEG compression artifacts."""
        # Convert to uint8
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Encode and decode with JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(60, 95)]
        _, encimg = cv2.imencode('.jpg', cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR), encode_param)
        img_compressed = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        img_compressed = cv2.cvtColor(img_compressed, cv2.COLOR_BGR2RGB)
        
        return img_compressed.astype(np.float32) / 255.0
    
    def apply_spatial_augmentations(self, img: np.ndarray, keypoints: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply spatial augmentations with optional keypoint preservation."""
        img = img.copy()
        keypoints_out = keypoints.copy() if keypoints is not None else None
        
        # Random horizontal flip
        if random.random() < 0.5:
            img = np.fliplr(img)
            if keypoints_out is not None:
                keypoints_out = np.fliplr(keypoints_out)
        
        # Random vertical flip
        if random.random() < 0.3:
            img = np.flipud(img)
            if keypoints_out is not None:
                keypoints_out = np.flipud(keypoints_out)
        
        # Random rotation
        if random.random() < 0.6:
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            if keypoints_out is not None:
                keypoints_out = cv2.warpAffine(keypoints_out, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Random scaling
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if keypoints_out is not None:
                keypoints_scaled = cv2.resize(keypoints_out, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            if scale > 1.0:
                # Crop from center
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                img = img_scaled[start_h:start_h+h, start_w:start_w+w]
                if keypoints_out is not None:
                    keypoints_out = keypoints_scaled[start_h:start_h+h, start_w:start_w+w]
            else:
                # Pad with reflection
                pad_h = (h - new_h) // 2
                pad_w = (w - new_w) // 2
                img = np.pad(img_scaled, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w), (0, 0)), mode='reflect')
                if keypoints_out is not None:
                    keypoints_out = np.pad(keypoints_scaled, ((pad_h, h-new_h-pad_h), (pad_w, w-new_w-pad_w)), mode='reflect')
        
        return img, keypoints_out
    
    def __call__(self, img: np.ndarray, keypoints: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply comprehensive augmentation to image and keypoints.
        
        Args:
            img: Input image as numpy array (H, W, C) with values in [0, 1]
            keypoints: Optional keypoint heatmap as numpy array (H, W)
            
        Returns:
            Tuple of (augmented_image, augmented_keypoints)
        """
        if self.use_albumentations:
            return self._apply_albumentations(img, keypoints)
        else:
            return self._apply_custom_augmentations(img, keypoints)
    
    def _apply_albumentations(self, img: np.ndarray, keypoints: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentations using albumentations library."""
        try:
            # Prepare data for albumentations
            data = {'image': (img * 255).astype(np.uint8)}
            if keypoints is not None:
                data['mask'] = (keypoints * 255).astype(np.uint8)
            
            # Apply transforms
            transformed = self.albumentations_transform(**data)
            
            # Convert back
            img_aug = transformed['image'].astype(np.float32) / 255.0
            keypoints_aug = transformed.get('mask', None)
            if keypoints_aug is not None:
                keypoints_aug = keypoints_aug.astype(np.float32) / 255.0
            
            return img_aug, keypoints_aug
        except Exception as e:
            print(f"Albumentations failed: {e}, falling back to custom augmentations")
            return self._apply_custom_augmentations(img, keypoints)
    
    def _apply_custom_augmentations(self, img: np.ndarray, keypoints: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply custom augmentations without albumentations."""
        # Apply spatial augmentations first
        img, keypoints = self.apply_spatial_augmentations(img, keypoints)
        
        # Apply lighting variations
        img = self.apply_lighting_variations(img)
        
        # Apply color variations
        img = self.apply_color_variations(img)
        
        # Apply advanced photometric augmentations
        img = self.apply_advanced_photometric_augmentations(img)
        
        return img, keypoints


class EnhancedClothAugmentation:
    """Enhanced augmentation specifically for cloth keypoint detection."""
    
    def __init__(self, image_size: int = 256, intensity: str = 'medium'):
        self.image_size = image_size
        self.lighting_color_aug = LightingColorAugmentation(
            image_size=image_size,
            intensity=intensity,
            use_albumentations=True,
            preserve_keypoints=True
        )
    
    def __call__(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply enhanced augmentation to a sample."""
        pixel_values = sample['pixel_values']  # (3, H, W)
        gt_heatmap = sample['gt_heatmap']  # (1, H, W)
        
        # Convert to numpy for augmentation
        img = pixel_values.permute(1, 2, 0).numpy()  # (H, W, 3)
        heatmap = gt_heatmap.squeeze(0).numpy()  # (H, W)
        
        # Apply comprehensive augmentation
        img_aug, heatmap_aug = self.lighting_color_aug(img, heatmap)
        
        # Convert back to tensors
        pixel_values_aug = torch.from_numpy(img_aug.transpose(2, 0, 1)).float()
        gt_heatmap_aug = torch.from_numpy(heatmap_aug).unsqueeze(0).float()
        
        # Update sample
        sample['pixel_values'] = pixel_values_aug
        sample['gt_heatmap'] = gt_heatmap_aug
        
        return sample


class EnhancedBedsheetAugmentation:
    """Enhanced augmentation for bedsheet keypoint detection."""
    
    def __init__(self, image_size: int = 256, intensity: str = 'medium'):
        self.image_size = image_size
        self.lighting_color_aug = LightingColorAugmentation(
            image_size=image_size,
            intensity=intensity,
            use_albumentations=True,
            preserve_keypoints=True
        )
    
    def __call__(self, img_tensor: torch.Tensor, keypoints_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply enhanced augmentation to image and keypoints."""
        # Convert to numpy for augmentation
        img = img_tensor.permute(1, 2, 0).numpy()  # (H, W, 3)
        keypoints = keypoints_tensor.squeeze(0).numpy()  # (H, W)
        
        # Apply comprehensive augmentation
        img_aug, keypoints_aug = self.lighting_color_aug(img, keypoints)
        
        # Convert back to tensors
        img_tensor_aug = torch.from_numpy(img_aug.transpose(2, 0, 1)).float()
        keypoints_tensor_aug = torch.from_numpy(keypoints_aug).unsqueeze(0).float()
        
        return img_tensor_aug, keypoints_tensor_aug


def create_lighting_color_augmentation(image_size: int = 256, 
                                     intensity: str = 'medium',
                                     augmentation_type: str = 'cloth') -> object:
    """
    Factory function to create appropriate augmentation class.
    
    Args:
        image_size: Target image size
        intensity: Augmentation intensity ('light', 'medium', 'strong')
        augmentation_type: Type of augmentation ('cloth', 'bedsheet', 'mattress', 'fitted_sheet')
    
    Returns:
        Appropriate augmentation class instance
    """
    if augmentation_type == 'cloth':
        return EnhancedClothAugmentation(image_size, intensity)
    elif augmentation_type in ['bedsheet', 'mattress', 'fitted_sheet']:
        return EnhancedBedsheetAugmentation(image_size, intensity)
    else:
        raise ValueError(f"Unknown augmentation type: {augmentation_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test the augmentation system
    import matplotlib.pyplot as plt
    
    # Create sample data
    sample_img = np.random.rand(256, 256, 3).astype(np.float32)
    sample_keypoints = np.random.rand(256, 256).astype(np.float32)
    
    # Test different intensities
    for intensity in ['light', 'medium', 'strong']:
        aug = LightingColorAugmentation(intensity=intensity)
        img_aug, kp_aug = aug(sample_img, sample_keypoints)
        
        print(f"Intensity: {intensity}")
        print(f"Original image shape: {sample_img.shape}")
        print(f"Augmented image shape: {img_aug.shape}")
        print(f"Original keypoints shape: {sample_keypoints.shape}")
        print(f"Augmented keypoints shape: {kp_aug.shape}")
        print("-" * 50)

