#!/usr/bin/env python3
"""
Standalone keypoint visualization script
Creates visualizations of keypoints on bedsheet images
"""

import cv2
import numpy as np
import os
import argparse

def create_keypoint_visualization(image_path, keypoint_file, output_path):
    """Create visualization of keypoints on the image"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            return False
        
        # Load keypoints
        keypoints = []
        if os.path.exists(keypoint_file):
            with open(keypoint_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.strip():
                    x, y = line.strip().split(',')
                    if x == '-1' and y == '-1':
                        keypoints.append(None)
                    else:
                        keypoints.append((int(x), int(y)))
        else:
            print(f"Keypoint file not found: {keypoint_file}")
            return False
        
        # Create a copy for visualization
        vis_img = img.copy()
        
        # Draw keypoints with different colors and labels
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Green, Red, Blue, Yellow
        labels = ['BL', 'BR', 'TL', 'TR']  # Bottom-Left, Bottom-Right, Top-Left, Top-Right
        full_labels = ['Bottom-Left', 'Bottom-Right', 'Top-Left', 'Top-Right']
        
        visible_count = 0
        for i, (kp, color, label, full_label) in enumerate(zip(keypoints, colors, labels, full_labels)):
            if kp is not None:
                x, y = kp
                visible_count += 1
                
                # Draw larger circle with outline
                cv2.circle(vis_img, (x, y), 12, (0, 0, 0), 3)  # Black outline
                cv2.circle(vis_img, (x, y), 10, color, -1)     # Filled circle
                
                # Draw label with background
                text = f"{label} ({x},{y})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                
                # Draw background rectangle
                cv2.rectangle(vis_img, 
                            (x + 15, y - text_height - 5), 
                            (x + 15 + text_width, y + 5), 
                            (255, 255, 255), -1)
                
                # Draw text
                cv2.putText(vis_img, text, (x + 15, y - 2), font, font_scale, color, thickness)
        
        # Add summary text at the top
        summary_text = f"Visible Keypoints: {visible_count}/4"
        cv2.putText(vis_img, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(vis_img, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)
        
        # Add image info
        info_text = f"Image: {os.path.basename(image_path)}"
        cv2.putText(vis_img, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_img, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Save visualization
        cv2.imwrite(output_path, vis_img)
        print(f"Keypoint visualization saved to: {output_path}")
        print(f"  - {visible_count}/4 keypoints visible")
        
        return True
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Visualize keypoints on bedsheet images')
    parser.add_argument('--image', required=True, help='Path to the image file')
    parser.add_argument('--keypoints', required=True, help='Path to the keypoint .txt file')
    parser.add_argument('--output', required=True, help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Create visualization
    success = create_keypoint_visualization(args.image, args.keypoints, args.output)
    
    if success:
        print("✅ Visualization created successfully!")
    else:
        print("❌ Failed to create visualization")

if __name__ == '__main__':
    main()
