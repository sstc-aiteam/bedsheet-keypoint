#!/usr/bin/env python3
"""
Demo script for the Keypoint Annotator Tool

This script demonstrates how to use the enhanced keypoint annotator
with bulk image loading and dropdown navigation features.
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from keypoint_annotator import KeypointAnnotator
import tkinter as tk

def main():
    """Run the keypoint annotator demo."""
    print("=" * 60)
    print("Keypoint Annotator Tool - Enhanced Version")
    print("=" * 60)
    print()
    print("Features:")
    print("✓ Bulk image loading from multiple directories")
    print("✓ Dropdown menu for quick image navigation")
    print("✓ VIA project format output")
    print("✓ Keyboard shortcuts for efficient annotation")
    print()
    print("Usage Instructions:")
    print("1. Click 'Select Image Directory' to load all images from a folder")
    print("   OR")
    print("   Click 'Select Multiple Images' to choose individual images")
    print("2. Click 'Select Output File' to choose where to save annotations")
    print("3. Left-click on images to add keypoints")
    print("4. Right-click near keypoints to remove them")
    print("5. Use the dropdown menu to jump to any image")
    print("6. Use keyboard shortcuts: ← → (navigate), R (reset), S (save)")
    print("7. Click 'Save Annotations' when done")
    print()
    print("Starting the annotator...")
    print()
    
    # Create and run the annotator
    root = tk.Tk()
    app = KeypointAnnotator(root)
    
    # Center the window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()