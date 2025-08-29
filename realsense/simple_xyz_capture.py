#!/usr/bin/env python3
"""
Simple RealSense XYZ coordinate capture with Tkinter GUI.

This script provides a simple GUI to capture a photo from RealSense camera
and extract X, Y, Z coordinates from user-specified pixel locations.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime


class RealSenseCaptureGUI:
    """Simple GUI for RealSense coordinate capture."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("RealSense XYZ Coordinate Capture")
        self.root.geometry("900x700")
        
        # RealSense setup
        self.pipeline = None
        self.depth_scale = None
        self.running = False
        self.current_frame = None
        self.current_depth = None
        self.depth_intrinsics = None
        
        # GUI elements
        self.setup_gui()
        
        # Start RealSense
        self.start_realsense()
    
    def setup_gui(self):
        """Setup the GUI elements."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="RealSense XYZ Coordinate Capture", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Camera view
        self.camera_label = ttk.Label(main_frame, text="Camera not connected", 
                                     borderwidth=2, relief="solid")
        self.camera_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # Pixel input frame
        pixel_frame = ttk.LabelFrame(main_frame, text="Pixel Coordinates Input", padding="10")
        pixel_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # X coordinate input
        ttk.Label(pixel_frame, text="X (0-639):").grid(row=0, column=0, padx=(0, 5))
        self.x_var = tk.StringVar(value="320")
        self.x_entry = ttk.Entry(pixel_frame, textvariable=self.x_var, width=10)
        self.x_entry.grid(row=0, column=1, padx=(0, 20))
        
        # Y coordinate input
        ttk.Label(pixel_frame, text="Y (0-479):").grid(row=0, column=2, padx=(0, 5))
        self.y_var = tk.StringVar(value="240")
        self.y_entry = ttk.Entry(pixel_frame, textvariable=self.y_var, width=10)
        self.y_entry.grid(row=0, column=3, padx=(0, 20))
        
        # Quick preset buttons
        preset_frame = ttk.Frame(pixel_frame)
        preset_frame.grid(row=1, column=0, columnspan=4, pady=(10, 0))
        
        ttk.Label(preset_frame, text="Quick Presets:").pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(preset_frame, text="Center", 
                  command=lambda: self.set_pixel_coords(320, 240)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Top-Left", 
                  command=lambda: self.set_pixel_coords(50, 50)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Top-Right", 
                  command=lambda: self.set_pixel_coords(590, 50)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Bottom-Left", 
                  command=lambda: self.set_pixel_coords(50, 430)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="Bottom-Right", 
                  command=lambda: self.set_pixel_coords(590, 430)).pack(side=tk.LEFT, padx=2)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=(0, 20))
        
        self.capture_btn = ttk.Button(button_frame, text="Capture Coordinates", 
                                     command=self.capture_coordinates, state="disabled")
        self.capture_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = ttk.Button(button_frame, text="Clear All", 
                                   command=self.clear_coordinates)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(button_frame, text="Save Coordinates", 
                                  command=self.save_coordinates, state="disabled")
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_label = ttk.Label(main_frame, text="Status: Initializing...", 
                                     font=("Arial", 10))
        self.status_label.grid(row=4, column=0, columnspan=3)
        
        # Coordinates display
        coord_frame = ttk.LabelFrame(main_frame, text="Captured Coordinates", padding="10")
        coord_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.coord_text = tk.Text(coord_frame, height=10, width=80, state="disabled")
        self.coord_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for coordinates
        coord_scrollbar = ttk.Scrollbar(coord_frame, orient=tk.VERTICAL, command=self.coord_text.yview)
        coord_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.coord_text.configure(yscrollcommand=coord_scrollbar.set)
        
        # Stored coordinates
        self.captured_coordinates = []
    
    def set_pixel_coords(self, x, y):
        """Set pixel coordinates from preset buttons."""
        self.x_var.set(str(x))
        self.y_var.set(str(y))
    
    def start_realsense(self):
        """Start RealSense camera."""
        try:
            # Configure RealSense
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start pipeline
            profile = self.pipeline.start(config)
            
            # Get depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Get depth intrinsics
            depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
            self.depth_intrinsics = depth_profile.get_intrinsics()
            
            self.status_label.config(text=f"Status: RealSense connected (Depth scale: {self.depth_scale})")
            self.capture_btn.config(state="normal")
            
            # Start camera preview
            self.start_preview()
            
        except Exception as e:
            self.status_label.config(text=f"Status: Error connecting to RealSense - {str(e)}")
            messagebox.showerror("Error", f"Failed to connect to RealSense camera:\n{str(e)}")
    
    def start_preview(self):
        """Start camera preview in a separate thread."""
        self.running = True
        preview_thread = threading.Thread(target=self.preview_loop, daemon=True)
        preview_thread.start()
    
    def preview_loop(self):
        """Camera preview loop."""
        try:
            while self.running:
                # Wait for frames
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Convert to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Store current frames
                self.current_frame = color_frame
                self.current_depth = depth_frame
                
                # Convert BGR to RGB for PIL
                color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                
                # Draw crosshair at current pixel coordinates
                try:
                    x = int(self.x_var.get())
                    y = int(self.y_var.get())
                    if 0 <= x < 640 and 0 <= y < 480:
                        # Draw crosshair
                        cv2.line(color_image_rgb, (x-10, y), (x+10, y), (255, 0, 0), 2)
                        cv2.line(color_image_rgb, (x, y-10), (x, y+10), (255, 0, 0), 2)
                        cv2.circle(color_image_rgb, (x, y), 15, (255, 0, 0), 2)
                        
                        # Add coordinate text
                        cv2.putText(color_image_rgb, f"({x},{y})", (x+20, y-20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                except ValueError:
                    pass  # Invalid coordinates
                
                # Resize for display
                height, width = color_image_rgb.shape[:2]
                display_width = 400
                display_height = int(height * display_width / width)
                color_image_resized = cv2.resize(color_image_rgb, (display_width, display_height))
                
                # Convert to PIL Image
                pil_image = Image.fromarray(color_image_resized)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update GUI (must be done in main thread)
                self.root.after(0, self.update_camera_display, photo)
                
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            print(f"Preview error: {e}")
            self.root.after(0, lambda: self.status_label.config(text=f"Status: Preview error - {str(e)}"))
    
    def update_camera_display(self, photo):
        """Update camera display in main thread."""
        self.camera_label.configure(image=photo, text="")
        self.camera_label.image = photo  # Keep a reference
    
    def capture_coordinates(self):
        """Capture coordinates from user-specified pixel location."""
        if not self.current_frame or not self.current_depth:
            messagebox.showwarning("Warning", "No frame available for capture")
            return
        
        try:
            # Get pixel coordinates from input
            x = int(self.x_var.get())
            y = int(self.y_var.get())
            
            # Validate coordinates
            if x < 0 or x >= 640 or y < 0 or y >= 480:
                messagebox.showerror("Error", "Invalid pixel coordinates. X must be 0-639, Y must be 0-479.")
                return
            
            # Get depth value
            depth = self.current_depth.get_distance(x, y)
            
            if depth == 0:
                messagebox.showwarning("Warning", f"No valid depth at pixel ({x}, {y})")
                return
            
            # Convert to 3D coordinates
            point = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], depth)
            
            # Create coordinate entry
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            coordinate_entry = {
                "timestamp": timestamp,
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "pixel_x": x,
                "pixel_y": y,
                "depth_meters": depth
            }
            
            # Add to list
            self.captured_coordinates.append(coordinate_entry)
            
            # Update display
            self.update_coordinate_display()
            
            # Enable save button
            self.save_btn.config(state="normal")
            
            self.status_label.config(text=f"Status: Captured coordinate at pixel ({x}, {y}) at {timestamp}")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer coordinates")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture coordinates:\n{str(e)}")
    
    def clear_coordinates(self):
        """Clear all captured coordinates."""
        self.captured_coordinates = []
        self.update_coordinate_display()
        self.save_btn.config(state="disabled")
        self.status_label.config(text="Status: All coordinates cleared")
    
    def update_coordinate_display(self):
        """Update the coordinate display text."""
        self.coord_text.config(state="normal")
        self.coord_text.delete(1.0, tk.END)
        
        if not self.captured_coordinates:
            self.coord_text.insert(tk.END, "No coordinates captured yet.\n")
            self.coord_text.insert(tk.END, "Enter pixel coordinates above and click 'Capture Coordinates'.\n")
        else:
            for i, coord in enumerate(self.captured_coordinates, 1):
                text = f"Point {i} ({coord['timestamp']}):\n"
                text += f"  Pixel: ({coord['pixel_x']}, {coord['pixel_y']})\n"
                text += f"  3D Coordinates: X={coord['x']:.3f}m, Y={coord['y']:.3f}m, Z={coord['z']:.3f}m\n"
                text += f"  Depth: {coord['depth_meters']:.3f}m\n\n"
                
                self.coord_text.insert(tk.END, text)
        
        self.coord_text.config(state="disabled")
        self.coord_text.see(tk.END)
    
    def save_coordinates(self):
        """Save coordinates to JSON file."""
        if not self.captured_coordinates:
            messagebox.showwarning("Warning", "No coordinates to save")
            return
        
        # Ask for filename
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save coordinates to JSON file"
        )
        
        if not filename:
            return
        
        try:
            # Prepare data for saving
            save_data = {
                "capture_info": {
                    "total_points": len(self.captured_coordinates),
                    "capture_date": datetime.now().isoformat(),
                    "depth_scale": self.depth_scale,
                    "camera_resolution": "640x480"
                },
                "coordinates": self.captured_coordinates
            }
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("Success", f"Saved {len(self.captured_coordinates)} coordinates to:\n{filename}")
            self.status_label.config(text=f"Status: Saved coordinates to {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save coordinates:\n{str(e)}")
    
    def on_closing(self):
        """Handle window closing."""
        self.running = False
        if self.pipeline:
            self.pipeline.stop()
        self.root.destroy()


def main():
    """Main function."""
    root = tk.Tk()
    app = RealSenseCaptureGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start GUI
    root.mainloop()


if __name__ == "__main__":
    main()
