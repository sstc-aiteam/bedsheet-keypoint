#!/usr/bin/env python3
"""
Keypoint Annotator Tool

A simple tool for annotating keypoints on images, saving in VIA project JSON format.
Features:
- Click to add keypoints
- Right-click to remove keypoints
- Reset button to clear all keypoints for current image
- Save annotations in VIA project format
- Load annotations from existing JSON config files
- Navigate through images in a directory
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import uuid
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image, ImageTk
except ImportError:
    try:
        from PIL import Image
        import tkinter as tk
        # Fallback for systems where ImageTk is not available
        class ImageTk:
            @staticmethod
            def PhotoImage(image):
                return tk.PhotoImage(image)
    except ImportError:
        print("Error: PIL/Pillow is required. Install with: pip install Pillow")
        sys.exit(1)

class KeypointAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Keypoint Annotator")
        self.root.geometry("1200x800")
        
        # Data storage
        self.images = []
        self.current_image_index = 0
        self.annotations = {}  # {image_id: [keypoints]}
        self.image_directory = ""
        self.output_file = ""
        
        # Current image data
        self.current_image = None
        self.current_image_id = None
        self.current_keypoints = []
        
        # UI setup
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Directory selection
        ttk.Button(control_frame, text="Select Image Directory", 
                  command=self.select_directory).pack(side=tk.LEFT, padx=(0, 5))
        
        # Bulk image selection
        ttk.Button(control_frame, text="Select Multiple Images", 
                  command=self.select_multiple_images).pack(side=tk.LEFT, padx=(0, 10))
        
        # Output file selection
        ttk.Button(control_frame, text="Select Output File", 
                  command=self.select_output_file).pack(side=tk.LEFT, padx=(0, 10))
        
        # Load JSON config
        ttk.Button(control_frame, text="Load JSON Config", 
                  command=self.load_json_config).pack(side=tk.LEFT, padx=(0, 10))
        
        # Navigation controls
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(side=tk.LEFT, padx=(20, 0))
        
        ttk.Button(nav_frame, text="← Previous", 
                  command=self.previous_image).pack(side=tk.LEFT, padx=(0, 5))
        
        # Image selection dropdown
        self.image_var = tk.StringVar()
        self.image_dropdown = ttk.Combobox(nav_frame, textvariable=self.image_var, 
                                          state="readonly", width=30)
        self.image_dropdown.pack(side=tk.LEFT, padx=5)
        self.image_dropdown.bind("<<ComboboxSelected>>", self.on_image_selected)
        
        ttk.Button(nav_frame, text="Next →", 
                  command=self.next_image).pack(side=tk.LEFT, padx=(5, 0))
        
        # Action buttons
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(side=tk.RIGHT)
        
        ttk.Button(action_frame, text="Reset Current", 
                  command=self.reset_current_image).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(action_frame, text="Save Annotations", 
                  command=self.save_annotations).pack(side=tk.LEFT, padx=(0, 5))
        
        # Image display area
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        self.canvas = tk.Canvas(self.image_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        
        # Bind keyboard shortcuts
        self.root.bind("<Left>", lambda e: self.previous_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("<r>", lambda e: self.reset_current_image())
        self.root.bind("<s>", lambda e: self.save_annotations())
        self.root.focus_set()  # Enable keyboard focus
        
        # Instructions
        instructions = ttk.Label(main_frame, 
                               text="Left-click to add keypoint | Right-click to remove keypoint | Reset to clear all keypoints | ← → to navigate | R to reset | S to save | Use dropdown to jump to any image")
        instructions.pack(pady=(10, 0))
        
    def select_directory(self):
        """Select directory containing images."""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            self.image_directory = directory
            self.load_images()
            
    def select_multiple_images(self):
        """Select multiple images from different locations."""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=filetypes
        )
        
        if files:
            self.images = [Path(f) for f in files]
            self.images.sort()  # Sort by filename
            self.current_image_index = 0
            self.load_current_image()
            self.update_image_dropdown()
            
    def select_output_file(self):
        """Select output JSON file."""
        filename = filedialog.asksaveasfilename(
            title="Save Annotations As",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.output_file = filename
            
    def load_json_config(self):
        """Load annotations from a JSON config file (VIA project format)."""
        filename = filedialog.askopenfilename(
            title="Load JSON Config",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filename:
            return
            
        try:
            with open(filename, 'r') as f:
                config_data = json.load(f)
            
            # Check if it's a VIA project format
            if not isinstance(config_data, dict):
                messagebox.showerror("Error", "Invalid JSON format. Expected a dictionary.")
                return
            
            # Try to load files from config if available
            files_loaded = False
            if "file" in config_data and config_data["file"]:
                # Extract file information
                file_list = []
                for file_id, file_info in config_data["file"].items():
                    if isinstance(file_info, dict) and "fname" in file_info:
                        file_list.append((file_id, file_info["fname"]))
                
                # Try to find images based on filenames
                if file_list:
                    # Ask user if they want to load images
                    response = messagebox.askyesno(
                        "Load Images",
                        f"Found {len(file_list)} file(s) in config. Would you like to search for these images?\n\n"
                        "Click 'Yes' to search in the current directory or select a directory.\n"
                        "Click 'No' to only load annotations."
                    )
                    
                    if response:
                        # Try to find images
                        search_dir = filedialog.askdirectory(
                            title="Select Directory to Search for Images"
                        )
                        if search_dir:
                            image_paths = []
                            search_path = Path(search_dir)
                            
                            # Search recursively for matching images
                            for file_id, fname in file_list:
                                found = False
                                name_without_ext = Path(fname).stem
                                
                                # Try exact match first
                                candidate = search_path / fname
                                if candidate.exists() and candidate.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                                    image_paths.append((file_id, candidate))
                                    found = True
                                else:
                                    # Try with extension variations
                                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                                        candidate = search_path / f"{name_without_ext}{ext}"
                                        if candidate.exists():
                                            image_paths.append((file_id, candidate))
                                            found = True
                                            break
                                    
                                    # If still not found, search recursively
                                    if not found:
                                        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                                            matches = list(search_path.rglob(f"{name_without_ext}{ext}"))
                                            if matches:
                                                image_paths.append((file_id, matches[0]))  # Take first match
                                                found = True
                                                break
                            
                            if image_paths:
                                # Sort by file_id to maintain order
                                image_paths.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 0)
                                self.images = [path for _, path in image_paths]
                                self.current_image_index = 0
                                files_loaded = True
                                messagebox.showinfo(
                                    "Images Loaded",
                                    f"Successfully loaded {len(self.images)} image(s)."
                                )
                            else:
                                messagebox.showwarning(
                                    "Images Not Found",
                                    "Could not find matching images. Annotations will be loaded, but you'll need to load images manually."
                                )
            
            # Load annotations from metadata
            annotations_loaded = False
            if "metadata" in config_data and config_data["metadata"]:
                # Clear existing annotations
                self.annotations = {}
                
                # Parse metadata
                for metadata_id, metadata_info in config_data["metadata"].items():
                    if isinstance(metadata_info, dict):
                        vid = metadata_info.get("vid")  # video/image ID
                        xy = metadata_info.get("xy", [])
                        
                        if vid and len(xy) >= 3:
                            # VIA format: xy = [shape_id, x, y]
                            x, y = float(xy[1]), float(xy[2])
                            
                            # Initialize list if needed
                            if vid not in self.annotations:
                                self.annotations[vid] = []
                            
                            # Add keypoint
                            self.annotations[vid].append((int(x), int(y)))
                            annotations_loaded = True
                
                if annotations_loaded:
                    messagebox.showinfo(
                        "Annotations Loaded",
                        f"Successfully loaded annotations for {len(self.annotations)} image(s)."
                    )
            
            # Set output file to the loaded config file
            self.output_file = filename
            
            # Update UI if images were loaded
            if files_loaded:
                self.update_image_dropdown()
                self.load_current_image()
            elif annotations_loaded:
                # Refresh current image if images are already loaded
                if self.images:
                    self.load_current_image()
                    messagebox.showinfo(
                        "Config Loaded",
                        f"Annotations loaded successfully for {len(self.annotations)} image(s). Current image refreshed."
                    )
                else:
                    messagebox.showinfo(
                        "Config Loaded",
                        "Annotations loaded successfully. Please load images manually to view them."
                    )
            else:
                messagebox.showwarning(
                    "No Data Found",
                    "The config file doesn't contain file or metadata information."
                )
                
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Failed to parse JSON file: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load config: {e}")
            
    def load_images(self):
        """Load all images from the selected directory."""
        if not self.image_directory:
            return
            
        # Supported image extensions
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        self.images = []
        for file_path in Path(self.image_directory).iterdir():
            if file_path.suffix.lower() in extensions:
                self.images.append(file_path)
        
        self.images.sort()  # Sort by filename
        
        if self.images:
            self.current_image_index = 0
            self.load_current_image()
            self.update_image_dropdown()
        else:
            self.update_image_dropdown()
            messagebox.showwarning("No Images", "No supported image files found in the selected directory.")
            
    def load_current_image(self):
        """Load and display the current image."""
        if not self.images or self.current_image_index >= len(self.images):
            return
            
        image_path = self.images[self.current_image_index]
        self.current_image_id = str(self.current_image_index + 1)
        
        # Load image
        try:
            self.current_image = Image.open(image_path)
            
            # Resize image to fit canvas while maintaining aspect ratio
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # Canvas is initialized
                self.display_image()
            else:
                # Canvas not ready, schedule display after window is ready
                self.root.after(100, self.display_image)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            
    def display_image(self):
        """Display the current image on canvas."""
        if not self.current_image:
            return
            
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return
            
        # Calculate scaling to fit image in canvas
        img_width, img_height = self.current_image.size
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        self.scale = min(scale_x, scale_y, 1.0)  # Don't scale up
        
        # Calculate new size
        new_width = int(img_width * self.scale)
        new_height = int(img_height * self.scale)
        
        # Resize image
        resized_image = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
        
        # Load and display existing keypoints
        self.load_keypoints_for_current_image()
        
    def load_keypoints_for_current_image(self):
        """Load keypoints for the current image."""
        if self.current_image_id in self.annotations:
            self.current_keypoints = self.annotations[self.current_image_id].copy()
        else:
            self.current_keypoints = []
            
        self.draw_keypoints()
        
    def draw_keypoints(self):
        """Draw all keypoints on the canvas."""
        if not self.current_image:
            return
            
        # Clear existing keypoints first
        self.canvas.delete('keypoint')
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.current_image.size
        
        # Calculate offset to center image
        offset_x = (canvas_width - img_width * self.scale) // 2
        offset_y = (canvas_height - img_height * self.scale) // 2
        
        for i, (x, y) in enumerate(self.current_keypoints):
            # Convert image coordinates to canvas coordinates
            canvas_x = offset_x + x * self.scale
            canvas_y = offset_y + y * self.scale
            
            # Draw keypoint
            self.canvas.create_oval(canvas_x-5, canvas_y-5, canvas_x+5, canvas_y+5, 
                                  fill='red', outline='white', width=2, tags='keypoint')
            
            # Draw keypoint number
            self.canvas.create_text(canvas_x, canvas_y-15, text=str(i+1), 
                                  fill='white', font=('Arial', 10, 'bold'), tags='keypoint')
            
    def on_left_click(self, event):
        """Handle left mouse click to add keypoint."""
        if not self.current_image:
            return
            
        # Convert canvas coordinates to image coordinates
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.current_image.size
        
        # Calculate offset to center image
        offset_x = (canvas_width - img_width * self.scale) // 2
        offset_y = (canvas_height - img_height * self.scale) // 2
        
        # Convert to image coordinates
        image_x = (event.x - offset_x) / self.scale
        image_y = (event.y - offset_y) / self.scale
        
        # Check if click is within image bounds
        if 0 <= image_x <= img_width and 0 <= image_y <= img_height:
            self.current_keypoints.append((int(image_x), int(image_y)))
            self.annotations[self.current_image_id] = self.current_keypoints.copy()
            self.draw_keypoints()
            
    def on_right_click(self, event):
        """Handle right mouse click to remove nearest keypoint."""
        if not self.current_keypoints:
            return
            
        # Convert canvas coordinates to image coordinates
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = self.current_image.size
        
        # Calculate offset to center image
        offset_x = (canvas_width - img_width * self.scale) // 2
        offset_y = (canvas_height - img_height * self.scale) // 2
        
        # Convert click to image coordinates
        click_x = (event.x - offset_x) / self.scale
        click_y = (event.y - offset_y) / self.scale
        
        # Find nearest keypoint
        min_distance = float('inf')
        nearest_index = -1
        
        for i, (x, y) in enumerate(self.current_keypoints):
            distance = ((x - click_x) ** 2 + (y - click_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_index = i
                
        # Remove nearest keypoint if within reasonable distance
        if nearest_index >= 0 and min_distance < 50:  # 50 pixel threshold
            self.current_keypoints.pop(nearest_index)
            self.annotations[self.current_image_id] = self.current_keypoints.copy()
            self.draw_keypoints()
            
    def reset_current_image(self):
        """Reset all keypoints for the current image."""
        if self.current_image_id:
            self.current_keypoints = []
            self.annotations[self.current_image_id] = []
            # Clear the canvas and redraw the image without keypoints
            self.display_image()
            
    def previous_image(self):
        """Go to previous image."""
        if self.images and self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            self.update_image_dropdown()
            
    def next_image(self):
        """Go to next image."""
        if self.images and self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.load_current_image()
            self.update_image_dropdown()
            
    def update_image_dropdown(self):
        """Update the image dropdown with available images."""
        if self.images:
            # Create list of display names for dropdown
            display_names = []
            for i, image_path in enumerate(self.images):
                display_name = f"{i+1:03d}: {image_path.name}"
                display_names.append(display_name)
            
            self.image_dropdown['values'] = display_names
            self.image_dropdown.current(self.current_image_index)
            self.image_var.set(display_names[self.current_image_index])
        else:
            self.image_dropdown['values'] = ["No images loaded"]
            self.image_dropdown.current(0)
            self.image_var.set("No images loaded")
            
    def on_image_selected(self, event=None):
        """Handle image selection from dropdown."""
        if not self.images:
            return
            
        selected_index = self.image_dropdown.current()
        if 0 <= selected_index < len(self.images):
            self.current_image_index = selected_index
            self.load_current_image()
            
    def save_annotations(self):
        """Save annotations in VIA project format."""
        if not self.output_file:
            messagebox.showwarning("No Output File", "Please select an output file first.")
            return
            
        if not self.images:
            messagebox.showwarning("No Images", "No images loaded.")
            return
            
        # Create VIA project format
        via_data = self.create_via_project()
        
        try:
            with open(self.output_file, 'w') as f:
                json.dump(via_data, f, indent=2)
            messagebox.showinfo("Success", f"Annotations saved to {self.output_file}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save annotations: {e}")
            
    def create_via_project(self):
        """Create VIA project format data."""
        # Generate unique IDs
        project_id = str(uuid.uuid4())
        timestamp = int(datetime.now().timestamp() * 1000)
        
        # Create project structure
        via_data = {
            "project": {
                "pid": project_id,
                "rev": project_id,
                "rev_timestamp": timestamp,
                "pname": "Keypoint Annotations",
                "creator": "Keypoint Annotator Tool",
                "created": timestamp,
                "vid_list": [str(i+1) for i in range(len(self.images))]
            },
            "config": {
                "file": {
                    "loc_prefix": {"1": "", "2": "", "3": "", "4": ""}
                },
                "ui": {
                    "file_content_align": "center",
                    "file_metadata_editor_visible": True,
                    "spatial_metadata_editor_visible": True,
                    "temporal_segment_metadata_editor_visible": True,
                    "spatial_region_label_attribute_id": "",
                    "gtimeline_visible_row_count": "4"
                }
            },
            "attribute": {},
            "file": {},
            "metadata": {},
            "view": {}
        }
        
        # Add file entries
        for i, image_path in enumerate(self.images):
            file_id = str(i + 1)
            via_data["file"][file_id] = {
                "fid": file_id,
                "fname": image_path.name,
                "type": 2,
                "loc": 1,
                "src": ""
            }
            via_data["view"][file_id] = {
                "fid_list": [file_id]
            }
            
        # Add metadata (keypoints)
        for image_id, keypoints in self.annotations.items():
            for i, (x, y) in enumerate(keypoints):
                metadata_id = f"{image_id}_{str(uuid.uuid4())[:8]}"
                via_data["metadata"][metadata_id] = {
                    "vid": image_id,
                    "flg": 0,
                    "z": [],
                    "xy": [1, x, y],  # VIA format: [shape_id, x, y]
                    "av": {}
                }
                
        return via_data

def main():
    """Main function to run the annotator."""
    root = tk.Tk()
    app = KeypointAnnotator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
