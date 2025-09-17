import json
import numpy as np
import bpy
import bmesh
from mathutils import Vector, Matrix
import bpy_extras
from bpy_extras.object_utils import world_to_camera_view
import os

class KeypointTracker:
    """Track bedsheet corner keypoints by mapping 3D vertices to 2D pixel coordinates"""
    
    def __init__(self, image_width=512, image_height=512):
        self.image_width = image_width
        self.image_height = image_height
        self.margin = 20  # Pixel margin for visibility check
        
    def get_corner_vertices_from_vertex_groups(self, bedsheet_obj):
        """Extract the 4 corner vertex coordinates from Blender vertex groups"""
        try:
            corner_vertices = []
            corner_names = ['Corner_BL', 'Corner_BR', 'Corner_TL', 'Corner_TR']
            
            for corner_name in corner_names:
                if corner_name in bedsheet_obj.vertex_groups:
                    vertex_group = bedsheet_obj.vertex_groups[corner_name]
                    # Get the vertex indices from the vertex group
                    vertex_indices = []
                    for i, vertex in enumerate(bedsheet_obj.data.vertices):
                        for group in vertex.groups:
                            if group.group == vertex_group.index:
                                vertex_indices.append(i)
                                break
                    
                    if vertex_indices:
                        vertex_idx = vertex_indices[0]  # Should be only one vertex per corner group
                        # Get world coordinates of the vertex
                        world_co = bedsheet_obj.matrix_world @ bedsheet_obj.data.vertices[vertex_idx].co
                        corner_vertices.append([world_co.x, world_co.y, world_co.z])
                        print(f"  - {corner_name}: vertex {vertex_idx} at world coords ({world_co.x:.3f}, {world_co.y:.3f}, {world_co.z:.3f})")
                    else:
                        print(f"Warning: No vertices found in vertex group {corner_name}")
                        corner_vertices.append([0, 0, 0])  # Fallback
                else:
                    print(f"Warning: Vertex group {corner_name} not found")
                    corner_vertices.append([0, 0, 0])  # Fallback
            
            return np.array(corner_vertices)
            
        except Exception as e:
            print(f"Error reading vertex groups: {e}")
            return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    
    def get_corner_vertices_from_simulation(self, warp_data_file):
        """Extract the 4 corner vertex coordinates from Warp simulation data (fallback method)"""
        try:
            with open(warp_data_file, 'r') as f:
                data = json.load(f)
            
            vertices = np.array(data['vertices'])
            nx = data['grid_size'][0]  # Number of vertices in X direction
            ny = data['grid_size'][1]  # Number of vertices in Y direction
            
            # Calculate corner indices
            # Corner order: bottom-left, bottom-right, top-left, top-right
            corner_indices = [
                0,                    # bottom-left (0, 0)
                ny - 1,              # bottom-right (0, ny-1) 
                (nx - 1) * ny,       # top-left (nx-1, 0)
                (nx - 1) * ny + ny - 1  # top-right (nx-1, ny-1)
            ]
            
            corner_vertices = []
            for idx in corner_indices:
                if idx < len(vertices):
                    corner_vertices.append(vertices[idx])
                else:
                    print(f"Warning: Corner index {idx} out of range for {len(vertices)} vertices")
                    corner_vertices.append([0, 0, 0])  # Fallback
            
            return np.array(corner_vertices)
            
        except Exception as e:
            print(f"Error reading simulation data: {e}")
            return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    
    def world_to_screen_coords(self, world_coords, camera):
        """Convert world coordinates to screen coordinates using Blender's built-in functions"""
        screen_coords = []
        
        # Get render resolution
        render = bpy.context.scene.render
        render_width = render.resolution_x
        render_height = render.resolution_y
        
        for i, coord in enumerate(world_coords):
            world_vec = Vector(coord)
            
            print(f"  - Keypoint {i}: world={coord}")
            
            # Use Blender's built-in world_to_camera_view function
            try:
                # Get normalized coordinates (-1 to 1)
                normalized_coords = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, camera, world_vec
                )
                
                print(f"    -> Normalized coords: ({normalized_coords.x}, {normalized_coords.y}, {normalized_coords.z})")
                
                # Check if point is behind camera or outside field of view
                if normalized_coords.z < 0:  # Behind camera
                    print(f"    -> Behind camera (z={normalized_coords.z})")
                    screen_coords.append(None)
                    continue
                
                # Check if point is within field of view
                if (normalized_coords.x < 0 or normalized_coords.x > 1 or 
                    normalized_coords.y < 0 or normalized_coords.y > 1):
                    print(f"    -> Outside field of view (x={normalized_coords.x}, y={normalized_coords.y})")
                    screen_coords.append(None)
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                x = normalized_coords.x * render_width
                y = (1.0 - normalized_coords.y) * render_height  # Flip Y coordinate
                
                # Ensure coordinates are within image bounds
                x = max(0, min(render_width - 1, x))
                y = max(0, min(render_height - 1, y))
                
                print(f"    -> Screen coords: ({x}, {y})")
                screen_coords.append((x, y))
                
            except Exception as e:
                print(f"    -> Error in world_to_camera_view: {e}")
                screen_coords.append(None)
        
        return screen_coords
    
    def check_visibility(self, screen_coords):
        """Check if keypoints are visible within the image frame"""
        visible_keypoints = []
        
        for i, coord in enumerate(screen_coords):
            if coord is None:
                visible_keypoints.append(None)
                continue
            
            x, y = coord
            
            # Check if within frame with margin
            if (self.margin <= x <= self.image_width - self.margin and 
                self.margin <= y <= self.image_height - self.margin):
                visible_keypoints.append((int(x), int(y)))
            else:
                visible_keypoints.append(None)
        
        return visible_keypoints
    
    def track_keypoints(self, bedsheet_obj, camera, warp_data_file=None):
        """Main function to track keypoints from Blender vertex groups to rendered image"""
        # Try to get corner vertices from vertex groups first
        corner_vertices = self.get_corner_vertices_from_vertex_groups(bedsheet_obj)
        
        # Fallback to simulation data if vertex groups don't work
        if np.allclose(corner_vertices, 0):
            print("  - Vertex groups not available, falling back to simulation data")
            if warp_data_file:
                corner_vertices = self.get_corner_vertices_from_simulation(warp_data_file)
            else:
                print("  - No fallback data available")
                return [None, None, None, None]
        
        # Convert directly from world coordinates to screen coordinates using Blender's built-in function
        screen_coords = self.world_to_screen_coords(corner_vertices, camera)
        
        # Check visibility
        visible_keypoints = self.check_visibility(screen_coords)
        
        return visible_keypoints
    
    def save_keypoints_to_txt(self, keypoints, output_file):
        """Save keypoints to .txt file in x,y format"""
        with open(output_file, 'w') as f:
            for kp in keypoints:
                if kp is not None:
                    f.write(f"{kp[0]},{kp[1]}\n")
                else:
                    f.write("-1,-1\n")  # Not visible
    
    def create_keypoint_visualization(self, image_path, keypoints, output_path):
        """Create visualization of keypoints on the image"""
        try:
            import cv2
            import numpy as np
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not load image: {image_path}")
                return
            
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
            
            # Save visualization
            cv2.imwrite(output_path, vis_img)
            print(f"Keypoint visualization saved to: {output_path}")
            print(f"  - {visible_count}/4 keypoints visible")
            
        except ImportError:
            print("OpenCV not available for visualization in Blender environment.")
            print("Visualization will be created separately using the standalone script.")
            # Don't try to install OpenCV in Blender environment - it won't work
        except Exception as e:
            print(f"Error creating visualization: {e}")

def track_bedsheet_keypoints(bedsheet_obj, camera, output_dir, image_name, warp_data_file=None):
    """Convenience function to track keypoints and save results"""
    # Use the same resolution as the render script (1024x1024)
    tracker = KeypointTracker(image_width=1024, image_height=1024)
    
    # Track keypoints
    keypoints = tracker.track_keypoints(bedsheet_obj, camera, warp_data_file)
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, 'keypoints'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # Save keypoints to .txt file
    keypoint_file = os.path.join(output_dir, 'keypoints', f"{image_name}.txt")
    tracker.save_keypoints_to_txt(keypoints, keypoint_file)
    
    # Create visualization if image exists
    image_path = os.path.join(output_dir, 'imgs', f"{image_name}.png")
    if os.path.exists(image_path):
        viz_path = os.path.join(output_dir, 'visualizations', f"{image_name}_keypoints.png")
        tracker.create_keypoint_visualization(image_path, keypoints, viz_path)
    
    # Print results
    print(f"Keypoints for {image_name}:")
    labels = ['Bottom-Left', 'Bottom-Right', 'Top-Left', 'Top-Right']
    for i, (kp, label) in enumerate(zip(keypoints, labels)):
        if kp is not None:
            print(f"  {label}: ({kp[0]}, {kp[1]}) - Visible")
        else:
            print(f"  {label}: Not visible")
    
    return keypoints
