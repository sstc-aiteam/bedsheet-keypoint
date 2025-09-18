#!/usr/bin/env python3
"""
Blender Integration for Warp Bedsheet Simulation
Imports Warp simulation data and renders in Blender
"""

import bpy
import bmesh
import json
import os
import numpy as np
import random
from mathutils import Vector, Euler
import argparse
from keypoint_tracker import track_bedsheet_keypoints


def clear_scene():
    """Clear the Blender scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Clear materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    
    # Clear meshes
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)


def load_mesh_data(json_path):
    """Load mesh data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    vertices = np.array(data['vertices'])
    faces = data['faces']
    grid_size = data.get('grid_size', None)
    
    return vertices, faces, grid_size


def create_bedsheet_mesh(vertices, faces, name="Bedsheet", grid_size=None):
    """Create a bedsheet mesh in Blender with corner vertex groups"""
    # Create new mesh
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    
    # Link to scene
    bpy.context.collection.objects.link(obj)
    
    # Create mesh from vertices and faces
    mesh.from_pydata(vertices.tolist(), [], faces)
    mesh.update()
    
    # Create vertex groups for corner vertices if grid size is provided
    if grid_size and len(grid_size) == 2:
        nx, ny = grid_size
        total_vertices = len(vertices)
        
        # Calculate corner indices
        corner_indices = [
            0,                    # Bottom-left (0, 0)
            ny - 1,              # Bottom-right (0, ny-1) 
            (nx - 1) * ny,       # Top-left (nx-1, 0)
            (nx - 1) * ny + ny - 1  # Top-right (nx-1, ny-1)
        ]
        
        # Create vertex groups for each corner
        corner_names = ['Corner_BL', 'Corner_BR', 'Corner_TL', 'Corner_TR']
        corner_labels = ['Bottom-Left', 'Bottom-Right', 'Top-Left', 'Top-Right']
        
        for i, (corner_idx, corner_name, corner_label) in enumerate(zip(corner_indices, corner_names, corner_labels)):
            if corner_idx < total_vertices:
                # Create vertex group
                vertex_group = obj.vertex_groups.new(name=corner_name)
                vertex_group.add([corner_idx], 1.0, 'REPLACE')
                print(f"  - Created vertex group '{corner_name}' for {corner_label} corner (vertex {corner_idx})")
            else:
                print(f"  - Warning: Corner index {corner_idx} out of range for {total_vertices} vertices")
    
    # Enter edit mode to recalculate normals
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return obj


def create_realistic_material(obj, material_type="cotton"):
    """Create realistic material for bedsheet"""
    # Create material
    mat = bpy.data.materials.new(name=f"{material_type}_material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    
    # Material properties based on type
    if material_type == "cotton":
        base_color = random.choice([
            (0.9, 0.9, 0.95, 1.0),  # White
            (0.8, 0.9, 1.0, 1.0),   # Light blue
            (1.0, 0.9, 0.8, 1.0),   # Cream
        ])
        roughness = 0.6  # Less rough for better edge definition
        specular = 0.3   # Higher specular for better edge visibility
        metallic = 0.0   # Non-metallic
        transmission = 0.0  # No transmission
        emission_strength = 0.0  # No emission
    elif material_type == "linen":
        base_color = random.choice([
            (0.95, 0.9, 0.8, 1.0),  # Linen
            (0.9, 0.85, 0.7, 1.0),  # Beige
            (0.8, 0.9, 0.8, 1.0),   # Light green
        ])
        roughness = 0.7  # Less rough for better edge definition
        specular = 0.25  # Higher specular for better edge visibility
        metallic = 0.0   # Non-metallic
        transmission = 0.0  # No transmission
        emission_strength = 0.0  # No emission
    else:  # silk
        base_color = random.choice([
            (1.0, 0.8, 0.8, 1.0),   # Pink
            (0.8, 0.8, 1.0, 1.0),   # Light purple
            (0.9, 1.0, 0.8, 1.0),   # Light green
        ])
        roughness = 0.1  # Very smooth for better edge definition
        specular = 0.5   # High specular for better edge visibility
        metallic = 0.0   # Non-metallic
        transmission = 0.0  # No transmission
        emission_strength = 0.0  # No emission
    
    # Set material properties for better crease visibility
    bsdf.inputs["Base Color"].default_value = base_color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = specular
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Transmission Weight"].default_value = transmission
    bsdf.inputs["Emission Strength"].default_value = emission_strength
    
    # Add a normal map for better edge definition
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Create a noise texture for normal mapping
    noise_tex = nodes.new(type='ShaderNodeTexNoise')
    noise_tex.inputs["Scale"].default_value = 50.0  # Fine detail
    noise_tex.inputs["Detail"].default_value = 15.0  # High detail
    noise_tex.inputs["Roughness"].default_value = 0.5
    noise_tex.inputs["Distortion"].default_value = 0.0
    
    # Create a normal map node
    normal_map = nodes.new(type='ShaderNodeNormalMap')
    normal_map.inputs["Strength"].default_value = 0.3  # Subtle normal mapping
    
    # Connect noise to normal map
    links.new(noise_tex.outputs["Fac"], normal_map.inputs["Color"])
    
    # Connect normal map to BSDF
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Alpha"].default_value = 1.0  # Fully opaque
    
    # Add slight subsurface scattering for more realistic fabric appearance
    bsdf.inputs["Subsurface Weight"].default_value = 0.1  # Slight subsurface
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.2, 0.1)  # Red subsurface
    
    # Assign material
    obj.data.materials.append(mat)
    
    return mat


def setup_lighting():
    """Setup varied lighting for natural shadow variations on bedsheet"""
    # Clear existing lights
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Primary overhead light (main light source - enhanced for better edge definition)
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 4))
    main_light = bpy.context.active_object
    main_light.name = "Main Overhead Light"
    main_light.data.energy = 10.0  # Increased for better contrast
    main_light.data.color = (1.0, 0.98, 0.95)  # Warm daylight
    main_light.data.size = 2.5  # Smaller for sharper shadows
    main_light.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')  # Pointing straight down
    
    # Secondary angled light (creates varied shadows on bedsheet surface)
    bpy.ops.object.light_add(type='AREA', location=(1.5, 1.5, 3.5))
    secondary_light = bpy.context.active_object
    secondary_light.name = "Secondary Light"
    secondary_light.data.energy = 5.0  # Increased for better edge definition
    secondary_light.data.color = (0.98, 1.0, 0.95)  # Slightly different color
    secondary_light.data.size = 2.0  # Smaller for sharper shadows
    secondary_light.rotation_euler = Euler((0.2, 0.3, 0.0), 'XYZ')  # Angled
    
    # Accent light from different angle (adds more shadow complexity)
    bpy.ops.object.light_add(type='AREA', location=(-1.2, 1.8, 3.2))
    accent_light = bpy.context.active_object
    accent_light.name = "Accent Light"
    accent_light.data.energy = 3.0  # Reduced for more dramatic shadows
    accent_light.data.color = (0.95, 0.95, 1.0)  # Slightly cool
    accent_light.data.size = 2.0  # Smaller for more defined shadows
    accent_light.rotation_euler = Euler((0.1, -0.4, 0.0), 'XYZ')  # Different angle
    
    # Additional fill light from opposite side (minimal to preserve shadows)
    bpy.ops.object.light_add(type='AREA', location=(2, -1.5, 3))
    fill_light2 = bpy.context.active_object
    fill_light2.name = "Fill Light 2"
    fill_light2.data.energy = 1.5  # Much reduced to preserve shadow contrast
    fill_light2.data.color = (0.95, 0.98, 1.0)  # Cool fill
    fill_light2.data.size = 2.0  # Smaller for more defined shadows
    fill_light2.rotation_euler = Euler((0.3, -0.2, 0.0), 'XYZ')
    
    # Soft fill light (minimal to preserve shadow variation)
    bpy.ops.object.light_add(type='AREA', location=(0, -2, 2.5))
    fill_light = bpy.context.active_object
    fill_light.name = "Fill Light"
    fill_light.data.energy = 1.0  # Much reduced to preserve shadows
    fill_light.data.color = (0.9, 0.95, 1.0)  # Cool fill
    fill_light.data.size = 2.5  # Smaller for more defined shadows
    fill_light.rotation_euler = Euler((0.4, 0.0, 0.0), 'XYZ')
    
    # Additional side fill lights to eliminate black areas
    bpy.ops.object.light_add(type='AREA', location=(-2, 0, 3))
    side_fill1 = bpy.context.active_object
    side_fill1.name = "Side Fill 1"
    side_fill1.data.energy = 1.0  # Reduced for more dramatic shadows
    side_fill1.data.color = (0.95, 0.95, 1.0)
    side_fill1.data.size = 2.0  # Smaller for more defined shadows
    side_fill1.rotation_euler = Euler((0.2, 0.0, 0.0), 'XYZ')
    
    bpy.ops.object.light_add(type='AREA', location=(0, 2, 3))
    side_fill2 = bpy.context.active_object
    side_fill2.name = "Side Fill 2"
    side_fill2.data.energy = 1.0  # Reduced for more dramatic shadows
    side_fill2.data.color = (0.95, 0.95, 1.0)
    side_fill2.data.size = 2.0  # Smaller for more defined shadows
    side_fill2.rotation_euler = Euler((0.2, 0.0, 0.0), 'XYZ')
    
    # Minimal ambient light (preserves shadow detail)
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 1))
    ambient_light = bpy.context.active_object
    ambient_light.name = "Ambient Light"
    ambient_light.data.energy = 0.8  # Much reduced to preserve shadow contrast
    ambient_light.data.color = (0.9, 0.92, 0.95)
    ambient_light.data.size = 8.0  # Very large for even ambient lighting
    ambient_light.rotation_euler = Euler((1.57, 0.0, 0.0), 'XYZ')  # Pointing up


def analyze_bedsheet_bounds(bedsheet):
    """Analyze the bedsheet's bounding box using only the 4 corner vertex groups"""
    bpy.context.view_layer.update()  # Ensure mesh data is up-to-date
    
    world_matrix = bedsheet.matrix_world
    
    # Get the corner vertex groups
    corner_groups = ['Corner_BL', 'Corner_BR', 'Corner_TL', 'Corner_TR']
    corner_positions = []
    
    for group_name in corner_groups:
        if group_name in bedsheet.vertex_groups:
            vertex_group = bedsheet.vertex_groups[group_name]
            # Get the first vertex in the group (should be only one)
            for vertex in bedsheet.data.vertices:
                for group in vertex.groups:
                    if group.group == vertex_group.index:
                        world_co = world_matrix @ vertex.co
                        corner_positions.append(world_co)
                        break
                else:
                    continue
                break
    
    if len(corner_positions) != 4:
        print(f"Warning: Expected 4 corner vertices, found {len(corner_positions)}")
        # Fallback to original method if corners not found
        return analyze_bedsheet_bounds_fallback(bedsheet)
    
    # Calculate bounding box from corner positions
    min_x = min(pos.x for pos in corner_positions)
    max_x = max(pos.x for pos in corner_positions)
    min_y = min(pos.y for pos in corner_positions)
    max_y = max(pos.y for pos in corner_positions)
    min_z = min(pos.z for pos in corner_positions)
    max_z = max(pos.z for pos in corner_positions)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    
    print(f"  - Corner-based bounding box: center=({center_x:.2f}, {center_y:.2f}, {center_z:.2f}), size=({size_x:.2f}, {size_y:.2f}, {size_z:.2f})")
    
    return Vector((center_x, center_y, center_z)), Vector((size_x, size_y, size_z))

def analyze_bedsheet_bounds_fallback(bedsheet):
    """Fallback method using entire mesh if corner groups not available"""
    bpy.context.view_layer.update()  # Ensure mesh data is up-to-date
    
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')
    min_z, max_z = float('inf'), float('-inf')

    mesh = bedsheet.data
    for vertex in mesh.vertices:
        world_co = bedsheet.matrix_world @ vertex.co
        min_x = min(min_x, world_co.x)
        max_x = max(max_x, world_co.x)
        min_y = min(min_y, world_co.y)
        max_y = max(max_y, world_co.y)
        min_z = min(min_z, world_co.z)
        max_z = max(max_z, world_co.z)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    
    return Vector((center_x, center_y, center_z)), Vector((size_x, size_y, size_z))


def setup_adaptive_camera(bedsheet):
    """Setup adaptive camera that dynamically positions based on bedsheet bounds for optimal keypoint visibility."""
    center, size = analyze_bedsheet_bounds(bedsheet)
    
    # Get the bedsheet's bounding box dimensions
    bedsheet_width = size.x  # X dimension (width)
    bedsheet_height = size.y  # Y dimension (height)
    max_bedsheet_dim = max(bedsheet_width, bedsheet_height)
    
    # Calculate adaptive camera positioning based on bedsheet bounding box
    # We want the bedsheet to fill about 60-80% of the frame
    target_frame_ratio = random.uniform(0.6, 0.8)
    
    # Calculate required camera distance based on bedsheet size and target frame ratio
    # For a top-down view, we need to consider the diagonal of the bedsheet
    bedsheet_diagonal = np.sqrt(size.x**2 + size.y**2)
    
    # Calculate required distance to fit bedsheet in frame
    # Using similar triangles: bedsheet_diagonal / distance = frame_ratio
    required_distance = bedsheet_diagonal / (2 * target_frame_ratio)
    
    # Add some margin and random variation
    camera_distance = required_distance * random.uniform(1.2, 1.5)
    camera_height = camera_distance  # For top-down view, height = distance
    
    # Ensure minimum values
    camera_distance = max(camera_distance, 2.0)
    camera_height = max(camera_height, 2.0)
    
    # Position camera directly above the bedsheet center
    camera_x = center.x
    camera_y = center.y
    camera_z = center.z + camera_height
    
    bpy.ops.object.camera_add(
        location=(camera_x, camera_y, camera_z), 
        rotation=(0, 0, 0)  # Initial rotation, will be adjusted to look at bedsheet
    )
    cam = bpy.context.active_object
    cam.name = "Camera"
    
    # Use normal perspective camera
    cam.data.type = 'PERSP'  # Use perspective projection
    
    # Calculate appropriate lens based on distance and bedsheet size
    # For a given distance, we can calculate the field of view needed
    fov_angle = 2 * np.arctan(bedsheet_diagonal / (2 * camera_distance))
    # Convert FOV to lens focal length (approximate)
    sensor_width = 32  # Standard 35mm sensor width
    focal_length = sensor_width / (2 * np.tan(fov_angle / 2))
    
    # Add some random variation to the lens
    cam.data.lens = focal_length * random.uniform(0.8, 1.2)
    
    # Ensure lens is within reasonable bounds
    cam.data.lens = max(10, min(100, cam.data.lens))
    
    bpy.context.scene.camera = cam
    
    # Use Blender's built-in look_at functionality to ensure camera points at bedsheet
    # First, select the camera and make it active
    bpy.context.view_layer.objects.active = cam
    cam.select_set(True)
    
    # Use Blender's built-in look_at to point camera at bedsheet center
    bpy.ops.object.constraint_add(type='TRACK_TO')
    constraint = cam.constraints[-1]
    constraint.target = bpy.data.objects.get('Bedsheet')
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    # Update the constraint to apply the rotation
    bpy.context.view_layer.update()
    
    # Remove the constraint after positioning
    cam.constraints.remove(constraint)
    
    print(f"  - ADAPTIVE Camera positioned at ({camera_x:.2f}, {camera_y:.2f}, {camera_z:.2f})")
    print(f"  - Camera height: {camera_height:.2f}m, distance: {camera_distance:.2f}m")
    print(f"  - Bedsheet diagonal: {bedsheet_diagonal:.2f}m, target frame ratio: {target_frame_ratio:.2f}")
    print(f"  - Camera lens: {cam.data.lens:.1f}mm (calculated from FOV: {np.degrees(fov_angle):.1f}°)")
    print(f"  - Bedsheet center: ({center.x:.2f}, {center.y:.2f}, {center.z:.2f}), size: ({size.x:.2f}, {size.y:.2f}, {size.z:.2f})")
    print(f"  - Camera rotation: ({np.degrees(cam.rotation_euler.x):.1f}°, {np.degrees(cam.rotation_euler.y):.1f}°, {np.degrees(cam.rotation_euler.z):.1f}°) - ADAPTIVE TOP-DOWN")
    
    return cam


def setup_world():
    """Setup world background"""
    world = bpy.context.scene.world
    world.use_nodes = True
    
    # Clear existing nodes
    world.node_tree.nodes.clear()
    
    # Add background node
    bg_node = world.node_tree.nodes.new(type='ShaderNodeBackground')
    output_node = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
    
    # Connect nodes
    world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    
    # Set background color - much brighter to reduce overall darkness
    bg_color = random.choice([
        (0.3, 0.35, 0.4, 1.0),  # Brighter blue-gray
        (0.35, 0.3, 0.35, 1.0), # Brighter purple-gray
        (0.3, 0.4, 0.3, 1.0),   # Brighter green-gray
    ])
    bg_node.inputs['Color'].default_value = bg_color
    bg_node.inputs['Strength'].default_value = 0.8  # Much brighter background


def create_floor():
    """Create a floor plane with better visibility"""
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    
    # Create floor material with better contrast
    mat = bpy.data.materials.new(name="Floor Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    
    # Use dark purple for better contrast with bedsheet
    floor_color = random.choice([
        (0.2, 0.1, 0.3, 1.0),  # Dark purple
        (0.25, 0.1, 0.35, 1.0),  # Slightly lighter dark purple
        (0.15, 0.05, 0.25, 1.0),  # Very dark purple
        (0.3, 0.15, 0.4, 1.0),  # Medium dark purple
    ])
    bsdf.inputs["Base Color"].default_value = floor_color
    bsdf.inputs["Roughness"].default_value = 0.9  # Very rough for realistic floor
    bsdf.inputs["Specular IOR Level"].default_value = 0.0  # No specular reflection
    
    floor.data.materials.append(mat)
    
    return floor


def render_bedsheet(json_path, output_path, material_type="cotton"):
    """Render a single bedsheet frame with adaptive camera focusing on the bedsheet"""
    # Clear scene
    clear_scene()
    
    # Load mesh data
    vertices, faces, grid_size = load_mesh_data(json_path)
    
    # Create bedsheet mesh with corner vertex groups
    bedsheet = create_bedsheet_mesh(vertices, faces, grid_size=grid_size)
    
    # Create material
    create_realistic_material(bedsheet, material_type)
    
    # Setup scene
    setup_lighting()
    setup_world()
    create_floor()
    
    # Setup adaptive camera that focuses on the bedsheet
    camera = setup_adaptive_camera(bedsheet)
    
    # Set render settings - balanced for quality and inference consistency
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 1024  # Balanced resolution for consistent keypoint inference
    scene.render.resolution_y = 1024
    scene.render.filepath = output_path
    
    # Configure GPU acceleration properly
    scene.cycles.device = 'GPU'
    
    # Set up GPU compute devices - proper headless configuration
    try:
        # Enable Cycles addon if not already enabled
        bpy.ops.preferences.addon_enable(module='cycles')
        
        # Get preferences and refresh devices
        prefs = bpy.context.preferences.addons['cycles'].preferences
        
        # Try different compute device types in order of preference
        device_types = ['OPTIX', 'CUDA', 'HIP', 'OPENCL', 'ONEAPI']
        gpu_devices_found = False
        
        for device_type in device_types:
            try:
                prefs.compute_device_type = device_type
                prefs.refresh_devices()
                
                # Check if any GPU devices are available for this type
                available_gpus = [d for d in prefs.devices if d.type == device_type]
                if available_gpus:
                    print(f"  - Found {device_type} devices, configuring...")
                    
                    # Enable all available GPU devices and disable CPU
                    for device in prefs.devices:
                        if device.type == device_type:
                            device.use = True
                            gpu_devices_found = True
                            print(f"  - Enabled GPU device: {device.name} ({device.type})")
                        elif device.type == 'CPU':
                            device.use = False  # Disable CPU to force GPU usage
                            print(f"  - Disabled CPU device: {device.name}")
                    
                    if gpu_devices_found:
                        break
                        
            except Exception as e:
                print(f"  - Failed to configure {device_type}: {e}")
                continue
        
        if not gpu_devices_found:
            print("  - No GPU devices found, falling back to CPU")
            # Re-enable CPU if no GPU found
            for device in prefs.devices:
                if device.type == 'CPU':
                    device.use = True
            scene.cycles.device = 'CPU'
        else:
            # Force GPU compute device selection
            scene.cycles.device = 'GPU'
            print(f"  - GPU rendering enabled with {len([d for d in prefs.devices if d.use and d.type in ['CUDA', 'OPTIX', 'OPENCL', 'HIP']])} devices")
            
    except Exception as e:
        print(f"  - Error configuring GPU: {e}")
        print("  - Falling back to CPU rendering")
        # Re-enable CPU on error
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            for device in prefs.devices:
                if device.type == 'CPU':
                    device.use = True
        except:
            pass
        scene.cycles.device = 'CPU'
    
    # Optimize render settings for GPU rendering - balanced for inference consistency
    if gpu_devices_found:
        scene.cycles.samples = 512  # Balanced samples for consistent quality and speed
        scene.cycles.use_denoising = True  # Enable denoising
        scene.cycles.denoiser = 'OPTIX'  # Use OptiX denoiser for NVIDIA GPUs
        
        # GPU memory optimization settings
        scene.cycles.tile_size = 256  # Optimal tile size for 1024x1024 resolution
        scene.cycles.debug_use_spatial_splits = True  # Better memory management
        scene.cycles.debug_use_hair_bvh = True  # Enable hair BVH for better memory usage
        
        print("  - GPU render settings: 512 samples, 1024x1024, OptiX denoising, tile size 256")
    else:
        scene.cycles.samples = 128  # Lower samples for CPU rendering
        scene.cycles.use_denoising = True  # Enable denoising
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'  # Use OpenImageDenoise for CPU
        print("  - CPU render settings: 128 samples, OpenImageDenoise")
    
    # Enable better shadow and lighting features
    scene.cycles.feature_set = 'SUPPORTED'  # Use supported features only
    # Note: debug settings are set above based on GPU/CPU mode
    
    # Enable contact shadows for better ground contact
    scene.cycles.contact_shadow_distance = 0.1
    scene.cycles.contact_shadow_bias = 0.03
    
    # Better light bounces for natural lighting and soft shadows - optimized for GPU
    if gpu_devices_found:
        # Initial bounce settings - will be overridden below for maximum GPU usage
        scene.cycles.max_bounces = 16  # More bounces for better quality and GPU utilization
        scene.cycles.diffuse_bounces = 8  # More diffuse bounces for natural light
        scene.cycles.glossy_bounces = 8  # More glossy bounces for realistic reflections
        scene.cycles.transmission_bounces = 6
        scene.cycles.volume_bounces = 4
    else:
        scene.cycles.max_bounces = 12  # More bounces for softer shadows
        scene.cycles.diffuse_bounces = 6  # More diffuse bounces for natural light
        scene.cycles.glossy_bounces = 6  # More glossy bounces for realistic reflections
        scene.cycles.transmission_bounces = 4
        scene.cycles.volume_bounces = 2
    
    # Enable light sampling for better area light shadows
    scene.cycles.use_light_tree = True
    
    # Additional GPU memory optimization
    if gpu_devices_found:
        scene.cycles.use_adaptive_sampling = True  # Adaptive sampling for better GPU utilization
        scene.cycles.adaptive_threshold = 0.01  # Lower threshold for better quality
        scene.cycles.adaptive_min_samples = 64  # Minimum samples before adaptive kicks in
        
        # Force GPU memory allocation
        scene.cycles.device = 'GPU'
        scene.cycles.tile_size = 256  # Optimal tile size for 1024x1024 resolution
        scene.cycles.debug_use_spatial_splits = True  # Better memory management
        scene.cycles.debug_use_hair_bvh = True  # Enable hair BVH for better memory usage
        
        # Additional GPU memory settings - balanced for consistency
        scene.cycles.use_light_tree = True  # Enable light tree for better GPU utilization
        scene.cycles.max_bounces = 16  # Balanced bounces for quality and speed
        scene.cycles.diffuse_bounces = 8
        scene.cycles.glossy_bounces = 8
        scene.cycles.transmission_bounces = 6
        scene.cycles.volume_bounces = 4
        
        print("  - GPU memory optimization: tile size 256, 16 bounces, adaptive sampling, 1024x1024")
    
    # Render with GPU memory monitoring
    print("  - Starting GPU rendering...")
    if gpu_devices_found:
        print("  - Monitor GPU usage with: nvidia-smi")
    
    bpy.ops.render.render(write_still=True)
    
    print(f"Rendered: {output_path}")
    
    # Final GPU status check
    if gpu_devices_found:
        print("  - GPU rendering completed - check nvidia-smi for memory usage")
    
    # Track keypoints after rendering
    try:
        # Extract image name from output path
        image_name = os.path.splitext(os.path.basename(output_path))[0]
        output_dir = os.path.dirname(output_path)
        
        # Create imgs directory if it doesn't exist
        imgs_dir = os.path.join(output_dir, 'imgs')
        os.makedirs(imgs_dir, exist_ok=True)
        
        # Move rendered image to imgs directory
        img_path = os.path.join(imgs_dir, f"{image_name}.png")
        if os.path.exists(output_path):
            os.rename(output_path, img_path)
            print(f"Moved rendered image to: {img_path}")
        
        # Track keypoints using vertex groups
        keypoints = track_bedsheet_keypoints(bedsheet, camera, output_dir, image_name, json_path)
        print(f"Keypoints tracked and saved for {image_name}")
        
    except Exception as e:
        print(f"Error tracking keypoints: {e}")


def main():
    parser = argparse.ArgumentParser(description='Blender rendering for Warp bedsheet simulation')
    parser.add_argument('--input', required=True, help='Input JSON file or directory')
    parser.add_argument('--output', default='blender_render_output', help='Output directory')
    parser.add_argument('--material', choices=['cotton', 'linen', 'silk'], default='cotton', help='Material type')
    parser.add_argument('--frames', type=int, help='Number of frames to render (if input is directory)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if os.path.isfile(args.input):
        # Single file
        output_path = os.path.join(args.output, 'bedsheet_render.png')
        render_bedsheet(args.input, output_path, args.material)
    else:
        # Directory of files
        json_files = [f for f in os.listdir(args.input) if f.endswith('.json')]
        json_files.sort()
        
        if args.frames:
            json_files = json_files[:args.frames]
        
        for i, json_file in enumerate(json_files):
            input_path = os.path.join(args.input, json_file)
            output_path = os.path.join(args.output, f'bedsheet_{i:04d}.png')
            render_bedsheet(input_path, output_path, args.material)
    
    print(f"Rendering completed! Images saved to {args.output}")


if __name__ == '__main__':
    main()
