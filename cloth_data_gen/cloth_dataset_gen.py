import bpy
import numpy as np
import os
import random
from mathutils import Vector, Euler

# Parameters
output_dir = "./output"
n_samples = 3000
length_range = (3.5, 5.0)
width_range = (3.5, 5.0)
res = 40  # Subdivisions per side

def ensure_dirs():
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/keypoints", exist_ok=True)

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

def create_cloth(length, width):
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=res, y_subdivisions=res, size=1)
    cloth = bpy.context.active_object
    cloth.scale = (length/2, width/2, 1)
    cloth.name = "Cloth"
    bpy.ops.object.shade_smooth()
    return cloth

def assign_random_color(obj):
    # Create and assign a material with either light or dark color
    mat = bpy.data.materials.new("ClothColorMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    # 隨機選擇亮色或暗色
    if random.random() < 0.5:
        # 暗色
        color = (
            random.uniform(0.0, 0.4),
            random.uniform(0.0, 0.4),
            random.uniform(0.0, 0.4),
            1
        )
    else:
        # 亮色
        color = (
            random.uniform(0.7, 1.0),
            random.uniform(0.7, 1.0),
            random.uniform(0.7, 1.0),
            1
        )
    bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.clear()
    obj.data.materials.append(mat)

# def assign_random_color(obj):
#     # Create and assign a pure white material to the cloth
#     mat = bpy.data.materials.new("ClothColorMat")
#     mat.use_nodes = True
#     bsdf = mat.node_tree.nodes["Principled BSDF"]
#     # Set to pure white
#     color = (1.0, 1.0, 1.0, 1.0)
#     bsdf.inputs["Base Color"].default_value = color
#     obj.data.materials.clear()
#     obj.data.materials.append(mat)

def bend_cloth(obj):
    bend = obj.modifiers.new(name="Bend", type='SIMPLE_DEFORM')
    bend.deform_method = 'BEND'
    bend.angle = np.radians(random.uniform(60, 150))
    bend.origin = None
    bend.limits = (0, 1)
    bend.deform_axis = random.choice(['X', 'Y'])

def add_random_crease(obj):
    disp = obj.modifiers.new(name="RandCrease", type='DISPLACE')
    tex = bpy.data.textures.new("RandCreaseTex", type='CLOUDS')
    tex.noise_scale = random.uniform(0.2, 0.6) # Random frequency
    disp.texture = tex
    disp.strength = random.uniform(0.05, 0.2) # Random crease strength
    disp.mid_level = 0.5
    disp.direction = 'NORMAL'

def add_bend_fold(obj):
    bend = obj.modifiers.new(name="RandBend", type='SIMPLE_DEFORM')
    bend.deform_method = 'BEND'
    bend.deform_axis = random.choice(['X', 'Y'])
    bend.angle = np.radians(random.uniform(-25, 25))
    bend.limits = (random.uniform(0, 0.5), random.uniform(0.5, 1))

def set_black_background_with_min_fill(strength=0.03):
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg_node = world.node_tree.nodes['Background']
    bg_node.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)  # RGBA pure black
    bg_node.inputs[1].default_value = strength  # very small ambient strength for shadow lift (0.01–0.05)

def add_steady_light():
    # Create a soft, always-on fill light above the cloth
    light_data = bpy.data.lights.new(name="ClothSteadyLight", type='AREA')
    light_obj = bpy.data.objects.new(name="ClothSteadyLightObj", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    # Place above the scene, pointing down
    light_obj.location = (0, 0, 6)
    light_obj.rotation_euler = (0, 0, 0)  # Directly down
    # Optionally: make area light face -Z for Blender, if needed
    # light_obj.rotation_euler = (np.radians(90), 0, 0)
    # Set intensity and color
    light_data.energy = 100  # Tune as needed; 50–200 is typical for AREA
    light_data.color = (1.0, 1.0, 1.0)  # White light
    light_data.shape = 'SQUARE'
    light_data.size = 3.0  # Covers most cloth; adjust as needed
    light_data.shadow_soft_size = 0.7
    return light_obj

def add_random_sunlight():
    # Remove any old 'ClothSun' lights if present
    for obj in bpy.data.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN' and obj.name.startswith('ClothSun'):
            bpy.data.objects.remove(obj, do_unlink=True)

    # Add a randomized SUN light to the scene
    sun_data = bpy.data.lights.new(name="ClothSunLight", type='SUN')
    sun_obj = bpy.data.objects.new(name="ClothSun", object_data=sun_data)
    bpy.context.collection.objects.link(sun_obj)
    # Point from high above, with some random in-plane rotation
    sun_obj.location = (0, 0, 20)
    # Random downward direction with gentle angle variance
    theta = np.radians(random.uniform(35, 85))  # angle from vertical (zenith)
    phi = np.radians(random.uniform(0, 360))    # azimuth
    sun_obj.rotation_euler = (theta, 0, phi)
    # Energy and color (simulate daylight/times)
    sun_data.energy = random.uniform(2, 8)
    # White-yellowish sun
    sun_data.color = (
        random.uniform(0.92, 1.00),  # R
        random.uniform(0.90, 1.00),  # G
        random.uniform(0.85, 1.00)   # B
    )
    sun_data.angle = np.radians(random.uniform(1, 10))   # Sun "softness" (shadow blur)
    return sun_obj

def randomize_world_background():
    # Randomly set the Blender world background (simulate sky/surrounding)
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg_node = world.node_tree.nodes['Background']
    # Light blue, warm, or neutral
    hue = random.uniform(0.50, 0.67)    # 0.50 = blue, 0.67 = sky blue
    sat = random.uniform(0.2, 0.5)
    val = random.uniform(0.5, 0.9)
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    bg_node.inputs[0].default_value = (r, g, b, 1.0)
    bg_node.inputs[1].default_value = random.uniform(0.9, 2.0)  # brightness


def random_deform(obj):
    for i in range(10):
        add_random_crease(obj)
        add_bend_fold(obj)

def get_corner_indices(grid_res):
    n = grid_res
    return [
        0,               # Bottom-left
        n - 1,           # Bottom-right
        n * n - 1,     # Top-left
        n * n - 2        # Top-right
    ]

def add_fill_light():
    light_data = bpy.data.lights.new(name="FillLight", type='AREA')
    light_obj = bpy.data.objects.new(name="FillLightObj", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (3, -4, 4)
    light_data.energy = 200
    light_data.color = (1.0, 1.0, 1.0)
    light_data.shape = 'SQUARE'
    light_data.size = 2.5
    return light_obj

def apply_random_rotation(obj):
    # Apply a random 3D rotation
    rot_x = np.radians(random.uniform(-10, 10))      # keep cloth mostly upright, but slight x/y tilt is ok
    rot_y = np.radians(random.uniform(-10, 10))
    rot_z = np.radians(random.uniform(0, 360))
    obj.rotation_euler = Euler((rot_x, rot_y, rot_z), 'XYZ')

def visualize_keypoints(obj, indices):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.to_mesh()
    mat = bpy.data.materials.new("RedMat")
    mat.diffuse_color = (1, 0, 0, 1)
    for idx in indices:
        co = mesh_eval.vertices[idx].co
        glob = obj.matrix_world @ co
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.06, location=glob)
        kp = bpy.context.active_object
        kp.data.materials.append(mat)
        kp.name = "Keypoint"
    obj_eval.to_mesh_clear()

def setup_camera():
    bpy.ops.object.camera_add(location=(0, -6, 3), rotation=(1.25, 0, 0))
    cam = bpy.context.active_object
    bpy.context.scene.camera = cam
    bpy.context.scene.render.resolution_x = 128
    bpy.context.scene.render.resolution_y = 128
    return cam

from bpy_extras.object_utils import world_to_camera_view

def world_to_pixel(coord_world, camera, scene):
    co_ndc = world_to_camera_view(scene, camera, coord_world)
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    x_pixel = int(co_ndc.x * width)
    y_pixel = int((1 - co_ndc.y) * height)
    return (x_pixel, y_pixel)


def render_sample(i, cloth, cam, indices):
    scene = bpy.context.scene
    bpy.context.scene.render.filepath = f"{output_dir}/images/cloth_{i:04d}.png"
    bpy.ops.render.render(write_still=True)
    # Save 3D and 2D (pixel-space) keypoints
    with open(f"{output_dir}/keypoints/cloth_{i:04d}.txt", "w") as f:
        f.write("x_world,y_world,z_world,x_pixel,y_pixel\n")
        # --- Modifier-aware robust keypoint export ---
        depsgraph = bpy.context.evaluated_depsgraph_get()
        cloth_eval = cloth.evaluated_get(depsgraph)
        mesh_eval = cloth_eval.to_mesh()

        for idx in indices:
            vertex = mesh_eval.vertices[idx]
            global_co = cloth_eval.matrix_world @ vertex.co
            px, py = world_to_pixel(global_co, cam, scene)
            f.write(f"{global_co.x:.4f},{global_co.y:.4f},{global_co.z:.4f},{px},{py}\n")

        cloth_eval.to_mesh_clear()


def main():
    ensure_dirs()
    for i in range(n_samples):
        clear_scene()
        l = random.uniform(*length_range)
        w = random.uniform(*width_range)
        cloth = create_cloth(l, w)
        assign_random_color(cloth)
        random_deform(cloth)
        apply_random_rotation(cloth)
        cam = setup_camera()
        sun_obj = add_random_sunlight()  # <<<<<<
        # (optionally: set_black_background_with_min_fill() here)
        set_black_background_with_min_fill(0.02)  # or 0.05 if too harsh
        indices = get_corner_indices(res)
        render_sample(i, cloth, cam, indices)
        bpy.data.objects.remove(sun_obj, do_unlink=True)
        # bpy.data.objects.remove(light_obj_fill, do_unlink=True)


    print("Dataset generation complete!")

if __name__ == "__main__":
    main()