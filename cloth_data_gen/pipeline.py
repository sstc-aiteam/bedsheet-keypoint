#!/usr/bin/env python3
"""
Warp + Blender Pipeline for Bedsheet Generation
Complete pipeline: Warp simulation -> Blender rendering
"""

import subprocess
import os
import argparse
from pathlib import Path


def run_warp_simulation(width=2.0, height=1.0, resolution=32, steps=300, output_dir="warp_output"):
    """Run Enhanced Warp bedsheet simulation with chaotic winds"""
    print("🚀 Running Enhanced Warp bedsheet simulation with chaotic winds...")
    
    cmd = [
        "python3", "warp_sim.py",
        "--width", str(width),
        "--height", str(height),
        "--resolution", str(resolution),
        "--max_steps", str(steps),
        "--output", output_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Warp simulation completed successfully")
        print(result.stdout)
        return True
    else:
        print("❌ Warp simulation failed")
        print(result.stderr)
        return False


def run_blender_rendering(input_dir, output_dir="blender_output", material="cotton", frames=None):
    """Run Blender rendering"""
    print("🎨 Running Blender rendering...")
    
    # Create a temporary script to handle the rendering
    render_script = f"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render import render_bedsheet
import json
import glob

def main():
    input_dir = "{input_dir}"
    output_dir = "{output_dir}"
    material = "{material}"
    frames = {frames if frames else 'None'}
    
    # Get list of JSON files
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    json_files.sort()
    
    # Only render the final settled frame (bedsheet_final.json)
    final_frame = None
    for json_file in json_files:
        if "final" in json_file.lower():
            final_frame = json_file
            break
    
    # If no final frame found, use the last frame
    if not final_frame and json_files:
        final_frame = json_files[-1]
    
    if not final_frame:
        print("No simulation frames found to render!")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Render only the final settled frame
    output_path = os.path.join(output_dir, 'bedsheet_final.png')
    render_bedsheet(final_frame, output_path, material)
    print(f"Rendered final settled bedsheet: {{final_frame}}")
    
    print(f"Blender rendering completed! Final image saved to {{output_dir}}")

if __name__ == '__main__':
    main()
"""
    
    # Write the temporary script
    with open("temp_blender_render.py", "w") as f:
        f.write(render_script)
    
    # Run Blender with the temporary script - show GPU rendering output
    cmd = ["blender", "-b", "--python", "temp_blender_render.py"]
    
    print("🎨 Starting Blender GPU rendering...")
    result = subprocess.run(cmd, text=True)  # Remove capture_output to show real-time output
    
    # Clean up temporary script
    if os.path.exists("temp_blender_render.py"):
        os.remove("temp_blender_render.py")
    
    if result.returncode == 0:
        print("✅ Blender rendering completed successfully")
        
        # Create visualizations using standalone script
        try:
            rendered_dir = output_dir  # output_dir is already the rendered_images directory
            if os.path.exists(rendered_dir):
                imgs_dir = os.path.join(rendered_dir, 'imgs')
                keypoints_dir = os.path.join(rendered_dir, 'keypoints')
                viz_dir = os.path.join(rendered_dir, 'visualizations')
                
                if os.path.exists(imgs_dir) and os.path.exists(keypoints_dir):
                    os.makedirs(viz_dir, exist_ok=True)
                    
                    # Find the image and keypoint files
                    img_files = [f for f in os.listdir(imgs_dir) if f.endswith('.png')]
                    
                    for img_file in img_files:
                        base_name = img_file.replace('.png', '')
                        keypoint_file = os.path.join(keypoints_dir, f"{base_name}.txt")
                        
                        if os.path.exists(keypoint_file):
                            img_path = os.path.join(imgs_dir, img_file)
                            viz_path = os.path.join(viz_dir, f"{base_name}_keypoints.png")
                            
                            # Create visualization
                            subprocess.run([
                                "python", "visualize_keypoints.py",
                                "--image", img_path,
                                "--keypoints", keypoint_file,
                                "--output", viz_path
                            ], check=True, capture_output=True, text=True)
                            print(f"✅ Created visualization: {viz_path}")
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
        
        return True
    else:
        print("❌ Blender rendering failed")
        print(result.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Warp + Blender bedsheet generation pipeline')
    parser.add_argument('--width', type=float, default=2.0, help='Bedsheet width (m)')
    parser.add_argument('--height', type=float, default=1.0, help='Bedsheet height (m)')
    parser.add_argument('--resolution', type=int, default=40, help='Grid resolution (higher = more vertices, slower)')
    parser.add_argument('--steps', type=int, default=300, help='Simulation steps')
    parser.add_argument('--material', choices=['cotton', 'linen', 'silk'], default='cotton', help='Material type')
    parser.add_argument('--frames', type=int, help='Number of frames to render')
    parser.add_argument('--output', default='warp_blender_output', help='Output directory')
    args = parser.parse_args()
    
    # Create output directories
    warp_output = os.path.join(args.output, 'warp_data')
    blender_output = os.path.join(args.output, 'rendered_images')
    
    Path(warp_output).mkdir(parents=True, exist_ok=True)
    Path(blender_output).mkdir(parents=True, exist_ok=True)
    
    print("🎯 Starting Warp + Blender bedsheet generation pipeline")
    print(f"Bedsheet size: {args.width}m x {args.height}m")
    print(f"Resolution: {args.resolution}")
    print(f"Simulation steps: {args.steps}")
    print(f"Material: {args.material}")
    print(f"Output: {args.output}")
    print()
    
    # Step 1: Run Warp simulation
    if not run_warp_simulation(
        width=args.width,
        height=args.height,
        resolution=args.resolution,
        steps=args.steps,
        output_dir=warp_output
    ):
        print("❌ Pipeline failed at Warp simulation step")
        return False
    
    print()
    
    # Step 2: Run Blender rendering
    if not run_blender_rendering(
        input_dir=warp_output,
        output_dir=blender_output,
        material=args.material,
        frames=args.frames
    ):
        print("❌ Pipeline failed at Blender rendering step")
        return False
    
    print()
    print("🎉 Pipeline completed successfully!")
    print(f"Warp data: {warp_output}")
    print(f"Rendered images: {blender_output}")
    
    # Count output files
    warp_files = len([f for f in os.listdir(warp_output) if f.endswith('.json')])
    blender_files = len([f for f in os.listdir(blender_output) if f.endswith('.png')])
    
    print(f"Generated {warp_files} simulation frames and {blender_files} rendered images")
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
