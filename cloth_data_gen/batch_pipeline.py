#!/usr/bin/env python3
"""
Fast Batch Pipeline for Bedsheet Generation
Ultra-fast generation using optimized fast pipeline with full GPU utilization
"""

import subprocess
import os
import argparse
import time
from pathlib import Path
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


def generate_fast_bedsheet(args):
    """Generate a single bedsheet using the fast pipeline"""
    width, height, resolution, steps, material, output_dir, index = args
    
    # Create unique output directory for this bedsheet
    bedsheet_output = os.path.join(output_dir, f"bedsheet_{index:04d}")
    
    # Run the fast pipeline
    cmd = [
        "python", "pipeline.py",
        "--width", str(width),
        "--height", str(height),
        "--resolution", str(resolution),
        "--steps", str(steps),
        "--material", material,
        "--output", bedsheet_output
    ]
    
    start_time = time.time()
    print(f"  🔧 Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    generation_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"  ❌ Pipeline failed with return code {result.returncode}")
        print(f"  📝 Error output: {result.stderr}")
    else:
        print(f"  ✅ Pipeline completed successfully")
    
    if result.returncode == 0:
        # Organize output into imgs/ and keypoints/ directories
        rendered_dir = os.path.join(bedsheet_output, "rendered_images")
        print(f"  📁 Looking for rendered_dir: {rendered_dir}")
        print(f"  📁 Rendered dir exists: {os.path.exists(rendered_dir)}")
        if os.path.exists(rendered_dir):
            print(f"  📁 Contents of rendered_dir: {os.listdir(rendered_dir)}")
            # Create imgs and keypoints directories
            imgs_dir = os.path.join(output_dir, "imgs")
            keypoints_dir = os.path.join(output_dir, "keypoints")
            os.makedirs(imgs_dir, exist_ok=True)
            os.makedirs(keypoints_dir, exist_ok=True)
            
            # Move images and keypoints from subdirectories
            image_moved = False
            keypoint_moved = False
            
            # Check imgs subdirectory
            imgs_subdir = os.path.join(rendered_dir, "imgs")
            if os.path.exists(imgs_subdir):
                for file in os.listdir(imgs_subdir):
                    if file.endswith('.png'):
                        # Move image to main imgs directory
                        src = os.path.join(imgs_subdir, file)
                        dst = os.path.join(imgs_dir, f"bedsheet_{index:04d}.png")
                        os.rename(src, dst)
                        image_moved = True
                        print(f"  📸 Moved image: {src} -> {dst}")
            
            # Check keypoints subdirectory
            keypoints_subdir = os.path.join(rendered_dir, "keypoints")
            if os.path.exists(keypoints_subdir):
                for file in os.listdir(keypoints_subdir):
                    if file.endswith('.txt'):
                        # Move keypoints to main keypoints directory
                        src = os.path.join(keypoints_subdir, file)
                        dst = os.path.join(keypoints_dir, f"bedsheet_{index:04d}.txt")
                        os.rename(src, dst)
                        keypoint_moved = True
                        print(f"  📝 Moved keypoints: {src} -> {dst}")
            
            # Create visualization if both image and keypoints are available
            if image_moved and keypoint_moved:
                try:
                    viz_dir = os.path.join(output_dir, "visualizations")
                    os.makedirs(viz_dir, exist_ok=True)
                    
                    image_path = os.path.join(imgs_dir, f"bedsheet_{index:04d}.png")
                    keypoint_path = os.path.join(keypoints_dir, f"bedsheet_{index:04d}.txt")
                    viz_path = os.path.join(viz_dir, f"bedsheet_{index:04d}_keypoints.png")
                    
                    subprocess.run([
                        "python", "visualize_keypoints.py",
                        "--image", image_path,
                        "--keypoints", keypoint_path,
                        "--output", viz_path
                    ], check=True, capture_output=True)
                    
                    # Copy visualization to main output directory for persistence
                    import shutil
                    main_viz_dir = os.path.join(output_dir, "visualizations")
                    os.makedirs(main_viz_dir, exist_ok=True)
                    main_viz_path = os.path.join(main_viz_dir, f"bedsheet_{index:04d}_keypoints.png")
                    if viz_path != main_viz_path:  # Only copy if different paths
                        shutil.copy2(viz_path, main_viz_path)
                        print(f"  📸 Visualization saved to: {main_viz_path}")
                    else:
                        print(f"  📸 Visualization already in correct location: {main_viz_path}")
                    
                except Exception as e:
                    print(f"Warning: Could not create visualization for bedsheet_{index:04d}: {e}")
        
        # Clean up intermediate directories
        subprocess.run(["rm", "-rf", bedsheet_output], capture_output=True)
        
        return {
            'index': index,
            'success': True,
            'time': generation_time,
            'output': f"bedsheet_{index:04d}.png"
        }
    else:
        return {
            'index': index,
            'success': False,
            'time': generation_time,
            'error': result.stderr
        }


def generate_fast_parameters(n_images, base_width=2.0, base_height=1.0, base_resolution=24, base_steps=120):
    """Generate optimized parameters for fast batch generation with enhanced simulation"""
    parameters = []
    
    for i in range(n_images):
        # Optimized parameters for speed with enhanced simulation
        width = random.uniform(base_width * 0.8, base_width * 1.2)  # More variation for diversity
        height = random.uniform(base_height * 0.8, base_height * 1.2)
        resolution = random.choice([48, 56, 64])  # High resolutions for more vertices and better quality
        steps = random.randint(base_steps // 3, base_steps)  # Fewer steps since enhanced sim settles faster
        material = random.choice(['cotton', 'linen', 'silk'])
        
        parameters.append((width, height, resolution, steps, material, i))
    
    return parameters


def main():
    parser = argparse.ArgumentParser(description='Fast batch bedsheet generation pipeline')
    parser.add_argument('--n_images', type=int, default=10, help='Number of images to generate')
    parser.add_argument('--width', type=float, default=2.0, help='Base bedsheet width (m)')
    parser.add_argument('--height', type=float, default=1.0, help='Base bedsheet height (m)')
    parser.add_argument('--resolution', type=int, default=56, help='Base grid resolution (higher = more vertices)')
    parser.add_argument('--steps', type=int, default=120, help='Base simulation steps (optimized for enhanced simulation)')
    parser.add_argument('--material', choices=['cotton', 'linen', 'silk'], default='cotton', help='Base material type')
    parser.add_argument('--output', default='fast_batch_output', help='Output directory')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (1 for GPU optimization)')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    print("⚡ Starting FAST Batch Bedsheet Generation Pipeline")
    print(f"Generating {args.n_images} images")
    print(f"Base parameters: {args.width}m x {args.height}m, resolution {args.resolution}, {args.steps} steps")
    print(f"Material: {args.material}")
    print(f"Workers: {args.workers} (1 recommended for GPU optimization)")
    print(f"Output: {args.output}")
    print("=" * 70)
    
    # Generate parameters for all images
    parameters = generate_fast_parameters(
        args.n_images, 
        args.width, 
        args.height, 
        args.resolution, 
        args.steps
    )
    
    # Prepare arguments for each generation
    generation_args = []
    for i, (width, height, resolution, steps, material, _) in enumerate(parameters):
        generation_args.append((
            width, height, resolution, steps, material, args.output, i
        ))
    
    start_time = time.time()
    results = []
    
    if args.workers > 1:
        # Parallel generation (use with caution - may not utilize GPU fully)
        print(f"Running {args.workers} parallel workers...")
        print("⚠️  Warning: Multiple workers may not fully utilize GPU memory")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(generate_fast_bedsheet, arg) for arg in generation_args]
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result['success']:
                    print(f"✅ Generated bedsheet_{result['index']:04d}.png ({result['time']:.1f}s)")
                else:
                    print(f"❌ Failed bedsheet_{result['index']:04d}: {result.get('error', 'Unknown error')}")
    else:
        # Sequential generation (optimal for GPU utilization)
        print("Running sequential generation (optimal for GPU utilization)...")
        for i, arg in enumerate(generation_args):
            print(f"Generating bedsheet {i+1}/{args.n_images}...")
            result = generate_fast_bedsheet(arg)
            results.append(result)
            
            if result['success']:
                print(f"✅ Generated bedsheet_{result['index']:04d}.png ({result['time']:.1f}s)")
                # Show progress
                progress = (i + 1) / args.n_images * 100
                print(f"   Progress: {progress:.1f}% complete")
            else:
                print(f"❌ Failed bedsheet_{result['index']:04d}: {result.get('error', 'Unknown error')}")
    
    total_time = time.time() - start_time
    
    # Summary
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print("=" * 70)
    print("⚡ FAST Batch Generation Completed!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Successful: {len(successful)}/{args.n_images}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_time = sum(r['time'] for r in successful) / len(successful)
        print(f"Average generation time: {avg_time:.1f}s per image")
        print(f"Throughput: {len(successful)/total_time:.2f} images/second")
        
        # Performance analysis
        if avg_time < 12:
            print("🚀 Excellent performance! Enhanced simulation + GPU fully utilized.")
        elif avg_time < 20:
            print("⚡ Good performance! Enhanced simulation working well.")
        elif avg_time < 35:
            print("✅ Decent performance. Consider reducing resolution for speed.")
        else:
            print("🐌 Performance could be improved. Check GPU utilization and reduce steps.")
    
    if failed:
        print("\nFailed generations:")
        for result in failed:
            print(f"  - bedsheet_{result['index']:04d}: {result.get('error', 'Unknown error')}")
    
    print(f"\nImages saved to: {args.output}")


if __name__ == '__main__':
    main()
