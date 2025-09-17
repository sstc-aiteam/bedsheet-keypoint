# Bedsheet Keypoint Generation Pipeline

## Essential Files

### Core Pipeline Scripts
- **`fast_warp_blender_pipeline.py`** - Main pipeline for single image generation (recommended)
- **`fast_batch_pipeline.py`** - Batch generation pipeline for multiple images
- **`enhanced_warp_bedsheet_sim.py`** - Enhanced Warp simulation with realistic physics
- **`blender_warp_render.py`** - Blender rendering with improved lighting and camera

### Alternative/Backup Scripts
- **`warp_blender_pipeline.py`** - Original pipeline (slower)
- **`batch_pipeline.py`** - Original batch pipeline (slower)
- **`warp_bedsheet_sim.py`** - Original Warp simulation
- **`enhanced_blender_warp_render.py`** - Alternative Blender renderer
- **`fast_gpu_blender_render.py`** - Ultra-fast GPU renderer

## Usage

### Single Image Generation
```bash
python fast_warp_blender_pipeline.py --width 2.0 --height 1.0 --resolution 48 --steps 100 --output my_bedsheet
```

### Batch Generation
```bash
python fast_batch_pipeline.py --n_images 10 --width 2.0 --height 1.0 --resolution 48 --steps 100 --output my_batch
```

## Features
- **Realistic cloth physics** with Warp simulation
- **GPU-accelerated rendering** with Blender Cycles
- **Adaptive camera positioning** for optimal framing
- **Enhanced lighting** with 5-light setup
- **Dark purple floor** for better contrast
- **Automatic keypoint detection** for 4 corner points
- **COCO format annotations** for training data

## Test Outputs
- **`test_dark_purple_floor/`** - Latest single image test with dark purple floor
- **`test_dark_purple_floor_batch/`** - Latest batch test with dark purple floor
