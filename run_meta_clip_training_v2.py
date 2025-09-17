#!/usr/bin/env python3
"""
Runner script for Meta CLIP Multi-Tier Training V2

This script runs the multi-tier keypoint detection training pipeline.
"""

import os
import sys

def main():
    print("Starting Meta CLIP multi-tier training V2...")
    print("="*60)
    print("Pre-trained model: models/meta_clip_style_cloth_v2")
    print("Output: models/meta_clip_style_bedsheet_post_v2")
    print("Results: results_meta_clip_bedsheet_post_v2")
    print("="*60)
    
    # Check if pre-trained model exists
    if os.path.exists("models/meta_clip_style_cloth_v2"):
        print("✓ Pre-trained model found")
    else:
        print("⚠️  Pre-trained model not found. Please run cloth training first.")
        return
    
    # Run the post-training pipeline
    try:
        from post_meta_clip_style_training_v2 import main_meta_clip_post_training_pipeline_v2, config
        trained_model, history = main_meta_clip_post_training_pipeline_v2(config)
        print("\n✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
