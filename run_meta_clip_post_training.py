#!/usr/bin/env python3
"""
Simple runner script for Meta CLIP post-training on bedsheet data.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from post_meta_clip_style_training import main_meta_clip_post_training_pipeline, DEFAULT_CONFIG

if __name__ == '__main__':
    print("Starting Meta CLIP post-training on bedsheet data...")
    print("=" * 60)
    print("Pre-trained model: models/meta_clip_style_cloth")
    print("Output: models/meta_clip_style_bedsheet_post")
    print("Results: results_meta_clip_bedsheet_post")
    print("=" * 60)
    
    # Check if pre-trained model exists
    pretrained_path = DEFAULT_CONFIG['pretrained_model_path']
    if not os.path.exists(pretrained_path):
        print(f"❌ Pre-trained model not found at: {pretrained_path}")
        print("Please run Meta CLIP cloth training first:")
        print("  python meta_clip_style_cloth_training.py")
        sys.exit(1)
    
    # Check if head weights exist
    head_path = os.path.join(pretrained_path, 'head.pth')
    if not os.path.exists(head_path):
        print(f"❌ Pre-trained head weights not found at: {head_path}")
        print("Please run Meta CLIP cloth training first:")
        print("  python meta_clip_style_cloth_training.py")
        sys.exit(1)
    
    print("✓ Pre-trained model found")
    
    try:
        # Run post-training
        model, history = main_meta_clip_post_training_pipeline(DEFAULT_CONFIG)
        
        print("\n" + "=" * 60)
        print("✅ Meta CLIP post-training completed successfully!")
        print("=" * 60)
        print(f"📁 Model saved to: {DEFAULT_CONFIG['output_dir']}")
        print(f"📊 Results saved to: {DEFAULT_CONFIG['results_dir']}")
        print(f"📈 Training history saved to: {DEFAULT_CONFIG['output_dir']}/training_history.json")
        
        # Show final metrics
        if history and 'val_loss' in history:
            final_val_loss = history['val_loss'][-1] if history['val_loss'] else 'N/A'
            print(f"🎯 Final validation loss: {final_val_loss}")
        
    except Exception as e:
        print(f"\n❌ Post-training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)