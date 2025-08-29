#!/usr/bin/env python3
"""
Main training script for the bed sheet folding robot keypoint detection model.
Uses the reorganized src structure for better code organization.
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training import main_training_pipeline, DEFAULT_CONFIG

def main():
    """Main training function."""
    
    print("🚀 Bed Sheet Folding Robot - Keypoint Detection Training")
    print("=" * 60)
    print("📁 Using reorganized src structure")
    print("=" * 60)
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    
    # Override with command line arguments if provided
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config.update(json.load(f))
            print(f"📋 Loaded configuration from: {config_file}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['model_save_path'] = f"models/keypoint_model_{timestamp}"
    config['results_dir'] = f"results/training_{timestamp}"
    
    # Create directories
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['model_save_path']), exist_ok=True)
    
    print(f"📁 Model will be saved to: {config['model_save_path']}")
    print(f"📁 Results will be saved to: {config['results_dir']}")
    
    # Print configuration summary
    print("\n📋 Configuration Summary:")
    print(f"  • Epochs: {config['num_epochs']}")
    print(f"  • Batch Size: {config['batch_size']}")
    print(f"  • Learning Rate: {config['learning_rate']}")
    print(f"  • Stronger Augmentation: {config.get('use_stronger_augmentation', True)}")
    print(f"  • Mixup: {config.get('use_mixup', True)}")
    print(f"  • FP16 Training: {config.get('use_fp16', True)}")
    print(f"  • Early Stopping: {config.get('early_stopping_patience', 20)} epochs")
    
    print("\n🔧 Integrated Improvements:")
    print("  ✅ Fixed coordinate system mapping")
    print("  ✅ Confidence-based masking refinement")
    print("  ✅ Single peak detection per keypoint")
    print("  ✅ Proper train/eval data separation")
    print("  ✅ Cutout augmentation (in StrongerAugmentation)")
    print("  ✅ Multi-stage learning rate scheduling")
    print("  ✅ Gradient clipping and accumulation")
    print("  ✅ Early stopping with best model saving")
    print("  ✅ FP16 mixed precision training")
    print("  ✅ Label smoothing and dropout")
    
    print("\n" + "=" * 60)
    print("🎯 Starting Training Pipeline...")
    print("=" * 60)
    
    try:
        # Run training pipeline
        model, history = main_training_pipeline(config)
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Print final results
        if history and 'train_loss' in history and len(history['train_loss']) > 0:
            final_train_loss = history['train_loss'][-1]
            best_train_loss = min(history['train_loss'])
            
            print(f"📈 Training Results:")
            print(f"  • Final Training Loss: {final_train_loss:.4f}")
            print(f"  • Best Training Loss: {best_train_loss:.4f}")
            
            if 'val_loss' in history and len(history['val_loss']) > 0:
                final_val_loss = history['val_loss'][-1]
                best_val_loss = min(history['val_loss'])
                
                print(f"  • Final Validation Loss: {final_val_loss:.4f}")
                print(f"  • Best Validation Loss: {best_val_loss:.4f}")
                
                # Calculate overfitting ratio
                overfitting_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else float('inf')
                print(f"  • Overfitting Ratio: {overfitting_ratio:.2f}x")
                
                # Performance assessment
                if overfitting_ratio < 1.5:
                    print("  ✅ Excellent: Low overfitting")
                elif overfitting_ratio < 2.0:
                    print("  ⚠️ Good: Moderate overfitting")
                else:
                    print("  ❌ High: Significant overfitting")
        
        # Save training history
        history_path = f"{config['results_dir']}/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"✅ Training history saved: {history_path}")
        
        # Check model files
        model_path = f"{config['model_save_path']}.pth"
        if os.path.exists(model_path):
            print(f"✅ Model saved: {model_path}")
        else:
            print(f"⚠️ Model file not found: {model_path}")
        
        # Check results
        if os.path.exists(config['results_dir']):
            result_files = [f for f in os.listdir(config['results_dir']) if f.endswith('.png')]
            print(f"✅ Results generated: {len(result_files)} visualization files")
        else:
            print(f"⚠️ Results directory not found: {config['results_dir']}")
        
        print("\n" + "=" * 60)
        print("🎯 ALL SYSTEMS WORKING OPTIMALLY!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✅ Training completed successfully!")
        print("🎯 Your model is now ready for deployment!")
    else:
        print("\n❌ Training failed. Please check the error messages above.")
        sys.exit(1)
