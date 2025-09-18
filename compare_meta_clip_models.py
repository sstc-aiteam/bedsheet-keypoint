#!/usr/bin/env python3
"""
Comparison script for Meta CLIP models: Original vs Pre-trained

This script runs both the original Meta CLIP model and the pre-trained Meta CLIP model
on the same bedsheet dataset to compare their performance.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from post_meta_clip_style_training import main_meta_clip_post_training_pipeline, DEFAULT_CONFIG

def run_model_comparison():
    """Run comparison between original and pre-trained Meta CLIP models."""
    
    print("=" * 80)
    print("META CLIP MODEL COMPARISON: Original vs Pre-trained")
    print("=" * 80)
    
    # Configuration for original Meta CLIP
    config_original = DEFAULT_CONFIG.copy()
    config_original['use_original_metaclip'] = True
    config_original['num_epochs'] = 10  # Reduced for comparison
    
    # Configuration for pre-trained Meta CLIP
    config_pretrained = DEFAULT_CONFIG.copy()
    config_pretrained['use_original_metaclip'] = False
    config_pretrained['num_epochs'] = 10  # Same epochs for fair comparison
    
    results = {}
    
    # Run original Meta CLIP
    print("\n" + "=" * 60)
    print("TRAINING ORIGINAL META CLIP MODEL")
    print("=" * 60)
    
    start_time = time.time()
    try:
        model_original, history_original = main_meta_clip_post_training_pipeline(config_original)
        training_time_original = time.time() - start_time
        
        results['original'] = {
            'model': model_original,
            'history': history_original,
            'training_time': training_time_original,
            'config': config_original,
            'success': True
        }
        
        print(f"✓ Original Meta CLIP training completed in {training_time_original:.2f} seconds")
        
    except Exception as e:
        print(f"✗ Original Meta CLIP training failed: {e}")
        results['original'] = {
            'error': str(e),
            'success': False
        }
    
    # Run pre-trained Meta CLIP
    print("\n" + "=" * 60)
    print("TRAINING PRE-TRAINED META CLIP MODEL")
    print("=" * 60)
    
    start_time = time.time()
    try:
        model_pretrained, history_pretrained = main_meta_clip_post_training_pipeline(config_pretrained)
        training_time_pretrained = time.time() - start_time
        
        results['pretrained'] = {
            'model': model_pretrained,
            'history': history_pretrained,
            'training_time': training_time_pretrained,
            'config': config_pretrained,
            'success': True
        }
        
        print(f"✓ Pre-trained Meta CLIP training completed in {training_time_pretrained:.2f} seconds")
        
    except Exception as e:
        print(f"✗ Pre-trained Meta CLIP training failed: {e}")
        results['pretrained'] = {
            'error': str(e),
            'success': False
        }
    
    # Generate comparison report
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    if results['original']['success'] and results['pretrained']['success']:
        print(f"Training Time Comparison:")
        print(f"  Original Meta CLIP:    {results['original']['training_time']:.2f} seconds")
        print(f"  Pre-trained Meta CLIP: {results['pretrained']['training_time']:.2f} seconds")
        print(f"  Time Difference:       {abs(results['original']['training_time'] - results['pretrained']['training_time']):.2f} seconds")
        
        # Compare final validation losses
        if 'val_loss' in results['original']['history'] and 'val_loss' in results['pretrained']['history']:
            original_final_loss = results['original']['history']['val_loss'][-1]
            pretrained_final_loss = results['pretrained']['history']['val_loss'][-1]
            
            print(f"\nFinal Validation Loss Comparison:")
            print(f"  Original Meta CLIP:    {original_final_loss:.4f}")
            print(f"  Pre-trained Meta CLIP: {pretrained_final_loss:.4f}")
            print(f"  Loss Difference:       {abs(original_final_loss - pretrained_final_loss):.4f}")
            
            if pretrained_final_loss < original_final_loss:
                improvement = ((original_final_loss - pretrained_final_loss) / original_final_loss) * 100
                print(f"  Performance Gain:      {improvement:.2f}% improvement with pre-trained model")
            else:
                degradation = ((pretrained_final_loss - original_final_loss) / original_final_loss) * 100
                print(f"  Performance Loss:      {degradation:.2f}% degradation with pre-trained model")
    
    # Save comparison results
    comparison_dir = Path("comparison_results")
    comparison_dir.mkdir(exist_ok=True)
    
    # Save detailed results (without model objects)
    save_results = {}
    for key, value in results.items():
        save_results[key] = {k: v for k, v in value.items() if k != 'model'}
    
    with open(comparison_dir / "meta_clip_comparison_results.json", 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    
    print(f"\nComparison results saved to: {comparison_dir / 'meta_clip_comparison_results.json'}")
    
    return results

def run_single_model(use_original: bool = False, epochs: int = 20):
    """Run a single model for testing purposes."""
    
    config = DEFAULT_CONFIG.copy()
    config['use_original_metaclip'] = use_original
    config['num_epochs'] = epochs
    
    model_type = "Original" if use_original else "Pre-trained"
    print(f"Running {model_type} Meta CLIP model with {epochs} epochs...")
    
    model, history = main_meta_clip_post_training_pipeline(config)
    
    return model, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Meta CLIP models")
    parser.add_argument("--mode", choices=["compare", "original", "pretrained"], 
                       default="compare", help="Run mode")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    
    args = parser.parse_args()
    
    if args.mode == "compare":
        run_model_comparison()
    elif args.mode == "original":
        run_single_model(use_original=True, epochs=args.epochs)
    elif args.mode == "pretrained":
        run_single_model(use_original=False, epochs=args.epochs)
