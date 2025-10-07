#!/usr/bin/env python3
"""
Clean TensorRT Keypoint Detection Demo.
Modularized version with all functionality under 500 lines.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.demo import (
    TensorRTDemo, 
    CLIPDemo, 
    create_argument_parser, 
    validate_config,
    list_available_models
)


def main():
    """Main demo function with command line interface."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        list_available_models()
        return
    
    # Validate configuration
    if not validate_config(args):
        return
    
    # Create configuration dictionary
    config = {
        'pytorch_model': args.pytorch_model,
        'tensorrt_model': args.tensorrt_model,
        'image_dir': args.image_dir,
        'model_type': args.model_type,
        'benchmark': args.benchmark,
        'num_runs': args.num_runs
    }
    
    # Run demo
    demo = TensorRTDemo(config)
    success = demo.run()
    
    if not success:
        sys.exit(1)


def demo_clip():
    """Run CLIP-specific demo."""
    demo = CLIPDemo()
    success = demo.run()
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo-clip":
        # Remove the --demo-clip argument and run CLIP demo
        sys.argv.pop(1)
        demo_clip()
    else:
        # Run main demo with command line arguments
        main()