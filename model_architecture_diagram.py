#!/usr/bin/env python3
"""
Model Architecture Diagram Generator
Creates a visual representation of the HybridKeypointNet architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_model_architecture_diagram():
    """Create a comprehensive diagram of the model architecture."""
    
    # Set up the figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors for different components
    colors = {
        'input': '#E8F4FD',
        'backbone': '#FFE6E6',
        'fusion': '#E6FFE6',
        'encoder': '#FFF2E6',
        'decoder': '#F0E6FF',
        'output': '#FFE6F2'
    }
    
    # Title
    ax.text(5, 11.5, 'HybridKeypointNet Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    ax.text(5, 11.2, 'YOLO + Vision Transformer Hybrid Model for Keypoint Detection', 
            fontsize=12, ha='center', style='italic')
    
    # Input Layer
    input_box = FancyBboxPatch((0.5, 10), 2, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 10.4, 'Input Image\n(3, 128, 128)', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # YOLO Backbone
    backbone_box = FancyBboxPatch((3.5, 10), 3, 0.8, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['backbone'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(backbone_box)
    ax.text(5, 10.4, 'YOLO Backbone\n(YOLOv8l layers 0-9)', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Multi-scale Features
    features_box = FancyBboxPatch((7.5, 10), 2, 0.8, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor=colors['backbone'], 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(features_box)
    ax.text(8.5, 10.4, 'Multi-scale\nFeatures', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # MultiScaleFusion
    fusion_box = FancyBboxPatch((3.5, 8.5), 3, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['fusion'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(fusion_box)
    ax.text(5, 8.9, 'MultiScaleFusion\n(128 channels)', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # PatchViTEncoder
    encoder_box = FancyBboxPatch((0.5, 7), 3, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['encoder'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(encoder_box)
    ax.text(2, 7.4, 'PatchViTEncoder\n(ViT-B/16)', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Pre-conv
    preconv_box = FancyBboxPatch((0.5, 5.5), 1.5, 0.6, 
                                boxstyle="round,pad=0.05", 
                                facecolor=colors['encoder'], 
                                edgecolor='black', linewidth=1)
    ax.add_patch(preconv_box)
    ax.text(1.25, 5.8, 'Conv2d\n(128→3)', 
            fontsize=8, ha='center', va='center')
    
    # ViT
    vit_box = FancyBboxPatch((2.5, 5.5), 1.5, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['encoder'], 
                            edgecolor='black', linewidth=1)
    ax.add_patch(vit_box)
    ax.text(3.25, 5.8, 'ViT-B/16\n(768 dim)', 
            fontsize=8, ha='center', va='center')
    
    # Decoder
    decoder_box = FancyBboxPatch((5.5, 7), 3, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['decoder'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(decoder_box)
    ax.text(7, 7.4, 'SingleHeatmapDecoder\n(768→1)', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Decoder stages
    up1_box = FancyBboxPatch((5.5, 5.5), 1.5, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['decoder'], 
                            edgecolor='black', linewidth=1)
    ax.add_patch(up1_box)
    ax.text(6.25, 5.8, 'Up1\n(768→128)', 
            fontsize=8, ha='center', va='center')
    
    up2_box = FancyBboxPatch((7.5, 5.5), 1.5, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['decoder'], 
                            edgecolor='black', linewidth=1)
    ax.add_patch(up2_box)
    ax.text(8.25, 5.8, 'Up2\n(128→32)', 
            fontsize=8, ha='center', va='center')
    
    up3_box = FancyBboxPatch((5.5, 4), 1.5, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['decoder'], 
                            edgecolor='black', linewidth=1)
    ax.add_patch(up3_box)
    ax.text(6.25, 4.3, 'Up3\n(32→16)', 
            fontsize=8, ha='center', va='center')
    
    up4_box = FancyBboxPatch((7.5, 4), 1.5, 0.6, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['decoder'], 
                            edgecolor='black', linewidth=1)
    ax.add_patch(up4_box)
    ax.text(8.25, 4.3, 'Up4\n(16→16)', 
            fontsize=8, ha='center', va='center')
    
    # Final conv
    final_box = FancyBboxPatch((6.5, 2.5), 1.5, 0.6, 
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['decoder'], 
                              edgecolor='black', linewidth=1)
    ax.add_patch(final_box)
    ax.text(7.25, 2.8, 'Final Conv\n(16→1)', 
            fontsize=8, ha='center', va='center')
    
    # Output
    output_box = FancyBboxPatch((3.5, 1), 3, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 1.4, 'Keypoint Heatmap\n(1, 128, 128)', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Connections
    # Input to Backbone
    ax.arrow(2.5, 10.4, 0.8, 0, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # Backbone to Features
    ax.arrow(6.5, 10.4, 0.8, 0, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # Features to Fusion
    ax.arrow(8.5, 9.8, -1.8, -0.5, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # Fusion to Encoder
    ax.arrow(3.5, 8.9, -2.8, -1.4, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # Fusion to Decoder
    ax.arrow(6.5, 8.9, 0.8, -1.4, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # Encoder internal connections
    ax.arrow(2, 6.8, 0, -0.8, head_width=0.05, head_length=0.05, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(1.25, 5.5, 1.8, 0, head_width=0.05, head_length=0.05, 
             fc='black', ec='black', linewidth=1)
    
    # Decoder internal connections
    ax.arrow(7, 6.8, 0, -0.8, head_width=0.05, head_length=0.05, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(6.25, 5.5, 1.8, 0, head_width=0.05, head_length=0.05, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(7.25, 5.2, 0, -0.8, head_width=0.05, head_length=0.05, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(6.25, 4, 1.8, 0, head_width=0.05, head_length=0.05, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(7.25, 3.7, 0, -0.8, head_width=0.05, head_length=0.05, 
             fc='black', ec='black', linewidth=1)
    ax.arrow(6.25, 2.5, 1.8, 0, head_width=0.05, head_length=0.05, 
             fc='black', ec='black', linewidth=1)
    
    # Decoder to Output
    ax.arrow(7.25, 2.5, -3.5, -1.4, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # Add feature dimensions
    ax.text(1.5, 9.5, 'RGB Image', fontsize=9, ha='center', style='italic')
    ax.text(5, 9.5, 'YOLO Features\nMultiple scales', fontsize=9, ha='center', style='italic')
    ax.text(8.5, 9.5, 'Feature Maps\nVarious channels', fontsize=9, ha='center', style='italic')
    
    # Add processing details
    ax.text(2, 6.2, 'Resize to\n128×128', fontsize=8, ha='center', style='italic')
    ax.text(3.25, 6.2, 'Patch embedding\n+ Self-attention', fontsize=8, ha='center', style='italic')
    
    # Add decoder details
    ax.text(6.25, 6.2, 'Conv + Deconv\n2× upsampling', fontsize=8, ha='center', style='italic')
    ax.text(8.25, 6.2, 'Conv + Deconv\n2× upsampling', fontsize=8, ha='center', style='italic')
    ax.text(6.25, 4.7, 'Conv + Deconv\n2× upsampling', fontsize=8, ha='center', style='italic')
    ax.text(8.25, 4.7, 'Conv + Deconv\n2× upsampling', fontsize=8, ha='center', style='italic')
    ax.text(7.25, 3.2, 'Final 1×1 conv\n+ Interpolate', fontsize=8, ha='center', style='italic')
    
    # Add key features box
    features_text = """Key Features:
• YOLO backbone for feature extraction
• Multi-scale feature fusion
• Vision Transformer for global context
• U-Net style decoder for precise localization
• Spatial softmax for keypoint probability
• Single heatmap per keypoint type"""
    
    ax.text(0.5, 0.2, features_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('model_architecture.pdf', bbox_inches='tight')
    plt.show()
    
    print("✅ Model architecture diagram saved as 'model_architecture.png' and 'model_architecture.pdf'")

def create_data_flow_diagram():
    """Create a data flow diagram showing the complete pipeline."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Complete Data Processing Pipeline', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Colors
    colors = {
        'input': '#E8F4FD',
        'preprocessing': '#FFE6E6', 
        'model': '#E6FFE6',
        'postprocessing': '#FFF2E6',
        'output': '#FFE6F2'
    }
    
    # Input
    input_box = FancyBboxPatch((0.5, 8), 2, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 8.4, 'Original Image\n(Any size)', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # YOLO Detection
    yolo_box = FancyBboxPatch((3.5, 8), 2, 0.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['preprocessing'], 
                             edgecolor='black', linewidth=2)
    ax.add_patch(yolo_box)
    ax.text(4.5, 8.4, 'YOLO Detection\n(Best.pt finetuned)', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Masking
    mask_box = FancyBboxPatch((6.5, 8), 2, 0.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['preprocessing'], 
                             edgecolor='black', linewidth=2)
    ax.add_patch(mask_box)
    ax.text(7.5, 8.4, 'Confidence-based\nMasking', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Resize
    resize_box = FancyBboxPatch((1.5, 6.5), 2, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['preprocessing'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(resize_box)
    ax.text(2.5, 6.9, 'Resize to\n128×128', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Data Augmentation
    aug_box = FancyBboxPatch((4.5, 6.5), 2, 0.8, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['preprocessing'], 
                            edgecolor='black', linewidth=2)
    ax.add_patch(aug_box)
    ax.text(5.5, 6.9, 'Data Augmentation\n(Stronger/Minimal)', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Keypoint Heatmap
    heatmap_box = FancyBboxPatch((7.5, 6.5), 2, 0.8, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['preprocessing'], 
                                edgecolor='black', linewidth=2)
    ax.add_patch(heatmap_box)
    ax.text(8.5, 6.9, 'Keypoint Heatmap\nGeneration', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Model
    model_box = FancyBboxPatch((3.5, 5), 3, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['model'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(model_box)
    ax.text(5, 5.4, 'HybridKeypointNet\n(YOLO + ViT)', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Model Output
    output_box = FancyBboxPatch((3.5, 3.5), 3, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], 
                               edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 3.9, 'Predicted Heatmap\n(1, 128, 128)', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Threshold
    threshold_box = FancyBboxPatch((1.5, 2), 2, 0.8, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=colors['postprocessing'], 
                                  edgecolor='black', linewidth=2)
    ax.add_patch(threshold_box)
    ax.text(2.5, 2.4, 'Fixed Threshold\nLocations', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Coordinate Fix
    coord_box = FancyBboxPatch((4.5, 2), 2, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['postprocessing'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(coord_box)
    ax.text(5.5, 2.4, 'Coordinate\nMapping', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Final Output
    final_box = FancyBboxPatch((7.5, 2), 2, 0.8, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['output'], 
                              edgecolor='black', linewidth=2)
    ax.add_patch(final_box)
    ax.text(8.5, 2.4, 'Keypoint\nCoordinates', 
            fontsize=10, ha='center', va='center', fontweight='bold')
    
    # Connections
    # Input to YOLO
    ax.arrow(2.5, 8.4, 0.8, 0, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # YOLO to Masking
    ax.arrow(5.5, 8.4, 0.8, 0, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # Masking to Resize
    ax.arrow(7.5, 7.8, -5.8, -1.3, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # Masking to Augmentation
    ax.arrow(7.5, 7.8, -2.8, -1.3, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # Masking to Heatmap
    ax.arrow(7.5, 7.8, 0.8, -1.3, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # All to Model
    ax.arrow(2.5, 6.5, 0.8, -1.4, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    ax.arrow(5.5, 6.5, -1.8, -1.4, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    ax.arrow(8.5, 6.5, -4.8, -1.4, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # Model to Output
    ax.arrow(5, 5, 0, -0.8, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # Output to Post-processing
    ax.arrow(4.5, 3.5, -2.8, -1.4, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    ax.arrow(5.5, 3.5, -1.8, -1.4, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    ax.arrow(6.5, 3.5, 0.8, -1.4, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # Post-processing to Final
    ax.arrow(3.5, 2.4, 0.8, 0, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    ax.arrow(6.5, 2.4, 0.8, 0, head_width=0.1, head_length=0.1, 
             fc='black', ec='black', linewidth=2)
    
    # Add optimization highlights
    optimizations = """Key Optimizations:
• Confidence-based masking refinement
• Fixed coordinate system mapping
• Single peak detection per keypoint
• Proper train/eval data separation
• FP16 mixed precision training
• Gradient accumulation & early stopping"""
    
    ax.text(0.5, 0.5, optimizations, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('data_flow_pipeline.png', dpi=300, bbox_inches='tight')
    plt.savefig('data_flow_pipeline.pdf', bbox_inches='tight')
    plt.show()
    
    print("✅ Data flow pipeline diagram saved as 'data_flow_pipeline.png' and 'data_flow_pipeline.pdf'")

if __name__ == "__main__":
    print("Creating model architecture diagrams...")
    create_model_architecture_diagram()
    create_data_flow_diagram()
    print("✅ All diagrams created successfully!")
