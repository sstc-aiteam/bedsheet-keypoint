#!/usr/bin/env python3
"""
Analyze test_mattress.json to check keypoint distribution per image.
"""

import json
from collections import defaultdict
import sys

def analyze_mattress_json(json_file):
    """Analyze the mattress JSON file for keypoint distribution."""
    
    print(f"Analyzing {json_file}...")
    print("=" * 60)
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Extract file information
    files = data.get('file', {})
    metadata = data.get('metadata', {})
    
    print(f"Total files in project: {len(files)}")
    print(f"Total metadata entries: {len(metadata)}")
    print()
    
    # Count keypoints per image
    keypoints_per_image = defaultdict(int)
    image_names = {}
    
    # Map file IDs to names
    for file_id, file_info in files.items():
        image_names[file_id] = file_info.get('fname', f'Unknown_{file_id}')
    
    # Count keypoints for each image
    for metadata_id, metadata_info in metadata.items():
        vid = metadata_info.get('vid')
        if vid:
            keypoints_per_image[vid] += 1
    
    # Analyze distribution
    print("Keypoint Distribution Analysis:")
    print("-" * 40)
    
    # Count images by keypoint count
    distribution = defaultdict(int)
    images_with_keypoints = 0
    images_without_keypoints = 0
    
    for file_id in files.keys():
        keypoint_count = keypoints_per_image.get(file_id, 0)
        distribution[keypoint_count] += 1
        
        if keypoint_count > 0:
            images_with_keypoints += 1
        else:
            images_without_keypoints += 1
    
    # Print distribution
    for keypoint_count in sorted(distribution.keys()):
        count = distribution[keypoint_count]
        percentage = (count / len(files)) * 100
        print(f"Images with {keypoint_count} keypoints: {count:3d} ({percentage:5.1f}%)")
    
    print()
    print(f"Images with keypoints: {images_with_keypoints}")
    print(f"Images without keypoints: {images_without_keypoints}")
    print(f"Total images: {len(files)}")
    
    # Check for anomalies
    print()
    print("Anomaly Check:")
    print("-" * 40)
    
    anomalies = []
    for file_id, keypoint_count in keypoints_per_image.items():
        if keypoint_count > 4:
            image_name = image_names.get(file_id, f'Unknown_{file_id}')
            anomalies.append((image_name, keypoint_count))
    
    if anomalies:
        print(f"⚠️  Found {len(anomalies)} images with more than 4 keypoints:")
        for image_name, count in anomalies:
            print(f"   {image_name}: {count} keypoints")
    else:
        print("✅ All images have 4 or fewer keypoints")
    
    # Show some examples
    print()
    print("Sample Images:")
    print("-" * 40)
    
    sample_count = 0
    for file_id, file_info in files.items():
        if sample_count >= 10:
            break
        keypoint_count = keypoints_per_image.get(file_id, 0)
        image_name = file_info.get('fname', f'Unknown_{file_id}')
        print(f"{image_name}: {keypoint_count} keypoints")
        sample_count += 1
    
    if len(files) > 10:
        print(f"... and {len(files) - 10} more images")

def main():
    """Main function."""
    json_file = "via_proj/mattress/test_mattress.json"
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    
    analyze_mattress_json(json_file)

if __name__ == "__main__":
    main()
