import numpy as np
import os
import json
from scipy.ndimage import label
import cv2

def depth_map_to_image(depth_map):
    """
    Convert a raw depth map (np.ndarray, any dtype, possibly >255) into an 8-bit image.
    Handles missing/zero values gracefully.
    Returns an uint8 grayscale image (0=far, 255=near).
    """
    valid = depth_map > 0
    if np.any(valid):
        dmin = np.min(depth_map[valid])
        dmax = np.max(depth_map)
        # Avoid division by zero
        if dmax > dmin:
            depth_img = ((depth_map - dmin) / (dmax - dmin) * 255)
        else:
            depth_img = np.zeros_like(depth_map)
    else:
        depth_img = np.zeros_like(depth_map)
    depth_img = np.clip(depth_img, 0, 255).astype(np.uint8)
    return depth_img

def thresholded_locations(data_2d, threshold):
    """
    Find centroids of connected components above a threshold.
    Args:
        data_2d: 2D numpy array (can be logits or probabilities).
        threshold: threshold for segmentation.
        from_logits: If True, applies sigmoid to convert logits to probabilities.
    Returns:
        List of centroid coordinates (as np arrays).
    """
    # If input is logits, convert to probability first!
    probs = data_2d

    # Create a binary mask
    thresholded_2d = (probs >= threshold).astype(np.uint8)

    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, num_features = label(thresholded_2d, structure=structure)
    centroids = []
    for mesh_label in range(1, num_features + 1):
        positions = np.argwhere(labeled == mesh_label)
        centroid = positions.mean(axis=0)
        centroids.append(centroid)
    return centroids

def combine_nearby_peaks(peaks, distance_threshold=10):
    """
    Combine nearby peaks into single keypoints using clustering.
    
    Args:
        peaks: List of peak coordinates [[y1, x1], [y2, x2], ...]
        distance_threshold: Maximum distance to consider peaks as part of same cluster
    
    Returns:
        List of combined peak coordinates
    """
    if not peaks:
        return []
    
    # Convert to numpy array for easier manipulation
    peaks = np.array(peaks)
    
    # If only one peak, return it
    if len(peaks) == 1:
        return peaks.tolist()
    
    # Calculate pairwise distances
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(peaks))
    
    # Create clusters
    clusters = []
    used = set()
    
    for i in range(len(peaks)):
        if i in used:
            continue
            
        # Start a new cluster
        cluster = [i]
        used.add(i)
        
        # Find all peaks within distance_threshold
        for j in range(i + 1, len(peaks)):
            if j not in used and distances[i, j] <= distance_threshold:
                cluster.append(j)
                used.add(j)
        
        clusters.append(cluster)
    
    # Calculate centroid for each cluster
    combined_peaks = []
    for cluster in clusters:
        cluster_peaks = peaks[cluster]
        centroid = cluster_peaks.mean(axis=0)
        combined_peaks.append(centroid)
    
    return combined_peaks

def extract_mask_compare(image, yolo_model_finetuned, allowed_classes):
    # image_name = os.path.basename(image_path)
    # 推論
    results = yolo_model_finetuned(image, task="segment")[0]

    # 原圖
    orig_img = image.copy()
    h, w = orig_img.shape[:2]

    # 空白遮罩
    mask_all = np.zeros((h, w), dtype=np.uint8)
    for r in results:
        if r.masks is None:
            continue
        masks = r.masks.data.cpu().numpy()     # [N, H_pred, W_pred]
        classes = r.boxes.cls.cpu().numpy()    # [N] 物件的類別 ID
        for m, cls_id in zip(masks, classes):
            if int(cls_id) not in allowed_classes:
                continue  # 跳過不在清單內的類別
            m = (m * 255).astype(np.uint8)
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_all = cv2.bitwise_or(mask_all, m)
    masked_image = orig_img.copy()
    masked_image[mask_all==0] = 0
    return mask_all

def get_keypoints_for_image(image_filename, keypoints_data_src):
    """
    Get keypoints for a specific image from the VIA JSON file.
    
    Args:
        image_filename: Name of the image file
        keypoints_data_src: Path to the VIA JSON file
    
    Returns:
        List of keypoint coordinates or None if not found
    """
    try:
        with open(keypoints_data_src, 'r') as f:
            data = json.load(f)
        
        # Find the image ID by filename
        image_id = None
        for fid, file_data in data['file'].items():
            if file_data['fname'] == image_filename:
                image_id = fid
                break
        
        if image_id is None:
            return None
        
        # Extract keypoints from metadata
        keypoints = []
        for metadata_id, metadata in data['metadata'].items():
            if metadata['vid'] == image_id:
                # Extract coordinates from xy field
                xy = metadata['xy']
                if len(xy) >= 3:  # [shape_type, x, y, ...]
                    x = xy[1]
                    y = xy[2]
                    keypoints.append([x, y])
        
        return keypoints if keypoints else None
        
    except Exception as e:
        print(f"Error loading keypoints for {image_filename}: {e}")
        return None

def resize_image_and_keypoints(image, keypoints, target_width, target_height):
    """
    Resize image and adjust keypoint coordinates accordingly.
    
    Args:
        image: Input image
        keypoints: List of keypoint coordinates [[x, y], ...]
        target_width: Target width
        target_height: Target height
    
    Returns:
        Tuple of (resized_image, adjusted_keypoints)
    """
    if keypoints is None:
        return image, None
    
    # Get original dimensions
    orig_height, orig_width = image.shape[:2]
    
    # Resize image
    resized_image = cv2.resize(image, (target_width, target_height))
    
    # Adjust keypoint coordinates
    adjusted_keypoints = []
    for kp in keypoints:
        x, y = kp
        # Scale coordinates
        new_x = int(x * target_width / orig_width)
        new_y = int(y * target_height / orig_height)
        adjusted_keypoints.append([new_x, new_y])
    
    return resized_image, adjusted_keypoints