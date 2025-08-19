import numpy as np
from skimage.feature import peak_local_max

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

import numpy as np
from scipy.ndimage import label

def thresholded_locations(data_2d, threshold):
    thresholded_2d = np.where(data_2d >= threshold, data_2d, 0)
    # thresholded_2d: your binary/thresholded 2D array
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity for 2D

    labeled, num_features = label(thresholded_2d, structure=structure)
    centroids = []

    for mesh_label in range(1, num_features + 1):
        positions = np.argwhere(labeled == mesh_label)
        # Calculate centroid as the average coordinates
        centroid = positions.mean(axis=0)
        centroids.append(centroid)

    return centroids