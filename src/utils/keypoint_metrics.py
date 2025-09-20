"""
Keypoint matching and evaluation metrics.

This module provides streamlined functions for calculating keypoint match rates
and related evaluation metrics used in training scripts.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Any, Optional
from shared.functions import thresholded_locations, combine_nearby_peaks


def match_keypoints(gt_keypoints: List[Tuple[int, int]], 
                   pred_keypoints: List[Tuple[int, int]], 
                   threshold: float = 10.0) -> Tuple[int, List[float]]:
    """
    Match predicted keypoints to ground truth keypoints using Euclidean distance.
    
    This function implements a greedy matching algorithm where each ground truth
    keypoint is matched to the closest predicted keypoint within the threshold.
    Each predicted keypoint can only be matched once.
    
    Args:
        gt_keypoints: List of ground truth keypoint coordinates [(x, y), ...]
        pred_keypoints: List of predicted keypoint coordinates [(x, y), ...]
        threshold: Maximum distance for a valid match (in pixels)
    
    Returns:
        Tuple of (matched_count, list_of_distances_for_matched_pairs)
    """
    matched = 0
    distances = []
    used_pred = set()
    
    for gt_kp in gt_keypoints:
        best_dist = float('inf')
        best_pred_idx = -1
        
        # Find closest predicted keypoint
        for i, pred_kp in enumerate(pred_keypoints):
            if i in used_pred:
                continue
            
            # Calculate Euclidean distance
            dist = np.sqrt((gt_kp[0] - pred_kp[0])**2 + (gt_kp[1] - pred_kp[1])**2)
            if dist < best_dist:
                best_dist = dist
                best_pred_idx = i
        
        # If we found a match within threshold, record it
        if best_pred_idx != -1 and best_dist < threshold:
            matched += 1
            used_pred.add(best_pred_idx)
            distances.append(best_dist)
    
    return matched, distances


def calculate_keypoint_match_rate(gt_heatmap: np.ndarray,
                                pred_heatmap: np.ndarray,
                                gt_threshold: float = 0.5,
                                pred_threshold: float = 0.3,
                                match_threshold: float = 10.0,
                                combine_distance: float = 10.0) -> Dict[str, Any]:
    """
    Calculate keypoint match rate between ground truth and predicted heatmaps.
    
    This function extracts keypoints from both heatmaps, combines nearby peaks,
    and calculates the match rate using the match_keypoints function.
    
    Args:
        gt_heatmap: Ground truth heatmap (2D numpy array)
        pred_heatmap: Predicted heatmap (2D numpy array)
        gt_threshold: Threshold for extracting GT keypoints
        pred_threshold: Threshold for extracting predicted keypoints
        match_threshold: Distance threshold for matching keypoints
        combine_distance: Distance threshold for combining nearby peaks
    
    Returns:
        Dictionary containing:
        - match_rate: Overall match rate (matched / total_gt)
        - matched_count: Number of matched keypoints
        - total_gt: Total number of ground truth keypoints
        - total_pred: Total number of predicted keypoints
        - avg_distance: Average distance for matched pairs
        - gt_keypoints: List of GT keypoint coordinates
        - pred_keypoints: List of predicted keypoint coordinates
    """
    # Normalize predicted heatmap if needed
    if pred_heatmap.max() > 0:
        pred_heatmap = pred_heatmap / pred_heatmap.max()
    
    # Extract keypoints from ground truth heatmap
    gt_peaks = thresholded_locations(gt_heatmap, threshold=gt_threshold)
    gt_keypoints = [(int(p[1]), int(p[0])) for p in gt_peaks]  # Convert to (x, y)
    
    # Extract keypoints from predicted heatmap
    pred_peaks = thresholded_locations(pred_heatmap, threshold=pred_threshold)
    # Combine nearby peaks to reduce duplicates
    combined_peaks = combine_nearby_peaks(pred_peaks, distance_threshold=combine_distance)
    pred_keypoints = [(int(p[1]), int(p[0])) for p in combined_peaks]  # Convert to (x, y)
    
    # Calculate matching
    matched_count, distances = match_keypoints(gt_keypoints, pred_keypoints, match_threshold)
    
    # Calculate metrics
    total_gt = len(gt_keypoints)
    total_pred = len(pred_keypoints)
    match_rate = matched_count / max(1, total_gt)  # Avoid division by zero
    avg_distance = np.mean(distances) if distances else None
    
    return {
        'match_rate': match_rate,
        'matched_count': matched_count,
        'total_gt': total_gt,
        'total_pred': total_pred,
        'avg_distance': avg_distance,
        'gt_keypoints': gt_keypoints,
        'pred_keypoints': pred_keypoints,
        'distances': distances
    }


def calculate_batch_match_rate(gt_heatmaps: torch.Tensor,
                             pred_heatmaps: torch.Tensor,
                             gt_threshold: float = 0.5,
                             pred_threshold: float = 0.3,
                             match_threshold: float = 10.0,
                             combine_distance: float = 10.0) -> Dict[str, Any]:
    """
    Calculate match rate for a batch of heatmaps.
    
    Args:
        gt_heatmaps: Ground truth heatmaps tensor (B, 1, H, W) or (B, H, W)
        pred_heatmaps: Predicted heatmaps tensor (B, 1, H, W) or (B, H, W)
        gt_threshold: Threshold for extracting GT keypoints
        pred_threshold: Threshold for extracting predicted keypoints
        match_threshold: Distance threshold for matching keypoints
        combine_distance: Distance threshold for combining nearby peaks
    
    Returns:
        Dictionary containing batch-level metrics:
        - overall_match_rate: Average match rate across batch
        - total_matched: Total matched keypoints across batch
        - total_gt_points: Total GT keypoints across batch
        - avg_distance: Average distance for all matched pairs
        - per_sample_results: List of individual sample results
    """
    # Ensure tensors are on CPU and convert to numpy
    if gt_heatmaps.is_cuda:
        gt_heatmaps = gt_heatmaps.cpu()
    if pred_heatmaps.is_cuda:
        pred_heatmaps = pred_heatmaps.cpu()
    
    # Remove batch and channel dimensions if present
    if gt_heatmaps.dim() == 4:  # (B, 1, H, W)
        gt_heatmaps = gt_heatmaps.squeeze(1)  # (B, H, W)
    if pred_heatmaps.dim() == 4:  # (B, 1, H, W)
        pred_heatmaps = pred_heatmaps.squeeze(1)  # (B, H, W)
    
    batch_size = gt_heatmaps.size(0)
    per_sample_results = []
    total_matched = 0
    total_gt_points = 0
    all_distances = []
    
    for i in range(batch_size):
        gt_heatmap = gt_heatmaps[i].numpy()
        pred_heatmap = pred_heatmaps[i].numpy()
        
        # Calculate metrics for this sample
        sample_result = calculate_keypoint_match_rate(
            gt_heatmap, pred_heatmap, 
            gt_threshold, pred_threshold, 
            match_threshold, combine_distance
        )
        
        per_sample_results.append(sample_result)
        total_matched += sample_result['matched_count']
        total_gt_points += sample_result['total_gt']
        all_distances.extend(sample_result['distances'])
    
    # Calculate overall metrics
    overall_match_rate = total_matched / max(1, total_gt_points)
    avg_distance = np.mean(all_distances) if all_distances else None
    
    return {
        'overall_match_rate': overall_match_rate,
        'total_matched': total_matched,
        'total_gt_points': total_gt_points,
        'avg_distance': avg_distance,
        'per_sample_results': per_sample_results
    }


def evaluate_model_keypoints(model: torch.nn.Module,
                           data_loader: torch.utils.data.DataLoader,
                           device: torch.device,
                           gt_threshold: float = 0.5,
                           pred_threshold: float = 0.3,
                           match_threshold: float = 10.0,
                           combine_distance: float = 10.0) -> Dict[str, Any]:
    """
    Evaluate a model's keypoint detection performance on a dataset.
    
    Args:
        model: PyTorch model for keypoint detection
        data_loader: DataLoader containing test/validation data
        device: Device to run evaluation on
        gt_threshold: Threshold for extracting GT keypoints
        pred_threshold: Threshold for extracting predicted keypoints
        match_threshold: Distance threshold for matching keypoints
        combine_distance: Distance threshold for combining nearby peaks
    
    Returns:
        Dictionary containing evaluation results
    """
    model.eval()
    all_results = []
    total_matched = 0
    total_gt_points = 0
    all_distances = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Extract images and keypoints from batch
            if isinstance(batch, dict):
                images = batch.get('image', batch.get('pixel_values'))
                keypoints = batch.get('keypoints', batch.get('gt_points'))
            else:
                images, keypoints = batch
            
            images = images.to(device)
            keypoints = keypoints.to(device)
            
            # Get model predictions
            if keypoints.dim() == 4:  # (B, 1, H, W)
                gt_heatmaps = keypoints.squeeze(1)  # (B, H, W)
            else:
                gt_heatmaps = keypoints
            
            # Forward pass
            pred_heatmaps = model(images)
            if pred_heatmaps.dim() == 4:  # (B, 1, H, W)
                pred_heatmaps = pred_heatmaps.squeeze(1)  # (B, H, W)
            
            # Calculate batch metrics
            batch_results = calculate_batch_match_rate(
                gt_heatmaps, pred_heatmaps,
                gt_threshold, pred_threshold,
                match_threshold, combine_distance
            )
            
            all_results.extend(batch_results['per_sample_results'])
            total_matched += batch_results['total_matched']
            total_gt_points += batch_results['total_gt_points']
            all_distances.extend([d for result in batch_results['per_sample_results'] 
                                for d in result['distances']])
    
    # Calculate overall metrics
    overall_match_rate = total_matched / max(1, total_gt_points)
    avg_distance = np.mean(all_distances) if all_distances else None
    
    return {
        'overall_match_rate': overall_match_rate,
        'total_matched': total_matched,
        'total_gt_points': total_gt_points,
        'avg_distance': avg_distance,
        'num_samples': len(all_results),
        'per_sample_results': all_results
    }
