"""
Bar length measurement module.
Estimates the approximate length of detected bars based on bounding box dimensions.
"""

from typing import List, Dict, Optional
import numpy as np


def estimate_bar_length(
    detections: List[Dict],
    reference_pixel_to_cm: float = 1.0
) -> List[Dict]:
    """
    Estimate the length of detected bars based on bounding box dimensions.
    
    Args:
        detections: List of detection dictionaries with box coordinates
        reference_pixel_to_cm: Conversion factor from pixels to cm
                              (adjust based on camera calibration)
        
    Returns:
        List[Dict]: Detections with added 'estimated_length_cm' field
    """
    for detection in detections:
        # Use the longer dimension of the bounding box as bar length
        width_px = detection['width']
        height_px = detection['height']
        
        # Assume the bar is the longer dimension
        bar_length_px = max(width_px, height_px)
        
        # Convert to cm using calibration factor
        bar_length_cm = bar_length_px * reference_pixel_to_cm
        
        detection['estimated_length_cm'] = bar_length_cm
        detection['length_source'] = 'bbox_max_dim'
    
    return detections


def calculate_average_bar_length(detections: List[Dict]) -> Optional[float]:
    """
    Calculate the average length of all detected bars.
    
    Args:
        detections: List of detection dictionaries with estimated lengths
        
    Returns:
        Optional[float]: Average bar length in cm, or None if no detections
    """
    if not detections:
        return None
    
    lengths = [d.get('estimated_length_cm', 0) for d in detections]
    valid_lengths = [l for l in lengths if l > 0]
    
    if not valid_lengths:
        return None
    
    return np.mean(valid_lengths)


def get_bar_length_stats(detections: List[Dict]) -> Dict[str, float]:
    """
    Get statistical information about detected bar lengths.
    
    Args:
        detections: List of detection dictionaries with estimated lengths
        
    Returns:
        Dict: Dictionary with min, max, mean, and median bar lengths
    """
    if not detections:
        return {
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            'count': 0
        }
    
    lengths = [d.get('estimated_length_cm', 0) for d in detections]
    valid_lengths = [l for l in lengths if l > 0]
    
    if not valid_lengths:
        return {
            'min': 0,
            'max': 0,
            'mean': 0,
            'median': 0,
            'count': 0
        }
    
    valid_lengths_arr = np.array(valid_lengths)
    
    return {
        'min': float(np.min(valid_lengths_arr)),
        'max': float(np.max(valid_lengths_arr)),
        'mean': float(np.mean(valid_lengths_arr)),
        'median': float(np.median(valid_lengths_arr)),
        'count': len(valid_lengths)
    }
