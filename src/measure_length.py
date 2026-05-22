"""
Bar length measurement module with advanced calibration and robust smoothing.
Estimates the approximate length of detected bars based on bounding box dimensions.

This module uses a METERS_PER_PIXEL calibration factor to convert pixel measurements
to real-world meters. Steel bars are standardized to be exactly 3.0 meters long.

Features:
  - Median-based smoothing (robust to outliers)
  - 30-frame history window
  - ROI masking for furnace glow rejection
"""

from typing import List, Dict, Optional
from collections import deque
import numpy as np

# ============================================================================
# STRICT ROI RECTANGLE CONFIGURATION
# ============================================================================

# CRITICAL: These percentages define a STRICT bounding rectangle for ALL processing
# The physical distance between START_LINE_Y and END_LINE_Y = EXACTLY 3.00 meters
START_LINE_Y_PERCENT = 0.50  # Top of valid zone = 50% of frame height (lower for stable bars on rollers)
END_LINE_Y_PERCENT = 0.90    # Bottom of valid zone = 90% of frame height (keep exact)

# Horizontal boundaries (tight central conveyor focus)
MARGIN_LEFT_PERCENT = 0.46   # Left boundary = 46% of frame width
MARGIN_RIGHT_PERCENT = 0.60  # Right boundary = 60% of frame width

# CALCULATED AT RUNTIME based on frame dimensions
START_LINE_Y = None  # Will be: int(frame_height * 0.50)
END_LINE_Y = None    # Will be: int(frame_height * 0.90)
MARGIN_LEFT = None   # Will be: int(frame_width * 0.46)
MARGIN_RIGHT = None  # Will be: int(frame_width * 0.60)

# AUTOMATIC CALIBRATION: METERS_PER_PIXEL is calculated from the rectangle height
# The rectangle height (pixels) represents EXACTLY 3.00 meters
METERS_PER_PIXEL = None  # Will be: 3.00 / (END_LINE_Y - START_LINE_Y)
TARGET_BAR_LENGTH_METERS = 3.0

# Minimum detection height to filter out sparks/reflections (pixels)
MIN_DETECTION_HEIGHT = 50  # Skip detections smaller than this

# Advanced smoothing parameters
LENGTH_HISTORY_WINDOW = 30  # Keep last 30 frames for median calculation
MEDIAN_SMOOTHING_ENABLED = True


def initialize_roi_config(frame_width: int, frame_height: int) -> Dict[str, float]:
    """
    Initialize ROI configuration with frame-dependent calculations.
    CRITICAL: This must be called once per video to set up the strict rectangle.
    
    The rectangle height (pixels) represents EXACTLY 3.00 meters.
    METERS_PER_PIXEL is automatically calculated from this.
    
    Args:
        frame_width: Width of video frames in pixels
        frame_height: Height of video frames in pixels
        
    Returns:
        Dict with calculated ROI parameters
    """
    global START_LINE_Y, END_LINE_Y, MARGIN_LEFT, MARGIN_RIGHT, METERS_PER_PIXEL
    
    # Calculate absolute pixel coordinates from percentages
    START_LINE_Y = int(frame_height * START_LINE_Y_PERCENT)
    END_LINE_Y = int(frame_height * END_LINE_Y_PERCENT)
    MARGIN_LEFT = int(frame_width * MARGIN_LEFT_PERCENT)
    MARGIN_RIGHT = int(frame_width * MARGIN_RIGHT_PERCENT)
    
    # AUTOMATIC CALIBRATION: Rectangle height = 3.00 meters
    roi_height_pixels = END_LINE_Y - START_LINE_Y
    METERS_PER_PIXEL = 3.00 / roi_height_pixels if roi_height_pixels > 0 else 0.009890
    
    config = {
        'start_line_y': START_LINE_Y,
        'end_line_y': END_LINE_Y,
        'margin_left': MARGIN_LEFT,
        'margin_right': MARGIN_RIGHT,
        'meters_per_pixel': METERS_PER_PIXEL,
        'roi_height_pixels': roi_height_pixels,
        'measurement_start_y': START_LINE_Y
    }
    
    return config


def calculate_measurement_roi(frame_height: int) -> Dict[str, float]:
    """
    Calculate the Region of Interest (ROI) for bar measurement.
    
    USES the STRICT RECTANGLE coordinates defined at module initialization.
    
    Args:
        frame_height: Height of the video frame in pixels
        
    Returns:
        Dict with 'start_line_y', 'end_line_y', 'measurement_start_y'
    """
    if START_LINE_Y is None or END_LINE_Y is None:
        # Fallback if not initialized
        start = int(frame_height * START_LINE_Y_PERCENT)
        end = int(frame_height * END_LINE_Y_PERCENT)
    else:
        start = START_LINE_Y
        end = END_LINE_Y
    
    return {
        'start_line_y': start,
        'end_line_y': end,
        'measurement_start_y': start,
        'measurement_end_y': end
    }


def is_detection_in_roi(detection: Dict, roi_config: Dict) -> bool:
    """
    Check if a detection's centroid is within the valid measurement ROI.
    
    Args:
        detection: Detection dictionary with 'box' coordinates
        roi_config: ROI configuration from calculate_measurement_roi()
        
    Returns:
        bool: True if centroid is in valid measurement region
    """
    x1, y1, x2, y2 = detection['box']
    centroid_y = (y1 + y2) / 2
    
    measurement_start_y = roi_config.get('measurement_start_y', 0)
    
    return centroid_y >= measurement_start_y


def is_detection_in_valid_zone(detection: Dict, frame_width: int, frame_height: int) -> bool:
    """
    Check if a detection is within the STRICT ROI Rectangle.
    
    STRICT validation:
    - Horizontal: center_x must be in [MARGIN_LEFT, MARGIN_RIGHT]
    - Vertical: Must fit within [START_LINE_Y, END_LINE_Y]
    - Size: Height must be >= MIN_DETECTION_HEIGHT (to reject sparks/reflections)
    
    Args:
        detection: Detection dictionary with 'box' coordinates
        frame_width: Width of the frame in pixels
        frame_height: Height of the frame in pixels
        
    Returns:
        bool: True if detection is fully within the strict rectangle
    """
    if MARGIN_LEFT is None or MARGIN_RIGHT is None:
        # Not initialized - use percentages
        left = int(frame_width * MARGIN_LEFT_PERCENT)
        right = int(frame_width * MARGIN_RIGHT_PERCENT)
    else:
        left = MARGIN_LEFT
        right = MARGIN_RIGHT
    
    x1, y1, x2, y2 = detection['box']
    center_x = (x1 + x2) / 2
    height = y2 - y1
    
    # Check horizontal bounds
    if not (left <= center_x <= right):
        return False
    
    # Check minimum height (filter sparks/reflections)
    if height < MIN_DETECTION_HEIGHT:
        return False
    
    return True


def estimate_bar_length(
    detections: List[Dict],
    meters_per_pixel: float = None,  # Will use calculated value
    roi_config: Optional[Dict] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None
) -> List[Dict]:
    """
    Estimate the length of detected bars based on bounding box dimensions.
    
    CRITICAL FILTERS (in order):
    1. Size check: Skip if height < MIN_DETECTION_HEIGHT (sparks/reflections)
    2. Horizontal bounds: center_x must be in [MARGIN_LEFT, MARGIN_RIGHT]
    3. STRICT DUAL-CROP Y: y1 = max(y1, START_LINE_Y), y2 = min(y2, END_LINE_Y)
    4. Calculate length from cropped box
    
    Args:
        detections: List of detection dictionaries with box coordinates
        meters_per_pixel: Conversion factor (uses calculated value if None)
        roi_config: ROI configuration with strict rectangle bounds
        frame_width: Width of frame (for horizontal boundary check)
        frame_height: Height of frame (for boundary calculations)
        
    Returns:
        List[Dict]: Detections with 'estimated_length_m' field
    """
    # Use global calculated meters_per_pixel if not provided
    if meters_per_pixel is None:
        meters_per_pixel = METERS_PER_PIXEL if METERS_PER_PIXEL is not None else 0.009890
    
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        center_x = (x1 + x2) / 2
        height = y2 - y1
        
        # FILTER 1: Minimum size check (reject sparks/reflections)
        if height < MIN_DETECTION_HEIGHT:
            detection['estimated_length_m'] = 0.0
            detection['in_valid_zone'] = False
            detection['in_measurement_roi'] = False
            continue
        
        # FILTER 2: Horizontal bounds check (strict rectangle left/right)
        if frame_width is not None:
            left = MARGIN_LEFT if MARGIN_LEFT is not None else int(frame_width * MARGIN_LEFT_PERCENT)
            right = MARGIN_RIGHT if MARGIN_RIGHT is not None else int(frame_width * MARGIN_RIGHT_PERCENT)
            
            if not (left <= center_x <= right):
                # OUTSIDE valid zone - IGNORE completely
                detection['estimated_length_m'] = 0.0
                detection['in_valid_zone'] = False
                detection['in_measurement_roi'] = False
                continue
            
            detection['in_valid_zone'] = True
        
        # FILTER 3: STRICT DUAL-CROP Y coordinates to rectangle bounds
        # CRITICAL: This prevents furnace glow and bottom artifacts
        if roi_config is not None:
            start_y = roi_config.get('start_line_y', 0)
            end_y = roi_config.get('end_line_y', frame_height)
            
            # Hard crop both top and bottom
            y1 = max(y1, start_y)
            y2 = min(y2, end_y)
            
            detection['box'] = (x1, y1, x2, y2)
            detection['in_measurement_roi'] = True
        
        # Calculate length from the CROPPED bounding box
        width_px = x2 - x1
        height_px = y2 - y1
        
        # Use the longer dimension (bars are rotated ~90 degrees relative to frame)
        bar_length_px = max(width_px, height_px)
        bar_length_m = bar_length_px * meters_per_pixel
        
        detection['estimated_length_m'] = bar_length_m
        detection['estimated_length_cm'] = bar_length_m * 100
        detection['length_source'] = 'bbox_max_dim'
    
    return detections


class LengthSmoother:
    """
    Smooths bar length measurements using a median filter over a 30-frame window.
    Much more robust to sudden spikes (like furnace glow) than exponential smoothing.
    """
    
    def __init__(self, window_size: int = LENGTH_HISTORY_WINDOW):
        """
        Initialize the length smoother.
        
        Args:
            window_size: Number of frames to keep for median calculation
        """
        self.window_size = window_size
        self.length_history = {}  # {track_id: deque of length measurements}
    
    def add_measurement(self, track_id: int, length_m: float):
        """
        Add a new length measurement for a track.
        
        Args:
            track_id: ID of the tracked object
            length_m: Measured length in meters
        """
        if track_id not in self.length_history:
            self.length_history[track_id] = deque(maxlen=self.window_size)
        
        # Only add valid measurements (non-zero)
        if length_m > 0:
            self.length_history[track_id].append(length_m)
    
    def get_smoothed_length(self, track_id: int) -> Optional[float]:
        """
        Get the smoothed length for a track using median filter.
        
        Args:
            track_id: ID of the tracked object
            
        Returns:
            Optional[float]: Median of last N measurements, or None if no data
        """
        if track_id not in self.length_history or len(self.length_history[track_id]) == 0:
            return None
        
        history = list(self.length_history[track_id])
        if len(history) == 0:
            return None
        
        # Use median (robust to outliers) instead of mean
        return float(np.median(history))
    
    def cleanup_track(self, track_id: int):
        """
        Remove history for a track (memory management).
        
        Args:
            track_id: ID of the track to clean up
        """
        if track_id in self.length_history:
            del self.length_history[track_id]


def smooth_length(current_length: float, previous_length: Optional[float], factor: float = 0.7) -> float:
    """
    Legacy exponential moving average smoothing (deprecated in favor of LengthSmoother).
    Kept for backwards compatibility.
    
    Args:
        current_length: Current measured length in meters
        previous_length: Previously smoothed length in meters
        factor: Smoothing factor (0.7 recommended)
        
    Returns:
        float: Smoothed length value
    """
    if previous_length is None:
        return current_length
    
    return factor * current_length + (1 - factor) * previous_length


def calculate_average_bar_length(detections: List[Dict]) -> Optional[float]:
    """
    Calculate the average length of all detected bars in meters.
    
    Args:
        detections: List of detection dictionaries with estimated lengths
        
    Returns:
        Optional[float]: Average bar length in meters, or None if no detections
    """
    if not detections:
        return None
    
    lengths = [d.get('estimated_length_m', 0) for d in detections if d.get('in_measurement_roi', True)]
    valid_lengths = [l for l in lengths if l > 0]
    
    if not valid_lengths:
        return None
    
    return np.mean(valid_lengths)


def get_bar_length_stats(detections: List[Dict]) -> Dict[str, float]:
    """
    Get statistical information about detected bar lengths in meters.
    
    Args:
        detections: List of detection dictionaries with estimated lengths
        
    Returns:
        Dict: Dictionary with min, max, mean, and median bar lengths in meters
    """
    if not detections:
        return {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'median': 0.0,
            'count': 0
        }
    
    lengths = [d.get('estimated_length_m', 0) for d in detections if d.get('in_measurement_roi', True)]
    valid_lengths = [l for l in lengths if l > 0]
    
    if not valid_lengths:
        return {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'median': 0.0,
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
