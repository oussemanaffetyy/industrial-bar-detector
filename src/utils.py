"""
Utility functions for video processing and visualization.

Features:
  - ROI visualization (measurement start line)
  - Median-smoothed length display
  - Professional UI with color-coded tracking
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


COLOR_PALETTE = [
    (255, 0, 0),       # Blue (strong, high-contrast)
    (0, 255, 0),       # Green (strong, high-contrast)
    (0, 255, 255),     # Yellow (bright, high-contrast)
    (255, 255, 0),     # Cyan (bright, high-contrast)
    (255, 0, 255),     # Magenta (bold, high-contrast)
    (0, 165, 255),     # Orange (bold, high-contrast)
    (128, 0, 128),     # Purple (bold, high-contrast)
    (0, 128, 128),     # Teal (distinct)
    (255, 165, 0),     # Deep Orange (warm, high-contrast)
    (200, 0, 0),       # Strong Blue (darker shade)
    (0, 200, 0),       # Strong Green (darker shade)
    (0, 200, 200),     # Strong Cyan (darker shade)
    (200, 200, 0),     # Strong Yellow (darker shade)
    (200, 0, 200),     # Strong Magenta (darker shade)
]


def get_color_by_id(track_id: Optional[int]) -> Tuple[int, int, int]:
    """
    Generate a distinct color from the palette based on the track ID.
    
    Args:
        track_id: Tracking ID of the object (None for untracked)
        
    Returns:
        Tuple: BGR color tuple for OpenCV
    """
    if track_id is None:
        return (255, 255, 255)  # White for untracked objects
    return COLOR_PALETTE[int(track_id) % len(COLOR_PALETTE)]


def get_motion_color(motion_state: str) -> Tuple[int, int, int]:
    """
    Get color based on motion state for visual distinction.
    
    Args:
        motion_state: Motion state string ('Moving', 'Stopped', etc.)
        
    Returns:
        Tuple: BGR color tuple
    """
    if motion_state == 'Moving':
        return (0, 255, 0)  # Green for moving
    elif motion_state == 'Stopped':
        return (0, 0, 255)  # Red for stopped
    else:
        return (255, 255, 255)  # White for unknown


def draw_dashed_line(frame: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], 
                     color: Tuple[int, int, int], thickness: int = 2, dash_length: int = 15, gap_length: int = 10) -> None:
    """
    Draw a dashed line on a frame (since OpenCV cv2.line doesn't support dashed natively).
    
    Args:
        frame: Input frame (will be modified in-place)
        pt1: Starting point (x, y)
        pt2: Ending point (x, y)
        color: Line color (BGR tuple)
        thickness: Line thickness in pixels
        dash_length: Length of each dash segment in pixels
        gap_length: Length of gap between dashes in pixels
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Calculate distance and direction
    dx = x2 - x1
    dy = y2 - y1
    distance = np.sqrt(dx**2 + dy**2)
    
    if distance == 0:
        return
    
    # Normalize direction
    dx /= distance
    dy /= distance
    
    # Draw dashes
    segment_length = dash_length + gap_length
    num_segments = int(distance / segment_length) + 1
    
    for i in range(num_segments):
        start_dist = i * segment_length
        end_dist = min(start_dist + dash_length, distance)
        
        if start_dist >= distance:
            break
        
        # Calculate actual points
        dash_start = (int(x1 + dx * start_dist), int(y1 + dy * start_dist))
        dash_end = (int(x1 + dx * end_dist), int(y1 + dy * end_dist))
        
        cv2.line(frame, dash_start, dash_end, color, thickness)


def draw_dashed_rectangle(frame: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                          color: Tuple[int, int, int], thickness: int = 3, 
                          dash_length: int = 20, gap_length: int = 10) -> None:
    """
    Draw a dashed rectangle on a frame.
    
    Args:
        frame: Input frame (will be modified in-place)
        pt1: Top-left corner (x, y)
        pt2: Bottom-right corner (x, y)
        color: Rectangle color (BGR tuple)
        thickness: Line thickness in pixels
        dash_length: Length of each dash segment
        gap_length: Length of gap between dashes
    """
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw four dashed sides
    # Top side
    draw_dashed_line(frame, (x1, y1), (x2, y1), color, thickness, dash_length, gap_length)
    # Right side
    draw_dashed_line(frame, (x2, y1), (x2, y2), color, thickness, dash_length, gap_length)
    # Bottom side
    draw_dashed_line(frame, (x2, y2), (x1, y2), color, thickness, dash_length, gap_length)
    # Left side
    draw_dashed_line(frame, (x1, y2), (x1, y1), color, thickness, dash_length, gap_length)


def draw_bounding_boxes(
    frame: np.ndarray,
    detections: List[Dict],
    roi_config: Optional[Dict] = None,
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None,
    meters_per_pixel: float = 0.009890
) -> np.ndarray:
    """
    Draw bounding boxes with motion state and smoothed length information on the frame.
    
    Args:
        frame: Input frame (BGR image)
        detections: List of detection dictionaries
        roi_config: ROI configuration for visualization
        
    Returns:
        np.ndarray: Frame with drawn annotations
    """
    frame_h, frame_w = frame.shape[:2]
    
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        track_id = detection.get('track_id')
        confidence = detection.get('confidence', 0.0)
        
        # Use smoothed length if available, otherwise use raw estimate
        length_m = detection.get('smoothed_length_m', 0.0)
        if length_m == 0.0:
            length_m = detection.get('estimated_length_m', 0.0)
        
        motion_state = detection.get('motion_state', 'Unknown')
        in_roi = detection.get('in_measurement_roi', True)
        
        # Get color based on track ID
        color = get_color_by_id(track_id)
        
        # Dim the box if outside measurement ROI
        if not in_roi:
            color = tuple(int(c * 0.5) for c in color)  # Darken color
        
        # Draw main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)
        
        # Draw corner markers for better visibility
        corner_length = 15
        corner_thickness = 2
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
        
        # Prepare label text - show smoothed length
        if track_id is not None:
            label = f"Bar {track_id} | {length_m:.2f}m"
        else:
            label = f"{length_m:.2f}m | {confidence:.2f}"
        
        info_label = ""
        if track_id is not None:
            info_label = f"Conf: {confidence:.2f}"
            if not in_roi:
                info_label += " [ROI]"
        
        # DYNAMIC TEXT PLACEMENT - avoid overlap and improve readability
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        info_size = cv2.getTextSize(info_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        
        # Calculate box center
        center_x = (x1 + x2) // 2
        
        # Determine if bar is on left or right side of screen
        if center_x < frame_w / 2:
            # LEFT SIDE: Draw label to the LEFT of the box
            text_x = max(10, x1 - label_size[0] - 10)
            text_y = max(25, y1 - 5)
        else:
            # RIGHT SIDE: Draw label to the RIGHT of the box
            text_x = min(frame_w - label_size[0] - 10, x2 + 10)
            text_y = max(25, y1 - 5)
        
        # Ensure text stays within frame bounds
        text_x = max(5, min(text_x, frame_w - label_size[0] - 5))
        text_y = max(label_size[1] + 5, min(text_y, frame_h - 5))
        
        # Draw dark background for label (solid, fully opaque for maximum readability)
        text_bg_x1 = text_x - 5
        text_bg_y1 = text_y - label_size[1] - 5
        text_bg_x2 = text_x + label_size[0] + 5
        text_bg_y2 = text_y + 5
        
        cv2.rectangle(
            frame,
            (max(0, text_bg_x1), max(0, text_bg_y1)),
            (min(frame_w, text_bg_x2), min(frame_h, text_bg_y2)),
            (0, 0, 0),  # Solid black background
            -1
        )
        
        # Draw text (bright white, maximum contrast)
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # Bright white
            2,  # Thicker font for better readability
            cv2.LINE_AA
        )
        
        # Draw info label below if exists
        if info_label:
            info_y = text_y + label_size[1] + 12
            info_bg_y1 = info_y - info_size[1] - 2
            info_bg_y2 = info_y + 3
            
            if info_bg_y2 <= frame_h:
                # Dark background for info label
                cv2.rectangle(
                    frame,
                    (max(0, text_x - 5), max(0, info_bg_y1)),
                    (min(frame_w, text_x + info_size[0] + 5), min(frame_h, info_bg_y2)),
                    (0, 0, 0),  # Solid black background
                    -1
                )
                
                cv2.putText(
                    frame,
                    info_label,
                    (text_x, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),  # Bright white
                    1,
                    cv2.LINE_AA
                )
    
    # Draw STRICT ROI RECTANGLE with dashed lines (yellow/cyan, thick)
    if roi_config is not None and frame_width is not None and frame_height is not None:
        from src.measure_length import MARGIN_LEFT, MARGIN_RIGHT, START_LINE_Y, END_LINE_Y
        
        frame_h, frame_w = frame.shape[:2]
        
        # Use the calculated strict rectangle coordinates
        if START_LINE_Y is not None and END_LINE_Y is not None and MARGIN_LEFT is not None and MARGIN_RIGHT is not None:
            left = MARGIN_LEFT
            right = MARGIN_RIGHT
            top = START_LINE_Y
            bottom = END_LINE_Y
        else:
            # Fallback calculation if not initialized
            from src.measure_length import START_LINE_Y_PERCENT, END_LINE_Y_PERCENT, MARGIN_LEFT_PERCENT, MARGIN_RIGHT_PERCENT
            left = int(frame_w * MARGIN_LEFT_PERCENT)
            right = int(frame_w * MARGIN_RIGHT_PERCENT)
            top = int(frame_h * START_LINE_Y_PERCENT)
            bottom = int(frame_h * END_LINE_Y_PERCENT)
        
        # Draw DASHED RECTANGLE with DARK RED color + black outline for visibility
        color_rect_main = (0, 0, 180)  # Dark Red in BGR (strong, reserved for ROI)
        color_rect_outline = (0, 0, 0)  # Black outline
        thickness = 4
        dash_length = 20
        gap_length = 10
        
        # Draw black outline first (behind the dark red) for pop effect
        draw_dashed_rectangle(frame, (left, top), (right, bottom), color_rect_outline, thickness + 2, dash_length, gap_length)
        
        # Draw dark red dashed rectangle on top
        draw_dashed_rectangle(frame, (left, top), (right, bottom), color_rect_main, thickness, dash_length, gap_length)
        
        # NO TEXT OVERLAY - clean appearance (removed by user request)
    
    return frame


def add_header_overlay(
    frame: np.ndarray,
    total_detections: int = 0,
    unique_tracks: int = 0,
    target_length: float = 3.0,
    fps: float = 0.0,
    title: str = "YOLOv8 Steel Bar Detection - Live Feed"
) -> np.ndarray:
    """
    Add a professional header overlay at the top of the frame.
    
    Args:
        frame: Input frame (BGR image)
        total_detections: Number of detections in current frame
        unique_tracks: Number of unique tracked objects
        target_length: Target bar length in meters
        fps: Current processing FPS
        title: Title to display in header
        
    Returns:
        np.ndarray: Frame with header overlay
    """
    frame_h, frame_w = frame.shape[:2]
    
    # RIGHT: "Bar detection" title (bold, top-right corner)
    title_text = "Bar detection"
    title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]
    
    # Position: top-right corner with small margin (10px from right, 15px from top)
    title_x = frame_w - title_size[0] - 10
    title_y = 28
    
    cv2.putText(
        frame,
        title_text,
        (title_x, title_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        3,  # Thicker font (bold)
        cv2.LINE_AA
    )
    
    # RIGHT: Status information (below title, same right alignment)
    status_text = f"Detections: {total_detections} | Target: {target_length:.2f}m | FPS: {fps:.1f}"
    status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    
    # Position below title, right-aligned
    status_x = frame_w - status_size[0] - 10
    status_y = title_y + 25
    
    # Draw semi-transparent black rectangle behind status text for readability
    padding = 5
    bg_x1 = max(0, status_x - padding)
    bg_y1 = max(0, status_y - status_size[1] - padding)
    bg_x2 = min(frame_w, status_x + status_size[0] + padding)
    bg_y2 = min(frame_h, status_y + padding)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Draw status text (bright white for visibility)
    cv2.putText(
        frame,
        status_text,
        (status_x, status_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    
    return frame


def load_model(model_path: str):
    """
    Load a YOLOv8 model.
    
    Args:
        model_path: Path to model weights file
        
    Returns:
        YOLO: Loaded model object
    """
    from ultralytics import YOLO
    return YOLO(model_path)


def get_frame_info(frame: np.ndarray) -> Dict:
    """
    Get basic information about a frame.
    
    Args:
        frame: Input frame (BGR image)
        
    Returns:
        Dict: Frame information (height, width, channels)
    """
    h, w = frame.shape[:2]
    c = frame.shape[2] if len(frame.shape) == 3 else 1
    
    return {
        'height': h,
        'width': w,
        'channels': c,
        'shape': frame.shape
    }


def resize_frame(frame: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """
    Resize a frame by a scale factor.
    
    Args:
        frame: Input frame
        scale: Scale factor (0.5 = 50% size)
        
    Returns:
        np.ndarray: Resized frame
    """
    h, w = frame.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def draw_grid(frame: np.ndarray, grid_size: int = 50, color: Tuple[int, int, int] = (50, 50, 50)) -> np.ndarray:
    """
    Draw a reference grid on the frame for spatial reference.
    
    Args:
        frame: Input frame
        grid_size: Size of grid cells in pixels
        color: Color of grid lines (BGR)
        
    Returns:
        np.ndarray: Frame with grid overlay
    """
    h, w = frame.shape[:2]
    
    # Vertical lines
    for x in range(0, w, grid_size):
        cv2.line(frame, (x, 0), (x, h), color, 1)
    
    # Horizontal lines
    for y in range(0, h, grid_size):
        cv2.line(frame, (0, y), (w, y), color, 1)
    
    return frame
