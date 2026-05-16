"""
Utility functions for the bar detection pipeline.
Includes model loading, visualization, and measurement utilities.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from ultralytics import YOLO


def load_yolo_model(model_path: str) -> YOLO:
    """
    Load a YOLOv8 model from the specified path.
    
    Args:
        model_path: Path to the YOLOv8 model file (.pt)
        
    Returns:
        YOLO: Loaded model object
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model fails to load
    """
    try:
        print(f"Loading YOLOv8 model from: {model_path}")
        model = YOLO(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")


def draw_bounding_boxes(
    frame: np.ndarray,
    results,
    confidence_threshold: float = 0.5,
    class_names: Optional[Dict[int, str]] = None
) -> np.ndarray:
    """
    Draw bounding boxes and confidence scores on the frame.
    
    Args:
        frame: Input image frame
        results: YOLO detection results
        confidence_threshold: Minimum confidence to display detection
        class_names: Optional dictionary mapping class IDs to names
        
    Returns:
        np.ndarray: Frame with drawn bounding boxes
    """
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return frame
    
    # Copy frame to avoid modifying original
    annotated_frame = frame.copy()
    
    for box in results[0].boxes:
        # Get box coordinates (xyxy format: x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get confidence and class
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        
        # Skip if below confidence threshold
        if confidence < confidence_threshold:
            continue
        
        # Draw bounding box (green color: BGR format)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label text
        class_name = class_names.get(class_id, f"Class {class_id}") if class_names else f"Bar"
        label = f"{class_name}: {confidence:.2f}"
        
        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10
        
        cv2.rectangle(
            annotated_frame,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0], text_y + 5),
            (0, 255, 0),
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated_frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )
    
    return annotated_frame


def get_detections(results, confidence_threshold: float = 0.5) -> List[Dict]:
    """
    Extract detection information from YOLO results.
    
    Args:
        results: YOLO detection results
        confidence_threshold: Minimum confidence to include detection
        
    Returns:
        List[Dict]: List of detections with coordinates, confidence, and area
    """
    detections = []
    
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return detections
    
    for box in results[0].boxes:
        confidence = float(box.conf[0])
        
        if confidence < confidence_threshold:
            continue
        
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        detection = {
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'width': width,
            'height': height,
            'area': area,
            'confidence': confidence,
            'class_id': int(box.cls[0])
        }
        detections.append(detection)
    
    return detections


def add_text_overlay(frame: np.ndarray, text: str, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
    """
    Add text overlay to the frame.
    
    Args:
        frame: Input frame
        text: Text to display
        position: (x, y) position for text
        
    Returns:
        np.ndarray: Frame with text overlay
    """
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    return frame


def create_video_writer(output_path: str, fps: float, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """
    Create a video writer object for saving processed video.
    
    Args:
        output_path: Path to save the output video
        fps: Frames per second
        frame_size: (width, height) of frame
        
    Returns:
        cv2.VideoWriter: Video writer object
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {output_path}")
    
    return writer
