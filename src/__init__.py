"""
Industrial Bar Detector - YOLOv8 based steel bar detection system

A modular detection pipeline for identifying and measuring steel bars in video
using YOLOv8 object detection model.
"""

__version__ = "1.0.0"
__author__ = "Industrial Bar Detection Team"

from src.detect_video import detect_video, VideoDetector
from src.detect_camera import detect_camera, CameraDetector
from src.utils import load_yolo_model, draw_bounding_boxes, get_detections
from src.measure_length import estimate_bar_length, get_bar_length_stats

__all__ = [
    'detect_video',
    'detect_camera',
    'VideoDetector',
    'CameraDetector',
    'load_yolo_model',
    'draw_bounding_boxes',
    'get_detections',
    'estimate_bar_length',
    'get_bar_length_stats',
]
