"""
Camera detection module.
Provides real-time detection using webcam or other camera sources.
"""

import cv2
import sys
from pathlib import Path
from typing import Optional, Dict

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    draw_bounding_boxes,
    add_header_overlay,
)
from src.measure_length import estimate_bar_length


class CameraDetector:
    """Class for real-time object detection using camera."""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.3,
        camera_id: int = 0,
        meters_per_pixel: float = 0.009890
    ):
        """
        Initialize the camera detector.
        
        Args:
            model_path: Path to YOLOv8 model file
            confidence_threshold: Minimum confidence for detections
            iou_threshold: NMS IOU threshold
            camera_id: Camera device ID (0 for default camera)
            meters_per_pixel: Calibration factor for length estimation
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.camera_id = camera_id
        self.meters_per_pixel = meters_per_pixel
        
    def start_detection(
        self,
        target_length: float = 3.0,
        max_frames: Optional[int] = None
    ) -> Dict:
        """
        Start real-time detection from camera.
        
        Args:
            target_length: Target bar length in meters
            max_frames: Optional maximum number of frames to process
            
        Returns:
            dict: Detection statistics
        """
        print(f"\n{'='*60}")
        print(f"CAMERA DETECTION MODE")
        print(f"{'='*60}")
        
        # Open camera
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera with ID: {self.camera_id}")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        print(f"Camera: ID {self.camera_id}")
        print(f"Resolution: {frame_width}x{frame_height} @ {fps:.1f} FPS")
        print(f"Model: {self.model.model_name if hasattr(self.model, 'model_name') else 'YOLOv8'}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"IOU threshold: {self.iou_threshold}")
        print(f"\nPress 'q' to quit, 's' to save frame")
        print(f"{'='*60}\n")
        
        frame_count = 0
        detection_count = 0
        saved_frames = 0
        unique_ids = set()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                frame_count += 1
                
                # Check max frames limit
                if max_frames and frame_count > max_frames:
                    break
                
                # Run tracking
                results = self.model.track(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    persist=True,
                    tracker='bytetrack.yaml',
                    verbose=False
                )
                
                # Extract detections manually from results
                detections = []
                if results and results[0].boxes is not None:
                    for box, conf, cls in zip(
                        results[0].boxes.xyxy,
                        results[0].boxes.conf,
                        results[0].boxes.cls
                    ):
                        x1, y1, x2, y2 = box.tolist()
                        
                        # Get track ID if available
                        track_id = None
                        if results[0].boxes.id is not None:
                            idx = list(results[0].boxes.xyxy).index(box)
                            track_id = int(results[0].boxes.id[idx])
                            unique_ids.add(track_id)
                        
                        detection = {
                            'box': (int(x1), int(y1), int(x2), int(y2)),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1),
                            'confidence': float(conf),
                            'class': int(cls),
                            'track_id': track_id,
                            'motion_state': 'Streaming',
                            'estimated_length_m': 0.0
                        }
                        detections.append(detection)
                
                # Estimate lengths
                if detections:
                    detections = estimate_bar_length(detections, self.meters_per_pixel)
                    detection_count += len(detections)
                
                # Draw annotations
                annotated_frame = draw_bounding_boxes(frame, detections)
                
                # Add header overlay
                annotated_frame = add_header_overlay(
                    annotated_frame,
                    total_detections=len(detections),
                    unique_tracks=len(unique_ids),
                    target_length=target_length,
                    fps=fps,
                    title="YOLOv8 Steel Bar Detection - Camera Feed"
                )
                
                # Display frame
                cv2.imshow('Bar Detection - Camera Feed', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting camera detection")
                    break
                elif key == ord('s'):
                    # Save frame with detections
                    save_path = f"outputs/camera_frame_{frame_count}.jpg"
                    cv2.imwrite(save_path, annotated_frame)
                    print(f"Frame saved to: {save_path}")
                    saved_frames += 1
                
        except KeyboardInterrupt:
            print("\nDetection interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        results_dict = {
            'total_frames': frame_count,
            'total_detections': detection_count,
            'unique_track_ids': len(unique_ids),
            'avg_detections_per_frame': detection_count / frame_count if frame_count > 0 else 0,
            'frames_saved': saved_frames
        }
        
        print(f"\n{'='*60}")
        print(f"CAMERA DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total frames processed: {results_dict['total_frames']}")
        print(f"Total detections: {results_dict['total_detections']}")
        print(f"Unique tracked IDs: {results_dict['unique_track_ids']}")
        print(f"Avg detections/frame: {results_dict['avg_detections_per_frame']:.2f}")
        print(f"Frames saved: {results_dict['frames_saved']}")
        print(f"{'='*60}\n")
        
        return results_dict


def detect_camera(
    model_path: str,
    camera_id: int = 0,
    confidence_threshold: float = 0.4,
    iou_threshold: float = 0.3,
    meters_per_pixel: float = 0.009890,
    target_length: float = 3.0,
    max_frames: Optional[int] = None
) -> Dict:
    """
    Convenience function for camera detection.
    
    Args:
        model_path: Path to YOLOv8 model
        camera_id: Camera device ID
        confidence_threshold: Minimum detection confidence
        iou_threshold: NMS IOU threshold
        meters_per_pixel: Calibration factor for length estimation
        target_length: Target bar length in meters
        max_frames: Optional maximum number of frames
        
    Returns:
        dict: Detection statistics
    """
    detector = CameraDetector(
        model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        camera_id=camera_id,
        meters_per_pixel=meters_per_pixel
    )
    return detector.start_detection(target_length=target_length, max_frames=max_frames)
