"""
Camera detection module.
Provides real-time detection using webcam or other camera sources.
"""

import cv2
from typing import Optional, Callable
from src.utils import (
    load_yolo_model,
    draw_bounding_boxes,
    get_detections,
    add_text_overlay
)
from src.measure_length import estimate_bar_length


class CameraDetector:
    """Class for real-time object detection using camera."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, camera_id: int = 0):
        """
        Initialize the camera detector.
        
        Args:
            model_path: Path to YOLOv8 model file
            confidence_threshold: Minimum confidence for detections
            camera_id: Camera device ID (0 for default camera)
        """
        self.model = load_yolo_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.camera_id = camera_id
        
    def start_detection(
        self,
        reference_pixel_to_cm: float = 1.0,
        callback: Optional[Callable] = None,
        max_frames: Optional[int] = None
    ) -> dict:
        """
        Start real-time detection from camera.
        
        Args:
            reference_pixel_to_cm: Calibration factor for length estimation
            callback: Optional callback function called for each frame with results
            max_frames: Optional maximum number of frames to process
            
        Returns:
            dict: Detection statistics
        """
        print(f"\nStarting camera detection (Camera ID: {self.camera_id})...")
        
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
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Camera properties: {frame_width}x{frame_height} @ {fps} FPS")
        print("Press 'q' to quit, 's' to save frame with detections")
        
        frame_count = 0
        detection_count = 0
        saved_frames = 0
        
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
                
                # Run detection
                results = self.model(frame, verbose=False)
                
                # Extract detections
                detections = get_detections(results, self.confidence_threshold)
                
                # Estimate bar lengths
                if detections:
                    detections = estimate_bar_length(detections, reference_pixel_to_cm)
                    detection_count += len(detections)
                
                # Draw annotations
                annotated_frame = draw_bounding_boxes(frame, results, self.confidence_threshold)
                
                # Add statistics overlay
                text = f"Detections: {len(detections)} | FPS: {fps}"
                annotated_frame = add_text_overlay(annotated_frame, text, (10, 30))
                
                # Call callback if provided
                if callback:
                    callback(annotated_frame, detections)
                
                # Display frame
                cv2.imshow('Bar Detection - Camera Feed', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Quitting camera detection")
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
            'avg_detections_per_frame': detection_count / frame_count if frame_count > 0 else 0,
            'frames_saved': saved_frames
        }
        
        return results_dict


def detect_camera(
    model_path: str,
    camera_id: int = 0,
    confidence_threshold: float = 0.5,
    reference_pixel_to_cm: float = 1.0,
    max_frames: Optional[int] = None
) -> dict:
    """
    Convenience function for camera detection.
    
    Args:
        model_path: Path to YOLOv8 model
        camera_id: Camera device ID
        confidence_threshold: Minimum detection confidence
        reference_pixel_to_cm: Calibration factor for length estimation
        max_frames: Optional maximum number of frames
        
    Returns:
        dict: Detection statistics
    """
    detector = CameraDetector(model_path, confidence_threshold, camera_id)
    return detector.start_detection(
        reference_pixel_to_cm=reference_pixel_to_cm,
        max_frames=max_frames
    )
