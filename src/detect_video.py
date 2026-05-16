"""
Video detection module.
Processes video files and detects steel bars using YOLOv8.
"""

import cv2
import os
from pathlib import Path
from typing import Optional, Tuple
from src.utils import (
    load_yolo_model,
    draw_bounding_boxes,
    get_detections,
    create_video_writer,
    add_text_overlay
)
from src.measure_length import estimate_bar_length, get_bar_length_stats


class VideoDetector:
    """Class for detecting objects in video files."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the video detector.
        
        Args:
            model_path: Path to YOLOv8 model file
            confidence_threshold: Minimum confidence for detections
        """
        self.model = load_yolo_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.total_frames = 0
        self.frames_with_detections = 0
        
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
        reference_pixel_to_cm: float = 1.0
    ) -> dict:
        """
        Process a video file and detect steel bars.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save processed video (optional)
            display: Whether to display results (note: may not work in all environments)
            reference_pixel_to_cm: Calibration factor for length estimation
            
        Returns:
            dict: Processing statistics and results
        """
        # Validate input file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        print(f"\nOpening video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS")
        print(f"Total frames: {total_frames}")
        
        # Create video writer if output path specified
        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            writer = create_video_writer(output_path, fps, (frame_width, frame_height))
            print(f"Output video will be saved to: {output_path}")
        
        # Process video
        frame_count = 0
        all_detections = []
        all_lengths = []
        
        print("\nProcessing frames...")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = self.model(frame, verbose=False)
            
            # Extract detections
            detections = get_detections(results, self.confidence_threshold)
            
            # Add frame context to detections
            for det in detections:
                det['frame_number'] = frame_count
            
            all_detections.extend(detections)
            
            # Estimate bar lengths
            if detections:
                detections = estimate_bar_length(detections, reference_pixel_to_cm)
                lengths = [d.get('estimated_length_cm', 0) for d in detections]
                all_lengths.extend(lengths)
                self.frames_with_detections += 1
            
            # Draw annotations
            annotated_frame = draw_bounding_boxes(frame, results, self.confidence_threshold)
            
            # Add statistics overlay
            detection_count = len(detections)
            text = f"Detections: {detection_count} | Frame: {frame_count}/{total_frames}"
            annotated_frame = add_text_overlay(annotated_frame, text, (10, 30))
            
            # Write frame
            if writer:
                writer.write(annotated_frame)
            
            # Show progress
            if frame_count % max(1, total_frames // 10) == 0:
                print(f"  Progress: {frame_count}/{total_frames} frames processed")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        # Calculate statistics
        length_stats = get_bar_length_stats([{'estimated_length_cm': l} for l in all_lengths])
        
        results_dict = {
            'total_frames': frame_count,
            'frames_with_detections': self.frames_with_detections,
            'total_detections': len(all_detections),
            'avg_detections_per_frame': len(all_detections) / frame_count if frame_count > 0 else 0,
            'length_stats': length_stats,
            'output_path': output_path
        }
        
        return results_dict


def detect_video(
    video_path: str,
    model_path: str,
    output_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
    reference_pixel_to_cm: float = 1.0
) -> dict:
    """
    Convenience function to detect objects in a video.
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLOv8 model
        output_path: Path to save processed video
        confidence_threshold: Minimum detection confidence
        reference_pixel_to_cm: Calibration factor for length estimation
        
    Returns:
        dict: Processing results and statistics
    """
    detector = VideoDetector(model_path, confidence_threshold)
    return detector.process_video(
        video_path,
        output_path,
        reference_pixel_to_cm=reference_pixel_to_cm
    )
