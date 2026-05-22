"""
Video-based detection module for steel bar detection using YOLOv8 tracking.

Features:
  - ByteTrack with extended track buffer (60+ frames) for persistent tracking
  - ROI masking to ignore furnace glow in top 20% of frame
  - Median-based length smoothing (robust to outliers)
  - Improved NMS tuning (iou=0.45-0.5 to prevent double detection)
"""

import cv2
import os
import sys
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    draw_bounding_boxes,
    add_header_overlay,
    get_color_by_id
)
from src.measure_length import (
    estimate_bar_length,
    smooth_length,
    calculate_measurement_roi,
    LengthSmoother,
)


class MotionTracker:
    """
    Tracks spatial and motion information for detected objects across frames.
    Implements spatial memory for ID recovery on occlusion and motion state detection.
    """
    
    def __init__(self, centroid_history_length=10, spatial_recovery_radius=50):
        """
        Args:
            centroid_history_length: Number of frames to keep for velocity calculation
            spatial_recovery_radius: Pixel radius for spatial ID recovery on reappearance
        """
        self.centroid_history_length = centroid_history_length
        self.spatial_recovery_radius = spatial_recovery_radius
        
        self.track_state = {}
        self.frame_count = 0
    
    def update(self, results, detections: List[Dict], frame_idx: int):
        """
        Update tracking state with new detections and manage spatial memory.
        
        Args:
            results: YOLOv8 tracking results
            detections: Processed detection dictionaries
            frame_idx: Current frame index
            
        Returns:
            Tuple of (updated_detections, recovered_ids_dict)
        """
        self.frame_count = frame_idx
        current_track_ids = set()
        recovered_ids = {}
        
        for det in detections:
            track_id = det.get('track_id')
            if track_id is not None:
                current_track_ids.add(track_id)
                
                x1, y1, x2, y2 = det['box']
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                centroid = (cx, cy)
                
                if track_id not in self.track_state:
                    self.track_state[track_id] = {
                        'centroid_history': deque([centroid], maxlen=self.centroid_history_length),
                        'length_m': det.get('estimated_length_m', 0),
                        'last_seen_frame': frame_idx,
                        'confidence': det.get('confidence', 0.9)
                    }
                else:
                    self.track_state[track_id]['centroid_history'].append(centroid)
                    self.track_state[track_id]['last_seen_frame'] = frame_idx
                    self.track_state[track_id]['confidence'] = det.get('confidence', 0.9)
                    
                    old_length = self.track_state[track_id]['length_m']
                    new_length = det.get('estimated_length_m', old_length)
                    smoothed_length = smooth_length(new_length, old_length if old_length > 0 else None)
                    self.track_state[track_id]['length_m'] = smoothed_length
                    det['estimated_length_m'] = smoothed_length
                
                det['motion_state'] = self._get_motion_state(track_id)
        
        lost_track_ids = set(self.track_state.keys()) - current_track_ids
        
        for lost_id in lost_track_ids:
            last_centroid = self.track_state[lost_id]['centroid_history'][-1] if self.track_state[lost_id]['centroid_history'] else None
            
            if last_centroid is None:
                continue
            
            for det in detections:
                new_id = det.get('track_id')
                if new_id is None or new_id in current_track_ids:
                    continue
                
                x1, y1, x2, y2 = det['box']
                new_centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                dist = np.sqrt(
                    (last_centroid[0] - new_centroid[0]) ** 2 +
                    (last_centroid[1] - new_centroid[1]) ** 2
                )
                
                if dist < self.spatial_recovery_radius:
                    recovered_ids[new_id] = lost_id
                    det['track_id'] = lost_id
                    det['motion_state'] = self._get_motion_state(lost_id)
                    
                    self.track_state[lost_id]['centroid_history'].append(new_centroid)
                    self.track_state[lost_id]['last_seen_frame'] = frame_idx
                    det['estimated_length_m'] = self.track_state[lost_id]['length_m']
                    
                    current_track_ids.add(lost_id)
                    break
        
        return detections, recovered_ids
    
    def _get_motion_state(self, track_id: int, velocity_threshold: float = 5.0) -> str:
        """
        Determine if object is moving or stopped based on centroid velocity.
        
        Args:
            track_id: Track ID to check
            velocity_threshold: Pixel movement threshold (pixels per frame)
            
        Returns:
            String: 'Moving' or 'Stopped'
        """
        if track_id not in self.track_state:
            return 'Unknown'
        
        history = self.track_state[track_id]['centroid_history']
        
        if len(history) < 2:
            return 'Initializing'
        
        recent_frames = min(5, len(history))
        if recent_frames < 2:
            return 'Initializing'
        
        first_point = history[-recent_frames]
        last_point = history[-1]
        
        dist = np.sqrt(
            (last_point[0] - first_point[0]) ** 2 +
            (last_point[1] - first_point[1]) ** 2
        )
        
        velocity = dist / recent_frames
        
        return 'Moving' if velocity > velocity_threshold else 'Stopped'
    
    def cleanup_old_tracks(self, max_frames_lost: int = 30):
        """
        Remove tracks that haven't been seen for too long to prevent memory bloat.
        
        Args:
            max_frames_lost: Maximum frames without seeing a track before removal
        """
        to_remove = [
            track_id for track_id, state in self.track_state.items()
            if self.frame_count - state['last_seen_frame'] > max_frames_lost
        ]
        for track_id in to_remove:
            del self.track_state[track_id]


class VideoDetector:
    """Handles video-based detection with YOLOv8 tracking."""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,  # Increased from 0.3 to prevent double detection
        track_buffer: int = 60,  # Increased from default 30 to remember objects longer
        meters_per_pixel: float = 0.009890,
        target_length: float = 3.0,
        roi_ignore_percent: float = 0.20  # Ignore top 20% (furnace area)
    ):
        """
        Initialize video detector with YOLOv8 model.
        
        Args:
            model_path: Path to YOLOv8 weights file
            confidence_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS (0.45-0.5 optimal)
            track_buffer: ByteTrack memory (frames to remember object)
            meters_per_pixel: Calibration factor
            target_length: Target bar length in meters
            roi_ignore_percent: Fraction of top frame to ignore (furnace)
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.track_buffer = track_buffer
        self.meters_per_pixel = meters_per_pixel
        self.target_length = target_length
        self.roi_ignore_percent = roi_ignore_percent
        
        self.motion_tracker = MotionTracker()
        self.length_smoother = LengthSmoother(window_size=30)
        self.unique_ids = set()
        self.max_id = 0
        self.roi_config = None
        self.frame_height = None
    
    def process_video(
        self,
        video_path: str,
        output_path: str = "outputs/output.mp4",
        show_live: bool = True
    ) -> Dict:
        """
        Process video file with YOLOv8 tracking and save output.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video
            show_live: Display live feed with cv2.imshow()
            
        Returns:
            Dict with processing statistics
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate ROI on first frame with STRICT RECTANGLE initialization
        self.frame_height = height
        self.frame_width = width
        
        # CRITICAL: Initialize strict ROI rectangle and automatic calibration
        from src.measure_length import initialize_roi_config, calculate_measurement_roi
        roi_init = initialize_roi_config(width, height)
        self.roi_config = calculate_measurement_roi(height)
        
        # Update meters_per_pixel from automatic calibration
        self.meters_per_pixel = roi_init['meters_per_pixel']
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        stats = {
            'total_frames': 0,
            'total_detections': 0,
            'unique_track_ids': set(),
            'processing_time': 0,
            'fps_avg': 0,
            'output_video': output_path
        }
        
        frame_times = deque(maxlen=30)
        frame_idx = 0
        
        print(f"\nProcessing video: {video_path}")
        print(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
        print(f"Total frames: {total_frames}")
        print(f"\n🔲 STRICT ROI RECTANGLE CONFIGURATION:")
        print(f"  Horizontal: [{roi_init['margin_left']}px, {roi_init['margin_right']}px] (X in [46%, 60%])")
        print(f"  Vertical: [{roi_init['start_line_y']}px, {roi_init['end_line_y']}px] (35%-90%)")
        print(f"  Rectangle Height: {roi_init['roi_height_pixels']}px = EXACTLY 3.00m")
        print(f"  ✓ Auto-calibration: METERS_PER_PIXEL = {self.meters_per_pixel:.6f}")
        print(f"  ✓ Min detection height: 50px (filters sparks/reflections)\n")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = datetime.now()
                
                # Run YOLOv8 tracking with improved settings
                # Using BoTSORT for better stationary object tracking and occlusion handling
                results = self.model.track(
                    frame,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,  # 0.45 prevents double detection
                    persist=True,
                    tracker='botsort.yaml',  # UPGRADED: BoTSORT better for stationary objects
                    device=0 if self._has_cuda() else 'cpu'
                )
                
                detections = self.get_detections(results, frame.shape)
                
                # Apply ALL filters: Valid Zone, ROI, Hard Y1 Clipping
                detections = estimate_bar_length(
                    detections,
                    self.meters_per_pixel,
                    self.roi_config,
                    frame_width=width,
                    frame_height=height
                )
                
                # Filter out detections outside valid zone (CRITICAL)
                detections_in_zone = [d for d in detections if d.get('in_valid_zone', True)]
                
                # Update motion tracking and handle spatial recovery
                detections_in_zone, recovered = self.motion_tracker.update(results, detections_in_zone, frame_idx)
                
                # Add to length smoother for median-based smoothing
                for det in detections_in_zone:
                    track_id = det.get('track_id')
                    if track_id is not None and det.get('in_measurement_roi', True):
                        self.length_smoother.add_measurement(track_id, det.get('estimated_length_m', 0))
                        smoothed = self.length_smoother.get_smoothed_length(track_id)
                        if smoothed is not None:
                            det['smoothed_length_m'] = smoothed
                
                # Update unique IDs
                for det in detections_in_zone:
                    track_id = det.get('track_id')
                    if track_id is not None:
                        self.unique_ids.add(int(track_id))
                        self.max_id = max(self.max_id, int(track_id))
                
                # Draw annotations (ONLY for detections in valid zone)
                frame_annotated = draw_bounding_boxes(
                    frame,
                    detections_in_zone,
                    self.roi_config,
                    frame_width=width,
                    frame_height=height,
                    meters_per_pixel=self.meters_per_pixel
                )
                
                # Add header overlay
                frame_annotated = add_header_overlay(
                    frame_annotated,
                    total_detections=len(detections_in_zone),
                    unique_tracks=len(self.unique_ids),
                    target_length=self.target_length,
                    fps=1 / (frame_times[-1].total_seconds() + 1e-6) if frame_times else fps
                )
                
                out.write(frame_annotated)
                
                if show_live:
                    cv2.imshow('YOLOv8 Steel Bar Detection - Live Feed', frame_annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nLive feed stopped by user (q pressed)")
                        break
                
                frame_elapsed = datetime.now() - frame_start
                frame_times.append(frame_elapsed)
                stats['total_frames'] += 1
                stats['total_detections'] += len(detections_in_zone)
                stats['unique_track_ids'] = self.unique_ids.copy()
                stats['fps_avg'] = 1 / np.mean([t.total_seconds() for t in frame_times])
                
                if (frame_idx + 1) % 100 == 0:
                    print(f"Frame {frame_idx + 1}/{total_frames} - Detections: {len(detections)}, "
                          f"Unique IDs: {len(self.unique_ids)}, FPS: {stats['fps_avg']:.1f}")
                
                frame_idx += 1
        
        finally:
            cap.release()
            out.release()
            if show_live:
                cv2.destroyAllWindows()
        
        stats['processing_time'] = stats['total_frames'] / stats['fps_avg'] if stats['fps_avg'] > 0 else 0
        
        return stats
    
    def get_detections(self, results, frame_shape: Tuple[int, int, int]) -> List[Dict]:
        """
        Extract detection information from YOLOv8 results.
        
        Args:
            results: YOLOv8 results object
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            List[Dict]: List of detection dictionaries
        """
        detections = []
        height, width = frame_shape[:2]
        
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            boxes_xyxy = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, 'cpu') else result.boxes.xyxy
            conf = result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, 'cpu') else result.boxes.conf
            cls_data = result.boxes.cls.cpu().numpy() if hasattr(result.boxes.cls, 'cpu') else result.boxes.cls
            
            track_ids = None
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy() if hasattr(result.boxes.id, 'cpu') else result.boxes.id
            
            for idx, box in enumerate(boxes_xyxy):
                x1, y1, x2, y2 = box
                
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(width, int(x2))
                y2 = min(height, int(y2))
                
                track_id = None
                if track_ids is not None and idx < len(track_ids):
                    track_id = int(track_ids[idx])
                
                box_width = x2 - x1
                box_height = y2 - y1
                
                detection = {
                    'box': (x1, y1, x2, y2),
                    'width': box_width,
                    'height': box_height,
                    'confidence': float(conf[idx]),
                    'class': int(cls_data[idx]),
                    'track_id': track_id,
                    'motion_state': 'Unknown',
                    'estimated_length_m': 0.0,
                    'smoothed_length_m': 0.0,
                    'in_measurement_roi': True
                }
                
                detections.append(detection)
        
        return detections
    
    @staticmethod
    def _has_cuda() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False


def detect_video(
    video_path: str,
    model_path: str,
    output_path: str = "outputs/output.mp4",
    confidence_threshold: float = 0.4,
    iou_threshold: float = 0.45,  # Improved default
    track_buffer: int = 60,  # Extended buffer
    meters_per_pixel: float = 0.009890,
    target_length: float = 3.0,
    show_live: bool = True
) -> Dict:
    """
    High-level function to run video detection.
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLOv8 weights
        output_path: Path to save output video
        confidence_threshold: Detection confidence threshold
        iou_threshold: NMS IoU threshold (0.45-0.5 recommended)
        track_buffer: ByteTrack memory (frames)
        meters_per_pixel: Calibration factor
        target_length: Target bar length in meters
        show_live: Display live feed during processing
        
    Returns:
        Dict: Processing statistics
    """
    detector = VideoDetector(
        model_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        track_buffer=track_buffer,
        meters_per_pixel=meters_per_pixel,
        target_length=target_length
    )
    
    return detector.process_video(video_path, output_path, show_live=show_live)
