"""
Main entry point for the industrial bar detector application.
Handles command-line arguments and coordinates video/camera detection with tracking.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.detect_video import detect_video
from src.detect_camera import detect_camera
from src.measure_length import METERS_PER_PIXEL, TARGET_BAR_LENGTH_METERS


def find_model_path() -> str:
    """
    Find the best available model file.
    Prefers models/best.pt, falls back to yolov8n.pt
    
    Returns:
        str: Path to model file
        
    Raises:
        FileNotFoundError: If no model file is found
    """
    # Priority order for models
    candidates = [
        "models/best.pt"
        
    ]
    
    for model_path in candidates:
        if os.path.exists(model_path):
            print(f"Found model: {model_path}")
            return model_path
    
    raise FileNotFoundError(
        "No YOLOv8 model found. Expected one of: " + ", ".join(candidates)
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Industrial Bar Detector - YOLOv8 with Object Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --video video.mp4
  python main.py --video video.mp4 --output outputs/result.mp4
  python main.py --video video.mp4 --iou 0.3
  python main.py --camera
  python main.py --camera --model yolov8n.pt
  python main.py --video video.mp4 --meters-per-pixel 0.006 --iou 0.4
        """
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Path to input video file for detection'
    )
    
    parser.add_argument(
        '--camera',
        action='store_true',
        help='Use webcam for real-time detection (default camera ID: 0)'
    )
    
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to YOLOv8 model file (default: auto-detect best.pt or yolov8n.pt)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save processed video (for video mode)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Confidence threshold for detections (default: 0.5)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IOU threshold for NMS (default: 0.45). Higher values prevent double detection, lower values allow separate boxes.'
    )
    
    parser.add_argument(
        '--track-buffer',
        type=int,
        default=60,
        help='ByteTrack memory buffer in frames (default: 60). Higher values remember objects longer during occlusions.'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum number of frames to process (for camera mode)'
    )
    
    parser.add_argument(
        '--pixel-to-cm',
        type=float,
        default=None,
        help='Calibration factor: pixels to cm (DEPRECATED: use --meters-per-pixel instead)'
    )
    
    parser.add_argument(
        '--meters-per-pixel',
        type=float,
        default=METERS_PER_PIXEL,
        help=f'Calibration factor: pixels to meters (default: {METERS_PER_PIXEL})'
    )
    
    parser.add_argument(
        '--target-length',
        type=float,
        default=TARGET_BAR_LENGTH_METERS,
        help=f'Target bar length in meters (default: {TARGET_BAR_LENGTH_METERS})'
    )
    
    parser.add_argument(
        '--display',
        action='store_true',
        default=True,
        help='Display live feed during video processing (default: enabled)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable live feed display'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.video and not args.camera:
        parser.print_help()
        print("\nError: Please specify either --video or --camera")
        sys.exit(1)
    
    # Handle display flag
    display_enabled = args.display and not args.no_display
    
    try:
        # Find model
        model_path = args.model or find_model_path()
        print(f"Using model: {model_path}\n")
        
        # Video mode
        if args.video:
            print("=" * 70)
            print("VIDEO DETECTION MODE - LIVE FEED WITH TRACKING")
            print("=" * 70)
            
            # Validate video file
            if not os.path.exists(args.video):
                print(f"Error: Video file not found: {args.video}")
                sys.exit(1)
            
            # Set default output path if not specified
            output_path = args.output or "outputs/output.mp4"
            
            # Run detection with tracking and configurable IOU
            results = detect_video(
                video_path=args.video,
                model_path=model_path,
                output_path=output_path,
                confidence_threshold=args.confidence,
                meters_per_pixel=args.meters_per_pixel,
                target_length=args.target_length,
                iou_threshold=args.iou,
                track_buffer=args.track_buffer,
                show_live=display_enabled
            )
            
            # Print results
            print("\n" + "=" * 70)
            print("DETECTION & TRACKING RESULTS")
            print("=" * 70)
            print(f"Total frames processed: {results['total_frames']}")
            print(f"Total detections: {results['total_detections']}")
            print(f"Unique tracked IDs: {len(results['unique_track_ids'])}")
            print(f"Average FPS: {results['fps_avg']:.2f}")
            print(f"\nTracking Configuration:")
            print(f"  IOU threshold: {args.iou}")
            print(f"  Meters per pixel: {args.meters_per_pixel:.6f}")
            print(f"  Target length: {args.target_length:.2f}m")
            print(f"\nProcessed video saved to: {output_path}")
            print("=" * 70)
        
        # Camera mode
        elif args.camera:
            print("=" * 70)
            print("CAMERA DETECTION MODE - LIVE FEED")
            print("=" * 70)
            
            results = detect_camera(
                model_path=model_path,
                camera_id=args.camera_id,
                confidence_threshold=args.confidence,
                reference_pixel_to_cm=args.pixel_to_cm,
                max_frames=args.max_frames
            )
            
            # Print results
            print("\n" + "=" * 70)
            print("DETECTION RESULTS")
            print("=" * 70)
            print(f"Total frames processed: {results['total_frames']}")
            print(f"Total detections: {results['total_detections']}")
            print(f"Average detections per frame: {results['avg_detections_per_frame']:.2f}")
            print(f"Frames saved: {results['frames_saved']}")
            print("=" * 70)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
