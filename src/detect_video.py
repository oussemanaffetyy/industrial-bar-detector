import cv2
import numpy as np
from PIL import Image
from yolov8_ultralytics import YOLO

# Load YOLOv8 model
model = YOLO="yolov8n.pt")

# Load video
video_CAPTURE = cv2.VideoCapture("video.mp4")

if not video_CAPTURE.isOpened():
    print(")    cap.release()
    cv2.destroyAllWindows()
