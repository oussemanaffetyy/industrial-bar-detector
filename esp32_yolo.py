from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/train/weights/best.pt")

# 🔴 مهم: بدّل IP
cap = cv2.VideoCapture("http://192.168.1.100:81/stream")

REAL_LENGTH_MM = 3000

factors = []
factor = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame)

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            pixel_length = x2 - x1

            if len(factors) < 20:
                factors.append(REAL_LENGTH_MM / pixel_length)

            factor = sum(factors) / len(factors)

            length_mm = pixel_length * factor

            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,
                        f"{length_mm:.0f} mm",
                        (x1,y2+30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)

    cv2.imshow("ESP32 + YOLO", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()