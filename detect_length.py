from ultralytics import YOLO
import cv2
import numpy as np
import math

# تحميل الموديل
model = YOLO("runs/detect/train/weights/best.pt")

# فيديو
cap = cv2.VideoCapture("video.mp4")

# الطول الحقيقي (mm)
REAL_LENGTH_MM = 3000

# calibration
factors = []
factor = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 🟢 نحسب الطول بالـ bounding box
            pixel_length = x2 - x1

            # إذا البار مائل استعمل هذا بدل:
            # pixel_length = math.hypot(x2 - x1, y2 - y1)

            # 🟢 calibration (أول 20 frame)
            if len(factors) < 20 and pixel_length > 0:
                f = REAL_LENGTH_MM / pixel_length
                factors.append(f)

            # 🟢 نحسب المتوسط
            if len(factors) > 0:
                factor = sum(factors) / len(factors)

            # 🟢 حساب الطول
            if factor is not None:
                length_mm = pixel_length * factor

                # رسم box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

                # كتابة الطول
                cv2.putText(frame,
                            f"{length_mm:.0f} mm",
                            (x1, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            3)

    cv2.imshow("Length Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()