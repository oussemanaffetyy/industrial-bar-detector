import cv2
import numpy as np
import math

REAL_LENGTH_MM = 3000

cap = cv2.VideoCapture("video.mp4")

factor = None
smooth_len = 0
alpha = 0.2  # كل ما تصغّرها، الثبات يزيد

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_img, w_img = frame.shape[:2]
    center_y = h_img // 2  # وسط الصورة

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # فلترة الضوء فقط
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(thresh, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=400,
        maxLineGap=50
    )

    best_line = None
    max_len = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            dx = x2 - x1
            dy = y2 - y1

            length = math.hypot(dx, dy)

            # 🟢 شرط 1: خط أفقي
            if abs(dy) > 20:
                continue

            # 🟢 شرط 2: قريب لوسط الصورة
            y_avg = (y1 + y2) // 2
            if abs(y_avg - center_y) > 150:
                continue

            # 🟢 ناخذ الأطول
            if length > max_len:
                max_len = length
                best_line = (x1, y1, x2, y2)

    if best_line is not None:
        x1, y1, x2, y2 = best_line

        # smoothing
        smooth_len = alpha * max_len + (1 - alpha) * smooth_len

        if factor is None:
            factor = REAL_LENGTH_MM / smooth_len

        length_mm = smooth_len * factor

        # رسم الخط
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

        # الكتابة تحت البار
        cv2.putText(frame,
                    f"{length_mm:.0f} mm",
                    (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3)

    cv2.imshow("Stable & Accurate Measurement", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()