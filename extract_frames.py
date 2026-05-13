import cv2
import os

if not os.path.exists("images"):
    os.makedirs("images")

cap = cv2.VideoCapture("video.mp4")
i = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imwrite(f"images/frame_{i}.jpg", frame)
    i += 1

cap.release()
print("Done")
