import cv2
import urllib.request
import numpy as np

# IP متاع ESP32-CAM
url = "http://192.168.59.131/stream"

while True:

    try:
        img_resp = urllib.request.urlopen(url, timeout=5)

        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)

        frame = cv2.imdecode(img_np, -1)

        frame = cv2.resize(frame, (640, 480))

        cv2.imshow("ESP32-CAM", frame)

    except Exception as e:
        print("❌ Error:", e)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()