# Industrial Steel Bar Detector (El Fouladh Steel Plant)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111111)
![MQTT](https://img.shields.io/badge/MQTT-Mosquitto-660066)
![Dashboard](https://img.shields.io/badge/Dashboard-FlowFuse%20Vue.js-009688)

Master's PFE end-of-studies project for real-time industrial steel bar detection, tracking, length monitoring, and supervision at the El Fouladh steel plant.

The system detects steel bars with YOLOv8, tracks them with BoTSORT, monitors the 3.00 m cutting target, serves annotated video through an MJPEG HTTP stream, and publishes live production state to a FlowFuse / Node-RED Vue.js dashboard over MQTT.

## Architecture

```text
Video / ESP32 Stream
  -> YOLOv8 + BoTSORT
  -> Length Measurement + Smoothing
  -> MJPEG HTTP Stream: http://<host>:8081/stream
  -> MQTT Topic: factory/bars/data
  -> FlowFuse / Node-RED Vue.js Dashboard
```

## Features

- YOLOv8 inference with the included model: `models/best.pt`
- BoTSORT tracking for stable steel bar IDs
- 3.00 m target-length cut alert generation
- MJPEG annotated video stream for dashboard display
- MQTT JSON publishing to `factory/bars/data`
- Local video and ESP32 camera stream support
- Passive FlowFuse dashboard integration via `flows.json`

## Prerequisites

- Python 3.9+
- Mosquitto MQTT Broker
- Node-RED / FlowFuse Dashboard
- Local video file or ESP32 camera stream

macOS Mosquitto setup:

```bash
brew install mosquitto
brew services start mosquitto
```

Debian/Ubuntu Mosquitto setup:

```bash
sudo apt update
sudo apt install mosquitto mosquitto-clients
sudo systemctl enable --now mosquitto
```

## Installation

```bash
git clone <repository-url>
cd <repository-folder>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Windows Compatibility Checklist

- Create a fresh Windows virtual environment:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\activate
  pip install -r requirements.txt
  ```
- Install and start Mosquitto MQTT Broker for Windows, then keep the broker running before launching `main.py`.
- The code uses `pathlib` for project paths, so both relative paths like `video.mp4` and Windows paths like `C:\path\to\video.mp4` are supported.
- Run OpenCV demos from a normal desktop session. `cv2.imshow()` requires a graphical Windows session and will not display correctly in headless terminals or remote shells without GUI forwarding.
- If another device must open the dashboard stream, allow Python through Windows Firewall for port `8081` and run:
  ```powershell
  python main.py --source video.mp4 --stream-public-host <windows-pc-ip>
  ```

## Usage

Local video:

```bash
python main.py --source video.mp4
```

ESP32 stream:

```bash
python main.py --source http://<esp32-ip>:81/stream
```

Remote dashboard access:

```bash
python main.py --source video.mp4 --stream-public-host <python-host-ip>
```

The Python service publishes:

- MJPEG stream: `http://<python-host-ip>:8081/stream`
- MQTT state: `factory/bars/data`

## FlowFuse / Node-RED

Import `flows.json` into Node-RED / FlowFuse. The dashboard is passive and displays the `streamUrl`, active detections, cut alert, FPS, and history data published by `main.py`.

## Repository Structure

```text
.
├── main.py
├── requirements.txt
├── flows.json
├── models/
│   └── best.pt
└── src/
    ├── __init__.py
    ├── detect_camera.py
    ├── detect_video.py
    ├── measure_length.py
    └── utils.py
```

## Notes

- `models/best.pt` is committed so the project works immediately after cloning.
- `outputs/`, `runs/`, and video files are ignored by Git.
- This repository is production/inference focused; training scripts and generated experiment files are excluded.
