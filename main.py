"""
Industrial Steel Bar Detector - IoT entry point.

This file intentionally contains no YOLO tracking or length-measurement logic.
Detection is provided by src.detect_video.detect_video_stream(); this module only:

  - reads CLI configuration,
  - serves annotated frames as MJPEG,
  - publishes dashboard JSON to MQTT.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import paho.mqtt.client as mqtt

from src.detect_video import detect_video_stream
from src.utils import add_header_overlay, draw_bounding_boxes


PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_SOURCE = PROJECT_ROOT / "video.mp4"
MQTT_TOPIC = "factory/bars/data"
TARGET_LENGTH_M = 3.0
URL_SCHEMES = {"http", "https", "rtsp", "rtmp", "udp", "tcp"}


class FrameStore:
    """Thread-safe store for the latest JPEG frame."""

    def __init__(self, quality: int = 85):
        self.quality = quality
        self.condition = threading.Condition()
        self.jpeg: Optional[bytes] = None
        self.sequence = 0

    def update(self, frame) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.quality],
        )
        if not ok:
            return

        with self.condition:
            self.jpeg = encoded.tobytes()
            self.sequence += 1
            self.condition.notify_all()

    def wait(self, last_sequence: int) -> Tuple[Optional[bytes], int]:
        with self.condition:
            if self.sequence == last_sequence:
                self.condition.wait(timeout=2.0)
            return self.jpeg, self.sequence


class MjpegServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler, stop_event: threading.Event):
        super().__init__(address, handler)
        self.stop_event = stop_event


def make_mjpeg_handler(store: FrameStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path in {"/", "/health"}:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"OK. Stream is available at /stream\n")
                return

            if self.path != "/stream":
                self.send_error(404, "Use /stream")
                return

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            last_sequence = -1
            while not self.server.stop_event.is_set():  # type: ignore[attr-defined]
                jpeg, sequence = store.wait(last_sequence)
                if jpeg is None or sequence == last_sequence:
                    continue

                last_sequence = sequence
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    break

    return Handler


class MqttPublisher:
    def __init__(self, broker: str, port: int, topic: str, client_id: str):
        self.topic = topic
        self.connected = False
        self.client = self._new_client(client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.broker = broker
        self.port = port

    @staticmethod
    def _new_client(client_id: str) -> mqtt.Client:
        if hasattr(mqtt, "CallbackAPIVersion"):
            return mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION1,
                client_id=client_id,
            )
        return mqtt.Client(client_id=client_id)

    def _on_connect(self, client, userdata, flags, rc) -> None:
        self.connected = rc == 0
        print(f"MQTT {'connected' if self.connected else 'connection failed'}: {self.broker}:{self.port}")

    def _on_disconnect(self, client, userdata, rc) -> None:
        self.connected = False
        if rc:
            print(f"MQTT disconnected with code {rc}")

    def start(self) -> None:
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self.client.connect_async(self.broker, self.port, keepalive=60)
        self.client.loop_start()

    def publish(self, payload: Dict[str, Any]) -> None:
        if self.connected:
            self.client.publish(
                self.topic,
                json.dumps(payload, separators=(",", ":")),
                qos=1,
                retain=False,
            )

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()


class DashboardState:
    def __init__(self, stream_url: str, target_length: float = TARGET_LENGTH_M):
        self.stream_url = stream_url
        self.target_length = round(target_length, 2)
        self.history: List[Dict[str, Any]] = []
        self.cut_emitted_ids: set[int] = set()

    def reset_cut_edges(self) -> None:
        self.cut_emitted_ids.clear()

    def payload(self, detections: List[Dict[str, Any]], fps: float, mqtt_connected: bool) -> Dict[str, Any]:
        now = datetime.now()
        active_bars = []
        cut_event = None

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            length = round(float(det.get("smoothed_length_m") or det.get("estimated_length_m") or 0.0), 2)
            active_bars.append(
                {
                    "id": int(track_id),
                    "length": length,
                    "confidence": round(float(det.get("confidence", 0.0)), 3),
                    "motion": det.get("motion_state", "Unknown"),
                }
            )

            if length == self.target_length and int(track_id) not in self.cut_emitted_ids:
                self.cut_emitted_ids.add(int(track_id))
                cut_event = {
                    "id": int(track_id),
                    "length": self.target_length,
                    "date": now.strftime("%d/%m/%Y"),
                    "time": now.strftime("%H:%M:%S"),
                    "timestamp": now.isoformat(timespec="seconds"),
                    "command": "CUT",
                }
                self.history.insert(0, cut_event)
                self.history = self.history[:100]

        return {
            "active_count": len(active_bars),
            "active_bars": active_bars,
            "history": self.history,
            "cut_signal": cut_event is not None,
            "cut_event": cut_event,
            "target_length": self.target_length,
            "fps": round(fps, 1),
            "status": "RUNNING",
            "mqtt_connected": mqtt_connected,
            "streamUrl": self.stream_url,
            "timestamp": now.isoformat(timespec="seconds"),
        }


def is_url(value: str) -> bool:
    return urlparse(value).scheme.lower() in URL_SCHEMES


def resolve_path(value: Union[str, Path]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_source(value: str) -> Tuple[Union[int, str], bool]:
    if value.isdigit():
        return int(value), True
    if is_url(value):
        return value, True

    path = resolve_path(value)
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path}")
    return str(path), False


def public_stream_url(args: argparse.Namespace) -> str:
    if args.stream_url:
        return args.stream_url

    host = args.stream_public_host
    if not host:
        host = "127.0.0.1" if args.stream_host in {"", "0.0.0.0", "::"} else args.stream_host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{args.stream_port}/stream"


def run(args: argparse.Namespace) -> None:
    model_path = resolve_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    source, is_live_source = resolve_source(args.source)
    stream_url = public_stream_url(args)
    stop_event = threading.Event()

    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    frame_store = FrameStore(args.jpeg_quality)
    server = MjpegServer((args.stream_host, args.stream_port), make_mjpeg_handler(frame_store), stop_event)
    threading.Thread(target=server.serve_forever, name="mjpeg-server", daemon=True).start()

    mqtt_publisher = MqttPublisher(args.mqtt_broker, args.mqtt_port, args.mqtt_topic, args.mqtt_client_id)
    mqtt_publisher.start()
    state = DashboardState(stream_url)

    print(f"Input source: {source}")
    print(f"MJPEG output: {stream_url}")
    print(f"MQTT topic: {args.mqtt_topic}")

    last_publish = 0.0
    publish_interval = 1.0 / max(args.publish_hz, 0.1)

    try:
        stream = detect_video_stream(
            source,
            str(model_path),
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            track_buffer=args.track_buffer,
            target_length=TARGET_LENGTH_M,
            loop_video=args.loop_video and not is_live_source,
        )

        for item in stream:
            if stop_event.is_set():
                break
            if item.get("source_reset"):
                state.reset_cut_edges()

            detections = item["detections"]
            annotated = draw_bounding_boxes(
                item["frame"].copy(),
                detections,
                item["roi_config"],
                frame_width=item["frame_width"],
                frame_height=item["frame_height"],
                meters_per_pixel=item["meters_per_pixel"],
            )
            annotated = add_header_overlay(
                annotated,
                total_detections=len(detections),
                unique_tracks=item["unique_tracks"],
                target_length=TARGET_LENGTH_M,
                fps=item["fps"],
            )
            frame_store.update(annotated)

            payload = state.payload(detections, item["fps"], mqtt_publisher.connected)
            now = time.monotonic()
            if payload["cut_signal"] or now - last_publish >= publish_interval:
                mqtt_publisher.publish(payload)
                last_publish = now

            if args.show:
                cv2.imshow("Steel Bar Detector IoT", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        stop_event.set()
        mqtt_publisher.stop()
        server.shutdown()
        server.server_close()
        if args.show:
            cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Steel bar detector IoT bridge for FlowFuse/Node-RED.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Video path, camera index, or stream URL.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLOv8 model path.")
    parser.add_argument("--confidence", type=float, default=0.5, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--track-buffer", type=int, default=60, help="BoTSORT track buffer.")
    parser.add_argument("--mqtt-broker", default="localhost", help="MQTT broker host.")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port.")
    parser.add_argument("--mqtt-topic", default=MQTT_TOPIC, help="MQTT data topic.")
    parser.add_argument("--mqtt-client-id", default="steel-bar-detector-iot", help="MQTT client id.")
    parser.add_argument("--stream-host", default="0.0.0.0", help="MJPEG bind host.")
    parser.add_argument("--stream-port", type=int, default=8081, help="MJPEG bind port.")
    parser.add_argument("--stream-public-host", default=None, help="Host/IP published in streamUrl.")
    parser.add_argument("--stream-url", default=None, help="Full streamUrl override published to MQTT.")
    parser.add_argument("--publish-hz", type=float, default=4.0, help="MQTT publish frequency.")
    parser.add_argument("--jpeg-quality", type=int, default=85, help="MJPEG JPEG quality.")
    parser.add_argument("--loop-video", action="store_true", help="Loop local video files for demos.")
    parser.add_argument("--show", action="store_true", help="Show local OpenCV preview window.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
