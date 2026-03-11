#!/usr/bin/env python3
"""MJPEG HTTP stream from camera (Picamera2 on Pi, OpenCV elsewhere).

Uses same config as main app. Supports 1280x720-Y10_1X10/RAW on OV9281.

Run on the Raspberry Pi, then from your desktop use OpenCV:
    cap = cv2.VideoCapture("http://<Pi-IP>:8000/stream.mjpg")

Or run PySpectrometer3 with:
    python -m pyspectrometer --camera http://<Pi-IP>:8000/stream.mjpg

Usage:
    poetry run stream              # Pi: Picamera2 (config resolution), port 8000
    poetry run stream --ag --ae    # Auto gain and exposure (center of frame)
    poetry run stream 0            # OpenCV camera 0, port 8000
    poetry run stream 0 9000       # OpenCV camera 0, port 9000
    poetry run stream --config /path/to/config.toml
"""

import argparse
import io
import logging
import socketserver
import sys
import threading
from http import server
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingOutput(io.BufferedIOBase):
    """Holds the latest JPEG frame for MJPEG streaming."""

    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.condition = threading.Condition()

    def write(self, buf: bytes) -> int:
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)


class StreamingHandler(server.BaseHTTPRequestHandler):
    """HTTP handler for MJPEG stream and simple status page."""

    def do_GET(self) -> None:
        if self.path == "/":
            self.send_response(301)
            self.send_header("Location", "/index.html")
            self.end_headers()
            return

        if self.path == "/index.html":
            self._serve_index()
            return

        if self.path == "/stream.mjpg":
            self._serve_stream()
            return

        self.send_error(404)
        self.end_headers()

    def _serve_index(self) -> None:
        content = (
            "<html><head><title>PySpectrometer3 Camera Stream</title></head>"
            "<body><h1>MJPEG Stream</h1>"
            "<p>OpenCV: <code>cv2.VideoCapture(\"http://&lt;this-ip&gt;:{port}/stream.mjpg\")</code></p>"
            "<img src=\"/stream.mjpg\" width=\"800\" /></body></html>"
        ).format(port=self.server.stream_port).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", len(content))
        self.end_headers()
        self.wfile.write(content)

    def _serve_stream(self) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
        )
        self.end_headers()
        try:
            while True:
                with self.server.output.condition:
                    self.server.output.condition.wait()
                    frame = self.server.output.frame
                if frame is None:
                    continue
                self.wfile.write(b"--FRAME\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", len(frame))
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError) as e:
            logger.debug("Client disconnected: %s", e)
        except Exception as e:
            logger.warning(
                "Streaming client %s disconnected: %s",
                self.client_address,
                str(e),
            )

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("%s - %s", self.address_string(), format % args)


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    """Threaded HTTP server with streaming output and port info."""

    allow_reuse_address = True
    daemon_threads = True
    output: StreamingOutput
    stream_port: int


def _parse_camera(source: str) -> int | str:
    """Parse camera source to int or str for OpenCV."""
    s = source.strip()
    if s.isdigit():
        return int(s)
    if s.lower().startswith("v4l:"):
        return s[4:].strip()
    return s


def _run_opencv_stream(
    output: StreamingOutput,
    source: int | str,
    width: int,
    height: int,
) -> None:
    """Capture from OpenCV and write JPEG frames to output."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            _, jpeg = cv2.imencode(".jpg", frame)
            output.write(jpeg.tobytes())
    finally:
        cap.release()


def _extract_center_for_ae_ag(frame: np.ndarray, extractor, max_val: float) -> np.ndarray:
    """Crop center 10% of frame, run SpectrumExtractor, return intensity 0-1 for AE/AG."""
    h, w = frame.shape[:2]
    crop_h = max(5, int(h * 0.10))
    y0 = (h - crop_h) // 2
    y1 = y0 + crop_h
    crop = frame[y0:y1, :]
    result = extractor.extract(crop, max_val=max_val)
    return result.intensity


def _run_picamera_stream(
    output: StreamingOutput,
    width: int,
    height: int,
    monochrome: bool = True,
    bit_depth: int = 10,
    auto_gain: bool = False,
    auto_exposure: bool = False,
) -> None:
    """Capture from Picamera2 (raw when monochrome 10-bit) and write JPEG frames."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from pyspectrometer.capture.picamera import Capture
    from pyspectrometer.core.spectrum import SpectrumData
    from pyspectrometer.processing.auto_controls import AutoExposureController, AutoGainController
    from pyspectrometer.processing.extraction import ExtractionMethod, SpectrumExtractor
    from pyspectrometer.utils.display import scale_to_uint8

    cap = Capture(
        width=width,
        height=height,
        monochrome=monochrome,
        bit_depth=bit_depth,
    )
    cap.start()
    max_val = float((1 << cap.bit_depth) - 1)

    ag_controller = AutoGainController(verbose=True) if auto_gain else None
    ae_controller = AutoExposureController(verbose=True) if auto_exposure else None

    # Extractor for center 10% crop - no calibration, no rotation
    crop_h = max(5, int(height * 0.10))
    extractor = SpectrumExtractor(
        frame_width=width,
        frame_height=crop_h,
        method=ExtractionMethod.MEDIAN,
        rotation_angle=0.0,
        perpendicular_width=min(crop_h, 100),
        spectrum_y_center=crop_h // 2,
    )

    def noop(_: float | int) -> None:
        pass

    try:
        while True:
            frame = cap.capture()
            if auto_gain or auto_exposure:
                intensity = _extract_center_for_ae_ag(frame, extractor, max_val)
                data = SpectrumData(
                    intensity=intensity,
                    wavelengths=np.linspace(0.0, 1.0, len(intensity)),
                )
                if ae_controller:
                    ae_controller.adjust(
                        data,
                        lambda: cap.exposure,
                        lambda v: setattr(cap, "exposure", v),
                        noop,
                    )
                if ag_controller:
                    ag_controller.adjust(
                        data,
                        lambda: cap.gain,
                        lambda v: setattr(cap, "gain", v),
                        noop,
                    )
            if frame.ndim == 2:
                display = scale_to_uint8(frame, max_val)
                display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            else:
                display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, jpeg = cv2.imencode(".jpg", display)
            output.write(jpeg.tobytes())
    finally:
        cap.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream camera as MJPEG over HTTP (Picamera2 on Pi, OpenCV elsewhere)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port (default: 8000)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera source for OpenCV (0, v4l:/dev/video0). Omit for Picamera2 on Pi.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (TOML). Uses same config as main app.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Frame width (default: from config or 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Frame height (default: from config or 720)",
    )
    parser.add_argument(
        "--monochrome",
        action="store_true",
        default=True,
        help="Use monochrome mode (default: True)",
    )
    parser.add_argument(
        "--color",
        action="store_true",
        help="Use color mode (disables monochrome)",
    )
    parser.add_argument(
        "--bit-depth",
        type=int,
        default=10,
        choices=[8, 10, 16],
        help="Bit depth for monochrome (default: 10)",
    )
    parser.add_argument(
        "--ag",
        action="store_true",
        help="Enable auto gain (uses center of frame)",
    )
    parser.add_argument(
        "--ae",
        action="store_true",
        help="Enable auto exposure (uses center of frame)",
    )
    args = parser.parse_args()

    # Load config (same as main app)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from pyspectrometer.config import load_config

    config_path = Path(args.config) if args.config else None
    config, config_loaded = load_config(config_path)
    if config_loaded:
        logger.info("Config: %s", config_loaded)

    width = args.width or config.camera.frame_width
    height = args.height or config.camera.frame_height
    monochrome = args.monochrome and not args.color

    output = StreamingOutput()

    if args.camera is not None:
        source = _parse_camera(args.camera)
        thread = threading.Thread(
            target=_run_opencv_stream,
            args=(output, source, width, height),
            daemon=True,
        )
        thread.start()
    else:
        try:
            from picamera2 import Picamera2  # noqa: F401
        except ImportError as e:
            logger.error(
                "Picamera2 not found. Use --camera 0 for webcam: poetry run stream 0"
            )
            raise SystemExit(1) from e

        logger.info(
            "Picamera2: %dx%d monochrome=%s bit_depth=%d (supports 1280x720-Y10 raw)",
            width,
            height,
            monochrome,
            args.bit_depth,
        )
        if args.ag or args.ae:
            logger.info("Auto: AG=%s AE=%s (center 10%% extract)", args.ag, args.ae)
        thread = threading.Thread(
            target=_run_picamera_stream,
            args=(
                output,
                width,
                height,
                monochrome,
                args.bit_depth,
                args.ag,
                args.ae,
            ),
            daemon=True,
        )
        thread.start()

    server_instance = StreamingServer(
        ("", args.port),
        StreamingHandler,
    )
    server_instance.output = output
    server_instance.stream_port = args.port

    try:
        import socket

        def _lan_ip() -> str:
            """Get LAN IP (not loopback) for stream URL."""
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
            except OSError:
                return "127.0.0.1"
            finally:
                s.close()

        addr = _lan_ip()
        logger.info(
            "MJPEG stream at http://%s:%d/stream.mjpg",
            addr,
            args.port,
        )
        logger.info(
            "Connect: --camera http://%s:%d/stream.mjpg",
            addr,
            args.port,
        )
        server_instance.serve_forever()
    finally:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
