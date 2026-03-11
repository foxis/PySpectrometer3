#!/usr/bin/env python3
"""MJPEG HTTP stream from Picamera2 on Raspberry Pi.

Uses pyspectrometer: Config, Capture (picamera), SpectrumExtractor,
AutoGainController, AutoExposureController, Calibration, scale_to_uint8.
Only this script implements the MJPEG HTTP server; all capture and processing
comes from the library.

Connect from desktop:
    python -m pyspectrometer --camera http://<Pi-IP>:8000/stream.mjpg

Usage:
    poetry run stream              # Port 8000, config resolution
    poetry run stream --port 9000
    poetry run stream --ag --ae    # Auto gain and exposure
"""

import argparse
import io
import logging
import socket
import socketserver
import sys
import threading
from http import server
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- MJPEG streamer (only custom code) ---


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
    """HTTP handler for MJPEG stream and status page."""

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
        port = self.server.stream_port
        content = (
            (
                "<html><head><title>PySpectrometer3 Stream</title></head>"
                "<body><h1>MJPEG Stream</h1>"
                f'<p><code>cv2.VideoCapture("http://&lt;this-ip&gt;:{port}/stream.mjpg")</code></p>'
                '<img src="/stream.mjpg" width="800" /></body></html>'
            )

            .encode()
        )
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
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
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
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("Client disconnected")
        except Exception as e:
            logger.warning("Stream client %s: %s", self.client_address, e)

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("%s - %s", self.address_string(), format % args)


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    """Threaded HTTP server for MJPEG."""

    allow_reuse_address = True
    daemon_threads = True
    output: StreamingOutput
    stream_port: int


# --- Capture loop (all from pyspectrometer) ---


def _capture_loop(
    output: StreamingOutput,
    config_path: Path | None,
    auto_gain: bool,
    auto_exposure: bool,
) -> None:
    """Capture pure camera frames, optionally run AE/AG, write JPEGs to output."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    import numpy as np

    from pyspectrometer.capture.picamera import Capture
    from pyspectrometer.config import load_config
    from pyspectrometer.core.spectrum import SpectrumData
    from pyspectrometer.processing.auto_controls import AutoExposureController, AutoGainController
    from pyspectrometer.processing.extraction import ExtractionMethod, SpectrumExtractor
    from pyspectrometer.utils.display import scale_to_uint8

    config, _ = load_config(config_path)
    camera = Capture(
        width=config.camera.frame_width,
        height=config.camera.frame_height,
        gain=config.camera.gain,
        fps=config.camera.fps,
        monochrome=config.camera.monochrome,
        bit_depth=config.camera.bit_depth,
    )

    auto_gain_ctrl = AutoGainController(verbose=True) if auto_gain else None
    auto_exposure_ctrl = AutoExposureController(verbose=True) if auto_exposure else None
    extractor: SpectrumExtractor | None = None
    if auto_gain or auto_exposure:
        extractor = SpectrumExtractor(
            frame_width=config.camera.frame_width,
            frame_height=config.camera.frame_height,
            method=ExtractionMethod.MEDIAN,
            rotation_angle=0.0,
            perpendicular_width=config.extraction.perpendicular_width,
            spectrum_y_center=config.camera.frame_height // 2,
        )

    camera.start()
    actual_w, actual_h = camera.width, camera.height
    if extractor is not None and (
        actual_w != config.camera.frame_width or actual_h != config.camera.frame_height
    ):
        config.camera.frame_width = actual_w
        config.camera.frame_height = actual_h
        extractor.set_dimensions(actual_w, actual_h)

    max_val = float((1 << camera.bit_depth) - 1)

    def noop(_: float | int) -> None:
        pass

    try:
        while True:
            frame = camera.capture()

            if (auto_gain or auto_exposure) and extractor is not None:
                extraction = extractor.extract(frame, max_val=max_val)
                intensity = extraction.intensity.astype("float32")
                n = len(intensity)
                data = SpectrumData(
                    intensity=intensity,
                    wavelengths=np.linspace(0.0, 1.0, n),
                    raw_frame=frame,
                    cropped_frame=extraction.cropped_frame,
                )
                if auto_exposure_ctrl:
                    auto_exposure_ctrl.adjust(
                        data,
                        lambda: camera.exposure,
                        lambda v: setattr(camera, "exposure", v),
                        noop,
                    )
                if auto_gain_ctrl:
                    auto_gain_ctrl.adjust(
                        data,
                        lambda: camera.gain,
                        lambda v: setattr(camera, "gain", v),
                        noop,
                    )

            display = scale_to_uint8(frame, max_val)
            if display.ndim == 2:
                display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            else:
                display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
            _, jpeg = cv2.imencode(".jpg", display)
            output.write(jpeg.tobytes())
    finally:
        camera.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MJPEG stream from Picamera2 (Pi only). Uses pyspectrometer pipeline."
    )
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--config", type=str, default=None, help="Config file path (TOML)")
    parser.add_argument("--ag", action="store_true", help="Enable auto gain")
    parser.add_argument("--ae", action="store_true", help="Enable auto exposure")
    parser.add_argument("--camera", type=str, default=None, help="Ignored (Pi only)")
    args = parser.parse_args()

    if args.camera is not None:
        logger.warning("--camera ignored. Stream is Pi-only (Picamera2).")
        logger.info(
            "Connect from desktop: python -m pyspectrometer --camera http://<Pi-IP>:%d/stream.mjpg",
            args.port,
        )

    try:
        from picamera2 import Picamera2  # noqa: F401
    except ImportError:
        logger.error("Picamera2 not found. Run this script on Raspberry Pi.")
        return 1

    config_path = Path(args.config) if args.config else None
    output = StreamingOutput()

    thread = threading.Thread(
        target=_capture_loop,
        args=(output, config_path, args.ag, args.ae),
        daemon=True,
    )
    thread.start()

    server_instance = StreamingServer(("", args.port), StreamingHandler)
    server_instance.output = output
    server_instance.stream_port = args.port

    def _lan_ip() -> str:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"
        finally:
            s.close()

    addr = _lan_ip()
    logger.info("MJPEG stream: http://%s:%d/stream.mjpg", addr, args.port)
    logger.info(
        "Connect: python -m pyspectrometer --camera http://%s:%d/stream.mjpg", addr, args.port
    )
    server_instance.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
