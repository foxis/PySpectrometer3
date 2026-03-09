#!/usr/bin/env python3
"""MJPEG HTTP stream from Picamera2 for remote development.

Run on the Raspberry Pi, then from your desktop use OpenCV:
    cap = cv2.VideoCapture("http://<Pi-IP>:8000/stream.mjpg")

Or run PySpectrometer3 with:
    python -m pyspectrometer --camera http://<Pi-IP>:8000/stream.mjpg
"""

import argparse
import io
import logging
import socketserver
from http import server
from threading import Condition

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingOutput(io.BufferedIOBase):
    """Holds the latest JPEG frame for MJPEG streaming."""

    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.condition = Condition()

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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream Picamera2 as MJPEG over HTTP for remote development."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port (default: 8000)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=800,
        help="Frame width (default: 800)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=600,
        help="Frame height (default: 600)",
    )
    args = parser.parse_args()

    try:
        from picamera2 import Picamera2
        from picamera2.encoders import JpegEncoder
        from picamera2.outputs import FileOutput
    except ImportError as e:
        logger.error(
            "Picamera2 required. Install: sudo apt install python3-picamera2"
        )
        raise SystemExit(1) from e

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_video_configuration(
            main={"size": (args.width, args.height)}
        )
    )
    output = StreamingOutput()
    picam2.start_recording(JpegEncoder(), FileOutput(output))

    server_instance = StreamingServer(
        ("", args.port),
        StreamingHandler,
    )
    server_instance.output = output
    server_instance.stream_port = args.port

    try:
        import socket

        hostname = socket.gethostname()
        addr = socket.gethostbyname(hostname)
        logger.info(
            "MJPEG stream at http://%s:%d/stream.mjpg",
            addr,
            args.port,
        )
        logger.info(
            "On desktop: --camera http://%s:%d/stream.mjpg",
            addr,
            args.port,
        )
        server_instance.serve_forever()
    finally:
        picam2.stop_recording()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
