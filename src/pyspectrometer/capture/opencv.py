"""OpenCV capture backend for webcam, V4L, RTSP, and HTTP MJPEG streams."""

import cv2
import numpy as np

from ..config import CameraConfig
from .base import CameraInterface, mirror_horizontal


def _parse_source(source: int | str) -> int | str:
    """Parse to OpenCV-compatible value (int index or str path/URL)."""
    if isinstance(source, int):
        return source
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    if s.lower().startswith("v4l:"):
        return s[4:].strip()
    return s


def list_cameras(max_index: int = 10) -> list[tuple[int, str]]:
    """Enumerate available camera devices.

    Tries indices 0..max_index-1. On Linux, also reports /dev/videoN path
    when available.

    Returns:
        List of (index, description) tuples
    """
    result: list[tuple[int, str]] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to get backend name (CAP_PROP_BACKEND) - may not be available
            try:
                backend = int(cap.get(cv2.CAP_PROP_BACKEND))
                backend_name = _backend_name(backend)
            except (TypeError, ValueError):
                backend_name = "unknown"
            desc = f"Index {i}"
            try:
                import platform

                if platform.system() == "Linux":
                    desc = f"/dev/video{i}"
            except Exception:
                pass
            result.append((i, f"{desc} ({backend_name})"))
            cap.release()
    return result


def _backend_name(backend: int) -> str:
    """Map OpenCV backend constant to name."""
    names = {
        cv2.CAP_ANY: "ANY",
        cv2.CAP_V4L2: "V4L2",
        cv2.CAP_FFMPEG: "FFMPEG",
        cv2.CAP_MSMF: "MSMF",
        cv2.CAP_DSHOW: "DSHOW",
        cv2.CAP_GSTREAMER: "GSTREAMER",
    }
    return names.get(backend, f"backend_{backend}")


class Capture(CameraInterface):
    """Camera capture using OpenCV VideoCapture.

    Supports webcam (device index), V4L path (v4l:/dev/video0), RTSP,
    and HTTP MJPEG streams. Outputs 10-bit grayscale for pipeline compatibility.

    Gain and exposure are no-ops; many sources do not support them.
    """

    def __init__(self, camera: CameraConfig) -> None:
        """Initialize OpenCV capture from :class:`CameraConfig` (uses ``opencv_source``)."""
        self._config = camera
        self._source = _parse_source(camera.opencv_source)
        self._width = camera.frame_width
        self._height = camera.frame_height
        self._gain = camera.gain
        self._exposure = 10000
        self._fps = camera.fps
        self._flip_horizontal = camera.flip_horizontal
        self._running = False
        self._cap: cv2.VideoCapture | None = None

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        lo, hi = self._config.gain_min, self._config.gain_max
        self._gain = max(lo, min(hi, value))

    @property
    def exposure(self) -> int:
        """Exposure in microseconds. No-op for OpenCV sources."""
        return self._exposure

    @exposure.setter
    def exposure(self, value: int) -> None:
        """Set exposure time in microseconds. No upper cap here; driver limits apply. Min 100 µs."""
        self._exposure = max(100, value)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def bit_depth(self) -> int:
        """Always 10-bit (scaled from 8-bit source)."""
        return 10

    @property
    def flip_horizontal(self) -> bool:
        return self._flip_horizontal

    @flip_horizontal.setter
    def flip_horizontal(self, value: bool) -> None:
        self._flip_horizontal = bool(value)

    def start(self) -> None:
        """Start OpenCV capture and log capabilities."""
        if self._running:
            return

        self._cap = cv2.VideoCapture(self._source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self._source}")

        # Set requested resolution and fps (best-effort)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        # Use actual dimensions from first frame (streams may ignore requested size)
        ret, frame = self._cap.read()
        if ret and frame is not None:
            self._height, self._width = frame.shape[:2]
            print(f"OpenCV camera: source={self._source}")
            print(f"  Dimensions: {self._width}x{self._height} (from stream)")
        else:
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self._width
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self._height
            self._width = actual_w
            self._height = actual_h
            print(f"OpenCV camera: source={self._source}")
            print(f"  Dimensions: {self._width}x{self._height}")
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"  FPS: {actual_fps:.1f}")
        print("  Gain/exposure: no-op (source does not support)")

        self._running = True

    def stop(self) -> None:
        """Stop capture and release resources."""
        if not self._running:
            return
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._running = False

    def capture(self) -> np.ndarray:
        """Capture one valid frame as 10-bit grayscale.

        Skips corrupt/empty frames (MJPG boundary errors, zero-size decodes)
        until a real frame arrives. Never returns an invalid frame.

        Returns:
            2D uint16 array (height, width), values 0-1023.
        """
        if not self._running or self._cap is None:
            raise RuntimeError("Camera is not running. Call start() first.")

        while True:
            ret, frame = self._cap.read()
            if ret and frame is not None and frame.size > 0:
                break

        # Convert to grayscale if color
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Scale 8-bit (0-255) to 10-bit (0-1023) for pipeline contract
        out = (gray.astype(np.float32) * 1023.0 / 255.0).astype(np.uint16)
        if self._flip_horizontal:
            out = mirror_horizontal(out)
        return out
