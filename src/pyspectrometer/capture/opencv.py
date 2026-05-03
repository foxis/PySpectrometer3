"""OpenCV capture backend for webcam, V4L, RTSP, and HTTP MJPEG streams."""

import math
import platform
import time

import cv2
import numpy as np

from ..config import CameraConfig
from .base import CameraInterface, mirror_horizontal, scale_to_uint16_full_scale

_IS_WINDOWS = platform.system() == "Windows"

# Exposure log2(seconds) range accepted by DSHOW/MSMF on Windows.
_EXP_LOG2_MIN = -13
_EXP_LOG2_MAX = -1
_EXP_LOG2_DEFAULT = -5  # ~31 ms — reasonable starting point

# MSMF drops frames after exposure changes; retry before giving up.
_READ_RETRIES = 5
_READ_RETRY_DELAY = 0.05  # seconds

_BACKEND_NAMES: dict[int, str] = {
    cv2.CAP_ANY: "ANY",
    cv2.CAP_V4L2: "V4L2",
    cv2.CAP_FFMPEG: "FFMPEG",
    cv2.CAP_MSMF: "MSMF",
    cv2.CAP_DSHOW: "DSHOW",
    cv2.CAP_GSTREAMER: "GSTREAMER",
}


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


def _fourcc_str(code: int) -> str:
    """Convert FOURCC int to string."""
    if code == 0:
        return "?"
    return "".join([chr((code >> (8 * i)) & 0xFF) for i in range(4)])


def _backend_name(backend: int) -> str:
    """Map OpenCV backend constant to name."""
    return _BACKEND_NAMES.get(backend, f"backend_{backend}")


def _us_to_log2(us: int) -> int:
    """Convert microseconds to log2(seconds) clamped to OpenCV range."""
    return max(_EXP_LOG2_MIN, min(_EXP_LOG2_MAX, round(math.log2(us / 1_000_000.0))))


def list_cameras(max_index: int = 10) -> list[tuple[int, str]]:
    """Enumerate available camera devices."""
    result: list[tuple[int, str]] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            try:
                backend = int(cap.get(cv2.CAP_PROP_BACKEND))
                name = _backend_name(backend)
            except (TypeError, ValueError):
                name = "unknown"
            result.append((i, f"Index {i} ({name})"))
            cap.release()
    return result


class Capture(CameraInterface):
    """Camera capture using OpenCV VideoCapture.

    On Windows local cameras: MSMF backend (raw YUY2 buffer, monotonic exposure
    control, no GUI dialogs).  Falls back to DSHOW if MSMF fails to open.
    On Linux: default backend (V4L2 preferred, supports GREY/Y16 natively).
    For URL/RTSP/MJPEG streams: default backend, no exposure control.
    """

    def __init__(self, camera: CameraConfig) -> None:
        self._config = camera
        self._source = _parse_source(camera.opencv_source)
        self._width = camera.frame_width
        self._height = camera.frame_height
        self._gain = camera.gain
        self._exposure = 10000  # microseconds
        self._fps = camera.fps
        self._flip_horizontal = camera.flip_horizontal
        self._running = False
        self._cap: cv2.VideoCapture | None = None
        self._backend = cv2.CAP_ANY
        self._is_stream = not isinstance(self._source, int)
        self._manual_exposure_active = False
        self._gain_supported = False
        self._frame_count = 0

    # ── Properties ──────────────────────────────────────────────────

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
        if self._cap is not None and self._gain_supported:
            self._cap.set(cv2.CAP_PROP_GAIN, self._gain)

    @property
    def exposure(self) -> int:
        return self._exposure

    @exposure.setter
    def exposure(self, value: int) -> None:
        self._exposure = max(100, value)
        if self._cap is None or self._is_stream:
            return
        self._activate_manual_exposure()
        self._cap.set(cv2.CAP_PROP_EXPOSURE, _us_to_log2(self._exposure))

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def bit_depth(self) -> int:
        return 8

    @property
    def flip_horizontal(self) -> bool:
        return self._flip_horizontal

    @flip_horizontal.setter
    def flip_horizontal(self, value: bool) -> None:
        self._flip_horizontal = bool(value)

    # ── Lifecycle ───────────────────────────────────────────────────

    def start(self) -> None:
        """Start OpenCV capture."""
        if self._running:
            return

        self._open_camera()
        self._configure_format()
        self._set_initial_exposure()
        self._drain_buffers()
        self._detect_dimensions()
        self._log_camera_info()
        self._probe_gain()
        self._running = True

    def stop(self) -> None:
        """Stop capture."""
        if not self._running:
            return
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._running = False

    # ── Startup helpers ─────────────────────────────────────────────

    def _open_camera(self) -> None:
        """Open camera with the best backend for the platform."""
        if _IS_WINDOWS and not self._is_stream:
            self._open_windows_local()
        else:
            self._cap = cv2.VideoCapture(self._source)
            if self._cap.isOpened():
                try:
                    self._backend = int(self._cap.get(cv2.CAP_PROP_BACKEND))
                except (TypeError, ValueError):
                    pass

        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self._source}")

    def _open_windows_local(self) -> None:
        """MSMF first (raw buffer, monotonic exposure), DSHOW fallback."""
        self._cap = cv2.VideoCapture(self._source, cv2.CAP_MSMF)
        if self._cap.isOpened():
            self._backend = cv2.CAP_MSMF
            return

        print("  MSMF failed, trying DSHOW...")
        self._cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
        if self._cap.isOpened():
            self._backend = cv2.CAP_DSHOW

    def _configure_format(self) -> None:
        """Set resolution and negotiate best uncompressed pixel format."""
        assert self._cap is not None
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, max(self._fps, 30))

        for fourcc in ['GREY', 'Y800', 'Y16 ', 'YUY2', 'YUYV', 'NV12']:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            actual = int(self._cap.get(cv2.CAP_PROP_FOURCC))
            if actual == cv2.VideoWriter_fourcc(*fourcc):
                print(f"  Format: {fourcc.strip()}")
                break
        else:
            print(f"  Format: camera default "
                  f"({_fourcc_str(int(self._cap.get(cv2.CAP_PROP_FOURCC)))})")

        self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    def _set_initial_exposure(self) -> None:
        """Set a reasonable exposure before draining so first frames are not black."""
        if self._is_stream:
            return
        self._cap.set(cv2.CAP_PROP_EXPOSURE, _EXP_LOG2_DEFAULT)

    def _drain_buffers(self) -> None:
        """Give camera time to initialise and flush stale frames."""
        assert self._cap is not None
        time.sleep(0.3)
        for _ in range(5):
            self._cap.read()

    def _detect_dimensions(self) -> None:
        """Update stored dimensions from actual camera properties."""
        assert self._cap is not None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            self._width = w
            self._height = h

    def _log_camera_info(self) -> None:
        assert self._cap is not None
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        exp = self._cap.get(cv2.CAP_PROP_EXPOSURE)
        gain = self._cap.get(cv2.CAP_PROP_GAIN)
        auto_exp = self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)

        print(f"OpenCV camera: source={self._source}")
        print(f"  Backend: {_backend_name(self._backend)}")
        print(f"  Size: {self._width}x{self._height} @ {fps:.1f} FPS")
        print(f"  Format: {_fourcc_str(fourcc)}")
        print(f"  Exposure: {exp}, Auto: {auto_exp}, Gain: {gain}")

    def _probe_gain(self) -> None:
        """Test whether gain control is responsive."""
        if self._is_stream:
            return
        assert self._cap is not None
        gain = self._cap.get(cv2.CAP_PROP_GAIN)
        if gain < 0:
            return
        test_val = gain + 5
        self._cap.set(cv2.CAP_PROP_GAIN, test_val)
        readback = self._cap.get(cv2.CAP_PROP_GAIN)
        self._gain_supported = abs(readback - test_val) < 2
        if self._gain_supported:
            self._cap.set(cv2.CAP_PROP_GAIN, gain)
        print(f"  Gain control: {'OK' if self._gain_supported else 'NO'}")

    # ── Exposure helpers ────────────────────────────────────────────

    def _activate_manual_exposure(self) -> None:
        """Switch hardware auto-exposure to manual mode (once)."""
        if self._manual_exposure_active or self._cap is None:
            return
        # DSHOW uses 1=manual / 3=auto.  MSMF uses 0.25=manual / 0.75=auto.
        # Try all known manual-mode values; the first one that changes readback wins.
        initial = self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        for val in [1, 0.25, 0]:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
            if self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) != initial:
                break
        self._manual_exposure_active = True

    # ── Read helpers ──────────────────────────────────────────────────

    def _retry_read(self) -> np.ndarray | None:
        """Retry after a failed read (MSMF drops frames on exposure change)."""
        for attempt in range(_READ_RETRIES):
            time.sleep(_READ_RETRY_DELAY)
            ret, frame = self._cap.read()
            if ret and frame is not None and frame.size > 0:
                return frame
        return None

    # ── Frame conversion ────────────────────────────────────────────

    def _frame_to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Extract grayscale from raw frame based on format.

        With CONVERT_RGB=0, OpenCV gives us raw sensor data:
        - YUY2: flat (1, W*H*2) or (H, W*2) uint8 — Y at even bytes
        - Y16:  (H, W) uint16
        - GREY: (H, W) uint8
        - BGR:  (H, W, 3) uint8 — when CONVERT_RGB=0 is ignored by backend
        """
        if frame.ndim == 2 and frame.shape == (self._height, self._width):
            return frame

        if frame.ndim == 2 and frame.dtype == np.uint16:
            return frame[:self._height, :self._width]

        # Flat or near-flat YUY2 buffer (e.g. MSMF raw)
        expected_yuy2_bytes = self._width * self._height * 2
        if frame.size == expected_yuy2_bytes and frame.dtype == np.uint8:
            data = frame.reshape(self._height, self._width * 2)
            return data[:, 0::2].copy()

        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame.ndim == 3 and frame.shape[2] == 2:
            return frame[:, :, 0]

        if frame.size == self._width * self._height:
            return frame.reshape(self._height, self._width)
        return frame.flatten()[:self._width * self._height].reshape(
            self._height, self._width
        )

    # ── Capture ─────────────────────────────────────────────────────

    def capture(self) -> np.ndarray:
        """Capture one frame as uint16 grayscale."""
        if not self._running or self._cap is None:
            raise RuntimeError("Camera is not running. Call start() first.")

        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            frame = self._retry_read()

        if frame is None:
            raise RuntimeError("Failed to capture frame")

        self._frame_count += 1

        if self._frame_count == 1:
            print(f"[RAW] shape={frame.shape}, dtype={frame.dtype}, size={frame.size}")

        gray = self._frame_to_gray(frame)

        if self._frame_count <= 5:
            print(f"[Frame {self._frame_count}] {gray.shape} "
                  f"min={gray.min()} max={gray.max()} mean={gray.mean():.1f}")

        in_max = 65535 if gray.dtype == np.uint16 else 255
        out = scale_to_uint16_full_scale(gray, in_max)
        if self._flip_horizontal:
            out = mirror_horizontal(out)
        return out
