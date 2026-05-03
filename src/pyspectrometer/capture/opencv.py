"""OpenCV capture backend for webcam, V4L, RTSP, and HTTP MJPEG streams."""

import cv2
import numpy as np
import time

from ..config import CameraConfig
from .base import CameraInterface, mirror_horizontal, scale_to_uint16_full_scale


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
    names = {
        cv2.CAP_ANY: "ANY",
        cv2.CAP_V4L2: "V4L2",
        cv2.CAP_FFMPEG: "FFMPEG",
        cv2.CAP_MSMF: "MSMF",
        cv2.CAP_DSHOW: "DSHOW",
        cv2.CAP_GSTREAMER: "GSTREAMER",
    }
    return names.get(backend, f"backend_{backend}")


def list_cameras(max_index: int = 10) -> list[tuple[int, str]]:
    """Enumerate available camera devices."""
    result: list[tuple[int, str]] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            try:
                backend = int(cap.get(cv2.CAP_PROP_BACKEND))
                backend_name = _backend_name(backend)
            except (TypeError, ValueError):
                backend_name = "unknown"
            result.append((i, f"Index {i} ({backend_name})"))
            cap.release()
    return result


class Capture(CameraInterface):
    """Camera capture using OpenCV VideoCapture.

    Uses MSMF on Windows for stable uncompressed capture.
    OpenCV handles YUV->BGR conversion internally; we convert BGR->Gray.
    """

    def __init__(self, camera: CameraConfig) -> None:
        """Initialize OpenCV capture from :class:`CameraConfig`."""
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
        self._exposure_supported = False
        self._gain_supported = False
        self._frame_count = 0

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
        if self._cap is not None and self._exposure_supported:
            import math
            exp_sec = self._exposure / 1_000_000.0
            log2_val = max(-13, min(-1, round(math.log2(exp_sec))))
            # Ensure manual mode before setting
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self._cap.set(cv2.CAP_PROP_EXPOSURE, log2_val)

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

    def start(self) -> None:
        """Start OpenCV capture."""
        if self._running:
            return

        self._open_camera()
        self._configure_format()
        self._wait_for_init()
        self._detect_dimensions()
        self._log_camera_info()
        self._probe_controls()
        self._running = True

    def _open_camera(self) -> None:
        """Open camera with best available backend."""
        import platform
        is_local = isinstance(self._source, int)

        if platform.system() == "Windows" and is_local:
            # DSHOW first: matches AMCap, better FPS and exposure control
            self._cap = cv2.VideoCapture(self._source, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                print("  DSHOW failed, trying MSMF...")
                self._cap = cv2.VideoCapture(self._source, cv2.CAP_MSMF)
        else:
            self._cap = cv2.VideoCapture(self._source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self._source}")

    def _configure_format(self) -> None:
        """Set resolution and request uncompressed format."""
        assert self._cap is not None
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, max(self._fps, 30))

        # Request grayscale/uncompressed format (NO MJPEG!)
        # Prefer native grayscale, then 16-bit, then YUV as last resort
        for fourcc in ['GREY', 'Y800', 'Y16 ', 'YUY2', 'YUYV', 'NV12']:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            actual = int(self._cap.get(cv2.CAP_PROP_FOURCC))
            if actual == cv2.VideoWriter_fourcc(*fourcc):
                print(f"  Format: {fourcc.strip()}")
                break
        else:
            print(f"  Format: camera default ({_fourcc_str(int(self._cap.get(cv2.CAP_PROP_FOURCC)))})")

        # Disable auto RGB conversion - we want raw sensor data
        self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    def _wait_for_init(self) -> None:
        """Give camera time to initialize and drain stale buffers."""
        assert self._cap is not None

        # On Windows local cameras, open native settings dialog for exposure/gain
        import platform
        if platform.system() == "Windows" and isinstance(self._source, int):
            self._cap.set(cv2.CAP_PROP_SETTINGS, 1)

        time.sleep(0.3)
        for _ in range(5):
            self._cap.read()

    def _detect_dimensions(self) -> None:
        """Get actual frame dimensions from camera properties."""
        assert self._cap is not None
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w > 0 and h > 0:
            self._width = w
            self._height = h

    def _log_camera_info(self) -> None:
        assert self._cap is not None
        try:
            backend = int(self._cap.get(cv2.CAP_PROP_BACKEND))
        except (TypeError, ValueError):
            backend = cv2.CAP_ANY

        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        exp = self._cap.get(cv2.CAP_PROP_EXPOSURE)
        gain = self._cap.get(cv2.CAP_PROP_GAIN)
        auto_exp = self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)

        print(f"OpenCV camera: source={self._source}")
        print(f"  Backend: {_backend_name(backend)}")
        print(f"  Size: {self._width}x{self._height} @ {fps:.1f} FPS")
        print(f"  Format: {_fourcc_str(fourcc)}")
        print(f"  Exposure: {exp}, Auto: {auto_exp}, Gain: {gain}")

    def _probe_controls(self) -> None:
        """Disable auto-exposure and test if exposure/gain can be changed."""
        assert self._cap is not None

        # Step 1: Force manual exposure mode
        # DirectShow: 1=manual, 3=auto. MSMF: 0.25=manual, 0.75=auto
        auto_exp = self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
        print(f"  Auto-exposure initial: {auto_exp}")
        for val in [1, 0.25, 0]:
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
            readback = self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            if readback != auto_exp:
                print(f"  Auto-exposure -> manual (set {val}, got {readback})")
                break
        else:
            print(f"  Auto-exposure: could not switch to manual")

        # Step 2: Test exposure by checking if frame brightness actually changes
        exp = self._cap.get(cv2.CAP_PROP_EXPOSURE)
        
        # Try setting two different exposures and compare frame brightness
        self._cap.set(cv2.CAP_PROP_EXPOSURE, -10)  # very short ~1ms
        time.sleep(0.1)
        ret1, f1 = self._cap.read()
        bright_short = f1.mean() if ret1 and f1 is not None else -1

        self._cap.set(cv2.CAP_PROP_EXPOSURE, -4)   # longer ~62ms
        time.sleep(0.1)
        ret2, f2 = self._cap.read()
        bright_long = f2.mean() if ret2 and f2 is not None else -1

        # If brightness changed significantly, exposure control works
        if bright_short >= 0 and bright_long >= 0:
            ratio = bright_long / max(bright_short, 0.01)
            self._exposure_supported = ratio > 1.5 or ratio < 0.67
            print(f"  Exposure test: short={bright_short:.1f}, long={bright_long:.1f}, "
                  f"ratio={ratio:.2f} -> {'WORKS' if self._exposure_supported else 'NO EFFECT'}")
        else:
            self._exposure_supported = False
            print(f"  Exposure test: could not read frames")

        # Restore a reasonable exposure
        self._cap.set(cv2.CAP_PROP_EXPOSURE, -5)

        # Test gain
        gain = self._cap.get(cv2.CAP_PROP_GAIN)
        if gain >= 0:
            test_val = gain + 5
            self._cap.set(cv2.CAP_PROP_GAIN, test_val)
            readback = self._cap.get(cv2.CAP_PROP_GAIN)
            self._gain_supported = abs(readback - test_val) < 2
            print(f"  Gain control: {'OK' if self._gain_supported else 'NO'}")

    def stop(self) -> None:
        """Stop capture."""
        if not self._running:
            return
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._running = False

    def _frame_to_gray(self, frame: np.ndarray) -> np.ndarray:
        """Extract grayscale from raw frame based on format.

        With CONVERT_RGB=0, OpenCV gives us raw sensor data:
        - YUY2: flat (1, W*H*2) or (H, W*2) uint8 - Y at even bytes
        - Y16:  (H, W) uint16
        - GREY: (H, W) uint8
        - BGR:  (H, W, 3) uint8 - when CONVERT_RGB=0 fails
        """
        # Already proper 2D grayscale
        if frame.ndim == 2 and frame.shape == (self._height, self._width):
            return frame

        # uint16 2D - Y16 format
        if frame.ndim == 2 and frame.dtype == np.uint16:
            return frame[:self._height, :self._width]

        # Flat or near-flat buffer from raw YUY2
        # YUY2 = 2 bytes/pixel: Y0 U0 Y1 V0 ...
        expected_yuy2_bytes = self._width * self._height * 2
        if frame.size == expected_yuy2_bytes and frame.dtype == np.uint8:
            data = frame.reshape(self._height, self._width * 2)
            return data[:, 0::2].copy()

        # 3-channel BGR (CONVERT_RGB=0 was ignored by backend)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2-channel (H, W, 2) - Y is channel 0
        if frame.ndim == 3 and frame.shape[2] == 2:
            return frame[:, :, 0]

        # Unknown - best effort
        return frame.reshape(self._height, self._width) if frame.size == self._width * self._height else frame.flatten()[:self._width * self._height].reshape(self._height, self._width)

    def capture(self) -> np.ndarray:
        """Capture one frame as uint16 grayscale."""
        if not self._running or self._cap is None:
            raise RuntimeError("Camera is not running. Call start() first.")

        ret, frame = self._cap.read()
        if not ret or frame is None or frame.size == 0:
            raise RuntimeError("Failed to capture frame")

        self._frame_count += 1

        if self._frame_count == 1:
            print(f"[RAW] shape={frame.shape}, dtype={frame.dtype}, size={frame.size}")

        gray = self._frame_to_gray(frame)

        if self._frame_count <= 5:
            print(f"[Frame {self._frame_count}] {gray.shape} min={gray.min()} max={gray.max()} mean={gray.mean():.1f}")

        in_max = 65535 if gray.dtype == np.uint16 else 255
        out = scale_to_uint16_full_scale(gray, in_max)
        if self._flip_horizontal:
            out = mirror_horizontal(out)
        return out
