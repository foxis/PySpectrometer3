"""Unit tests for capture.opencv backend."""

import numpy as np

from ..capture.base import CAPTURE_UINT16_MAX, mirror_horizontal, scale_to_uint16_full_scale
from ..capture.opencv import Capture, _parse_source
from ..config import CameraConfig


def test_mirror_horizontal_2d():
    """mirror_horizontal reverses columns (left-right)."""
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    m = mirror_horizontal(a)
    np.testing.assert_array_equal(m, np.array([[3, 2, 1], [6, 5, 4]]))


def test_parse_source_int():
    """Integer source returned as int."""
    assert _parse_source(0) == 0
    assert _parse_source(2) == 2


def test_parse_source_digit_string():
    """Numeric string converted to int."""
    assert _parse_source("0") == 0
    assert _parse_source("1") == 1


def test_parse_source_v4l():
    """v4l: prefix stripped to path."""
    assert _parse_source("v4l:/dev/video0") == "/dev/video0"


def test_parse_source_url():
    """URLs passed through as string."""
    url = "http://192.168.1.1:8000/stream.mjpg"
    assert _parse_source(url) == url
    assert _parse_source("rtsp://host/path") == "rtsp://host/path"


def test_capture_instantiation():
    """capture.opencv.Capture can be instantiated with CameraConfig."""
    cfg = CameraConfig(
        frame_width=800,
        frame_height=600,
        opencv_source=0,
    )
    cap = Capture(cfg)
    assert cap.width == 800
    assert cap.height == 600
    assert cap.gain == 10.0
    assert cap.bit_depth == 8
    assert not cap.is_running


def test_capture_gain_setter_noop():
    """Gain setter accepts value (no-op for opencv source)."""
    cap = Capture(CameraConfig(opencv_source=0))
    cap.gain = 25.0
    assert cap.gain == 25.0


def test_scale_to_uint16_full_scale_8bit():
    """8-bit samples map linearly to 0..CAPTURE_UINT16_MAX."""
    a = np.array([[0, 255], [128, 127]], dtype=np.uint8)
    out = scale_to_uint16_full_scale(a, 255)
    assert out.dtype == np.uint16
    assert out[0, 0] == 0
    assert out[0, 1] == CAPTURE_UINT16_MAX
    assert out.max() <= CAPTURE_UINT16_MAX


def test_capture_exposure_setter_noop():
    """Exposure setter accepts value (no-op for opencv source)."""
    cap = Capture(CameraConfig(opencv_source=0))
    cap.exposure = 5000
    assert cap.exposure == 5000
