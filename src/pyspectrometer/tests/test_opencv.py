"""Unit tests for capture.opencv backend."""

from ..capture.opencv import Capture, _parse_source


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
    """capture.opencv.Capture can be instantiated with valid source."""
    cap = Capture(0, width=800, height=600)
    assert cap.width == 800
    assert cap.height == 600
    assert cap.gain == 10.0
    assert cap.bit_depth == 10
    assert not cap.is_running


def test_capture_gain_setter_noop():
    """Gain setter accepts value (no-op for opencv source)."""
    cap = Capture(0)
    cap.gain = 25.0
    assert cap.gain == 25.0


def test_capture_exposure_setter_noop():
    """Exposure setter accepts value (no-op for opencv source)."""
    cap = Capture(0)
    cap.exposure = 5000
    assert cap.exposure == 5000
