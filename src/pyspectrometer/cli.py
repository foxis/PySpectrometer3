"""CLI entry points for Poetry scripts: calibrate, measure, colors, raman, lint, format, stream."""

import subprocess
import sys
from pathlib import Path


def _camera_arg() -> str | None:
    """First non-flag arg is camera source."""
    for a in sys.argv[1:]:
        if not a.startswith("-"):
            return a
    return None


def _run(mode: str, camera: str | None = None) -> int:
    """Run spectrometer in given mode with optional camera source."""
    argv = ["pyspectrometer", "--mode", mode]
    if camera is not None:
        argv.extend(["--camera", camera])
    passthrough = [a for a in sys.argv[1:] if a != camera]
    sys.argv = argv + passthrough
    from .__main__ import main

    return main()


def calibrate() -> int:
    """Run calibration mode. Usage: poetry run calibrate [camera_source]"""
    return _run("calibration", _camera_arg())


def measure() -> int:
    """Run measurement mode. Usage: poetry run measure [camera_source]"""
    return _run("measurement", _camera_arg())


def colors() -> int:
    """Run color science mode. Usage: poetry run colors [camera_source]"""
    return _run("colorscience", _camera_arg())


def raman() -> int:
    """Run Raman mode. Usage: poetry run raman [camera_source]"""
    return _run("raman", _camera_arg())


def lint() -> int:
    """Run ruff check on src/. Use `poetry run ruff` for custom commands."""
    return subprocess.call(["ruff", "check", "src/"])


def format_src() -> int:
    """Run ruff format on src/. Use `poetry run ruff` for custom commands."""
    return subprocess.call(["ruff", "format", "src/"])


def stream() -> int:
    """Stream MJPEG from camera. Usage: poetry run stream [camera] [port] [--config PATH]"""
    camera = _camera_arg()
    port = "8000"
    rest = [a for a in sys.argv[1:] if a != camera]
    for a in rest:
        if not a.startswith("-") and a.isdigit():
            port = a
            rest = [x for x in rest if x != a]
            break
    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "stream_camera.py"
    cmd = [sys.executable, str(script), "--port", port] + rest
    if camera is not None:
        cmd.extend(["--camera", camera])
    return subprocess.call(cmd)
