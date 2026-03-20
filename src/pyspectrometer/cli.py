"""CLI entry points for Poetry scripts: calibrate, measure, colors, raman, lint, format, stream, waterfall."""

import subprocess
import sys
from pathlib import Path

import numpy as np


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


def fit_csv() -> int:
    """Fit pixels to wavelengths from CSV using FL12. Usage: poetry run fit_csv <csv_path> [--save]"""
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    csv_path = args[0] if args else None
    if not csv_path or not Path(csv_path).exists():
        print("Usage: poetry run fit_csv <path/to/spectrum.csv> [--save]")
        print("  --save: write calibration to config")
        return 1
    save_to_config = "--save" in sys.argv[1:]

    from .data import get_reference_spectrum, load_spectrum_csv
    from .data.reference_spectra import ReferenceSource
    from .processing.auto_calibrator import calibrate

    measured, n, _ = load_spectrum_csv(csv_path)
    if n < 10:
        print(f"[fit_csv] Need at least 10 pixels, got {n}")
        return 1
    ref_wl = np.linspace(380.0, 750.0, 500)
    ref_int = get_reference_spectrum(ReferenceSource.FL12, ref_wl)
    points = calibrate(measured, ref_wl, ref_int)
    if not points:
        return 1
    print(f"FL12 calibration: {len(points)} points")
    for pixel, wl in points:
        print(f"  Pixel {pixel} -> {wl:.1f} nm")

    if save_to_config:
        from .config import load_config, save_config

        config, config_path = load_config()
        pixels = [p for p, _ in points]
        wavelengths = [w for _, w in points]
        config.calibration.cal_pixels = pixels
        config.calibration.cal_wavelengths = wavelengths
        if save_config(config, config_path):
            print(f"Saved to {config_path or 'pyspectrometer.toml'}")
        else:
            print("Failed to save config")
            return 1
    return 0


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


def waterfall() -> int:
    """Run waterfall mode (standalone). Usage: poetry run waterfall [camera_source]"""
    return _run("waterfall", _camera_arg())


def view_csv() -> int:
    """Open a spectrum CSV in the viewer.

    Usage:
      poetry run viewer              # opens file-browser dialog (defaults to output dir)
      poetry run viewer spectrum.csv # open directly
    """
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    csv_path_arg = args[0] if args else None

    if csv_path_arg and not Path(csv_path_arg).exists():
        print(f"Error: file not found: {csv_path_arg}", file=sys.stderr)
        return 1

    config_arg = None
    for i, a in enumerate(sys.argv[1:], 1):
        if a in ("--config", "-c") and i < len(sys.argv) - 1:
            config_arg = sys.argv[i + 1]
            break

    from .config import load_config
    from .csv_viewer.spectrometer import CsvViewerSpectrometer

    config, _ = load_config(Path(config_arg) if config_arg else None)
    try:
        viewer = CsvViewerSpectrometer(
            Path(csv_path_arg) if csv_path_arg else None,
            config=config,
        )
        viewer.run()
        return 0
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        import traceback
        print(f"Error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
