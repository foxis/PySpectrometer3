"""CLI entry points for Poetry scripts: calibrate, measure, colors, raman, lint, format, stream, waterfall, led."""

import subprocess
import sys
from pathlib import Path

import numpy as np


def _led_pin_from_argv() -> int:
    """Parse optional `--pin N` from argv; default BCM pin from hardware.led."""
    from .hardware.led import DEFAULT_PIN

    args = sys.argv[1:]
    if "--pin" in args:
        i = args.index("--pin")
        if i + 1 < len(args):
            return int(args[i + 1])
    return DEFAULT_PIN


def _led_freq_from_argv(default: int = 100) -> int:
    args = sys.argv[1:]
    if "--freq" in args:
        i = args.index("--freq")
        if i + 1 < len(args):
            return int(args[i + 1])
    return default


def _led_duty_positional() -> float | None:
    """First numeric argv token that is not a flag or flag value."""
    args = sys.argv[1:]
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a in ("--pin", "--freq"):
            skip = True
            continue
        if a.startswith("-"):
            continue
        try:
            return float(a)
        except ValueError:
            continue
    return None


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


def led_on() -> int:
    """Drive LED fully on via gpiozero. Usage: poetry run led-on [--pin N]"""
    from .hardware.led import turn_on

    try:
        turn_on(_led_pin_from_argv())
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1


def led_off() -> int:
    """Drive LED off via gpiozero. Usage: poetry run led-off [--pin N]"""
    from .hardware.led import turn_off

    try:
        turn_off(_led_pin_from_argv())
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1


def led_pwm() -> int:
    """Hold software PWM until Ctrl+C. Usage: poetry run led-pwm <duty> [--pin N] [--freq Hz]

    duty: 0–1 (fraction) or 2–100 (percent); e.g. 0.5 or 50 for half brightness.
    """
    from .hardware.led import hold_pwm

    duty = _led_duty_positional()
    if duty is None:
        print(
            "Usage: poetry run led-pwm <duty> [--pin N] [--freq Hz]",
            file=sys.stderr,
        )
        print("  duty: 0–1 (fraction) or 2–100 (percent, e.g. 50)", file=sys.stderr)
        return 1
    pin = _led_pin_from_argv()
    freq = _led_freq_from_argv()
    try:
        hold_pwm(duty, pin=pin, frequency=freq)
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1


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
      poetry run viewer              # open empty viewer, click Load to pick a CSV
      poetry run viewer spectrum.csv # open directly with the given CSV
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

    from .config import load_csv_viewer_config
    from .csv_viewer.spectrometer import CsvViewerSpectrometer

    if config_arg:
        from .config import load_config

        config, _ = load_config(Path(config_arg))
        config.apply_csv_viewer_preset()
        from .config import csv_viewer_config_path

        save_path = csv_viewer_config_path()
    else:
        config, save_path = load_csv_viewer_config()

    try:
        viewer = CsvViewerSpectrometer(
            Path(csv_path_arg) if csv_path_arg else None,
            config=config,
            config_path=save_path,
        )
        viewer.run()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        import traceback

        print(f"Error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
