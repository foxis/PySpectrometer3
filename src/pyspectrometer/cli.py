"""CLI entry points for Poetry scripts: calibrate, measure, colors, raman, lint, format, stream, waterfall, led, list-cameras."""

import subprocess
import sys
from pathlib import Path

import numpy as np


def _apply_led_runtime_from_config() -> None:
    from .config import explicit_config_path_from_argv, load_config
    from .hardware.led import apply_led_config_from_values

    cfg, _ = load_config(explicit_config_path_from_argv())
    apply_led_config_from_values(
        cfg.hardware.led_pin,
        cfg.hardware.led_pwm_frequency_hz,
    )


def _led_pin_from_argv() -> int:
    """Parse optional `--pin N` from argv; else effective pin from config.toml."""
    from .hardware.led import effective_pin

    args = sys.argv[1:]
    if "--pin" in args:
        i = args.index("--pin")
        if i + 1 < len(args):
            return int(args[i + 1])
    return effective_pin()


def _led_freq_from_argv() -> int:
    from .hardware.led import effective_frequency_hz

    args = sys.argv[1:]
    if "--freq" in args:
        i = args.index("--freq")
        if i + 1 < len(args):
            return int(args[i + 1])
    return effective_frequency_hz()


def _led_duty_positional() -> float | None:
    """First numeric argv token that is not a flag or flag value."""
    args = sys.argv[1:]
    skip = False
    skip_flags = frozenset({"--pin", "--freq", "--config", "-c"})
    for a in args:
        if skip:
            skip = False
            continue
        if a in skip_flags:
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
    """First positional token intended as camera source.

    Skips boolean flags and flags that take a value (e.g. ``--config path``) so
    ``calibrate --config garden.toml 1`` resolves camera ``1``, not ``garden.toml``.
    """
    args = sys.argv[1:]
    i = 0
    no_value = frozenset(
        {
            "--fullscreen",
            "--waterfall",
            "--waveshare",
            "--color",
            "--list-cameras",
            "--show-config",
            "--version",
        }
    )
    one_value = frozenset(
        {
            "--gain",
            "--width",
            "--mode",
            "--laser",
            "--bit-depth",
            "--camera",
            "--config",
            "-c",
        }
    )
    while i < len(args):
        a = args[i]
        if a in one_value:
            i += 2
            continue
        if a in no_value:
            i += 1
            continue
        if a == "--csv":
            i += 1
            if i < len(args) and not args[i].startswith("-"):
                i += 1
            continue
        if a.startswith("-"):
            i += 1
            continue
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
    """Fit pixels to wavelengths from CSV using FL12.

    Usage: poetry run fit_csv <csv_path> [--save] [--config PATH]
    """
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    csv_path = args[0] if args else None
    if not csv_path or not Path(csv_path).exists():
        print("Usage: poetry run fit_csv <path/to/spectrum.csv> [--save]")
        print("  --save: write calibration to config")
        return 1
    save_to_config = "--save" in sys.argv[1:]

    from .bootstrap import build_reference_file_loader
    from .config import explicit_config_path_from_argv, load_config
    from .data import get_reference_spectrum, load_spectrum_csv
    from .data.reference_spectra import ReferenceSource
    from .processing.auto_calibrator import calibrate

    measured, n, _ = load_spectrum_csv(csv_path)
    if n < 10:
        print(f"[fit_csv] Need at least 10 pixels, got {n}")
        return 1
    config, _ = load_config(explicit_config_path_from_argv())
    ref_loader = build_reference_file_loader(config)
    ref_wl = np.linspace(380.0, 750.0, 500)
    ref_int = get_reference_spectrum(ReferenceSource.FL12, ref_wl, file_loader=ref_loader)
    points = calibrate(measured, ref_wl, ref_int)
    if not points:
        return 1
    print(f"FL12 calibration: {len(points)} points")
    for pixel, wl in points:
        print(f"  Pixel {pixel} -> {wl:.1f} nm")

    if save_to_config:
        from .config import explicit_config_path_from_argv, load_config, save_config

        config, config_path = load_config(explicit_config_path_from_argv())
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
    """Drive LED fully on via gpiozero.

    Usage: poetry run led-on [--pin N] [--config PATH]
    Pin/frequency default from [hardware] in config when omitted.
    """
    from .hardware.led import turn_on

    try:
        _apply_led_runtime_from_config()
        turn_on(_led_pin_from_argv())
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1


def led_off() -> int:
    """Drive LED off via gpiozero.

    Usage: poetry run led-off [--pin N] [--config PATH]
    """
    from .hardware.led import turn_off

    try:
        _apply_led_runtime_from_config()
        turn_off(_led_pin_from_argv())
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1


def led() -> int:
    """Alias dispatcher so ``poetry run led off`` works (not only ``led-off``).

    Usage: poetry run led <on|off|pwm> [...] — same extra args as led-on / led-off / led-pwm.
    """
    if len(sys.argv) < 2:
        print(
            "Usage: poetry run led <on|off|pwm> [...]\n"
            "  Same options as: poetry run led-on | led-off | led-pwm",
            file=sys.stderr,
        )
        return 1
    sub = sys.argv[1].lower().replace("_", "-")
    base = sys.argv[0]
    sys.argv = [base, *sys.argv[2:]]
    match sub:
        case "on":
            return led_on()
        case "off":
            return led_off()
        case "pwm":
            return led_pwm()
        case _:
            print(
                f"Unknown subcommand {sub!r} (expected on, off, pwm).",
                file=sys.stderr,
            )
            return 1


def led_pwm() -> int:
    """Hold software PWM until Ctrl+C.

    Usage: poetry run led-pwm <duty> [--pin N] [--freq Hz] [--config PATH]
    duty: 0–1 (fraction) or 2–100 (percent); e.g. 0.5 or 50 for half brightness.
    """
    from .hardware.led import hold_pwm

    duty = _led_duty_positional()
    if duty is None:
        print(
            "Usage: poetry run led-pwm <duty> [--pin N] [--freq Hz] [--config PATH]",
            file=sys.stderr,
        )
        print("  duty: 0–1 (fraction) or 2–100 (percent, e.g. 50)", file=sys.stderr)
        return 1
    try:
        _apply_led_runtime_from_config()
        pin = _led_pin_from_argv()
        freq = _led_freq_from_argv()
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


def list_cameras() -> int:
    """Enumerate OpenCV camera indices (same as ``python -m pyspectrometer --list-cameras``).

    Usage: poetry run list-cameras
    """
    from .capture.opencv import list_cameras as enumerate_cameras

    cameras = enumerate_cameras()
    if not cameras:
        print("No cameras found.")
    else:
        print("Available cameras:")
        for idx, desc in cameras:
            print(f"  {idx}: {desc}")
    return 0


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
