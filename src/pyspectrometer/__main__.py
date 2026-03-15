#!/usr/bin/env python3
"""
PySpectrometer 3 - A modular spectrometer application.

This is a refactored and extended version of PySpectrometer2 by Les Wright.
Original: https://github.com/leswright1977/PySpectrometer

Features:
- Higher resolution (800px wide graph)
- 3 row pixel averaging of sensor data
- Fullscreen option for the Spectrometer graph
- 3rd order polynomial fit of calibration data
- Improved graph labelling
- Labelled measurement cursors
- Optional waterfall display
- Key bindings for all operations
- Peak hold, peak detect, Savitzky-Golay filter
- Save graphs as PNG and data as CSV
"""

import argparse
import sys


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="pyspectrometer",
        description="PySpectrometer 3 - A modular spectrometer application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Keyboard Shortcuts:
  q     Quit application
  h     Toggle peak hold
  s     Save spectrum (PNG + CSV)
  c     Perform calibration
  m     Toggle measure mode (wavelength)
  p     Toggle pixel mode (for calibration)
  o/l   Increase/decrease Savitzky-Golay order
  i/k   Increase/decrease peak width
  u/j   Increase/decrease threshold
  t/g   Increase/decrease camera gain

Operating Modes:
  calibration    - Wavelength calibration with reference spectra
  measurement    - General spectrum measurement (default)
  raman          - Raman spectroscopy with wavenumber display
  colorscience   - Transmittance/reflectance/CRI analysis

Examples:
  pyspectrometer                            # Measurement mode (default)
  pyspectrometer --mode calibration         # Calibration mode
  pyspectrometer --mode raman --laser 785   # Raman with 785nm laser
  pyspectrometer --mode colorscience        # Color science mode
  pyspectrometer --waveshare --mode measurement  # Waveshare display
  pyspectrometer --fullscreen               # Fullscreen mode
  pyspectrometer --gain 15                  # Custom camera gain
  pyspectrometer --list-cameras             # List available cameras
  pyspectrometer --camera 0                 # Use webcam
  pyspectrometer --camera http://pi:8000/stream.mjpg  # Remote Pi stream
""",
    )

    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Run in fullscreen mode",
    )

    parser.add_argument(
        "--waterfall",
        action="store_true",
        help="Enable waterfall display",
    )

    parser.add_argument(
        "--waveshare",
        action="store_true",
        help='Optimize for Waveshare 3.5" touchscreen (640x480)',
    )

    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        metavar="VALUE",
        help="Initial camera gain (0-50, default: 10)",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        metavar="PIXELS",
        help="Frame width in pixels (640 or 1280, default: 1280). Height derived from width.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="measurement",
        choices=["calibration", "measurement", "raman", "colorscience", "waterfall"],
        metavar="MODE",
        help="Operating mode: calibration, measurement, raman, colorscience, waterfall (default: measurement)",
    )

    parser.add_argument(
        "--laser",
        type=float,
        default=785.0,
        metavar="NM",
        help="Raman laser wavelength in nm (default: 785)",
    )

    parser.add_argument(
        "--color",
        action="store_true",
        help="Use color camera mode (RGB888, 8-bit) instead of monochrome",
    )

    parser.add_argument(
        "--bit-depth",
        type=int,
        default=10,
        choices=[8, 10, 16],
        metavar="BITS",
        help="Bit depth for monochrome mode (8, 10, or 16, default: 10)",
    )

    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        metavar="SOURCE",
        help="OpenCV camera source: 0 (webcam), v4l:/dev/video0, rtsp://..., http://... (Pi stream)",
    )

    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available OpenCV camera devices and exit",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to config file (TOML). Overrides PYSPECTROMETER_CONFIG.",
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print config file path and exit",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 3.0.0",
    )

    return parser.parse_args()


def _parse_source(value: str) -> int | str:
    """Parse --camera arg to int or str for capture.opencv.Capture."""
    v = value.strip()
    if v.isdigit():
        return int(v)
    return v


def main() -> int:
    """Main entry point for PySpectrometer 3."""
    args = parse_args()

    from pathlib import Path

    from .config import load_config

    config_path = Path(args.config) if args.config else None
    if args.show_config:
        from .config import config_search_paths

        print("Config search order:")
        for i, p in enumerate(config_search_paths(config_path), 1):
            exists = " (exists)" if p.exists() else ""
            print(f"  {i}. {p}{exists}")
        config, loaded = load_config(config_path)
        if loaded:
            print(f"\nLoaded: {loaded}")
            print(f"  camera: {config.camera.frame_width}x{config.camera.frame_height}")
        else:
            print("\nNo config file found, using defaults.")
        return 0

    if args.list_cameras:
        from .capture.opencv import list_cameras

        cameras = list_cameras()
        if not cameras:
            print("No cameras found.")
        else:
            print("Available cameras:")
            for idx, desc in cameras:
                print(f"  {idx}: {desc}")
        return 0

    # Full imports (needed only when running spectrometer)
    from .config import Config
    from .spectrometer import Spectrometer

    mode_names = {
        "calibration": "Calibration",
        "measurement": "Measurement",
        "raman": "Raman",
        "colorscience": "Color Science",
        "waterfall": "Waterfall",
    }

    print(f"PySpectrometer 3 - {mode_names.get(args.mode, args.mode)} Mode")

    if args.waveshare:
        print('Waveshare 3.5" display mode (640x480)')
    if args.fullscreen:
        print("Fullscreen Spectrometer enabled")
    if args.waterfall:
        print("Waterfall display enabled")
    if args.mode == "raman":
        print(f"Raman laser wavelength: {args.laser} nm")

    # Monochrome is default, --color disables it
    monochrome = not args.color
    if monochrome:
        print(f"Monochrome camera mode ({args.bit_depth}-bit)")
    else:
        print("Color camera mode (RGB888, 8-bit)")

    config, config_loaded = load_config(config_path)
    if config_loaded:
        print(f"Config: {config_loaded}")
    # Apply CLI overrides (takes precedence over file)
    config = Config.from_args(
        base=config,
        fullscreen=args.fullscreen,
        waterfall=args.waterfall,
        waveshare=args.waveshare,
        gain=args.gain,
        width=args.width,
        monochrome=monochrome,
        bit_depth=args.bit_depth,
    )

    stream_control_base_url = None
    if args.camera is not None:
        from urllib.parse import urlparse

        from .capture.opencv import Capture

        source = _parse_source(args.camera)
        camera = Capture(
            source=source,
            width=config.camera.frame_width,
            height=config.camera.frame_height,
            gain=config.camera.gain,
            fps=config.camera.fps,
        )
        print(f"Using camera: {args.camera}")
        if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
            parsed = urlparse(source)
            stream_control_base_url = f"{parsed.scheme}://{parsed.netloc}"
    else:
        from .capture.picamera import Capture

        camera = Capture(
            width=config.camera.frame_width,
            height=config.camera.frame_height,
            gain=config.camera.gain,
            fps=config.camera.fps,
            monochrome=config.camera.monochrome,
            bit_depth=config.camera.bit_depth,
        )

    camera.start()
    config_width_before = config.camera.frame_width
    config_height_before = config.camera.frame_height
    load_calibration = True
    if config_width_before != camera.width or config_height_before != camera.height:
        print(
            f"WARNING: Config dimensions ({config_width_before}x{config_height_before}) differ "
            f"from camera ({camera.width}x{camera.height}) - uncalibrated"
        )
        load_calibration = False
    config.camera.frame_width = camera.width
    config.camera.frame_height = camera.height
    config.calibration.default_pixels = (0, camera.width // 2, camera.width)

    try:
        spectrometer = Spectrometer(
            config,
            camera=camera,
            mode=args.mode,
            laser_nm=args.laser,
            load_calibration=load_calibration,
            config_path=config_loaded,
            stream_control_base_url=stream_control_base_url,
        )
        spectrometer.run()
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
