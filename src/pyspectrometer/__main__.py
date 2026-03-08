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

from .config import Config
from .spectrometer import Spectrometer


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
  x     Clear click points
  m     Toggle measure mode (wavelength)
  p     Toggle pixel mode (for calibration)
  o/l   Increase/decrease Savitzky-Golay order
  i/k   Increase/decrease peak width
  u/j   Increase/decrease threshold
  t/g   Increase/decrease camera gain

Examples:
  pyspectrometer                    # Normal windowed mode (800x480)
  pyspectrometer --fullscreen       # Fullscreen mode
  pyspectrometer --waterfall        # With waterfall display
  pyspectrometer --waveshare        # Waveshare 3.5" display (640x480)
  pyspectrometer --waveshare --fullscreen  # Waveshare fullscreen
  pyspectrometer --gain 15          # Custom camera gain
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
        help="Optimize for Waveshare 3.5\" touchscreen (640x480)",
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
        default=None,
        metavar="PIXELS",
        help="Frame width in pixels (default: 800)",
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        metavar="PIXELS",
        help="Frame height in pixels (default: 600)",
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 3.0.0",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point for PySpectrometer 3."""
    args = parse_args()
    
    if args.waveshare:
        print("Waveshare 3.5\" display mode (640x480)")
    if args.fullscreen:
        print("Fullscreen Spectrometer enabled")
    if args.waterfall:
        print("Waterfall display enabled")
    
    config = Config.from_args(
        fullscreen=args.fullscreen,
        waterfall=args.waterfall,
        waveshare=args.waveshare,
        gain=args.gain,
        width=args.width,
        height=args.height,
    )
    
    try:
        spectrometer = Spectrometer(config)
        spectrometer.run()
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
