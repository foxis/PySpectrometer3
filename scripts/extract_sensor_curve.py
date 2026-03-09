#!/usr/bin/env python3
"""
Extract spectral sensitivity curves from VC MIPI sensor chart image.

Samples legend colors (top right) to disambiguate curves by color, then extracts
each sensor's curve separately. Outputs CSV per sensor with 5 nm wavelength steps.

Sensors: OV9281 (orange/brown), IMX290 (purple), IMX252/273/296/297/392 (cyan).

Usage:
    python scripts/extract_sensor_curve.py [image_path]
    python scripts/extract_sensor_curve.py image.png -o output_dir
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np

try:
    import cv2
except ImportError:
    print("Error: cv2 required. Install with: pip install opencv-python", file=sys.stderr)
    sys.exit(1)


SENSOR_NAMES = ["OV9281", "IMX290", "IMX252_273_296_297_392"]


class LegendSwatch(NamedTuple):
    """Rectangle in image where legend color swatch appears."""

    x0: int
    y0: int
    x1: int
    y1: int


def load_image(path: Path) -> np.ndarray:
    """Load image as BGR array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def _sample_legend_color(img: np.ndarray, swatch: LegendSwatch) -> np.ndarray:
    """Sample dominant non-background color from legend swatch region. Returns BGR."""
    region = img[swatch.y0:swatch.y1, swatch.x0:swatch.x1]
    if region.size == 0:
        return np.array([0, 0, 0], dtype=np.float64)
    # Exclude dark background (grid/axis)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mask = gray > 60
    if not np.any(mask):
        return np.median(region.reshape(-1, 3), axis=0).astype(np.float64)
    pixels = region[mask]
    # Use median to avoid white grid/text
    return np.median(pixels, axis=0).astype(np.float64)


def _build_legend_swatches(h: int, w: int) -> dict[str, LegendSwatch]:
    """Define legend swatch regions (top right). Order: OV9281, IMX290, IMX252... from top."""
    x0 = int(0.72 * w)
    x1 = int(0.88 * w)
    y0 = int(0.05 * h)
    y1 = int(0.22 * h)
    row_h = max(1, (y1 - y0) // 3)
    swatch_w = max(1, int(0.18 * (x1 - x0)))
    swatches = {}
    for i, name in enumerate(SENSOR_NAMES):
        ry0 = y0 + i * row_h + row_h // 4
        ry1 = ry0 + row_h // 2
        swatches[name] = LegendSwatch(x0, ry0, x0 + swatch_w, ry1)
    return swatches


def _color_distance_bgr(pixel_bgr: np.ndarray, ref_bgr: np.ndarray) -> float:
    """Euclidean distance in BGR space."""
    return float(np.linalg.norm(pixel_bgr.astype(np.float64) - ref_bgr))


def _find_curve_value_in_column(
    img: np.ndarray,
    col: int,
    y0: int,
    y1: int,
    ref_bgr: np.ndarray,
    match_thresh: float = 45.0,
) -> float | None:
    """
    Find relative sensitivity (0-1) at wavelength column for one color.
    Top of chart = 1, bottom = 0. Returns topmost matching pixel's normalized y.
    """
    region = img[y0:y1, col : col + 1]
    if region.size == 0:
        return None
    h_region = y1 - y0
    best_y: float | None = None
    best_dist = match_thresh
    for row in range(h_region):
        pixel = region[row, 0]
        dist = _color_distance_bgr(pixel, ref_bgr)
        if dist < best_dist:
            best_dist = dist
            best_y = row
    if best_y is None:
        return None
    # y=0 at top of chart = high sensitivity; invert so top -> 1
    return 1.0 - (best_y / max(1, h_region - 1))


def extract_curves_by_color(
    img_path: Path,
    wl_min: float = 400.0,
    wl_max: float = 1000.0,
    wl_step: float = 5.0,
    x_margin_left: float = 0.08,
    x_margin_right: float = 0.25,
    y_margin_top: float = 0.12,
    y_margin_bottom: float = 0.12,
    match_thresh: float = 55.0,
) -> dict[str, list[tuple[float, float]]]:
    """
    Extract spectral sensitivity per sensor using color disambiguation from legend.
    Returns dict: sensor_name -> [(wavelength_nm, relative_sensitivity), ...].
    """
    img = load_image(img_path)
    h, w = img.shape[:2]

    # Legend swatches (top right)
    swatches = _build_legend_swatches(h, w)
    ref_colors: dict[str, np.ndarray] = {}
    for name, sw in swatches.items():
        ref_colors[name] = _sample_legend_color(img, sw)

    # Chart region (exclude legend on right)
    x0 = int(w * x_margin_left)
    x1 = int(w * (1 - x_margin_right))
    y0 = int(h * y_margin_top)
    y1 = int(h * (1 - y_margin_bottom))

    wavelengths = np.arange(wl_min, wl_max + wl_step / 2, wl_step)
    curves: dict[str, list[tuple[float, float]]] = {name: [] for name in SENSOR_NAMES}

    for wl in wavelengths:
        t = (wl - wl_min) / max(1e-6, wl_max - wl_min)
        col = x0 + int(t * (x1 - x0))
        col = np.clip(col, 0, w - 1)

        for name in SENSOR_NAMES:
            val = _find_curve_value_in_column(
                img, col, y0, y1, ref_colors[name], match_thresh
            )
            if val is not None:
                curves[name].append((float(wl), float(val)))

    # Normalize each curve so max = 1
    for name in SENSOR_NAMES:
        pts = curves[name]
        if not pts:
            continue
        max_s = max(s for _, s in pts)
        if max_s > 0:
            curves[name] = [(wl, s / max_s) for wl, s in pts]
        else:
            curves[name] = [(wl, 0.0) for wl, _ in pts]

    return curves


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract sensor spectral curves from chart (color disambiguation from legend)"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Path to mipi_sensors_spectral_curve.png",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory for CSVs (default: data/sensor_sensitivity)",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=5.0,
        help="Wavelength step in nm (default: 5)",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    img_path = (
        Path(args.image)
        if args.image
        else base / "data" / "sensor_sensitivity" / "mipi_sensors_spectral_curve.png"
    )
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else base / "data" / "sensor_sensitivity"
    )

    if not img_path.exists():
        print(f"Image not found: {img_path}", file=sys.stderr)
        return 1

    try:
        curves = extract_curves_by_color(img_path, wl_step=args.step)
    except Exception as e:
        print(f"Extraction failed: {e}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, pts in curves.items():
        if not pts:
            continue
        fname = f"{name}_spectral_sensitivity_extracted.csv"
        out_path = out_dir / fname
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["wavelength_nm", "relative_sensitivity"])
            for wl, s in pts:
                writer.writerow([f"{wl:.1f}", f"{s:.4f}"])
        print(f"Wrote {len(pts)} points to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
