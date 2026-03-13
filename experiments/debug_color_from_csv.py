"""Debug: compute color from measured spectrum CSVs and verify XYZ/LAB normalization.

Supports two CSV formats:
  Color Science / Measurement mode:  Pixel,Wavelength,Intensity
  Calibration mode:                  pixel,intensity,reference_wavelength,reference_intensity,
                                     calibrated_wavelength,calibrated_intensity

XYZ is now normalized relative to equal-energy illuminant E (x=y=1/3):
  - SPD divided by total integral (removes absolute brightness)
  - Scaled so E gives Y=100
  - L*a*b* reference white = (100,100,100) (equal energy E by CIE definition)

Run with:
    poetry run python experiments/debug_color_from_csv.py [path/to/spectrum.csv]
    poetry run python experiments/debug_color_from_csv.py --all   # all today's CSVs

Output: .tmp/chromaticity_debug.png
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import colour

from pyspectrometer.colorscience.chromaticity import (
    RefPoint,
    illuminant_xy,
    render_chromaticity,
)
from pyspectrometer.colorscience.xyz import calculate_XYZ, xyz_to_lab


# ---------------------------------------------------------------------------
# CSV loading — handles both formats
# ---------------------------------------------------------------------------

def load_spectrum(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Load spectrum from CSV (both Calibration and Color Science format).

    Returns (wavelengths, intensity, reference_intensity, metadata).
    reference_intensity is zero array when not present.
    """
    metadata: dict[str, str] = {}

    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                if ":" in line:
                    key, _, val = line.lstrip("# ").partition(":")
                    metadata[key.strip()] = val.strip()
            else:
                break

    rows: list[tuple[float, float, float]] = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(filter(lambda r: not r.startswith("#"), f))
        fieldnames = reader.fieldnames or []

        if "calibrated_wavelength" in fieldnames:
            # Calibration mode format
            for row in reader:
                wl = float(row["calibrated_wavelength"])
                cal = float(row["calibrated_intensity"])
                ref = float(row.get("reference_intensity") or 0.0)
                rows.append((wl, cal, ref))
        else:
            # Color Science / Measurement mode format
            wl_col = next((c for c in fieldnames if "wavelength" in c.lower()), "Wavelength")
            int_col = next((c for c in fieldnames if "intensity" in c.lower()), "Intensity")
            for row in reader:
                wl = float(row[wl_col])
                val = float(row[int_col])
                rows.append((wl, val, 0.0))

    data = np.array(rows)
    return data[:, 0], data[:, 1], data[:, 2], metadata


# ---------------------------------------------------------------------------
# Illuminant XYZ/xy helpers
# ---------------------------------------------------------------------------

def _xyz(wl: np.ndarray, sp: np.ndarray, observer: str = "2deg") -> tuple[float, float, float]:
    return calculate_XYZ(sp, wl, "illumination", observer=observer)


def _to_xy(X: float, Y: float, Z: float) -> tuple[float, float]:
    s = X + Y + Z
    return (X / s, Y / s) if s > 1e-12 else (0.0, 0.0)


def _ref_illuminant_xyz(name: str) -> tuple[float, float, float]:
    """Compute normalised (Y=100) XYZ of a colour-science illuminant via our method."""
    sd = colour.SDS_ILLUMINANTS[name]
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
    wl = cmfs.wavelengths
    vals = np.interp(wl, sd.wavelengths.astype(float), sd.values.astype(float),
                     left=0.0, right=0.0)
    return _xyz(wl, vals)


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def report(
    csv_path: Path,
    wavelengths: np.ndarray,
    intensity: np.ndarray,
    reference: np.ndarray,
    metadata: dict,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Print XYZ/xy/LAB.  Returns (intensity_xy, reference_xy)."""
    vis = (wavelengths >= 360) & (wavelengths <= 830)
    wl  = wavelengths[vis]
    spd = intensity[vis]
    ref = reference[vis]

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  CSV: {csv_path.name}")
    mtype = metadata.get("Measurement type", "?")
    mode  = metadata.get("Mode", "?")
    print(f"  Mode: {mode}  |  Measurement type: {mtype}")
    print(f"  Spectrum: {len(wl)} points, {wl[0]:.1f} to {wl[-1]:.1f} nm")
    print(f"  Intensity peak: {wl[int(np.argmax(spd))]:.1f} nm = {spd.max():.4f}")
    print(f"{sep}")

    # Equal energy reference: CIE E illuminant → X=Y=Z=100 by definition
    E_WHITE = (100.0, 100.0, 100.0)

    X, Y, Z = _xyz(wl, spd)
    xy = _to_xy(X, Y, Z)
    L, a, b = xyz_to_lab(X, Y, Z, reference_white_XYZ=E_WHITE)
    print(f"  XYZ (E-normalized): X={X:.2f}  Y={Y:.2f}  Z={Z:.2f}")
    print(f"  xy = ({xy[0]:.4f}, {xy[1]:.4f})")
    print(f"  L*a*b* vs E:  L*={L:.1f}  a*={a:.2f}  b*={b:.2f}")

    # Also show vs D65 for reference
    d65_xy = illuminant_xy("D65")
    Ld65, ad65, bd65 = xyz_to_lab(X, Y, Z, reference_white_XYZ=None)  # colour default = D65
    print(f"  L*a*b* vs D65: L*={Ld65:.1f}  a*={ad65:.2f}  b*={bd65:.2f}")

    # Closest standard illuminant by chromaticity distance
    print("  Closest illuminants (by xy distance):")
    dists = []
    for name in ["A", "D50", "D65", "FL1", "FL2", "FL11", "FL12"]:
        ixy = illuminant_xy(name)
        d = ((xy[0] - ixy[0]) ** 2 + (xy[1] - ixy[1]) ** 2) ** 0.5
        dists.append((d, name, ixy))
    for d, name, ixy in sorted(dists)[:3]:
        print(f"     {name:6s}: x={ixy[0]:.4f}  y={ixy[1]:.4f}  |d|={d:.4f}")

    ref_xy = (0.0, 0.0)
    if ref.max() > 1e-6:
        Xr, Yr, Zr = _xyz(wl, ref)
        ref_xy = _to_xy(Xr, Yr, Zr)
        Lr, ar, br = xyz_to_lab(Xr, Yr, Zr, reference_white_XYZ=E_WHITE)
        print(f"  [reference col]  xy=({ref_xy[0]:.4f},{ref_xy[1]:.4f})  "
              f"L*={Lr:.1f}  a*={ar:.2f}  b*={br:.2f}")

    print()
    return xy, ref_xy


# ---------------------------------------------------------------------------
# Reference point list for diagram annotation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = Path(__file__).parent.parent / ".tmp"
    out_dir.mkdir(exist_ok=True)

    # Collect CSV paths from args
    args = sys.argv[1:]
    output_dir = Path(__file__).parent.parent / "output"

    if "--all" in args or not args:
        # All today's Color Science CSVs + default
        csvs = sorted(output_dir.glob("Spectrum-20260313--*.csv"))
        if not csvs:
            csvs = [output_dir / "Spectrum-20260311--214345.csv"]
    else:
        csvs = [Path(a) for a in args]

    all_xys: list[tuple[float, float]] = []
    last_refs: list[RefPoint] = []

    for csv_path in csvs:
        if not csv_path.exists():
            print(f"  SKIP (not found): {csv_path.name}")
            continue
        wavelengths, intensity, reference, metadata = load_spectrum(csv_path)
        xy, ref_xy = report(csv_path, wavelengths, intensity, reference, metadata)
        if xy != (0.0, 0.0):
            all_xys.append(xy)

    if not all_xys:
        print("No valid spectra found.")
        return

    # Build standard reference points
    std_refs = [
        RefPoint(xy=illuminant_xy(n), label=n, color=c)
        for n, c in [("D65", (0, 230, 230)), ("D50", (100, 200, 255)),
                     ("FL1", (200, 255, 140)), ("FL12", (80, 180, 255)),
                     ("A", (0, 140, 255))]
    ]

    # Show the last / most interesting measurement point on the diagram
    current_xy = all_xys[-1]
    img = render_chromaticity(900, 500, xy_current=current_xy, refs=std_refs)

    # Draw all measurement points as small crosses
    _annotate_all(img, all_xys)

    out_path = out_dir / "chromaticity_debug.png"
    cv2.imwrite(str(out_path), img)
    print(f"  Diagram saved: {out_path}\n")


def _annotate_all(img: np.ndarray, xys: list[tuple[float, float]]) -> None:
    """Draw small cyan crosses for all measurement xy points."""
    from pyspectrometer.colorscience.chromaticity import _X_MIN, _X_MAX, _Y_MIN, _Y_MAX  # type: ignore[attr-defined]
    h, w = img.shape[:2]
    margin = 32
    draw_w = w - 2 * margin
    draw_h = h - 2 * margin

    for i, (x, y) in enumerate(xys):
        px = int(margin + (x - _X_MIN) / (_X_MAX - _X_MIN) * draw_w)
        py = int(margin + (_Y_MAX - y) / (_Y_MAX - _Y_MIN) * draw_h)
        cv2.drawMarker(img, (px, py), (255, 200, 0), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(img, str(i + 1), (px + 6, py - 6),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 200, 0), 1)


if __name__ == "__main__":
    main()
