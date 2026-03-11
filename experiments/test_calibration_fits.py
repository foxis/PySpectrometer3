"""Test harness for calibration: run auto-calibrator and generate fit graphs.

Usage:
  poetry run python .tmp/test_calibration_fits.py [csv_path] [--source FL12|HG|...] [--method correlation|peaks|both]

Output: .tmp/calibration_fit_*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root and .tmp for calibration_srp
ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / ".tmp"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(TMP))

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib: poetry add --group dev matplotlib")
    sys.exit(1)


def _source_from_name(name: str):
    from pyspectrometer.data.reference_spectra import ReferenceSource

    m = {"FL12": ReferenceSource.FL12, "FL1": ReferenceSource.FL1, "HG": ReferenceSource.HG, "D65": ReferenceSource.D65}
    return m.get(name.upper(), ReferenceSource.FL12)


def load_csv_intensity(csv_path: Path) -> tuple[np.ndarray, int, np.ndarray | None]:
    """Load pixel,intensity from CSV. Returns (intensity_array, n_pixels, wavelengths|None).

    If CSV has reference_wavelength or calibrated_wavelength column, returns it as wavelengths.
    """
    pixels_list: list[int] = []
    intensity_list: list[float] = []
    wl_list: list[float] = []
    intensity_col = 2
    wl_col = -1

    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if line.lower().startswith("pixel"):
                intensity_col = 1
                if "reference_wavelength" in line.lower():
                    wl_col = 2
                elif "calibrated_wavelength" in line.lower():
                    wl_col = 4
                elif "wavelength" in line.lower():
                    wl_col = 1
                continue
            if line.startswith("Pixel") and "Wavelength" in line:
                continue
            if len(parts) >= max(2, intensity_col + 1):
                try:
                    px = int(parts[0])
                    intensity = float(parts[intensity_col])
                    pixels_list.append(px)
                    intensity_list.append(intensity)
                    if wl_col >= 0 and len(parts) > wl_col:
                        wl_list.append(float(parts[wl_col]))
                    else:
                        wl_list.append(0.0)
                except (ValueError, IndexError):
                    continue

    if not pixels_list:
        return np.array([]), 0, None

    pairs = sorted(zip(pixels_list, intensity_list, wl_list), key=lambda x: x[0])
    n = pairs[-1][0] + 1 if pairs else 0
    measured = np.zeros(n, dtype=np.float64)
    wavelengths = np.zeros(n, dtype=np.float64) if wl_list and wl_col >= 0 else None
    for px, intensity, wl in pairs:
        measured[px] = intensity
        if wavelengths is not None:
            wavelengths[px] = wl
    if wavelengths is not None and np.any(wavelengths > 0):
        px_vals = np.array([p[0] for p in pairs])
        wl_vals = np.array([p[2] for p in pairs])
        for i in range(n):
            if wavelengths[i] == 0:
                wavelengths[i] = np.interp(i, px_vals, wl_vals)
    else:
        wavelengths = None
    return measured, n, wavelengths


def run_peaks_calibration(
    measured: np.ndarray,
    wavelengths: np.ndarray,
    source_name: str,
    *,
    debug: bool = True,
) -> list[tuple[int, float]]:
    """Run SRP peak-based calibration."""
    from pyspectrometer.data.reference_spectra import ReferenceSource

    source_map = {
        "FL12": ReferenceSource.FL12,
        "FL1": ReferenceSource.FL1,
        "HG": ReferenceSource.HG,
        "D65": ReferenceSource.D65,
    }
    source = source_map.get(source_name.upper(), ReferenceSource.FL12)

    from calibration_srp.calibrate import calibrate_peaks  # noqa: E402

    return calibrate_peaks(measured, wavelengths, source, debug=debug)


def run_extremum_calibration(
    measured: np.ndarray,
    wavelengths: np.ndarray,
    source_name: str,
    *,
    debug: bool = True,
    return_outcomes: bool = False,
):
    """Run feature-based extremum matching: peaks with peaks, dips with dips.

    When return_outcomes=True, returns (cal_points, [euclidean_outcome, cosine_outcome]).
    """
    from pyspectrometer.data.reference_spectra import ReferenceSource

    source_map = {
        "FL12": ReferenceSource.FL12,
        "FL1": ReferenceSource.FL1,
        "HG": ReferenceSource.HG,
        "D65": ReferenceSource.D65,
    }
    source = source_map.get(source_name.upper(), ReferenceSource.FL12)

    from calibration_srp.calibrate import calibrate_extremums  # noqa: E402

    return calibrate_extremums(
        measured, wavelengths, source, debug=debug, return_outcomes=return_outcomes
    )


def run_correlation_calibration(
    measured: np.ndarray,
    source_name: str,
) -> list[tuple[int, float]]:
    """Run current correlation-based auto-calibrator."""
    from pyspectrometer.data.reference_spectra import ReferenceSource, get_reference_spectrum
    from pyspectrometer.processing.auto_calibrator import AutoCalibrator

    source_map = {
        "FL12": ReferenceSource.FL12,
        "FL1": ReferenceSource.FL1,
        "FL2": ReferenceSource.FL2,
        "FL3": ReferenceSource.FL3,
        "HG": ReferenceSource.HG,
        "D65": ReferenceSource.D65,
        "LED": ReferenceSource.LED,
    }
    source = source_map.get(source_name.upper(), ReferenceSource.FL12)

    cal = AutoCalibrator()
    points = cal.calibrate(
        measured_intensity=measured,
        wavelengths=np.linspace(380, 750, len(measured)),  # dummy
        peak_indices=[],
        source=source,
        sensitivity=None,
        sensitivity_enabled=False,
    )
    return points


def plot_fit(
    measured: np.ndarray,
    cal_points: list[tuple[int, float]],
    out_path: Path,
    source_name: str,
    source=None,
) -> None:
    """Generate calibration fit graph: measured + reference SPD, peak alignment, polynomial, residuals."""
    if len(cal_points) < 3:
        print(f"[WARN] Only {len(cal_points)} cal points, skipping plot")
        return

    from pyspectrometer.data.reference_spectra import ReferenceSource, get_reference_spectrum

    px_arr = np.array([p for p, _ in cal_points], dtype=np.float64)
    wl_arr = np.array([w for _, w in cal_points], dtype=np.float64)

    deg = min(3, len(cal_points) - 1)
    coeffs = np.polyfit(px_arr, wl_arr, deg)
    poly = np.poly1d(coeffs)

    n = len(measured)
    pixels_full = np.arange(n, dtype=np.float64)
    wl_fit = poly(pixels_full)

    # Reference SPD on wavelength grid (350-750 nm)
    wl_ref = np.linspace(350, 750, n)
    ref_src = source or ReferenceSource.FL12
    ref_spd = get_reference_spectrum(ref_src, wl_ref)
    if ref_spd.max() > 0:
        ref_spd = ref_spd / ref_spd.max()

    # Measured in wavelength space (poly maps pixel -> wl)
    meas_wl = wl_fit
    meas_norm = measured.copy()
    if meas_norm.max() > 0:
        meas_norm = meas_norm / meas_norm.max()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), height_ratios=[1.5, 1, 1], sharex=False)

    # Top: measured SPD + reference SPD on wavelength axis, peak alignment lines (350-750 nm)
    ax0 = axes[0]
    mask = (meas_wl >= 350) & (meas_wl <= 750)
    ax0.plot(meas_wl[mask], meas_norm[mask], "b-", alpha=0.8, label="Measured", linewidth=1.5)
    ax0.plot(wl_ref, ref_spd, "orange", alpha=0.7, label="Reference", linewidth=1.2)

    # Draw alignment: vertical lines at each matched peak (measured vs reference at same wavelength)
    for px, wl in cal_points:
        wl_at_px = float(poly(px))
        if not 350 <= wl_at_px <= 750:
            continue
        y_meas = float(meas_norm[min(int(px), n - 1)]) if 0 <= px < n else 0
        y_ref = float(np.interp(wl_at_px, wl_ref, ref_spd))
        ax0.plot([wl_at_px, wl_at_px], [y_meas, y_ref], "r-", alpha=0.7, linewidth=1.5)
        ax0.scatter([wl_at_px], [y_meas], c="blue", s=50, zorder=6, edgecolors="darkblue")
        ax0.scatter([wl_at_px], [y_ref], c="orange", s=50, zorder=6, edgecolors="darkorange")

    ax0.set_xlabel("Wavelength (nm)")
    ax0.set_ylabel("Intensity (norm)")
    ax0.set_title(f"Calibration fit ({source_name}) — peak alignment")
    ax0.set_xlim(350, 750)
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.3)

    # Middle: pixel vs wavelength (cal points + polynomial)
    ax1 = axes[1]
    ax1.scatter(px_arr, wl_arr, c="red", s=80, label="Cal points", zorder=5)
    ax1.plot(pixels_full, wl_fit, "g-", alpha=0.8, label=f"Poly deg {deg}")
    ax1.set_ylabel("Wavelength (nm)")
    ax1.set_ylim(350, 750)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom: residuals from lower-degree fit (deg-1 when deg>1) so they are non-trivial
    ax2 = axes[2]
    deg_res = max(1, deg - 1) if len(cal_points) > deg else deg
    coeffs_res = np.polyfit(px_arr, wl_arr, deg_res)
    poly_res = np.poly1d(coeffs_res)
    wl_pred = poly_res(px_arr)
    residuals = wl_arr - wl_pred
    r_max = float(np.abs(residuals).max())
    ax2.bar(px_arr, residuals, width=5, color="orange", alpha=0.7)
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_xlabel("Pixel")
    ax2.set_ylabel("Residual (nm)")
    ax2.set_title(f"Residuals from poly deg {deg_res} (max={r_max:.2f} nm)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    ax0.set_xlim(350, 750)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_peak_matching(
    cal_points: list[tuple[int, float]],
    measured: np.ndarray,
    out_path: Path,
    source_name: str,
    source=None,
    *,
    initial_wavelengths: np.ndarray | None = None,
    all_peaks: "AllPeaksInfo | None" = None,
) -> None:
    """Draw both SPDs with lines connecting matched peaks. Show all peaks as points."""
    if not measured.size:
        return
    from pyspectrometer.data.reference_spectra import ReferenceSource, get_reference_spectrum

    n = len(measured)
    pixels = np.arange(n, dtype=np.float64)
    wl_ref = np.linspace(350, 750, n)
    ref_src = source or ReferenceSource.FL12
    ref_spd = get_reference_spectrum(ref_src, wl_ref)
    if ref_spd.max() > 0:
        ref_spd = ref_spd / ref_spd.max()
    meas_norm = measured.copy()
    if meas_norm.max() > 0:
        meas_norm = meas_norm / meas_norm.max()

    if initial_wavelengths is not None and len(initial_wavelengths) == n:
        meas_wl = np.asarray(initial_wavelengths, dtype=np.float64)
    else:
        meas_wl = np.linspace(380, 750, n)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(350, 750)
    ax.set_ylabel("Intensity (norm)")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_title(f"Peak matching ({source_name}) — lines connect matched peaks")

    mask_meas = (meas_wl >= 350) & (meas_wl <= 750)
    ax.plot(meas_wl[mask_meas], meas_norm[mask_meas], "b-", alpha=0.8, label="Measured", linewidth=1.5)
    ax.plot(wl_ref, ref_spd, color="orange", alpha=0.8, label="Reference", linewidth=1.5)

    matched_px = {int(px) for px, _ in cal_points}
    unmatched_wl: list[float] = []
    unmatched_y: list[float] = []

    if all_peaks and all_peaks.pixels:
        for px, pos in zip(all_peaks.pixels, all_peaks.positions):
            if px < 0 or px >= n:
                continue
            meas_wl_at_px = float(np.interp(px, pixels, meas_wl))
            if not (350 <= meas_wl_at_px <= 750):
                continue
            y_meas = float(meas_norm[px])
            if px in matched_px:
                continue
            unmatched_wl.append(meas_wl_at_px)
            unmatched_y.append(y_meas)
        if unmatched_wl:
            ax.scatter(
                unmatched_wl, unmatched_y,
                c="gray", s=40, zorder=4, alpha=0.7, marker="o", label="Unmatched",
            )

    colors = plt.cm.tab10(np.linspace(0, 1, min(len(cal_points), 10)))
    for i, (px, ref_wl) in enumerate(cal_points):
        px = int(px)
        if px < 0 or px >= n:
            continue
        meas_wl_at_px = float(np.interp(px, pixels, meas_wl))
        y_meas = float(meas_norm[px])
        y_ref = float(np.interp(ref_wl, wl_ref, ref_spd))
        c = colors[i % len(colors)]
        ax.plot(
            [meas_wl_at_px, ref_wl],
            [y_meas, y_ref],
            c=c,
            alpha=0.8,
            linewidth=2,
            zorder=5,
        )
        ax.scatter([meas_wl_at_px], [y_meas], c=[c], s=80, zorder=6, edgecolors="black")
        ax.scatter([ref_wl], [y_ref], c=[c], s=80, zorder=6, edgecolors="black")

    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test calibration and generate fit graphs")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(ROOT / "output" / "Spectrum-20260311--214345.csv"),
        help="Path to spectrum CSV",
    )
    parser.add_argument("--source", default="FL12", help="Reference source (FL12, HG, etc.)")
    parser.add_argument(
        "--method",
        choices=["correlation", "peaks", "extremum", "both"],
        default="both",
        help="Calibration method to run",
    )
    parser.add_argument("--no-debug", action="store_true", help="Hide peak detection debug")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    measured, n, wl_from_csv = load_csv_intensity(csv_path)
    if n < 10:
        print(f"Need at least 10 pixels, got {n}")
        return 1

    wavelengths = (
        wl_from_csv
        if wl_from_csv is not None and np.any(wl_from_csv > 0)
        else np.linspace(380, 750, n)
    )
    out_dir = ROOT / ".tmp"
    corr_points: list[tuple[int, float]] = []
    peaks_points: list[tuple[int, float]] = []
    extremum_points: list[tuple[int, float]] = []

    if args.method in ("correlation", "both"):
        print(f"Loaded {n} pixels from {csv_path}")
        print(f"Running correlation calibration (source={args.source})...")
        corr_points = run_correlation_calibration(measured, args.source)
        if corr_points:
            print(f"Correlation: {len(corr_points)} points")
            for p, w in corr_points:
                print(f"  Pixel {p} -> {w:.1f} nm")
            out_path = out_dir / f"calibration_fit_correlation_{csv_path.stem}.png"
            src = _source_from_name(args.source)
            plot_fit(measured, corr_points, out_path, f"{args.source} (correlation)", source=src)
        else:
            print("Correlation calibration failed")

    if args.method in ("peaks", "both"):
        print(f"Running peaks calibration (source={args.source})...")
        peaks_points = run_peaks_calibration(
            measured, wavelengths, args.source, debug=not args.no_debug
        )
        if peaks_points:
            print(f"Peaks: {len(peaks_points)} points")
            for p, w in peaks_points:
                print(f"  Pixel {p} -> {w:.1f} nm")
            out_path = out_dir / f"calibration_fit_peaks_{csv_path.stem}.png"
            src = _source_from_name(args.source)
            plot_fit(measured, peaks_points, out_path, f"{args.source} (peaks)", source=src)
            plot_peak_matching(
                peaks_points, measured,
                out_dir / f"calibration_peak_matches_peaks_{csv_path.stem}.png",
                f"{args.source} (peaks)", source=src,
                initial_wavelengths=wavelengths,
            )
        else:
            print("Peaks calibration failed")

    if args.method in ("extremum", "both"):
        print(f"Running extremum calibration (source={args.source})...")
        extremum_result = run_extremum_calibration(
            measured, wavelengths, args.source,
            debug=not args.no_debug,
            return_outcomes=True,
        )
        if isinstance(extremum_result, tuple):
            extremum_points, outcomes, all_peaks = extremum_result
        else:
            extremum_points = extremum_result
            outcomes = []
            all_peaks = None

        if extremum_points:
            print(f"Extremum (best): {len(extremum_points)} points")
            for p, w in extremum_points:
                print(f"  Pixel {p} -> {w:.1f} nm")
            for o in outcomes:
                print(f"  {o.metric}: {len(o.cal_points)} pts, score={o.score:.4f}")
            out_path = out_dir / f"calibration_fit_extremum_{csv_path.stem}.png"
            src = _source_from_name(args.source)
            plot_fit(measured, extremum_points, out_path, f"{args.source} (extremum)", source=src)
            plot_peak_matching(
                extremum_points, measured,
                out_dir / f"calibration_peak_matches_extremum_{csv_path.stem}.png",
                f"{args.source} (extremum)", source=src,
                initial_wavelengths=wavelengths,
                all_peaks=all_peaks,
            )
            for o in outcomes:
                plot_peak_matching(
                    o.cal_points, measured,
                    out_dir / f"calibration_peak_matches_extremum_{o.metric}_{csv_path.stem}.png",
                    f"{args.source} (extremum {o.metric})", source=src,
                    initial_wavelengths=wavelengths,
                    all_peaks=all_peaks,
                )
        else:
            print("Extremum calibration failed")

    if args.method == "both" and (corr_points or peaks_points or extremum_points):
        src = _source_from_name(args.source)
        _plot_comparison(
            measured,
            corr_points,
            peaks_points,
            extremum_points,
            out_dir / f"calibration_comparison_{csv_path.stem}.png",
            args.source,
            source=src,
        )

    return 0


def _plot_comparison(
    measured: np.ndarray,
    corr_points: list[tuple[int, float]],
    peaks_points: list[tuple[int, float]],
    extremum_points: list[tuple[int, float]],
    out_path: Path,
    source_name: str,
    source=None,
) -> None:
    """Plot correlation vs peaks vs extremum fit: measured + reference SPD, alignment."""
    from pyspectrometer.data.reference_spectra import ReferenceSource, get_reference_spectrum

    src = source or ReferenceSource.FL12
    n = len(measured)
    pixels = np.arange(n, dtype=np.float64)
    wl_ref = np.linspace(350, 750, n)
    ref_spd = get_reference_spectrum(src, wl_ref)
    if ref_spd.max() > 0:
        ref_spd = ref_spd / ref_spd.max()
    meas_norm = measured / measured.max() if measured.max() > 0 else measured

    rows = [(corr_points, "Correlation"), (peaks_points, "Peaks"), (extremum_points, "Extremum")]
    rows = [(p, l) for p, l in rows if len(p) >= 3]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 4 * len(rows)), sharex=True)
    if len(rows) == 1:
        axes = [axes]

    for ax, (points, label) in zip(axes, rows):
        px = np.array([p for p, _ in points], dtype=np.float64)
        wl = np.array([w for _, w in points], dtype=np.float64)
        deg = min(3, len(points) - 1)
        coeffs = np.polyfit(px, wl, deg)
        poly = np.poly1d(coeffs)
        wl_fit = poly(pixels)

        mask = (wl_fit >= 350) & (wl_fit <= 750)
        ax.plot(wl_fit[mask], meas_norm[mask], "b-", alpha=0.8, label="Measured")
        ax.plot(wl_ref, ref_spd, "orange", alpha=0.7, label="Reference")
        wl_at_px = poly(px)
        y_at_px = np.array([float(meas_norm[min(int(p), n - 1)]) for p in px])
        in_range = (wl_at_px >= 350) & (wl_at_px <= 750)
        ax.scatter(wl_at_px[in_range], y_at_px[in_range], c="blue", s=50, zorder=5)
        for p, w in zip(px, wl):
            wl_p = float(poly(p))
            if not 350 <= wl_p <= 750:
                continue
            y_m = float(meas_norm[min(int(p), n - 1)])
            y_r = float(np.interp(wl_p, wl_ref, ref_spd))
            ax.plot([wl_p, wl_p], [y_m, y_r], "r-", alpha=0.6)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (norm)")
        ax.set_title(f"{label} ({source_name}) — peak alignment")
        ax.set_xlim(350, 750)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ax in axes:
        ax.set_xlim(350, 750)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison {out_path}")


if __name__ == "__main__":
    sys.exit(main())
