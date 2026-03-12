"""Experiment: SPD-based peak/dip calibration with 3-panel visualization.

Uses calibrate_spectrum_anchors (2-anchor RANSAC, no emission line database).
1. Score grid
2. Measured linearly transformed overlaid on reference
3. After Cauchy tuning
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
OUT_DIR = ROOT / ".tmp"
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt

from pyspectrometer.data.reference_spectra import ReferenceSource, get_reference_spectrum
from pyspectrometer.processing.calibration.extremum import extract, valid_positions
from pyspectrometer.processing.calibration.cauchy_fit import fit_cal_points
from pyspectrometer.processing.calibration.hough_matching import (
    PeakDipResult,
    alignment_score_from_wavelengths,
    calibrate_spectrum_anchors,
    count_aligned_features,
)


def load_csv(csv_path: Path) -> tuple[np.ndarray, int, np.ndarray | None]:
    """Load intensity and wavelengths from CSV."""
    pixels_list: list[int] = []
    intensity_list: list[float] = []
    wl_list: list[float] = []
    intensity_col = 1
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
                continue
            if len(parts) >= max(2, intensity_col + 1):
                try:
                    px = int(parts[0])
                    intensity_list.append(float(parts[intensity_col]))
                    pixels_list.append(px)
                    wl_list.append(
                        float(parts[wl_col]) if wl_col >= 0 and len(parts) > wl_col else 0.0
                    )
                except (ValueError, IndexError):
                    continue

    if not pixels_list:
        return np.array([]), 0, None

    pairs = sorted(zip(pixels_list, intensity_list, wl_list), key=lambda x: x[0])
    n = pairs[-1][0] + 1 if pairs else 0
    measured = np.zeros(n, dtype=np.float64)
    wavelengths = np.zeros(n, dtype=np.float64) if wl_list else None
    for px, intensity, wl in pairs:
        measured[px] = intensity
        if wavelengths is not None:
            wavelengths[px] = wl
    if wavelengths is not None and np.any(wavelengths > 0):
        px_vals = np.array([p[0] for p in pairs])
        wl_vals = np.array([p[2] for p in pairs])
        for i in range(n):
            if wavelengths[i] == 0:
                wavelengths[i] = float(np.interp(i, px_vals, wl_vals))
    else:
        wavelengths = None
    return measured, n, wavelengths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hough-based calibration: accumulator, linear overlay, Cauchy overlay"
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=str(OUTPUT_DIR / "Spectrum-20260311--214345.csv"),
    )
    parser.add_argument("--source", default="FL12")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--delta-scale", type=float, default=15.0, help="Delta scale (nm) for alignment score")
    parser.add_argument("--delta-exp", type=float, default=2.0, help="Peakdip: exponent (2=original)")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1

    measured, n, wl_csv = load_csv(csv_path)
    if n < 10:
        print("Need at least 10 pixels")
        return 1

    positions = (
        wl_csv if wl_csv is not None and np.any(wl_csv > 0)
        else np.linspace(380, 750, n)
    )
    pos = valid_positions(positions if positions is not None else None, n)
    position_px = np.arange(n, dtype=np.intp)

    meas = extract(
        measured,
        pos,
        position_px=position_px,
        max_count=40,
    )
    if len(meas) < 2:
        print("Need at least 2 extremums")
        return 1

    try:
        src = ReferenceSource[args.source.upper()]
    except KeyError:
        src = ReferenceSource.FL12

    wl_ref_grid = np.linspace(380, 750, 500)
    ref_spd_full = get_reference_spectrum(src, wl_ref_grid)

    meas_px = np.array([e.position_px for e in meas if e.position_px is not None], dtype=np.float64)
    meas_int = np.array([abs(e.height) for e in meas if e.position_px is not None], dtype=np.float64)
    meas_is_dip = np.array([e.is_dip for e in meas if e.position_px is not None], dtype=bool)

    ref_extremums = extract(ref_spd_full, wl_ref_grid, position_px=None, max_count=40)
    ref_wl = np.array([e.position for e in ref_extremums], dtype=np.float64)
    ref_int = np.array([abs(e.height) for e in ref_extremums], dtype=np.float64)
    ref_is_dip = np.array([e.is_dip for e in ref_extremums], dtype=bool)

    print(f"  Measured features: {len(meas_px)} | Ref features (SPD): {len(ref_wl)}")

    result = calibrate_spectrum_anchors(
        n,
        meas_pixels=meas_px,
        meas_is_dip=meas_is_dip,
        meas_intensities=meas_int,
        ref_feat_wl=ref_wl,
        ref_feat_is_dip=ref_is_dip,
        ref_feat_intensities=ref_int,
        min_slope=0.15,
        max_slope=0.70,
        min_intercept=200.0,
        max_intercept=500.0,
        sigma_nm=10.0,
        top_k=18,
    )

    if result is None:
        print("Calibration failed")
        return 1

    print(f"  slope={result.slope:.5f}  intercept={result.intercept:.2f}  "
          f"peaks={result.n_peak_match}  dips={result.n_dip_match}  miss={result.n_mismatch}")
    if wl_csv is not None:
        m_gt = (wl_csv[n - 1] - wl_csv[0]) / (n - 1)
        c_gt = float(wl_csv[0])
        print(f"  GT   slope={m_gt:.5f}  intercept={c_gt:.2f}  "
              f"dm={result.slope - m_gt:+.5f}  dc={result.intercept - c_gt:+.2f}")

    # Reference SPD for overlay (full range)
    wl_ref = np.linspace(350, 750, 500)
    ref_spd = get_reference_spectrum(src, wl_ref)
    if ref_spd.max() > 0:
        ref_spd = ref_spd / ref_spd.max()
    meas_norm = measured.copy()
    if meas_norm.max() > 0:
        meas_norm = meas_norm / meas_norm.max()

    # Linear transform: wavelength = m * pixel + c
    wl_linear = result.slope * np.arange(n) + result.intercept

    # Cauchy (final) wavelengths
    wl_cauchy = result.wavelengths

    meas_px_for_align = meas_px
    ref_wl_for_align = ref_wl
    aligned_linear = (
        count_aligned_features(meas_px_for_align, ref_wl_for_align, wl_linear, tolerance_nm=5.0)
        if len(meas_px_for_align) > 0 and len(ref_wl_for_align) > 0
        else 0
    )
    aligned_cauchy = (
        count_aligned_features(meas_px_for_align, ref_wl_for_align, wl_cauchy, tolerance_nm=5.0)
        if len(meas_px_for_align) > 0 and len(ref_wl_for_align) > 0
        else len(result.cal_points)
    )

    score_linear = score_cauchy = None
    if isinstance(result, PeakDipResult) and len(meas_px) > 0 and len(ref_wl) > 0:
        score_linear = alignment_score_from_wavelengths(
            meas_px, meas_is_dip, meas_int, ref_wl, ref_is_dip, ref_int, wl_linear,
            delta_scale_nm=args.delta_scale, delta_exponent=2.0,
        )
        score_cauchy = alignment_score_from_wavelengths(
            meas_px, meas_is_dip, meas_int, ref_wl, ref_is_dip, ref_int, wl_cauchy,
            delta_scale_nm=args.delta_scale, delta_exponent=2.0,
        )

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), constrained_layout=True)

    # Panel 1: Score grid
    ax0 = axes[0]
    grid = result.score_grid.T
    im = ax0.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[
            result.m_bins[0],
            result.m_bins[-1],
            result.c_bins[0],
            result.c_bins[-1],
        ],
        cmap="viridis",
    )
    ax0.axvline(result.slope, color="red", linestyle="--", linewidth=1.5, label="peak")
    ax0.axhline(result.intercept, color="red", linestyle="--", linewidth=1.5)
    ax0.set_xlabel("Slope m (nm/pixel)")
    ax0.set_ylabel("Intercept c (nm)")
    ax0.set_title(
        f"Peak/dip alignment: score={result.score:.1f} "
        f"(+{result.n_peak_match} peaks +{result.n_dip_match} dips -{result.n_mismatch} mismatch)"
    )
    plt.colorbar(im, ax=ax0, label="Score")
    ax0.legend()

    # Panel 2: Measured linearly transformed on reference
    ax1 = axes[1]
    mask_ref = (wl_ref >= 350) & (wl_ref <= 750)
    ax1.plot(wl_ref[mask_ref], ref_spd[mask_ref], "orange", alpha=0.8, linewidth=2, label="Reference SPD")
    mask_meas = (wl_linear >= 350) & (wl_linear <= 750)
    ax1.plot(wl_linear[mask_meas], meas_norm[mask_meas], "b-", alpha=0.7, linewidth=1.5, label="Measured (linear)")
    for px, ref_wl in result.cal_points:
        if 0 <= px < n:
            wl = wl_linear[px]
            if 350 <= wl <= 750:
                y_meas = float(meas_norm[px])
                y_ref = float(np.interp(ref_wl, wl_ref, ref_spd))
                ax1.scatter([wl], [y_meas], c="blue", s=40, zorder=5, edgecolors="black")
                ax1.scatter([ref_wl], [y_ref], c="orange", s=40, zorder=5, edgecolors="black", marker="s")
                ax1.plot([wl, ref_wl], [y_meas, y_ref], "gray", alpha=0.5, linewidth=1)
    ax1.set_xlim(350, 750)
    ax1.set_ylabel("Intensity")
    align_str = ""
    if score_linear is not None:
        align_str = f" — align score={score_linear:.1f} ({aligned_linear} aligned)"
    ax1.set_title(f"Linear transform: λ = {result.slope:.4f}·pixel + {result.intercept:.1f}{align_str}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 3: After Cauchy tuning
    ax2 = axes[2]
    ax2.plot(wl_ref[mask_ref], ref_spd[mask_ref], "orange", alpha=0.8, linewidth=2, label="Reference SPD")
    mask_cauchy = (wl_cauchy >= 350) & (wl_cauchy <= 750)
    ax2.plot(wl_cauchy[mask_cauchy], meas_norm[mask_cauchy], "g-", alpha=0.7, linewidth=1.5, label="Measured (Cauchy)")
    for px, ref_wl in result.cal_points:
        if 0 <= px < n:
            wl = wl_cauchy[px]
            if 350 <= wl <= 750:
                y_meas = float(meas_norm[px])
                y_ref = float(np.interp(ref_wl, wl_ref, ref_spd))
                ax2.scatter([wl], [y_meas], c="green", s=40, zorder=5, edgecolors="black")
                ax2.scatter([ref_wl], [y_ref], c="orange", s=40, zorder=5, edgecolors="black", marker="s")
                ax2.plot([wl, ref_wl], [y_meas, y_ref], "gray", alpha=0.5, linewidth=1)
    ax2.set_xlim(350, 750)
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Intensity")
    cauchy_title = f"After Cauchy tuning — {len(result.cal_points)} cal points"
    if score_cauchy is not None:
        cauchy_title += f" — align score={score_cauchy:.1f} ({aligned_cauchy} aligned)"
    else:
        cauchy_title += f", {aligned_cauchy} aligned"
    ax2.set_title(cauchy_title)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 4: Pixel/nm calibration curve
    ax3 = axes[3]
    px_curve = np.arange(n, dtype=np.float64)
    ax3.plot(px_curve, wl_linear, "b-", alpha=0.7, linewidth=1.5, label="Linear")
    ax3.plot(px_curve, wl_cauchy, "g-", alpha=0.7, linewidth=1.5, label="Cauchy")
    ax3.scatter(
        [p for p, _ in result.cal_points],
        [w for _, w in result.cal_points],
        c="red",
        s=40,
        zorder=5,
        label="Cal points",
    )
    ax3.set_xlabel("Pixel")
    ax3.set_ylabel("Wavelength (nm)")
    ax3.set_title("Calibration curve: pixel → wavelength")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"Hough-based calibration ({args.source}) — {csv_path.name}",
        fontsize=12,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiment_hough_matching.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    if isinstance(result, PeakDipResult):
        print(f"  Peak/dip: score={result.score:.1f} (+{result.n_peak_match} peaks +{result.n_dip_match} dips -{result.n_mismatch} mismatch)")
        print(f"  slope={result.slope:.4f} nm/px  intercept={result.intercept:.2f} nm")
        if wl_csv is not None:
            m_gt = (wl_csv[n - 1] - wl_csv[0]) / (n - 1)
            c_gt = float(wl_csv[0])
            print(f"  GT:    slope={m_gt:.4f} nm/px  intercept={c_gt:.2f} nm  (algo err: dm={result.slope-m_gt:+.4f}  dc={result.intercept-c_gt:+.2f})")
        print(f"  Linear: {aligned_linear} aligned, Cauchy: {aligned_cauchy} aligned")
    print(f"  Cal points: {len(result.cal_points)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
