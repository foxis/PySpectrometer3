"""Benchmark: analyze ground-truth calibrations from saved CSV files.

Each CSV has reference_wavelength = calibrated_wavelength (already aligned).
We load each, identify the lamp source from peak wavelengths, compute ground-truth
(m, c), and score our algorithm against it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
sys.path.insert(0, str(ROOT / "src"))

from pyspectrometer.data.reference_spectra import ReferenceSource, get_reference_spectrum
from pyspectrometer.processing.calibration.extremum import extract
from pyspectrometer.processing.calibration.hough_matching import (
    _alignment_cost_all_features,
    alignment_score_from_wavelengths,
)

# Known Hg lines (nm)
HG_LINES = [253.65, 296.73, 302.15, 312.57, 334.15, 365.02,
            404.66, 407.78, 435.84, 546.07, 576.96, 578.97]

# Known FL12 prominent peaks (nm) - typical fluorescent lamp
FL12_APPROX_PEAKS = [405, 436, 487, 544, 577, 611, 671]


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Returns (intensity, calibrated_wl, metadata_lines, n_pixels)."""
    intensity: list[float] = []
    wl: list[float] = []
    meta: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("#"):
                meta.append(s)
                continue
            if s.lower().startswith("pixel"):
                continue
            parts = s.split(",")
            if len(parts) >= 5:
                try:
                    intensity.append(float(parts[1]))
                    wl.append(float(parts[4]))
                except (ValueError, IndexError):
                    pass
    arr_i = np.array(intensity, dtype=np.float64)
    arr_w = np.array(wl, dtype=np.float64)
    return arr_i, arr_w, meta, len(arr_i)


def identify_lamp(peak_wls: list[float]) -> str:
    """Guess lamp type from measured peak wavelengths."""
    hg_hits = sum(any(abs(p - h) < 5.0 for h in HG_LINES) for p in peak_wls)
    fl_hits = sum(any(abs(p - f) < 10.0 for f in FL12_APPROX_PEAKS) for p in peak_wls)
    if hg_hits >= 3:
        return f"Hg (matched {hg_hits} lines)"
    if fl_hits >= 3:
        return f"FL12 (matched {fl_hits} peaks)"
    return f"Unknown (hg={hg_hits}, fl={fl_hits})"


def score_calibration(
    intensity: np.ndarray,
    wl_gt: np.ndarray,
    src: ReferenceSource,
) -> tuple[float, float, int]:
    """Score ground-truth wavelength array against ref source.

    Returns (gaussian_score, alignment_score, n_features).
    """
    n = len(intensity)
    wl_ref_grid = np.linspace(380, 750, 500)
    ref_spd = get_reference_spectrum(src, wl_ref_grid)

    ref_extremums = extract(ref_spd, wl_ref_grid, position_px=None, max_count=25)
    ref_wl = np.array([e.position for e in ref_extremums], dtype=np.float64)
    ref_is_dip = np.array([e.is_dip for e in ref_extremums], dtype=bool)
    ref_int = np.array([abs(e.height) for e in ref_extremums], dtype=np.float64)

    pos = np.linspace(wl_gt[0], wl_gt[-1], n)
    px_arr = np.arange(n, dtype=np.intp)
    meas_extremums = extract(intensity, pos, position_px=px_arr, max_count=25)
    meas_px = np.array([e.position_px for e in meas_extremums if e.position_px is not None], dtype=np.float64)
    meas_is_dip = np.array([e.is_dip for e in meas_extremums if e.position_px is not None], dtype=bool)
    meas_int = np.array([abs(e.height) for e in meas_extremums if e.position_px is not None], dtype=np.float64)

    gauss = -_alignment_cost_all_features(meas_px, meas_is_dip, ref_wl, ref_is_dip, wl_gt, sigma=8.0)
    align = alignment_score_from_wavelengths(
        meas_px, meas_is_dip, meas_int, ref_wl, ref_is_dip, ref_int, wl_gt,
        delta_scale_nm=10.0, delta_exponent=3.0,
    )
    return float(gauss), float(align), len(meas_px)


def main() -> None:
    csv_files = [
        OUTPUT_DIR / "Spectrum-20260311--214345.csv",
        OUTPUT_DIR / "Spectrum-20260311--213319.csv",
        OUTPUT_DIR / "Spectrum-20260311--193856.csv",
    ]

    for csv_path in csv_files:
        if not csv_path.exists():
            print(f"MISSING: {csv_path.name}")
            continue

        intensity, wl_gt, meta, n = load_csv(csv_path)
        m_gt = (wl_gt[-1] - wl_gt[0]) / (n - 1)
        c_gt = float(wl_gt[0])

        peaks, pr = find_peaks(intensity, prominence=0.002, distance=15)
        top_peaks = sorted(
            [float(wl_gt[p]) for p in peaks[np.argsort(pr["prominences"])[-12:][::-1]]]
        )
        lamp = identify_lamp(top_peaks)

        print(f"\n{'='*60}")
        print(f"File: {csv_path.name}")
        for m in meta:
            print(f"  {m}")
        print(f"  n={n}, wl[0]={c_gt:.2f}, wl[-1]={wl_gt[-1]:.2f}")
        print(f"  slope={m_gt:.4f} nm/px, intercept={c_gt:.3f} nm")
        print(f"  Detected lamp: {lamp}")
        print(f"  Top peaks (nm): {[round(x, 1) for x in top_peaks]}")

        # Try scoring against likely sources
        for src_name in ("FL12", "HG"):
            try:
                src = ReferenceSource[src_name]
                gauss, align, nf = score_calibration(intensity, wl_gt, src)
                print(f"  vs {src_name}: gaussian={gauss:.2f}, alignment={align:.2f}, n_features={nf}")
            except Exception as e:
                print(f"  vs {src_name}: ERROR {e}")


if __name__ == "__main__":
    main()
