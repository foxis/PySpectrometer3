"""Simulate narrow-range spectrometer (like live Hg-verified setup) to verify fix."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pyspectrometer.data.reference_spectra import ReferenceSource, get_reference_spectrum
from pyspectrometer.processing.calibration.extremum import extract, valid_positions
from pyspectrometer.processing.calibration.hough_matching import calibrate_spectrum_anchors


def run(true_slope: float, true_c: float, source: ReferenceSource, label: str) -> None:
    n = 1280
    true_wl = true_slope * np.arange(n) + true_c

    wl_ref_full = np.linspace(380, 750, 500)
    ref_spd_full = get_reference_spectrum(source, wl_ref_full)

    measured = np.interp(true_wl, wl_ref_full, ref_spd_full)
    rng = np.random.default_rng(42)
    measured += rng.normal(0, 0.005, n)
    measured = np.clip(measured, 0, None)

    # Boost known emission lines for detectability
    for wl_line, boost in [(405, 3), (436, 5), (487, 2), (546, 4), (578, 2), (611, 3)]:
        px = int((wl_line - true_c) / true_slope)
        if 0 <= px < n:
            measured[max(0, px - 2) : px + 3] += boost * 0.5

    pos = valid_positions(None, n)
    meas_feats = extract(measured, pos, position_px=np.arange(n, dtype=np.intp), max_count=18)
    ref_feats = extract(ref_spd_full, wl_ref_full, position_px=None, max_count=18)

    meas_px = np.array([e.position_px for e in meas_feats if e.position_px is not None], dtype=np.float64)
    meas_int = np.array([abs(e.height) for e in meas_feats if e.position_px is not None], dtype=np.float64)
    meas_dip = np.array([e.is_dip for e in meas_feats if e.position_px is not None], dtype=bool)
    ref_wl = np.array([e.position for e in ref_feats], dtype=np.float64)
    ref_int = np.array([abs(e.height) for e in ref_feats], dtype=np.float64)
    ref_dip = np.array([e.is_dip for e in ref_feats], dtype=bool)

    result = calibrate_spectrum_anchors(
        n, meas_pixels=meas_px, meas_is_dip=meas_dip, meas_intensities=meas_int,
        ref_feat_wl=ref_wl, ref_feat_is_dip=ref_dip, ref_feat_intensities=ref_int,
        min_slope=0.15, max_slope=0.70, min_intercept=200.0, max_intercept=500.0,
        sigma_nm=10.0, top_k=18,
    )

    dm = result.slope - true_slope
    dc = result.intercept - true_c
    ok = abs(dm) < 0.020 and abs(dc) < 15.0
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {label}: found slope={result.slope:.4f} c={result.intercept:.2f}  "
          f"(true slope={true_slope:.4f} c={true_c:.2f})  dm={dm:+.4f} dc={dc:+.2f} nm  "
          f"score={result.score:.3f}")


if __name__ == "__main__":
    print("=== Narrow-range spectrometer simulation (Hg-verified parameters) ===")
    run(0.230, 278.0, ReferenceSource.FL12, "Live FL12, slope=0.230, c=278 (Hg-verified range 278-572nm)")
    run(0.230, 278.0, ReferenceSource.HG,   "Live Hg,  slope=0.230, c=278")
    run(0.492, 271.0, ReferenceSource.FL12, "CSV FL12, slope=0.492, c=271 (wide range 271-900nm)")
    run(0.268, 392.0, ReferenceSource.HG,   "Old CSV Hg, slope=0.268, c=392 (392-735nm)")
