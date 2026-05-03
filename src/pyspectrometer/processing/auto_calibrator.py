"""Standalone wavelength calibration.

Accepts measured spectrum (pixels, intensity) and reference SPD (nm, intensity).
Returns calibration points (pixel, wavelength). No CSV, no reference data import.
Uses calibrate_spectrum_anchors for peak/dip matching. Falls back to correlation
if anchors yield <4 points.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ..data.reference_spectra import REFERENCE_WL_MAX, REFERENCE_WL_MIN
from .calibration import calibrate_spectrum_anchors, extract
from .calibration.extremum import valid_positions


def calibrate(
    measured_intensity: np.ndarray,
    reference_wl: np.ndarray,
    reference_intensity: np.ndarray,
    *,
    max_extremums: int = 30,
) -> list[tuple[int, float]]:
    """Calibrate pixels to wavelengths from measured and reference spectra.

    Standalone: pass measured intensity (per pixel) and reference SPD (nm, intensity).
    Uses SPD-based peak/dip matching (calibrate_spectrum_anchors). Falls back to
    correlation if anchors yield <4 points.

    Args:
        measured_intensity: Intensity per pixel (0..n-1)
        reference_wl: Reference wavelengths (nm)
        reference_intensity: Reference SPD at those wavelengths
        max_extremums: Max peaks/dips for feature extraction

    Returns:
        List of (pixel, wavelength) calibration points, or [] on failure
    """
    n = len(measured_intensity)
    if n < 10 or len(reference_wl) < 4:
        return []

    ref_wl = np.asarray(reference_wl, dtype=np.float64)
    ref_int = np.asarray(reference_intensity, dtype=np.float64)
    if ref_wl.size != ref_int.size or ref_wl.size < 4:
        return []

    wl_ref = np.linspace(REFERENCE_WL_MIN, REFERENCE_WL_MAX, 500)
    ref_spd = np.interp(wl_ref, ref_wl, ref_int)
    pos = valid_positions(None, n)

    meas = extract(
        measured_intensity,
        pos,
        position_px=np.arange(n, dtype=np.intp),
        max_count=max_extremums,
    )
    ref_ext = extract(ref_spd, wl_ref, position_px=None, max_count=max_extremums)

    meas_px = np.array([e.position_px for e in meas if e.position_px is not None], dtype=np.float64)
    meas_int = np.array([abs(e.height) for e in meas if e.position_px is not None], dtype=np.float64)
    meas_is_dip = np.array([e.is_dip for e in meas if e.position_px is not None], dtype=bool)
    ref_feat_wl = np.array([e.position for e in ref_ext], dtype=np.float64)
    ref_int_arr = np.array([abs(e.height) for e in ref_ext], dtype=np.float64)
    ref_is_dip = np.array([e.is_dip for e in ref_ext], dtype=bool)

    if len(meas_px) < 2 or len(ref_feat_wl) < 2:
        return _correlation_fallback(measured_intensity, ref_wl, ref_int)

    result = calibrate_spectrum_anchors(
        n,
        meas_pixels=meas_px,
        meas_is_dip=meas_is_dip,
        meas_intensities=meas_int,
        ref_feat_wl=ref_feat_wl,
        ref_feat_is_dip=ref_is_dip,
        ref_feat_intensities=ref_int_arr,
    )

    if result is not None and len(result.cal_points) >= 4:
        return [(int(p), float(w)) for p, w in result.cal_points]

    return _correlation_fallback(measured_intensity, ref_wl, ref_int)


def _correlation_fallback(
    measured: np.ndarray,
    ref_wl: np.ndarray,
    ref_int: np.ndarray,
) -> list[tuple[int, float]]:
    """Correlation-based calibration when triplet matching fails."""
    n = len(measured)
    if n < 10 or ref_wl.size < 4:
        return []

    span = max(n - 1, 1)
    wl_lo, wl_hi = float(ref_wl.min()), float(ref_wl.max())
    c1_min, c1_max = 200.0 / span, 600.0 / span
    # Allow start wavelength 250-450 nm (Hg starts ~250, FL ~360)
    opt_bounds = [
        (250.0, 450.0),
        (c1_min, c1_max),
        (-5.0, 5.0),
        (-2.0, 2.0),
    ]

    def loss(x: np.ndarray) -> float:
        c0, c1, c2, c3 = x
        p = np.arange(n, dtype=np.float64)
        wl = c0 + c1 * p + c2 * (p * p) / (n * n) + c3 * (p * p * p) / (n * n * n)
        ref_at_wl = np.interp(wl, ref_wl, ref_int)
        if np.std(ref_at_wl) < 1e-9 or np.std(measured) < 1e-9:
            return 1e6
        corr = np.corrcoef(measured, ref_at_wl)[0, 1]
        if np.isnan(corr):
            return 1e6
        linearity_penalty = 0.2 * (c2**2 + c3**2)
        return -corr + linearity_penalty

    disp = 400.0 / span
    starts = [
        np.array([380.0, disp, 0.0, 0.0]),
        np.array([400.0, disp * 0.9, 0.0, 0.0]),
        np.array([360.0, disp * 1.1, 0.0, 0.0]),
    ]

    best_res = None
    best_loss = float("inf")
    for x0 in starts:
        try:
            res = minimize(
                loss,
                x0,
                method="L-BFGS-B",
                bounds=opt_bounds,
                options={"maxiter": 200, "ftol": 1e-8},
            )
            if res.fun < best_loss:
                best_loss = res.fun
                best_res = res
        except Exception:
            continue

    if best_res is None or best_loss > 1e5:
        return []

    c0, c1, c2, c3 = best_res.x
    p = np.arange(n, dtype=np.float64)
    wl_arr = c0 + c1 * p + c2 * (p**2) / (n * n) + c3 * (p**3) / (n * n * n)

    idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    return [(int(i), float(wl_arr[int(i)])) for i in idx]
