"""Standalone wavelength calibration.

Accepts measured spectrum (pixels, intensity) and reference SPD (nm, intensity).
Returns calibration points (pixel, wavelength). No CSV, no reference data import.
Uses calibration module for peak/extremum detection, matching, and fit.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .calibration import calibrate as triplet_calibrate


def calibrate(
    measured_intensity: np.ndarray,
    reference_wl: np.ndarray,
    reference_intensity: np.ndarray,
    *,
    max_extremums: int = 15,
) -> list[tuple[int, float]]:
    """Calibrate pixels to wavelengths from measured and reference spectra.

    Standalone: pass measured intensity (per pixel) and reference SPD (nm, intensity).
    Uses triplet-based matching from calibration module. Falls back to correlation
    if triplet fails.

    Args:
        measured_intensity: Intensity per pixel (0..n-1)
        reference_wl: Reference wavelengths (nm)
        reference_intensity: Reference SPD at those wavelengths
        max_extremums: Max peaks/dips for triplet matching

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

    # Initial positions: linear 380–750 nm over pixels (uncalibrated guess)
    positions = np.linspace(380.0, 750.0, n, dtype=np.float64)

    # Triplet-based calibration (uses calibration module: extract, match, fit)
    result = triplet_calibrate(
        measured_intensity,
        positions,
        ref_wl.tolist(),
        ref_int.tolist(),
        max_extremums=max_extremums,
    )

    if result is not None and len(result.cal_points) >= 4:
        return [(int(p), float(w)) for p, w in result.cal_points]

    # Fallback: correlation when triplet yields <4 points
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
