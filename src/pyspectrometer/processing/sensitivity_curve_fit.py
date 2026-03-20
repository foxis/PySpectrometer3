"""Derive instrument sensitivity curve from measured vs reference spectrum.

Assumes wavelength calibration is valid. Uses ratio measured/reference, smoothing,
scaling to the datasheet CMOS curve, and edge blending for extrapolation.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def _fade_weights(
    wavelengths: np.ndarray,
    valid: np.ndarray,
    fade_nm: float,
) -> np.ndarray:
    """Weights 1 in core valid region, ramping to 0 toward edges (for CMOS blend)."""
    n = len(wavelengths)
    w = np.zeros(n, dtype=np.float64)
    if not np.any(valid):
        return w

    idx = np.flatnonzero(valid)
    i0, i1 = int(idx[0]), int(idx[-1])
    w[i0 : i1 + 1] = 1.0

    if fade_nm <= 0:
        return w

    wl = wavelengths.astype(np.float64)

    def ramp_left(i_start: int) -> None:
        target = wl[i_start]
        for i in range(i_start - 1, -1, -1):
            dist = target - wl[i]
            if dist >= fade_nm:
                w[i] = 0.0
            else:
                w[i] = max(0.0, 1.0 - dist / fade_nm)

    def ramp_right(i_end: int) -> None:
        target = wl[i_end]
        for i in range(i_end + 1, n):
            dist = wl[i] - target
            if dist >= fade_nm:
                w[i] = 0.0
            else:
                w[i] = max(0.0, 1.0 - dist / fade_nm)

    ramp_left(i0)
    ramp_right(i1)
    return w


def fit_sensitivity_values(
    wavelengths: np.ndarray,
    measured: np.ndarray,
    reference: np.ndarray,
    cmos_at_wl: np.ndarray,
    *,
    ref_threshold_frac: float = 0.08,
    savgol_window: int = 21,
    savgol_poly: int = 3,
    fade_nm: float = 35.0,
) -> np.ndarray:
    """Compute per-pixel sensitivity values aligned to ``wavelengths``.

    ``cmos_at_wl`` is the datasheet curve interpolated to the same grid (for scale
    matching and extrapolation where reference signal is weak).

    Returns:
        Sensitivity array (same length as wavelengths), strictly positive.
    """
    n = min(len(wavelengths), len(measured), len(reference), len(cmos_at_wl))
    if n < 8:
        return np.maximum(cmos_at_wl[:n], 1e-9)

    wl = np.asarray(wavelengths[:n], dtype=np.float64)
    m = np.maximum(np.asarray(measured[:n], dtype=np.float64), 1e-12)
    r = np.maximum(np.asarray(reference[:n], dtype=np.float64), 1e-12)
    cmos = np.maximum(np.asarray(cmos_at_wl[:n], dtype=np.float64), 1e-12)

    raw = m / r

    rmax = float(np.max(r))
    valid = r > (ref_threshold_frac * rmax + 1e-12)

    if np.sum(valid) < 5:
        return cmos.copy()

    max_odd = n if (n % 2 == 1) else n - 1
    window = min(savgol_window, max_odd)
    if window % 2 == 0:
        window -= 1
    window = max(window, savgol_poly + 2)
    if window % 2 == 0:
        window -= 1
    window = min(window, max_odd)
    if window < 3:
        smoothed = raw.copy()
    else:
        smoothed = savgol_filter(raw, window, savgol_poly)

    med = np.median(cmos[valid] / (smoothed[valid] + 1e-12))
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    scaled = smoothed * med
    scaled = np.clip(scaled, 1e-9, None)

    weights = _fade_weights(wl, valid, fade_nm)
    out = weights * scaled + (1.0 - weights) * cmos
    return np.maximum(out, 1e-9)
