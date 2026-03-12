"""Inverse Cauchy fit: pixel → wavelength (prism dispersion model)."""

from __future__ import annotations

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
    _PCHIP_AVAILABLE = True
except ImportError:
    _PCHIP_AVAILABLE = False


def inverse_cauchy(
    pixels: np.ndarray,
    wavelengths: np.ndarray,
    n_pixels: int,
) -> np.ndarray:
    """Fit 1/λ² = poly(pixel), evaluate over [0, n_pixels-1].

    Prism dispersion: wavelength vs pixel is inverse Cauchy.
    Returns strictly monotonic wavelength array.
    """
    px = np.asarray(pixels, dtype=np.float64)
    wl = np.asarray(wavelengths, dtype=np.float64)
    order = np.argsort(px)
    px = px[order]
    wl = wl[order]

    if len(px) < 3:
        return _linear_extrap(px, wl, n_pixels)

    inv_wl_sq = 1.0 / ((wl * 1e-3) ** 2)
    deg = min(2, len(px) - 1)
    try:
        coeffs = np.polyfit(px, inv_wl_sq, deg)
        poly = np.poly1d(coeffs)
    except np.linalg.LinAlgError:
        return _linear_extrap(px, wl, n_pixels)

    pixel_grid = np.arange(n_pixels, dtype=np.float64)
    pred_inv = poly(pixel_grid)
    eps = 1e-12
    if np.any(pred_inv <= eps):
        return _fallback_interp(px, wl, n_pixels)

    full_wl = 1.0 / np.sqrt(pred_inv) * 1000.0
    d = np.diff(full_wl)
    if not (np.all(d > 0) or np.all(d < 0)):
        return _fallback_interp(px, wl, n_pixels)
    # Reject Cauchy if extrapolation is nonsensical (e.g. pixel 0 → 70 nm)
    if full_wl[0] < 350 or full_wl[-1] > 800 or full_wl[0] > full_wl[-1]:
        return _linear_extrap(px, wl, n_pixels)

    return full_wl


def _linear_extrap(px: np.ndarray, wl: np.ndarray, n: int) -> np.ndarray:
    """Linear interpolation with slope extrapolation."""
    out = np.interp(np.arange(n), px, wl)
    p_min, p_max = float(px.min()), float(px.max())
    if len(px) >= 2 and px[1] != px[0]:
        slope_lo = (wl[1] - wl[0]) / (px[1] - px[0])
    else:
        slope_lo = 0.0
    if len(px) >= 2 and px[-1] != px[-2]:
        slope_hi = (wl[-1] - wl[-2]) / (px[-1] - px[-2])
    else:
        slope_hi = 0.0
    grid = np.arange(n, dtype=np.float64)
    mask_lo = grid < p_min
    mask_hi = grid > p_max
    out[mask_lo] = wl[0] + (grid[mask_lo] - px[0]) * slope_lo
    out[mask_hi] = wl[-1] + (grid[mask_hi] - px[-1]) * slope_hi
    return out


def _fallback_interp(px: np.ndarray, wl: np.ndarray, n: int) -> np.ndarray:
    """PCHIP if available, else linear."""
    if _PCHIP_AVAILABLE and len(px) >= 3:
        p_min, p_max = float(px.min()), float(px.max())
        interp = PchipInterpolator(px, wl, extrapolate=False)
        out = _linear_extrap(px, wl, n)
        grid = np.arange(n, dtype=np.float64)
        mask = (grid >= p_min) & (grid <= p_max)
        out[mask] = interp(grid[mask])
        return out
    return _linear_extrap(px, wl, n)


def fit_cal_points(
    cal_points: list[tuple[int, float]],
    n_pixels: int,
) -> np.ndarray:
    """Convert (pixel, wavelength) cal points to full wavelength array."""
    if not cal_points:
        return np.linspace(380, 750, n_pixels)
    px = np.array([p[0] for p in cal_points], dtype=np.float64)
    wl = np.array([p[1] for p in cal_points], dtype=np.float64)
    return inverse_cauchy(px, wl, n_pixels)
