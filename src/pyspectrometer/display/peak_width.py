"""Peak/dip width (FWHM) and local-extremum detection for label display.

Used by PeaksRenderer and MarkersRenderer to show width below the wavelength
label when the marker is on a peak (local maximum).
"""

import numpy as np


def is_local_max(index: int, intensity: np.ndarray) -> bool:
    """True if intensity[index] is a local maximum (both neighbors lower)."""
    n = len(intensity)
    if n < 3 or index <= 0 or index >= n - 1:
        return False
    c = float(intensity[index])
    return c >= float(intensity[index - 1]) and c >= float(intensity[index + 1])


def fwhm_nm(
    index: int,
    intensity: np.ndarray,
    wavelengths: np.ndarray,
) -> float | None:
    """Full-width at half maximum in nm at the given peak index.

    Returns None if index is not a local max or FWHM cannot be computed
    (e.g. peak too flat or at edge).
    """
    n = len(intensity)
    if n < 3 or index <= 0 or index >= n - 1:
        return None
    if not is_local_max(index, intensity):
        return None
    peak_val = float(intensity[index])
    half = peak_val * 0.5
    if half <= 0:
        return None

    # Left crossing: first index left of peak where intensity <= half
    left_idx = index
    for i in range(index - 1, -1, -1):
        if intensity[i] <= half:
            left_idx = i
            break
    else:
        return None
    # Right crossing
    right_idx = index
    for i in range(index + 1, n):
        if intensity[i] <= half:
            right_idx = i
            break
    else:
        return None

    # Linear interpolation at half-max for finer wavelength
    def interp_wl(i_lo: int, i_hi: int) -> float:
        if i_lo == i_hi:
            return float(wavelengths[i_lo])
        y_lo = float(intensity[i_lo])
        y_hi = float(intensity[i_hi])
        if abs(y_hi - y_lo) < 1e-12:
            return float(wavelengths[i_lo])
        t = (half - y_lo) / (y_hi - y_lo)
        t = max(0.0, min(1.0, t))
        return float(wavelengths[i_lo]) + t * (float(wavelengths[i_hi]) - float(wavelengths[i_lo]))

    wl_left = interp_wl(left_idx, left_idx + 1) if left_idx < index else float(wavelengths[left_idx])
    wl_right = interp_wl(right_idx - 1, right_idx) if right_idx > index else float(wavelengths[right_idx])
    return abs(wl_right - wl_left)
