"""Illumination-only metrics: CCT (correlated colour temperature) and CRI (Ra).

Used in Color Science mode when measurement type is Illumination.
Both rely on colour-science; return None when unavailable or out of range.
"""

from __future__ import annotations

import numpy as np

try:
    import colour
    _COLOUR = True
except ImportError:
    _COLOUR = False


def cct_from_xyz(X: float, Y: float, Z: float) -> tuple[float, float] | None:
    """Correlated colour temperature (K) and Δuv from CIE XYZ.

    Uses Ohno 2013 (colour.temperature.XYZ_to_CCT_Ohno2013). XYZ scale 0–100.
    Returns None if colour is unavailable or chromaticity is not near the Planckian locus.
    """
    if not _COLOUR:
        return None
    try:
        if Y < 1e-10:
            return None
        xyz = np.array([X, Y, Z], dtype=float) / 100.0
        out = colour.temperature.XYZ_to_CCT_Ohno2013(xyz)
        cct_f = float(out[0])
        duv_f = float(out[1])
        if cct_f < 1000 or cct_f > 100_000:
            return None
        return (cct_f, duv_f)
    except (ValueError, TypeError, KeyError):
        return None


def cri_from_spectrum(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
) -> float | None:
    """Colour Rendering Index (CRI Ra) from a relative spectral power distribution.

    Builds a colour SpectralDistribution from (wavelengths, spectrum) and
    calls colour.colour_rendering_index. Returns None if colour is unavailable,
    data is invalid, or CRI cannot be computed (e.g. outside CCT range for CRI).
    """
    if not _COLOUR:
        return None
    if spectrum is None or wavelengths is None or len(spectrum) < 2:
        return None
    try:
        wl = np.asarray(wavelengths, dtype=float)
        sp = np.maximum(np.asarray(spectrum, dtype=float), 0.0)
        total = float(np.trapezoid(sp, wl))
        if total < 1e-12:
            return None
        sp_norm = sp / total
        data = dict(zip(wl.tolist(), sp_norm.tolist()))
        sd = colour.SpectralDistribution(data)
        ra = colour.colour_rendering_index(sd, additional_data=False)
        return float(ra)
    except (ValueError, TypeError, KeyError):
        return None
