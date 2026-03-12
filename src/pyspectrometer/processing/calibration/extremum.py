"""Extremum detection: re-export from peak_detection, add reference helpers."""

from __future__ import annotations

import numpy as np

from ..peak_detection import DEFAULT_MAX_EXTREMUMS, extract_extremums
from ...core.spectrum import Extremum

DEFAULT_MAX_COUNT = DEFAULT_MAX_EXTREMUMS


def extract(
    intensity,
    positions,
    *,
    position_px=None,
    max_count=DEFAULT_MAX_COUNT,
) -> list[Extremum]:
    """Thin wrapper: use peak_detection extract_extremums."""
    return extract_extremums(
        intensity,
        positions,
        position_px=position_px,
        max_count=max_count,
    )


def from_known_lines(
    wavelengths: list[float],
    intensities: list[float],
) -> list[Extremum]:
    """Build Extremum list from known spectral lines (reference)."""
    if not wavelengths or not intensities:
        return []
    max_int = max(intensities) if intensities else 1.0
    result: list[Extremum] = []
    for i, (wl, inten) in enumerate(zip(wavelengths, intensities)):
        h_norm = inten / max_int if max_int > 0 else 0.5
        result.append(
            Extremum(
                index=i,
                position=float(wl),
                position_px=None,
                height=h_norm,
                width=5.0,
                is_dip=False,
            )
        )
    return result


def valid_positions(wavelengths: np.ndarray | None, n: int) -> np.ndarray:
    """Valid monotonic wavelength positions or linear 380–750."""
    if wavelengths is None or len(wavelengths) < 2:
        return np.linspace(380, 750, n)
    wl = np.asarray(wavelengths)
    if wl.size != n:
        return np.linspace(380, 750, n)
    ok = bool(np.all(np.diff(wl) > 0) and 300 < wl.min() < wl.max() < 900)
    return wl if ok else np.linspace(380, 750, n)
