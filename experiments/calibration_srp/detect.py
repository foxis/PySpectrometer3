"""Extremum detection: re-export from pyspectrometer, add reference helpers."""

from __future__ import annotations

from pyspectrometer.core.spectrum import Extremum
from pyspectrometer.processing.peak_detection import extract_extremums
from pyspectrometer.processing.peak_detection import DEFAULT_MAX_EXTREMUMS

DEFAULT_MAX_COUNT = DEFAULT_MAX_EXTREMUMS


def extract(
    intensity,
    positions,
    *,
    position_px=None,
    max_count=DEFAULT_MAX_COUNT,
):
    """Thin wrapper: use pyspectrometer extract_extremums."""
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
