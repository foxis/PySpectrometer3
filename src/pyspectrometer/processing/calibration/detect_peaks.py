"""Detect peaks in SPD A (measured) and SPD B (reference).

Single responsibility: peak detection only. No matching logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MeasuredPeak:
    """Peak or dip in measured spectrum (SPD A)."""

    pixel: int
    intensity: float
    width_nm: float
    is_dip: bool = False


@dataclass
class ReferencePeak:
    """Peak or dip in reference spectrum (SPD B)."""

    wavelength: float
    intensity: float
    width_nm: float
    label: str = ""
    is_dip: bool = False


def _find_dips(
    intensity: np.ndarray,
    wavelengths: np.ndarray,
    *,
    min_dist: int = 15,
    prominence: float = 0.02,
) -> list[tuple[int, float, float]]:
    """Detect thin strong dips (local minima). Returns (pixel, depth, width_nm)."""
    try:
        from scipy.signal import find_peaks as scipy_find_peaks
        from scipy.signal import peak_widths as scipy_peak_widths
    except ImportError:
        return []

    arr = np.asarray(intensity, dtype=np.float64)
    if arr.size < 3:
        return []
    rng = float(np.max(arr) - np.min(arr))
    if rng <= 0:
        return []
    inverted = -arr
    height = float(np.min(inverted) + prominence * rng)
    idx, _ = scipy_find_peaks(
        inverted,
        height=height,
        prominence=prominence * rng,
        distance=max(1, min_dist),
    )
    if idx.size == 0:
        return []

    widths_px, _, _, _ = scipy_peak_widths(inverted, idx, rel_height=0.5)
    n = len(wavelengths)
    disp = (wavelengths[-1] - wavelengths[0]) / max(n - 1, 1)
    result = []
    for i, px in enumerate(idx):
        depth = float(np.max(arr) - arr[px])
        w_nm = float(widths_px[i]) * disp if i < len(widths_px) else 0.0
        result.append((int(px), depth, w_nm))
    return result


def detect_peaks_measured(
    intensity: np.ndarray,
    wavelengths: np.ndarray,
    *,
    threshold: float = 0.05,
    min_dist: int = 10,
    prominence: float = 0.005,
    include_dips: bool = True,
) -> list[MeasuredPeak]:
    """Detect peaks and optionally thin strong dips in measured intensity (SPD A)."""
    from ..peak_detection import find_peaks, peak_widths_nm

    peaks = find_peaks(
        intensity,
        wavelengths,
        threshold=threshold,
        min_dist=min_dist,
        prominence=prominence,
    )
    result: list[MeasuredPeak] = []
    if peaks:
        pixels = [p.index for p in peaks]
        widths = peak_widths_nm(intensity, wavelengths, pixels)
        for i, p in enumerate(peaks):
            result.append(
                MeasuredPeak(
                    pixel=p.index,
                    intensity=float(p.intensity),
                    width_nm=widths[i] if i < len(widths) else 0.0,
                    is_dip=False,
                )
            )

    if include_dips:
        dips = _find_dips(intensity, wavelengths, min_dist=min_dist, prominence=0.02)
        for px, depth, w_nm in dips:
            result.append(MeasuredPeak(pixel=px, intensity=depth, width_nm=w_nm, is_dip=True))
        result.sort(key=lambda x: x.pixel)

    return result


def get_reference_peaks(
    source,
    merge_threshold_nm: float = 8.0,
) -> list[ReferencePeak]:
    """Get peaks from reference spectrum (SPD B)."""
    from ...data.reference_spectra import SpectralLine, get_reference_peaks as get_ref

    raw = get_ref(source)
    if len(raw) < 2:
        return [ReferencePeak(r.wavelength, r.intensity, 0.0, r.label) for r in raw]

    sorted_ref = sorted(raw, key=lambda p: p.wavelength)
    merged: list[ReferencePeak] = []
    run: list[SpectralLine] = [sorted_ref[0]]

    for ref in sorted_ref[1:]:
        if ref.wavelength - run[-1].wavelength <= merge_threshold_nm:
            run.append(ref)
        else:
            avg_wl = sum(r.wavelength for r in run) / len(run)
            max_int = max(r.intensity for r in run)
            merged.append(ReferencePeak(avg_wl, max_int, 0.0, run[0].label))
            run = [ref]

    avg_wl = sum(r.wavelength for r in run) / len(run)
    max_int = max(r.intensity for r in run)
    merged.append(ReferencePeak(avg_wl, max_int, 0.0, run[0].label))
    return merged


def reference_peak_widths_from_spectrum(
    ref_spectrum: np.ndarray,
    wavelengths: np.ndarray,
    ref_peaks: list[ReferencePeak],
    rel_height: float = 0.5,
) -> dict[float, float]:
    """Compute FWHM in nm for each reference peak from spectrum curve."""
    if len(ref_spectrum) < 3 or len(wavelengths) < 3:
        return {}

    wl = np.asarray(wavelengths, dtype=np.float64)
    r = np.asarray(ref_spectrum, dtype=np.float64)
    result: dict[float, float] = {}

    for rp in ref_peaks:
        idx = int(np.clip(np.searchsorted(wl, rp.wavelength), 1, len(wl) - 2))
        peak_val = float(np.interp(rp.wavelength, wl, r))
        thresh = peak_val * rel_height
        if peak_val <= 1e-12:
            result[rp.wavelength] = 0.0
            continue
        left_ips = float(idx)
        for i in range(idx, 0, -1):
            if i > 0 and r[i - 1] < thresh <= r[i]:
                t = (thresh - r[i - 1]) / (r[i] - r[i - 1]) if r[i] != r[i - 1] else 0.5
                left_ips = i - 1 + t
                break
        right_ips = float(idx)
        for i in range(idx, len(r) - 1):
            if i + 1 < len(r) and r[i] >= thresh > r[i + 1]:
                t = (thresh - r[i]) / (r[i + 1] - r[i]) if r[i + 1] != r[i] else 0.5
                right_ips = i + t
                break
        wl_left = float(np.interp(left_ips, np.arange(len(wl)), wl))
        wl_right = float(np.interp(right_ips, np.arange(len(wl)), wl))
        result[rp.wavelength] = max(0.0, wl_right - wl_left)

    return result
