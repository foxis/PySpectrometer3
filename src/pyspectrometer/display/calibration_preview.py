"""Calibration preview: measured SPD, reference SPD, peaks and connecting lines.

Renders an overlay image for the calibration step (similar to autolevel).
Shows measured spectrum, reference spectrum, detected peaks, reference peaks,
and lines connecting matched pairs. User clicks to dismiss and continue.
"""

import numpy as np

from ..data.reference_spectra import (
    ReferenceSource,
    get_reference_peaks,
    get_reference_spectrum,
)
from ..processing.peak_detection import find_peaks
from ..utils.graph_scale import scale_intensity_to_graph


def _wavelength_to_pixel(wavelengths: np.ndarray, wl: float) -> float:
    """Map wavelength to pixel index via linear interpolation."""
    n = len(wavelengths)
    if n < 2:
        return 0.0
    idx = np.searchsorted(wavelengths, wl, side="left")
    if idx <= 0:
        return 0.0
    if idx >= n:
        return float(n - 1)
    w0, w1 = wavelengths[idx - 1], wavelengths[idx]
    if w1 <= w0:
        return float(idx)
    t = (wl - w0) / (w1 - w0)
    return float(idx - 1) + t


def _match_peaks_to_reference(
    measured_peaks: list,
    wavelengths: np.ndarray,
    ref_peaks: list,
    max_wl_diff_nm: float = 40.0,
) -> list[tuple[int, float, float, float]]:
    """Match measured peaks to reference by wavelength proximity.

    Returns list of (measured_pixel, measured_wl, ref_wl, ref_pixel).
    """
    if not measured_peaks or not ref_peaks:
        return []
    matches = []
    used_ref = set()
    for p in measured_peaks:
        pixel = p.index if hasattr(p, "index") else p
        wl = p.wavelength if hasattr(p, "wavelength") else wavelengths[min(pixel, len(wavelengths) - 1)]
        best_ref = None
        best_err = float("inf")
        for r in ref_peaks:
            if r.wavelength in used_ref:
                continue
            err = abs(wl - r.wavelength)
            if err < best_err and err <= max_wl_diff_nm:
                best_err = err
                best_ref = r
        if best_ref is not None:
            ref_px = _wavelength_to_pixel(wavelengths, best_ref.wavelength)
            matches.append((pixel, wl, best_ref.wavelength, ref_px))
            used_ref.add(best_ref.wavelength)
    return matches


def render(
    measured_intensity: np.ndarray,
    wavelengths: np.ndarray,
    source: ReferenceSource,
    width: int,
    height: int,
) -> np.ndarray:
    """Build calibration preview image: measured SPD, reference SPD, peaks, connecting lines.

    Args:
        measured_intensity: Measured spectrum (0-1 or raw, will be normalized)
        wavelengths: Wavelength per pixel from current calibration
        source: Reference light source
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        BGR image (uint8) ready for overlay display
    """
    import cv2

    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    n = len(measured_intensity)
    if n < 2:
        cv2.putText(img, "No spectrum data", (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return img

    # Scale data to fit graph area (leave margin for labels)
    margin_top = 25
    margin_bottom = 20
    graph_h = max(1, height - margin_top - margin_bottom)
    y_base = height - margin_bottom

    # Normalize measured
    meas = np.asarray(measured_intensity, dtype=np.float64)
    m_min, m_max = meas.min(), meas.max()
    if m_max > m_min:
        meas_norm = (meas - m_min) / (m_max - m_min)
    else:
        meas_norm = np.zeros_like(meas)

    # Reference spectrum at same wavelengths
    ref = get_reference_spectrum(source, wavelengths)
    ref = np.asarray(ref, dtype=np.float64)
    r_min, r_max = ref.min(), ref.max()
    if r_max > r_min:
        ref_norm = (ref - r_min) / (r_max - r_min)
    else:
        ref_norm = np.zeros_like(ref)

    # Scale to graph Y
    meas_scaled = scale_intensity_to_graph(meas_norm, graph_h, margin=1)
    ref_scaled = scale_intensity_to_graph(ref_norm, graph_h, margin=1)

    # Resample to width if needed
    if n != width:
        x_old = np.linspace(0, width - 1, num=n, dtype=np.float32)
        x_new = np.arange(width, dtype=np.float32)
        meas_scaled = np.interp(x_new, x_old, meas_scaled.astype(np.float32))
        ref_scaled = np.interp(x_new, x_old, ref_scaled.astype(np.float32))

    # Draw spectra
    def polyline(arr: np.ndarray, color: tuple, thickness: int = 2) -> None:
        pts = []
        for x in range(min(len(arr), width)):
            y_val = float(arr[x])
            sy = y_base - int(min(y_val, graph_h - 1))
            sy = max(margin_top, min(height - margin_bottom, sy))
            pts.append((x, sy))
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

    polyline(meas_scaled, (0, 0, 0), 2)  # Black: measured
    polyline(ref_scaled, (0, 150, 255), 1)  # Orange: reference

    # Peaks
    peaks = find_peaks(
        measured_intensity,
        wavelengths,
        threshold=0.08,
        min_dist=10,
        prominence=0.01,
    )
    ref_peaks = get_reference_peaks(source)
    matches = _match_peaks_to_reference(peaks, wavelengths, ref_peaks)

    # Map pixel to screen x (data spans full width)
    def px_to_sx(px: float) -> int:
        return int(round(px * (width - 1) / max(n - 1, 1)))

    # Draw connecting lines and peak markers
    for mp, mwl, rwl, rpx in matches:
        sx_meas = px_to_sx(mp)
        sx_ref = px_to_sx(rpx)
        if 0 <= sx_meas < width and 0 <= sx_ref < width:
            y_meas = y_base - int(min(meas_scaled[min(sx_meas, len(meas_scaled) - 1)], graph_h - 1))
            y_ref = y_base - int(min(ref_scaled[min(sx_ref, len(ref_scaled) - 1)], graph_h - 1))
            y_meas = max(margin_top, min(height - margin_bottom, y_meas))
            y_ref = max(margin_top, min(height - margin_bottom, y_ref))
            cv2.line(img, (sx_meas, y_meas), (sx_ref, y_ref), (100, 100, 255), 1, cv2.LINE_AA)

    # Vertical lines at measured peaks
    for p in peaks:
        sx = px_to_sx(p.index)
        if 0 <= sx < width:
            cv2.line(img, (sx, margin_top), (sx, height - margin_bottom), (0, 0, 0), 1, cv2.LINE_AA)

    # Vertical lines at reference peaks (mapped to pixel)
    for r in ref_peaks:
        rpx = _wavelength_to_pixel(wavelengths, r.wavelength)
        sx = px_to_sx(rpx)
        if 0 <= sx < width:
            cv2.line(img, (sx, margin_top), (sx, height - margin_bottom), (0, 150, 255), 1, cv2.LINE_AA)

    # Title
    cv2.putText(
        img,
        "Calibration preview: measured (black), reference (orange), matches",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return img
