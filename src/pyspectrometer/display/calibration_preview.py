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
from ..processing.auto_calibrator import (
    _merge_close_reference_peaks,
    _linearity_score,
    _intensity_similarity,
    _reference_peak_widths_nm,
    _width_similarity,
)
from ..processing.peak_detection import extract_extremums, peak_widths_nm
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
    measured_intensity: np.ndarray,
    ref_spectrum: np.ndarray,
    tolerance_nm: float = 35.0,
    w_linearity: float = 0.60,
    w_intensity: float = 0.20,
    w_width: float = 0.20,
) -> list[tuple[int, float, float, float]]:
    """Match measured peaks to reference by wavelength, intensity, and width similarity.

    Strong measured peaks prefer strong reference peaks; narrow prefers narrow.
    Merges close reference peaks. Returns list of (measured_pixel, measured_wl, ref_wl, ref_pixel).
    """
    if not measured_peaks or not ref_peaks:
        return []
    peak_pixels = [p.index if hasattr(p, "index") else p for p in measured_peaks]
    peak_wl = [
        getattr(p, "wavelength", None) or getattr(p, "position", None)
        or wavelengths[min(p.index, len(wavelengths) - 1)]
        for p in measured_peaks
    ]
    peak_intensities = [
        getattr(p, "intensity", None) or (abs(p.height) if hasattr(p, "height") else 0.0)
        for p in measured_peaks
    ]

    ref_effective = _merge_close_reference_peaks(ref_peaks)
    meas_widths = peak_widths_nm(measured_intensity, wavelengths, peak_pixels)
    ref_width_map = _reference_peak_widths_nm(ref_spectrum, wavelengths, ref_effective)
    ref_sorted = sorted(ref_effective, key=lambda r: r.wavelength)
    max_ref = max(r.intensity for r in ref_sorted) if ref_sorted else 1.0
    norm_ref_map = {
        r.wavelength: r.intensity / max_ref if max_ref > 1e-9 else 0.0
        for r in ref_sorted
    }
    max_meas = max(peak_intensities) if peak_intensities else 1.0
    norm_measured = [x / max_meas if max_meas > 1e-9 else 0.0 for x in peak_intensities]

    order = np.argsort(peak_pixels)
    peak_pixels = [peak_pixels[i] for i in order]
    peak_wl = [peak_wl[i] for i in order]
    norm_measured = [norm_measured[i] for i in order]
    meas_widths = [meas_widths[i] for i in order]

    matches: list[tuple[int, float, float, float]] = []
    used_refs: set[float] = set()

    for i, (pixel, approx_wl) in enumerate(zip(peak_pixels, peak_wl)):
        best_ref = None
        best_score = -1.0

        for ref in ref_sorted:
            if ref.wavelength in used_refs:
                continue
            if abs(approx_wl - ref.wavelength) > tolerance_nm:
                continue

            pts = [(p, rw) for p, _, rw, _ in matches] + [(pixel, ref.wavelength)]
            r2 = _linearity_score(pts)
            score = w_linearity * r2
            score += w_intensity * _intensity_similarity(norm_measured[i], norm_ref_map[ref.wavelength])
            ref_w = ref_width_map.get(ref.wavelength, 0.0)
            score += w_width * _width_similarity(meas_widths[i] if i < len(meas_widths) else 0.0, ref_w)

            if score > best_score:
                best_score = score
                best_ref = ref

        if best_ref is not None:
            ref_px = _wavelength_to_pixel(wavelengths, best_ref.wavelength)
            matches.append((pixel, approx_wl, best_ref.wavelength, ref_px))
            used_refs.add(best_ref.wavelength)

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

    # Extremums (peaks + dips) from canonical extraction
    extremums = extract_extremums(
        measured_intensity,
        wavelengths,
        position_px=np.arange(n, dtype=np.intp),
        max_count=20,
    )
    peaks = [e for e in extremums if not e.is_dip]
    dips = [e for e in extremums if e.is_dip]
    ref_peaks = get_reference_peaks(source)
    matches = _match_peaks_to_reference(
        peaks, wavelengths, ref_peaks, measured_intensity, ref
    )

    # Map pixel to screen x (data spans full width)
    def px_to_sx(px: float) -> int:
        return int(round(px * (width - 1) / max(n - 1, 1)))

    # Draw connecting lines at peak apex (not curve sample)
    for mp, mwl, rwl, rpx in matches:
        sx_meas = px_to_sx(mp)
        sx_ref = px_to_sx(rpx)
        if 0 <= sx_meas < width and 0 <= sx_ref < width:
            apex_val = float(measured_intensity[min(mp, n - 1)])
            m_norm = (apex_val - m_min) / (m_max - m_min) if m_max > m_min else 0.0
            y_meas_apex = float(scale_intensity_to_graph(m_norm, graph_h, margin=1))
            y_meas = y_base - int(min(y_meas_apex, graph_h - 1))

            ref_val = float(np.interp(rwl, wavelengths, ref))
            r_norm = (ref_val - r_min) / (r_max - r_min) if r_max > r_min else 0.0
            y_ref_apex = float(scale_intensity_to_graph(r_norm, graph_h, margin=1))
            y_ref = y_base - int(min(y_ref_apex, graph_h - 1))

            y_meas = max(margin_top, min(height - margin_bottom, y_meas))
            y_ref = max(margin_top, min(height - margin_bottom, y_ref))
            cv2.line(img, (sx_meas, y_meas), (sx_ref, y_ref), (100, 100, 255), 1, cv2.LINE_AA)

    # Vertical lines at measured peaks (black) and dips (red)
    for p in peaks:
        sx = px_to_sx(p.index)
        if 0 <= sx < width:
            cv2.line(img, (sx, margin_top), (sx, height - margin_bottom), (0, 0, 0), 1, cv2.LINE_AA)
    for d in dips:
        sx = px_to_sx(d.index)
        if 0 <= sx < width:
            cv2.line(img, (sx, margin_top), (sx, height - margin_bottom), (0, 0, 255), 1, cv2.LINE_AA)

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
