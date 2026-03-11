"""Automatic wavelength calibration via correlation-based spectrum matching.

Uses a single robust algorithm for both Hg (discrete lines) and FL (continuous)
reference sources: multi-start correlation optimization.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from ..data.reference_spectra import (
    ReferenceSource,
    SpectralLine,
    get_reference_peaks,
    get_reference_spectrum,
)

if TYPE_CHECKING:
    from .sensitivity_correction import SensitivityCorrection

DISPERSION_LO, DISPERSION_HI = 0.2, 0.5

# Peaks within this separation (nm) are merged to their average for matching.
# Limited spectrometer resolution may join or lose close peaks.
PEAK_MERGE_THRESHOLD_NM = 8.0

# Wavelength bounds by source: (start_lo, start_hi), (end_lo, end_hi)
# Allow shorter spectrometers (e.g. 380-600 nm) to calibrate.
_WL_BOUNDS: dict[ReferenceSource, tuple[tuple[float, float], tuple[float, float]]] = {
    ReferenceSource.HG: ((250.0, 320.0), (550.0, 780.0)),
    ReferenceSource.FL1: ((360.0, 440.0), (550.0, 780.0)),
    ReferenceSource.FL2: ((360.0, 440.0), (550.0, 780.0)),
    ReferenceSource.FL3: ((360.0, 440.0), (550.0, 780.0)),
    ReferenceSource.FL12: ((360.0, 440.0), (550.0, 780.0)),
    ReferenceSource.D65: ((360.0, 440.0), (550.0, 780.0)),
    ReferenceSource.LED: ((360.0, 440.0), (550.0, 780.0)),
    ReferenceSource.LED2: ((360.0, 440.0), (550.0, 780.0)),
    ReferenceSource.LED3: ((360.0, 440.0), (550.0, 780.0)),
}
_DEFAULT_BOUNDS = ((360.0, 440.0), (550.0, 780.0))


def _dispersion_score(pts: list[tuple[int, float]]) -> float:
    """Prefer dispersion in typical range. Returns 1 if ideal, 0 if far off."""
    if len(pts) < 2:
        return 1.0
    px_span = pts[-1][0] - pts[0][0]
    wl_span = pts[-1][1] - pts[0][1]
    if px_span <= 0:
        return 0.0
    disp = wl_span / px_span
    if DISPERSION_LO <= disp <= DISPERSION_HI:
        return 1.0
    d = min(abs(disp - DISPERSION_LO), abs(disp - DISPERSION_HI))
    return max(0.0, 1.0 - d / 0.2)


def _linearity_score(points: list[tuple[int, float]]) -> float:
    """R² of linear fit pixel→wavelength. Higher = more linear."""
    if len(points) < 2:
        return 1.0
    px = np.array([p[0] for p in points], dtype=np.float64)
    wl = np.array([p[1] for p in points], dtype=np.float64)
    ss_tot = np.sum((wl - wl.mean()) ** 2)
    if ss_tot < 1e-12:
        return 1.0
    slope, intercept = np.polyfit(px, wl, 1)
    y_pred = slope * px + intercept
    ss_res = np.sum((wl - y_pred) ** 2)
    return 1.0 - ss_res / ss_tot


def _merge_close_reference_peaks(
    reference_peaks: list[SpectralLine],
    min_separation_nm: float = PEAK_MERGE_THRESHOLD_NM,
) -> list[SpectralLine]:
    """Merge reference peaks that are too close for spectrometer resolution.

    When peaks are within min_separation_nm, the spectrometer may see one merged
    peak. Replace each such group with a single peak at their average wavelength.
    """
    if len(reference_peaks) < 2:
        return reference_peaks.copy()

    ref_sorted = sorted(reference_peaks, key=lambda p: p.wavelength)
    merged: list[SpectralLine] = []
    run: list[SpectralLine] = [ref_sorted[0]]

    for ref in ref_sorted[1:]:
        prev_wl = run[-1].wavelength
        if ref.wavelength - prev_wl <= min_separation_nm:
            run.append(ref)
        else:
            avg_wl = sum(r.wavelength for r in run) / len(run)
            max_int = max(r.intensity for r in run)
            merged.append(SpectralLine(avg_wl, max_int, run[0].label))
            run = [ref]

    avg_wl = sum(r.wavelength for r in run) / len(run)
    max_int = max(r.intensity for r in run)
    merged.append(SpectralLine(avg_wl, max_int, run[0].label))
    return merged


def _extract_peak_intensities(
    peak_indices: list,
    pixel_indices: list[int],
    measured_intensity: np.ndarray,
) -> list[float] | None:
    """Extract intensity at each peak. Returns None if all zero."""
    out = []
    for p, idx in zip(peak_indices, pixel_indices):
        if hasattr(p, "intensity"):
            out.append(float(p.intensity))
        elif 0 <= idx < len(measured_intensity):
            out.append(float(measured_intensity[idx]))
        else:
            out.append(0.0)
    return out if out and max(out) > 1e-9 else None


def _intensity_similarity(
    norm_measured: float,
    norm_ref: float,
) -> float:
    """Similarity of normalized intensities (0-1). 1 = identical, 0 = maximally different."""
    return 1.0 - abs(norm_measured - norm_ref)


def _match_peaks_to_reference(
    peak_pixels: list[int],
    peak_wavelengths: list[float],
    reference_peaks: list[SpectralLine],
    peak_intensities: list[float] | None = None,
) -> list[tuple[int, float, float]]:
    """Match detected peaks to reference spectral lines.

    Reference peaks within PEAK_MERGE_THRESHOLD_NM are merged to their average
    wavelength, since limited spectrometer resolution may join or lose close peaks.
    Prefers matches that yield linear pixel→wavelength fit and similar intensity
    (strong measured peak ↔ strong reference peak).
    """
    order = np.argsort(peak_pixels)
    peak_pixels = [peak_pixels[i] for i in order]
    peak_wavelengths = [peak_wavelengths[i] for i in order]
    if peak_intensities is not None:
        peak_intensities = [peak_intensities[i] for i in order]
        max_meas = max(peak_intensities) if peak_intensities else 1.0
        norm_measured = [x / max_meas if max_meas > 1e-9 else 0.0 for x in peak_intensities]
    else:
        norm_measured = None

    ref_effective = _merge_close_reference_peaks(reference_peaks)
    ref_sorted = sorted(ref_effective, key=lambda p: p.wavelength)
    max_ref = max(r.intensity for r in ref_sorted) if ref_sorted else 1.0
    norm_ref_map = {
        r.wavelength: r.intensity / max_ref if max_ref > 1e-9 else 0.0
        for r in ref_sorted
    }
    tolerance_nm = 35
    w_linearity = 0.75
    w_intensity = 0.25

    matches: list[tuple[int, float, float]] = []
    used_refs: set[float] = set()

    for i, (pixel, approx_wl) in enumerate(zip(peak_pixels, peak_wavelengths)):
        best_match = None
        best_score = -1.0

        for ref in ref_sorted:
            if ref.wavelength in used_refs:
                continue
            error = abs(approx_wl - ref.wavelength)
            if error > tolerance_nm:
                continue

            pts = [(p, w) for p, w, _ in matches] + [(pixel, ref.wavelength)]
            r2 = _linearity_score(pts)
            score = w_linearity * r2
            if norm_measured is not None:
                int_sim = _intensity_similarity(norm_measured[i], norm_ref_map[ref.wavelength])
                score += w_intensity * int_sim
            else:
                score += w_intensity  # no penalty when intensities unknown

            if score > best_score:
                best_score = score
                best_match = ref

        if best_match is not None:
            err = abs(approx_wl - best_match.wavelength)
            matches.append((pixel, best_match.wavelength, err))
            used_refs.add(best_match.wavelength)

    matches.sort(key=lambda m: m[0])
    return matches


def _correlation_calibrate(
    measured_intensity: np.ndarray,
    source: ReferenceSource,
    sensitivity: "SensitivityCorrection | None",
    sensitivity_enabled: bool,
) -> list[tuple[int, float]]:
    """Robust correlation-based calibration. Works for Hg and FL sources.

    Multi-start optimization with source-adaptive wavelength bounds.
    """
    n = len(measured_intensity)
    if n < 10:
        print("[Calibration] Need at least 10 pixels for correlation calibration")
        return []

    span = max(n - 1, 1)
    (wl_start_lo, wl_start_hi), (wl_end_lo, wl_end_hi) = _WL_BOUNDS.get(
        source, _DEFAULT_BOUNDS
    )
    c1_min, c1_max = 200.0 / span, 600.0 / span
    opt_bounds = [
        (wl_start_lo, wl_start_hi),
        (c1_min, c1_max),
        (-5.0, 5.0),
        (-2.0, 2.0),
    ]

    def loss(x: np.ndarray) -> float:
        c0, c1, c2, c3 = x
        p = np.arange(n, dtype=np.float64)
        wl = c0 + c1 * p + c2 * (p * p) / (n * n) + c3 * (p * p * p) / (n * n * n)
        wl_0, wl_n = wl[0], wl[-1]
        range_penalty = 0.0
        if wl_0 > wl_start_hi or wl_0 < wl_start_lo:
            range_penalty += 100.0 * min(
                abs(wl_0 - wl_start_lo), abs(wl_0 - wl_start_hi)
            )
        if wl_n < wl_end_lo or wl_n > wl_end_hi:
            range_penalty += 100.0 * min(
                abs(wl_n - wl_end_lo), abs(wl_n - wl_end_hi)
            )
        if wl_0 >= wl_n - 50:
            range_penalty += 1e6
        ref = get_reference_spectrum(source, wl)
        if sensitivity_enabled and sensitivity is not None:
            sens = sensitivity.interpolate(wl)
            if sens is not None:
                ref = ref * np.maximum(sens, 1e-9)
        if ref.size < 2 or np.std(ref) < 1e-9 or np.std(measured_intensity) < 1e-9:
            return 1e6
        corr = np.corrcoef(measured_intensity, ref)[0, 1]
        if np.isnan(corr):
            return 1e6
        # Light penalty for extreme non-linearity; prism dispersion is non-linear
        linearity_penalty = 0.2 * (c2**2 + c3**2)
        return -corr + linearity_penalty + range_penalty

    # Multi-start: different initial points for robustness
    disp_nm_per_px = 400.0 / span  # typical 0.2-0.5 nm/px
    starts = [
        np.array([380.0, disp_nm_per_px, 0.0, 0.0]),
        np.array([400.0, disp_nm_per_px * 0.9, 0.0, 0.0]),
        np.array([360.0, disp_nm_per_px * 1.1, 0.0, 0.0]),
        np.array([390.0, 350.0 / span, 0.01, 0.0]),  # slight curvature
    ]
    if source == ReferenceSource.HG:
        starts.append(np.array([267.0, 500.0 / span, 0.0, 0.0]))

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

    if best_res is None:
        print("[Calibration] Correlation optimization failed: no result")
        return []
    if not best_res.success and best_loss > 1e5:
        msg = best_res.message if best_res else "no result"
        print(f"[Calibration] Correlation optimization failed: {msg}")
        return []
    if not best_res.success:
        print(f"[Calibration] Optimization warning: {best_res.message} (using best result)")

    c0, c1, c2, c3 = best_res.x
    pixels_arr = np.arange(n, dtype=np.float64)
    wl_arr = (
        c0
        + c1 * pixels_arr
        + c2 * (pixels_arr**2) / (n * n)
        + c3 * (pixels_arr**3) / (n * n * n)
    )

    idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    points = [(int(i), float(wl_arr[int(i)])) for i in idx]
    ref_final = get_reference_spectrum(source, wl_arr)
    if sensitivity_enabled and sensitivity is not None:
        sens = sensitivity.interpolate(wl_arr)
        if sens is not None:
            ref_final = ref_final * np.maximum(sens, 1e-9)
    corr = (
        np.corrcoef(measured_intensity, ref_final)[0, 1]
        if np.std(ref_final) > 1e-9
        else 0.0
    )
    print(f"[Calibration] Correlation calibration: r={corr:.4f} ({points[0][1]:.1f}-{points[-1][1]:.1f} nm)")
    if corr < 0.3:
        print("[Calibration] Low correlation - check spectrum quality, reference source, and LEVEL")
    return points


class AutoCalibrator:
    """Performs automatic wavelength calibration via correlation-based spectrum matching."""

    def calibrate(
        self,
        measured_intensity: np.ndarray,
        wavelengths: np.ndarray,
        peak_indices: list,
        source: ReferenceSource,
        sensitivity: "SensitivityCorrection | None" = None,
        sensitivity_enabled: bool = False,
    ) -> list[tuple[int, float]]:
        """Auto-calibrate via SPD correlation. All references use spectral power distribution."""
        return _correlation_calibrate(
            measured_intensity, source, sensitivity, sensitivity_enabled
        )

    def calibrate_from_csv(
        self,
        csv_path: str | Path,
        source: ReferenceSource = ReferenceSource.FL12,
    ) -> list[tuple[int, float]]:
        """Fit pixels to wavelengths from CSV intensity using reference spectrum correlation.

        Loads CSV (Pixel,Wavelength,Intensity), ignores wavelengths, uses intensity only.
        Correlates measured intensity with reference spectrum to derive pixel→wavelength mapping.

        Args:
            csv_path: Path to CSV file
            source: Reference illuminant (default FL12)

        Returns:
            List of (pixel, wavelength) calibration points
        """
        path = Path(csv_path)
        if not path.exists():
            print(f"[Calibration] CSV not found: {path}")
            return []

        pixels_list: list[int] = []
        intensity_list: list[float] = []
        intensity_col = 2  # Default: Pixel,Wavelength,Intensity

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if line.lower().startswith("pixel"):
                    intensity_col = 1  # pixel,intensity,reference_wavelength,...
                    continue
                if line.startswith("Pixel") and "Wavelength" in line:
                    continue
                if len(parts) >= max(2, intensity_col + 1):
                    try:
                        px = int(parts[0])
                        intensity = float(parts[intensity_col])
                        pixels_list.append(px)
                        intensity_list.append(intensity)
                    except (ValueError, IndexError):
                        continue

        if len(intensity_list) < 10:
            print(f"[Calibration] Need at least 10 rows, got {len(intensity_list)}")
            return []

        pairs = sorted(zip(pixels_list, intensity_list), key=lambda x: x[0])
        n = pairs[-1][0] + 1 if pairs else 0
        measured = np.zeros(n, dtype=np.float64)
        for px, intensity in pairs:
            measured[px] = intensity
        return _correlation_calibrate(
            measured_intensity=measured,
            source=source,
            sensitivity=None,
            sensitivity_enabled=False,
        )

    def calibrate_from_peaks(
        self,
        measured_intensity: np.ndarray,
        wavelengths: np.ndarray,
        peak_indices: list[int],
        source: ReferenceSource,
    ) -> list[tuple[int, float]]:
        """Manual calibration by matching detected peaks to reference lines."""
        reference_peaks = get_reference_peaks(source)

        if len(peak_indices) < 4:
            print(f"[Calibration] Need at least 4 peaks, found {len(peak_indices)}")
            return []

        if len(reference_peaks) < 4:
            print(f"[Calibration] Reference has only {len(reference_peaks)} peaks")
            return []

        pixel_indices = [p.index if hasattr(p, "index") else p for p in peak_indices]
        peak_wavelengths = [wavelengths[min(idx, len(wavelengths) - 1)] for idx in pixel_indices]
        peak_intensities = _extract_peak_intensities(
            peak_indices, pixel_indices, measured_intensity
        )
        matches = _match_peaks_to_reference(
            pixel_indices,
            peak_wavelengths,
            reference_peaks,
            peak_intensities=peak_intensities,
        )

        if len(matches) < 4:
            print(f"[Calibration] Could only match {len(matches)} peaks, need 4")
            return []

        matches = sorted(matches, key=lambda m: m[2])[: min(6, len(matches))]
        return [(m[0], m[1]) for m in matches]
