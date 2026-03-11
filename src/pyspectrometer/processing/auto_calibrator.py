"""Automatic wavelength calibration via peak matching and correlation."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from ..data.reference_spectra import (
    ReferenceSource,
    SpectralLine,
    get_reference_name,
    get_reference_peaks,
    get_reference_spectrum,
)

if TYPE_CHECKING:
    from .sensitivity_correction import SensitivityCorrection

DISPERSION_LO, DISPERSION_HI = 0.2, 0.5


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


def _match_peaks_to_reference(
    peak_pixels: list[int],
    peak_wavelengths: list[float],
    reference_peaks: list[SpectralLine],
) -> list[tuple[int, float, float]]:
    """Match detected peaks to reference spectral lines.

    Prefers matches that yield the most linear pixel→wavelength fit.
    """
    order = np.argsort(peak_pixels)
    peak_pixels = [peak_pixels[i] for i in order]
    peak_wavelengths = [peak_wavelengths[i] for i in order]

    ref_sorted = sorted(reference_peaks, key=lambda p: p.wavelength)
    tolerance_nm = 35

    matches: list[tuple[int, float, float]] = []
    used_refs: set[float] = set()

    for pixel, approx_wl in zip(peak_pixels, peak_wavelengths):
        best_match = None
        best_r2 = -1.0

        for ref in ref_sorted:
            if ref.wavelength in used_refs:
                continue
            error = abs(approx_wl - ref.wavelength)
            if error > tolerance_nm:
                continue

            pts = [(p, w) for p, w, _ in matches] + [(pixel, ref.wavelength)]
            r2 = _linearity_score(pts)
            if r2 > best_r2:
                best_r2 = r2
                best_match = ref

        if best_match is not None:
            err = abs(approx_wl - best_match.wavelength)
            matches.append((pixel, best_match.wavelength, err))
            used_refs.add(best_match.wavelength)

    matches.sort(key=lambda m: m[0])
    return matches


class AutoCalibrator:
    """Performs automatic wavelength calibration via peak matching or correlation."""

    def calibrate(
        self,
        measured_intensity: np.ndarray,
        wavelengths: np.ndarray,
        peak_indices: list,
        source: ReferenceSource,
        sensitivity: "SensitivityCorrection | None" = None,
        sensitivity_enabled: bool = False,
    ) -> list[tuple[int, float]]:
        """Auto-calibrate. Tries peak matching first, falls back to correlation."""
        ref_peaks = get_reference_peaks(source)
        if len(ref_peaks) < 3:
            print(
                f"[Calibration] Reference '{get_reference_name(source)}' "
                f"has {len(ref_peaks)} peaks; use Hg or FL for auto-calibration"
            )
            return []

        result = self._peak_ordered(peak_indices, source)
        if result:
            return result
        return self._correlation(measured_intensity, source, sensitivity, sensitivity_enabled)

    def _peak_ordered(
        self,
        peak_indices: list,
        source: ReferenceSource,
    ) -> list[tuple[int, float]]:
        """Calibrate by matching peaks by order."""
        peak_pixels = [p.index if hasattr(p, "index") else int(p) for p in peak_indices]
        peak_pixels = sorted(peak_pixels)

        ref_peaks = get_reference_peaks(source)
        ref_sorted = sorted(ref_peaks, key=lambda p: p.wavelength)

        min_required = 3
        n_ref = len(ref_sorted)
        n_meas = len(peak_pixels)
        if n_ref < min_required or n_meas < min_required:
            print(
                f"[Calibration] Peak matching: need {min_required}+ peaks, "
                f"have {n_meas} measured, {n_ref} reference"
            )
            return []

        n_use = min(n_ref, n_meas)
        best_points: list[tuple[int, float]] = []
        best_r2 = -1.0
        best_dispersion_score = -1.0

        meas_shift_max = max(0, n_meas - n_use)
        ref_offset_max = max(0, n_ref - n_use)
        for m_shift in range(meas_shift_max + 1):
            for r_off in range(ref_offset_max + 1):
                pts = [
                    (peak_pixels[m_shift + i], ref_sorted[r_off + i].wavelength)
                    for i in range(n_use)
                ]
                r2 = _linearity_score(pts)
                dsc = _dispersion_score(pts)
                if r2 > best_r2 or (r2 == best_r2 and dsc > best_dispersion_score):
                    best_r2 = r2
                    best_dispersion_score = dsc
                    best_points = pts

        points = best_points[: min(6, len(best_points))]
        print(f"[Calibration] Peak-ordered calibration: {len(points)} points")
        return points

    def _correlation(
        self,
        measured_intensity: np.ndarray,
        source: ReferenceSource,
        sensitivity: "SensitivityCorrection | None",
        sensitivity_enabled: bool,
    ) -> list[tuple[int, float]]:
        """Optimize calibration polynomial to maximize spectrum correlation."""
        n = len(measured_intensity)
        if n < 10:
            print("[Calibration] Need at least 10 pixels for correlation calibration")
            return []

        span = max(n - 1, 1)
        c1_init = 370.0 / span
        c0_init = 380.0
        x0 = np.array([c0_init, c1_init, 0.0, 0.0])

        wl_start_lo, wl_start_hi = 360.0, 440.0
        wl_end_lo, wl_end_hi = 660.0, 780.0

        def loss(x: np.ndarray) -> float:
            c0, c1, c2, c3 = x
            p = np.arange(n, dtype=np.float64)
            wl = c0 + c1 * p + c2 * (p * p) / (n * n) + c3 * (p * p * p) / (n * n * n)
            wl_0 = wl[0]
            wl_n = wl[-1]
            range_penalty = 0.0
            if wl_0 > wl_start_hi or wl_0 < wl_start_lo:
                range_penalty += 100.0 * min(abs(wl_0 - wl_start_lo), abs(wl_0 - wl_start_hi))
            if wl_n < wl_end_lo or wl_n > wl_end_hi:
                range_penalty += 100.0 * min(abs(wl_n - wl_end_lo), abs(wl_n - wl_end_hi))
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
            linearity_penalty = 2.0 * (c2**2 + c3**2)
            return -corr + linearity_penalty + range_penalty

        c1_min = 250.0 / span
        c1_max = 500.0 / span
        bounds = [
            (wl_start_lo, wl_start_hi),
            (c1_min, c1_max),
            (-15.0, 15.0),
            (-15.0, 15.0),
        ]
        res = minimize(
            loss,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 60, "ftol": 1e-6},
        )

        if not res.success:
            print(f"[Calibration] Correlation optimization failed: {res.message}")
            return []

        c0, c1, c2, c3 = res.x
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
        corr = np.corrcoef(measured_intensity, ref_final)[0, 1] if np.std(ref_final) > 1e-9 else 0
        print(f"[Calibration] Correlation calibration: r={corr:.4f}")
        return points

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

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
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
        return self._correlation(
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

        matches = _match_peaks_to_reference(pixel_indices, peak_wavelengths, reference_peaks)

        if len(matches) < 4:
            print(f"[Calibration] Could only match {len(matches)} peaks, need 4")
            return []

        matches = sorted(matches, key=lambda m: m[2])[: min(6, len(matches))]
        return [(m[0], m[1]) for m in matches]
