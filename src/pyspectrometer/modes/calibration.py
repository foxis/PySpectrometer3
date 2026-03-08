"""Calibration mode for wavelength calibration.

Workflow:
1. Select reference source (FL, Hg, D65, LED)
2. Point spectrometer at light source
3. Use auto-level and auto-shift to fit spectrum in view
4. Freeze spectrum and click AutoCal
5. Correlation-based optimization finds polynomial that maximizes spectrum
   correlation with reference, regularized for linearity (prism dispersion)
6. Verify overlay aligns with measured spectrum
7. Save calibration manually when satisfied

Manual calibration (peak matching) via calibrate_from_peaks() for later use.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import numpy as np
from scipy.optimize import minimize

from .base import BaseMode, ModeType, ButtonDefinition
from ..data.reference_spectra import (
    ReferenceSource,
    get_reference_spectrum,
    get_reference_peaks,
    get_reference_name,
    SpectralLine,
)

# Data directory for CMOS sensitivity curve (up 3 levels from modes/calibration.py)
_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"


@dataclass
class CalibrationState:
    """State specific to calibration mode."""
    
    # Current reference source
    current_source: ReferenceSource = ReferenceSource.FL
    
    # Overlay visibility
    overlay_visible: bool = True
    
    # Sensitivity correction enabled
    sensitivity_correction_enabled: bool = False
    
    # Detected peaks from measured spectrum
    detected_peaks: list[int] = field(default_factory=list)  # pixel positions
    
    # Matched calibration points (pixel, wavelength)
    calibration_points: list[tuple[int, float]] = field(default_factory=list)
    
    # Y-axis shift (for auto-center)
    y_shift: int = 0
    
    # Frozen spectrum for calibration
    frozen_intensity: Optional[np.ndarray] = None


class CalibrationMode(BaseMode):
    """Calibration mode for wavelength calibration."""
    
    # Reference sources in cycle order
    SOURCES = [
        ReferenceSource.FL,
        ReferenceSource.FL2,
        ReferenceSource.FL3,
        ReferenceSource.FL4,
        ReferenceSource.FL5,
        ReferenceSource.A,
        ReferenceSource.D65,
        ReferenceSource.HG,
        ReferenceSource.LED,
        ReferenceSource.LED2,
        ReferenceSource.LED3,
    ]
    
    def __init__(self):
        """Initialize calibration mode."""
        super().__init__()
        self.cal_state = CalibrationState()
        
        # Enable auto-gain by default
        self.state.auto_gain_enabled = True
        
        # Load CMOS sensitivity curve
        self._cmos_sensitivity_wl: Optional[np.ndarray] = None
        self._cmos_sensitivity_val: Optional[np.ndarray] = None
        self._load_cmos_sensitivity()
        
        # Callbacks to be set by spectrometer
        self._on_save_calibration: Optional[Callable[[], None]] = None
        self._on_load_calibration: Optional[Callable[[], None]] = None
        self._on_auto_calibrate: Optional[Callable[[list[tuple[int, float]]], None]] = None
        self._on_gain_change: Optional[Callable[[float], None]] = None
        self._on_y_shift_change: Optional[Callable[[int], None]] = None
    
    @property
    def mode_type(self) -> ModeType:
        return ModeType.CALIBRATION
    
    @property
    def name(self) -> str:
        return "Calibration"
    
    def get_buttons(self) -> list[ButtonDefinition]:
        """Get calibration mode buttons."""
        return [
            # Row 1: Source selection and calibration actions
            ButtonDefinition("FL", "source_fl", row=1),
            ButtonDefinition("Hg", "source_hg", row=1),
            ButtonDefinition("D65", "source_d65", row=1),
            ButtonDefinition("LED1", "source_led", row=1),
            ButtonDefinition("LED2", "source_led2", row=1),
            ButtonDefinition("LED3", "source_led3", row=1),
            ButtonDefinition("AutoLvl", "auto_level", is_toggle=True, row=1),
            ButtonDefinition("AutoCal", "auto_calibrate", row=1),
            ButtonDefinition("Save", "save_cal", shortcut="w", row=1),
            ButtonDefinition("Load", "load_cal", row=1),
            
            # Row 2: Display and control
            ButtonDefinition("Freeze", "freeze", is_toggle=True, shortcut="f", row=2),
            ButtonDefinition("Overlay", "toggle_overlay", is_toggle=True, row=2),
            ButtonDefinition("AutoG", "auto_gain", is_toggle=True, row=2),
            ButtonDefinition("Gain+", "gain_up", shortcut="t", row=2),
            ButtonDefinition("Gain-", "gain_down", shortcut="g", row=2),
            ButtonDefinition("Clear", "clear_points", shortcut="x", row=2),
            ButtonDefinition("Quit", "quit", shortcut="q", row=2),
        ]
    
    def process_spectrum(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Process spectrum - apply freeze if enabled."""
        # If frozen, return the frozen spectrum
        if self.state.frozen and self.cal_state.frozen_intensity is not None:
            return self.cal_state.frozen_intensity
        
        # Apply averaging if enabled
        if self.state.averaging_enabled:
            intensity = self.accumulate_spectrum(intensity)
        
        # Store for freeze
        self.cal_state.frozen_intensity = intensity.copy()
        
        return intensity
    
    def get_overlay(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> Optional[tuple[np.ndarray, tuple[int, int, int]]]:
        """Get reference spectrum overlay."""
        if not self.cal_state.overlay_visible:
            return None
        
        # Generate reference spectrum
        ref_intensity = get_reference_spectrum(
            self.cal_state.current_source,
            wavelengths,
        )
        
        # Scale to graph height
        scaled = ref_intensity * (graph_height - 10)
        
        # Color based on source
        colors = {
            ReferenceSource.FL: (255, 200, 0),     # Yellow
            ReferenceSource.HG: (255, 0, 255),     # Magenta
            ReferenceSource.D65: (100, 200, 255),  # Daylight blue
            ReferenceSource.LED: (200, 200, 200),  # White/gray
            ReferenceSource.LED2: (180, 220, 220), # Light cyan
            ReferenceSource.LED3: (220, 200, 220), # Light purple
        }
        color = colors.get(self.cal_state.current_source, (200, 200, 200))
        
        return (scaled, color)
    
    def select_source(self, source: ReferenceSource) -> None:
        """Select a reference source."""
        self.cal_state.current_source = source
        print(f"[Calibration] Selected reference: {get_reference_name(source)}")
    
    def cycle_source(self) -> ReferenceSource:
        """Cycle to next reference source."""
        current_idx = self.SOURCES.index(self.cal_state.current_source)
        next_idx = (current_idx + 1) % len(self.SOURCES)
        self.cal_state.current_source = self.SOURCES[next_idx]
        print(f"[Calibration] Reference: {get_reference_name(self.cal_state.current_source)}")
        return self.cal_state.current_source
    
    def toggle_overlay(self) -> bool:
        """Toggle reference overlay visibility."""
        self.cal_state.overlay_visible = not self.cal_state.overlay_visible
        print(f"[Calibration] Overlay: {'ON' if self.cal_state.overlay_visible else 'OFF'}")
        return self.cal_state.overlay_visible
    
    def auto_calibrate(
        self,
        measured_intensity: np.ndarray,
        wavelengths: np.ndarray,
        peak_indices: list,
    ) -> list[tuple[int, float]]:
        """Perform automatic calibration. Tries peak matching first, falls back to correlation.

        Peak matching (ordered): Match detected peaks to reference peaks by order
        (leftmost measured → leftmost reference). Robust when spectrum shape differs
        slightly (e.g. different LED CCT, different fluorescent phosphor mix) but
        major peaks are present in the same order.

        Correlation: Optimizes polynomial to maximize spectrum correlation.
        Can misalign when dominant region (green-yellow-red) drives the fit and
        blue peak gets ignored.

        Args:
            measured_intensity: Measured spectrum intensity (per pixel)
            wavelengths: Current wavelength calibration (used only for fallback)
            peak_indices: Detected peaks (Peak objects or int pixel indices)

        Returns:
            List of (pixel, wavelength) calibration points for recalibrate
        """
        # Try peak-based ordered matching first (robust for slightly different spectra)
        result = self._auto_calibrate_peak_ordered(peak_indices)
        if result:
            return result
        # Fall back to correlation
        return self._auto_calibrate_correlation(measured_intensity)

    def _auto_calibrate_peak_ordered(
        self,
        peak_indices: list,
    ) -> list[tuple[int, float]]:
        """Calibrate by matching peaks by order (no wavelength guess needed).

        When measured has MORE peaks than reference (e.g. Hg lamp with extra
        lines), tries multiple shift hypotheses and picks the one with best
        linearity (spectrometer dispersion is roughly linear).
        """
        peak_pixels = [p.index if hasattr(p, "index") else int(p) for p in peak_indices]
        peak_pixels = sorted(peak_pixels)

        ref_peaks = get_reference_peaks(self.cal_state.current_source)
        ref_sorted = sorted(ref_peaks, key=lambda p: p.wavelength)

        min_required = 3
        n_ref = len(ref_sorted)
        n_meas = len(peak_pixels)
        if n_ref < min_required or n_meas < min_required:
            print(f"[Calibration] Peak matching: need {min_required}+ peaks, "
                  f"have {n_meas} measured, {n_ref} reference")
            return []

        n_use = min(n_ref, n_meas)
        best_points: list[tuple[int, float]] = []
        best_r2 = -1.0

        # Try shifts: measured can have extra peaks (skip leading); reference can
        # have extra (e.g. Hg 5 lines, we see 4). Pick mapping with best linearity.
        meas_shift_max = max(0, n_meas - n_use)
        ref_offset_max = max(0, n_ref - n_use)
        for m_shift in range(meas_shift_max + 1):
            for r_off in range(ref_offset_max + 1):
                pts = [
                    (peak_pixels[m_shift + i], ref_sorted[r_off + i].wavelength)
                    for i in range(n_use)
                ]
                r2 = self._linearity_score(pts)
                if r2 > best_r2:
                    best_r2 = r2
                    best_points = pts

        calibration_points = best_points[:min(6, len(best_points))]
        self.cal_state.calibration_points = calibration_points

        print(f"[Calibration] Peak-ordered calibration: {len(calibration_points)} points")
        for pixel, wl in calibration_points:
            print(f"  Pixel {pixel} -> {wl:.1f} nm")
        return calibration_points

    def _auto_calibrate_correlation(
        self,
        measured_intensity: np.ndarray,
    ) -> list[tuple[int, float]]:
        """Optimize calibration polynomial to maximize spectrum correlation.

        Constraints: visible 400-700 nm must be within measured pixel range.
        Mostly shift and scale (prism dispersion ~linear); c2, c3 are small tweaks.
        """
        n = len(measured_intensity)
        if n < 10:
            print("[Calibration] Need at least 10 pixels for correlation calibration")
            return []

        source = self.cal_state.current_source
        # Initial guess: linear 380-750 nm, mostly shift + scale
        span = max(n - 1, 1)
        c1_init = 370.0 / span
        c0_init = 380.0
        x0 = np.array([c0_init, c1_init, 0.0, 0.0])

        # Sensible wavelength range: 400-700 nm must lie within [pixel 0, pixel n-1]
        wl_start_lo, wl_start_hi = 360.0, 440.0
        wl_end_lo, wl_end_hi = 660.0, 780.0

        def loss(x: np.ndarray) -> float:
            c0, c1, c2, c3 = x
            p = np.arange(n, dtype=np.float64)
            wl = c0 + c1 * p + c2 * (p * p) / (n * n) + c3 * (p * p * p) / (n * n * n)
            wl_0 = wl[0]
            wl_n = wl[-1]
            # Penalty if visible 400-700 falls outside measured range
            range_penalty = 0.0
            if wl_0 > wl_start_hi or wl_0 < wl_start_lo:
                range_penalty += 100.0 * min(abs(wl_0 - wl_start_lo), abs(wl_0 - wl_start_hi))
            if wl_n < wl_end_lo or wl_n > wl_end_hi:
                range_penalty += 100.0 * min(abs(wl_n - wl_end_lo), abs(wl_n - wl_end_hi))
            if wl_0 >= wl_n - 50:
                range_penalty += 1e6
            ref = get_reference_spectrum(source, wl)
            if (
                self.cal_state.sensitivity_correction_enabled
                and self._cmos_sensitivity_wl is not None
                and self._cmos_sensitivity_val is not None
            ):
                sens = np.interp(
                    wl,
                    self._cmos_sensitivity_wl,
                    self._cmos_sensitivity_val,
                    left=0.0,
                    right=0.0,
                )
                ref = ref * np.maximum(sens, 1e-9)
            if ref.size < 2 or np.std(ref) < 1e-9 or np.std(measured_intensity) < 1e-9:
                return 1e6
            corr = np.corrcoef(measured_intensity, ref)[0, 1]
            if np.isnan(corr):
                return 1e6
            # Strong linearity penalty: mostly shift + scale, not drastic curvature
            linearity_penalty = 2.0 * (c2 ** 2 + c3 ** 2)
            return -corr + linearity_penalty + range_penalty

        # c0: wl at pixel 0; c1: nm/pixel (span ~350-400 nm over n pixels)
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
            options={"maxiter": 200, "ftol": 1e-8},
        )

        if not res.success:
            print(f"[Calibration] Correlation optimization failed: {res.message}")
            return []

        c0, c1, c2, c3 = res.x
        pixels_arr = np.arange(n, dtype=np.float64)
        wl_arr = c0 + c1 * pixels_arr + c2 * (pixels_arr ** 2) / (n * n) + c3 * (pixels_arr ** 3) / (n * n * n)

        # Generate 5 points for 3rd-order poly fit (existing recalibrate expects points)
        idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        calibration_points = [(int(i), float(wl_arr[int(i)])) for i in idx]
        self.cal_state.calibration_points = calibration_points

        ref_final = get_reference_spectrum(source, wl_arr)
        if (
            self.cal_state.sensitivity_correction_enabled
            and self._cmos_sensitivity_wl is not None
            and self._cmos_sensitivity_val is not None
        ):
            sens = np.interp(
                wl_arr,
                self._cmos_sensitivity_wl,
                self._cmos_sensitivity_val,
                left=0.0,
                right=0.0,
            )
            ref_final = ref_final * np.maximum(sens, 1e-9)
        corr_final = np.corrcoef(measured_intensity, ref_final)[0, 1] if np.std(ref_final) > 1e-9 else 0
        print(f"[Calibration] Correlation calibration: r={corr_final:.4f}")
        for pixel, wl in calibration_points:
            print(f"  Pixel {pixel} -> {wl:.1f} nm")
        return calibration_points

    def calibrate_from_peaks(
        self,
        measured_intensity: np.ndarray,
        wavelengths: np.ndarray,
        peak_indices: list[int],
    ) -> list[tuple[int, float]]:
        """Manual calibration by matching detected peaks to reference lines.

        Use this for manual calibration when peak matching is preferred.

        Args:
            measured_intensity: Measured spectrum intensity
            wavelengths: Current wavelength calibration
            peak_indices: Detected peak pixel positions

        Returns:
            List of (pixel, wavelength) calibration points
        """
        reference_peaks = get_reference_peaks(self.cal_state.current_source)

        if len(peak_indices) < 4:
            print(f"[Calibration] Need at least 4 peaks, found {len(peak_indices)}")
            return []

        if len(reference_peaks) < 4:
            print(f"[Calibration] Reference has only {len(reference_peaks)} peaks")
            return []

        pixel_indices = [p.index if hasattr(p, 'index') else p for p in peak_indices]
        peak_wavelengths = [wavelengths[min(idx, len(wavelengths) - 1)] for idx in pixel_indices]

        matches = self._match_peaks_to_reference(
            pixel_indices,
            peak_wavelengths,
            reference_peaks,
        )

        if len(matches) < 4:
            print(f"[Calibration] Could only match {len(matches)} peaks, need 4")
            return []

        matches = sorted(matches, key=lambda m: m[2])[:min(6, len(matches))]
        calibration_points = [(m[0], m[1]) for m in matches]
        self.cal_state.calibration_points = calibration_points

        print(f"[Calibration] Peak-matched {len(calibration_points)} points:")
        for pixel, wl in calibration_points:
            print(f"  Pixel {pixel} -> {wl:.1f} nm")
        return calibration_points
    
    def _linearity_score(self, points: list[tuple[int, float]]) -> float:
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
        self,
        peak_pixels: list[int],
        peak_wavelengths: list[float],
        reference_peaks: list[SpectralLine],
    ) -> list[tuple[int, float, float]]:
        """Match detected peaks to reference spectral lines.

        Prefers matches that yield the most linear pixel→wavelength fit
        (spectrometers have roughly linear dispersion).

        Args:
            peak_pixels: Detected peak pixel positions (sorted by pixel)
            peak_wavelengths: Approximate wavelengths from current calibration
            reference_peaks: Known reference spectral lines

        Returns:
            List of (pixel, reference_wavelength, error) tuples
        """
        # Sort by pixel so we process left-to-right
        order = np.argsort(peak_pixels)
        peak_pixels = [peak_pixels[i] for i in order]
        peak_wavelengths = [peak_wavelengths[i] for i in order]

        ref_sorted = sorted(reference_peaks, key=lambda p: p.wavelength)
        tolerance_nm = 35  # nm; tighter than before

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
                r2 = self._linearity_score(pts)
                if r2 > best_r2:
                    best_r2 = r2
                    best_match = ref

            if best_match is not None:
                err = abs(approx_wl - best_match.wavelength)
                matches.append((pixel, best_match.wavelength, err))
                used_refs.add(best_match.wavelength)

        matches.sort(key=lambda m: m[0])
        return matches
    
    def get_calibration_points(self) -> list[tuple[int, float]]:
        """Get current calibration points for saving.
        
        Returns:
            List of (pixel, wavelength) tuples
        """
        return self.cal_state.calibration_points.copy()
    
    def clear_calibration_points(self) -> None:
        """Clear all calibration points."""
        self.cal_state.calibration_points.clear()
        print("[Calibration] Points cleared")
    
    def get_current_source_name(self) -> str:
        """Get name of current reference source."""
        return get_reference_name(self.cal_state.current_source)
    
    def _load_cmos_sensitivity(self) -> None:
        """Load CMOS spectral sensitivity curve from CSV.
        Tries cwd/data first, then package-relative data dir.
        """
        name = "silicon_CMOS_spectral_sensitivity.csv"
        candidates = [
            Path.cwd() / "data" / name,
            _DATA_DIR / name,
        ]
        csv_path = None
        for p in candidates:
            if p.exists():
                csv_path = p
                break
        if csv_path is None:
            print(f"[Calibration] CMOS sensitivity file not found (tried: {[str(p) for p in candidates]})")
            return

        try:
            data = np.loadtxt(csv_path, delimiter=",", skiprows=2)
            self._cmos_sensitivity_wl = data[:, 0]
            self._cmos_sensitivity_val = data[:, 1]
            print(f"[Calibration] Loaded CMOS sensitivity: {len(self._cmos_sensitivity_wl)} points, {self._cmos_sensitivity_wl[0]:.0f}-{self._cmos_sensitivity_wl[-1]:.0f} nm")
        except Exception as e:
            print(f"[Calibration] Failed to load CMOS sensitivity: {e}")
            self._cmos_sensitivity_wl = None
            self._cmos_sensitivity_val = None
    
    def toggle_sensitivity_correction(self) -> bool:
        """Toggle CMOS sensitivity correction."""
        self.cal_state.sensitivity_correction_enabled = not self.cal_state.sensitivity_correction_enabled
        print(f"[Calibration] Sensitivity correction: {'ON' if self.cal_state.sensitivity_correction_enabled else 'OFF'}")
        return self.cal_state.sensitivity_correction_enabled
    
    def get_sensitivity_curve(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> Optional[tuple[np.ndarray, tuple[int, int, int]]]:
        """Get CMOS sensitivity curve for overlay.
        
        Args:
            wavelengths: Wavelength array (per pixel)
            graph_height: Height of graph in pixels
            
        Returns:
            Tuple of (scaled_sensitivity, color) or None if not available
        """
        if self._cmos_sensitivity_wl is None or self._cmos_sensitivity_val is None:
            return None
        
        if not self.cal_state.sensitivity_correction_enabled:
            return None
        
        sensitivity = np.interp(
            wavelengths,
            self._cmos_sensitivity_wl,
            self._cmos_sensitivity_val,
            left=0.0,
            right=0.0,
        )
        
        scaled = sensitivity * (graph_height - 10)
        color = (100, 255, 100)  # Green
        
        return (scaled, color)
    
    def apply_sensitivity_correction(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Apply CMOS sensitivity correction to linearize spectrum.

        Args:
            intensity: Measured spectrum intensity
            wavelengths: Wavelength array (per pixel), same length as intensity or longer

        Returns:
            Corrected intensity (linearized), float32
        """
        if not self.cal_state.sensitivity_correction_enabled:
            return intensity

        if self._cmos_sensitivity_wl is None or self._cmos_sensitivity_val is None:
            return intensity

        n_wl = len(wavelengths)
        n_int = len(intensity)
        n = min(n_wl, n_int)
        intensity = intensity[:n]
        wl = wavelengths[:n]

        sensitivity = np.interp(
            wl,
            self._cmos_sensitivity_wl,
            self._cmos_sensitivity_val,
            left=0.0,
            right=0.0,
        )

        intensity_f = np.asarray(intensity, dtype=np.float32)
        corrected = np.zeros(n, dtype=np.float32)
        mask = sensitivity > 1e-6
        corrected[mask] = intensity_f[mask] / sensitivity[mask]
        corrected[~mask] = intensity_f[~mask]

        return corrected
