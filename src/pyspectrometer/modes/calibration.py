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

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.spectrum import SpectrumData
import numpy as np
from scipy.optimize import minimize

from ..core.mode_context import ModeContext
from ..data.reference_spectra import (
    ReferenceSource,
    SpectralLine,
    get_reference_name,
    get_reference_peaks,
    get_reference_spectrum,
)
from ..utils.graph_scale import scale_intensity_to_graph
from .base import BaseMode, ButtonDefinition, ModeType

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
    frozen_intensity: np.ndarray | None = None


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
        self._cmos_sensitivity_wl: np.ndarray | None = None
        self._cmos_sensitivity_val: np.ndarray | None = None
        self._load_cmos_sensitivity()

    def setup(self, ctx: ModeContext) -> None:
        """Register calibration-specific handlers."""
        super().setup(ctx)
        self._register_calibration_handlers(ctx)

    def _register_calibration_handlers(self, ctx: ModeContext) -> None:
        """Register calibration mode button handlers."""
        for src in self.SOURCES:
            name = f"source_{src.name.lower()}"
            self.register_callback(name, lambda s=src: self._on_select_source(ctx, s))
        self.register_callback("toggle_overlay", lambda: self._on_toggle_overlay(ctx))
        self.register_callback("toggle_sensitivity", lambda: self._on_toggle_sensitivity(ctx))
        self.register_callback("correct_sensitivity", lambda: self._on_correct_sensitivity(ctx))
        self.register_callback("auto_level", lambda: self._on_auto_level(ctx))
        self.register_callback("auto_calibrate", lambda: self._on_auto_calibrate(ctx))
        self.register_callback("reset_calibration", lambda: self._on_reset_calibration(ctx))
        self.register_callback("save_cal", lambda: self._on_save_calibration(ctx))
        self.register_callback("save_spectrum", lambda: self._on_button_save(ctx))
        self.register_callback("load_cal", lambda: self._on_load_calibration(ctx))
        self.register_callback("freeze", lambda: self._on_toggle_freeze(ctx))
        self.register_callback("toggle_averaging", lambda: self._on_toggle_averaging(ctx))
        self.register_callback("toggle_accumulation", lambda: self._on_toggle_accumulation(ctx))
        self.register_callback("clear_points", lambda: self._on_clear_points(ctx))
        ctx.display.set_button_active("toggle_overlay", self.cal_state.overlay_visible)
        ctx.display.set_button_active(
            "toggle_sensitivity", self.cal_state.sensitivity_correction_enabled
        )
        ctx.display.set_button_active("freeze", not ctx.frozen_spectrum)

    def on_start(self, ctx: ModeContext) -> None:
        """Select FL as default source and update source button states."""
        self.select_source(ReferenceSource.FL)
        for s in ReferenceSource:
            ctx.display.set_button_active(f"source_{s.name.lower()}", s == ReferenceSource.FL)

    def update_display(
        self,
        ctx: ModeContext,
        processed: "SpectrumData",
        graph_height: int,
    ) -> None:
        """Update calibration overlay and status."""
        overlay = self.get_overlay(processed.wavelengths, graph_height)
        ctx.display.set_mode_overlay(overlay)
        sensitivity_overlay = self.get_sensitivity_curve(processed.wavelengths, graph_height)
        ctx.display.set_sensitivity_overlay(sensitivity_overlay)
        ctx.display.set_status("Ref", self.get_current_source_name())
        ctx.display.set_status("Pts", str(len(self.cal_state.calibration_points)))
        if ctx.frozen_spectrum:
            ctx.display.set_status("Status", "FROZEN")
        elif self.state.integration_mode != "none":
            ctx.display.set_status(
                "Status", f"{self.state.integration_mode.upper()}:{self.state.accumulated_frames}"
            )
        else:
            ctx.display.set_status("Status", "LIVE")

    def _on_select_source(self, ctx: ModeContext, source: ReferenceSource) -> None:
        """Handle reference source selection."""
        self.select_source(source)
        for s in ReferenceSource:
            ctx.display.set_button_active(f"source_{s.name.lower()}", source == s)

    def _on_toggle_overlay(self, ctx: ModeContext) -> None:
        """Toggle reference overlay visibility."""
        visible = self.toggle_overlay()
        ctx.display.set_button_active("toggle_overlay", visible)

    def _on_toggle_sensitivity(self, ctx: ModeContext) -> None:
        """Toggle CMOS sensitivity correction."""
        enabled = self.toggle_sensitivity_correction()
        ctx.display.set_button_active("toggle_sensitivity", enabled)

    def _on_correct_sensitivity(self, ctx: ModeContext) -> None:
        """Handle CORR button - placeholder for future sensitivity curve correction."""
        print("[CORR] Correct sensitivity curve (placeholder, noop)")

    def _on_auto_level(self, ctx: ModeContext) -> None:
        """Handle auto-level - detect rotation and Y center."""
        if ctx.last_frame is None:
            print("[AutoLevel] No frame available for angle detection")
            return
        angle, y_center, vis_image = ctx.extractor.detect_angle(ctx.last_frame, visualize=True)
        ctx.extractor.set_rotation_angle(angle)
        ctx.extractor.set_spectrum_y_center(y_center)
        print(f"[AutoLevel] Rotation: {angle:.2f}°  Y Center: {y_center}")
        if vis_image is not None:
            ctx.display.set_autolevel_overlay(vis_image)
            ctx.display.set_autolevel_click_dismiss(ctx.clear_autolevel_overlay)
            print("[AutoLevel] Click to close overlay")

    def _on_auto_calibrate(self, ctx: ModeContext) -> None:
        """Handle auto-calibrate action."""
        if ctx.last_data is None:
            print("[CAL] No data available for calibration")
            return
        now = time.monotonic()
        if now - ctx.last_auto_calibrate_time < ctx.auto_calibrate_debounce_sec:
            return
        ctx.last_auto_calibrate_time = now
        if not ctx.frozen_spectrum:
            print("[CAL] Freeze spectrum first (Play button), then click AutoCal")
            return
        measured = (
            ctx.last_intensity_pre_sensitivity
            if (
                self.cal_state.sensitivity_correction_enabled
                and ctx.last_intensity_pre_sensitivity is not None
                and len(ctx.last_intensity_pre_sensitivity) == len(ctx.last_data.intensity)
            )
            else ctx.last_data.intensity
        )
        cal_points = self.auto_calibrate(
            measured_intensity=measured,
            wavelengths=ctx.last_data.wavelengths,
            peak_indices=ctx.last_data.peaks,
        )
        if len(cal_points) >= 3:
            print(f"[CAL] Auto-calibration found {len(cal_points)} points (4+ preferred)")
            pixels = [p for p, w in cal_points]
            wavelengths = [w for p, w in cal_points]
            ctx.calibration.recalibrate(pixels, wavelengths)
        else:
            print("[CAL] Auto-calibration failed to find enough matches")

    def _on_reset_calibration(self, ctx: ModeContext) -> None:
        """Reset calibration to linear 1:1 pixels to spectrum."""
        n = ctx.calibration.width
        wl_min, wl_max = 380.0, 750.0
        pixels = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        wavelengths = [wl_min + (wl_max - wl_min) * p / max(n - 1, 1) for p in pixels]
        ctx.calibration.recalibrate(pixels, wavelengths)
        self.cal_state.calibration_points = list(zip(pixels, wavelengths))
        print(f"[R] Calibration reset to linear 1:1: {wl_min:.0f}-{wl_max:.0f} nm over {n} pixels")

    def _on_save_calibration(self, ctx: ModeContext) -> None:
        """Save calibration and extraction params."""
        cal_points = self.get_calibration_points()
        if len(cal_points) < 4:
            pixels = ctx.calibration.cal_pixels
            wavelengths = ctx.calibration.cal_wavelengths
        else:
            pixels = [p for p, w in cal_points]
            wavelengths = [w for p, w in cal_points]
        saved = False
        if len(pixels) >= 4:
            if ctx.calibration.save(pixels, wavelengths):
                print(f"[CAL] Wavelength calibration saved ({len(pixels)} points)")
                saved = True
        ctx.calibration.save_extraction_params(
            rotation_angle=ctx.extractor.rotation_angle,
            spectrum_y_center=ctx.extractor.spectrum_y_center,
            perpendicular_width=ctx.extractor.perpendicular_width,
        )
        print("[CAL] Extraction params saved")
        if not saved:
            print("[CAL] Need at least 4 calibration points to save wavelength cal")

    def _on_button_save(self, ctx: ModeContext) -> None:
        """Handle save spectrum button."""
        if ctx.last_data is not None:
            ctx.save_snapshot(ctx.last_data)
        else:
            print("[SAVE] No spectrum data available")

    def _on_load_calibration(self, ctx: ModeContext) -> None:
        """Load calibration from file."""
        if ctx.calibration.load():
            ctx.extractor.set_rotation_angle(ctx.calibration.rotation_angle)
            ctx.extractor.set_spectrum_y_center(ctx.calibration.spectrum_y_center)
            ctx.extractor.set_perpendicular_width(ctx.calibration.perpendicular_width)
            print("[CAL] Calibration loaded")
        else:
            print("[CAL] Failed to load calibration")

    def _on_toggle_freeze(self, ctx: ModeContext) -> None:
        """Toggle spectrum freeze."""
        frozen = self.toggle_freeze()
        ctx.frozen_spectrum = frozen
        if frozen and ctx.last_intensity_pre_sensitivity is not None:
            ctx.frozen_intensity = ctx.last_intensity_pre_sensitivity.copy()
        else:
            ctx.frozen_intensity = None
        ctx.display.set_button_active("freeze", not frozen)
        print(f"[FREEZE] Spectrum {'FROZEN' if frozen else 'LIVE'}")

    def _on_toggle_averaging(self, ctx: ModeContext) -> None:
        """Toggle averaging (off Acc if on)."""
        enabled = self.toggle_averaging()
        ctx.display.set_button_active("toggle_averaging", enabled)
        ctx.display.set_button_active("toggle_accumulation", False)
        print(f"[AVG] Averaging {'ON' if enabled else 'OFF'}")

    def _on_toggle_accumulation(self, ctx: ModeContext) -> None:
        """Toggle accumulation (off Avg if on)."""
        enabled = self.toggle_accumulation()
        ctx.display.set_button_active("toggle_accumulation", enabled)
        ctx.display.set_button_active("toggle_averaging", False)
        print(f"[ACC] Accumulation {'ON' if enabled else 'OFF'}")

    def _on_clear_points(self, ctx: ModeContext) -> None:
        """Clear calibration points."""
        self.clear_calibration_points()
        ctx.display.state.click_points.clear()
        print("[CAL] Points cleared")

    @property
    def mode_type(self) -> ModeType:
        return ModeType.CALIBRATION

    @property
    def name(self) -> str:
        return "Calibration"

    def get_buttons(self) -> list[ButtonDefinition]:
        """Get calibration mode buttons."""
        return [
            # Row 1: Source selection, spacer, SENS/LEVEL/CALIB/R
            ButtonDefinition("FL", "source_fl", row=1),
            ButtonDefinition("Hg", "source_hg", row=1),
            ButtonDefinition("D65", "source_d65", row=1),
            ButtonDefinition("LED1", "source_led", row=1),
            ButtonDefinition("LED2", "source_led2", row=1),
            ButtonDefinition("LED3", "source_led3", row=1),
            ButtonDefinition("FL2", "source_fl2", row=1),
            ButtonDefinition("FL3", "source_fl3", row=1),
            ButtonDefinition("FL4", "source_fl4", row=1),
            ButtonDefinition("FL5", "source_fl5", row=1),
            ButtonDefinition("A", "source_a", row=1),
            ButtonDefinition("Overlay", "toggle_overlay", is_toggle=True, row=1),
            ButtonDefinition("__spacer__", "__spacer_left__", row=1),
            ButtonDefinition("S", "toggle_sensitivity", is_toggle=True, row=1),
            ButtonDefinition("CORR", "correct_sensitivity", row=1),
            ButtonDefinition("LEVEL", "auto_level", is_toggle=True, row=1),
            ButtonDefinition("CALIB", "auto_calibrate", row=1),
            ButtonDefinition("R", "reset_calibration", row=1),
            # Row 2: Playback (freeze), Peak, Avg, G, E, AG, AE, Prev, Clear, Save, CSV, Load
            ButtonDefinition("Play", "freeze", is_toggle=True, row=2, icon_type="playback"),
            ButtonDefinition("Peak", "capture_peak", is_toggle=True, row=2),
            ButtonDefinition("Avg", "toggle_averaging", is_toggle=True, row=2),
            ButtonDefinition("Acc", "toggle_accumulation", is_toggle=True, row=2),
            ButtonDefinition("G", "show_gain_slider", is_toggle=True, row=2),
            ButtonDefinition("E", "show_exposure_slider", is_toggle=True, row=2),
            ButtonDefinition("AG", "auto_gain", is_toggle=True, row=2),
            ButtonDefinition("AE", "auto_exposure", is_toggle=True, row=2),
            ButtonDefinition("Prev", "cycle_preview", shortcut="v", row=2),
            ButtonDefinition("Clear", "clear_points", shortcut="x", row=2),
            ButtonDefinition("Save", "save_cal", shortcut="w", row=2),
            ButtonDefinition("CSV", "save_spectrum", shortcut="s", row=2),
            ButtonDefinition("Load", "load_cal", row=2),
            ButtonDefinition("Quit", "quit", row=2),
            ButtonDefinition("__spacer__", "__spacer_right__", row=2),
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

        if self.state.integration_mode != "none":
            intensity = self.accumulate_spectrum(intensity)

        # Store for freeze
        self.cal_state.frozen_intensity = intensity.copy()

        return intensity

    def get_overlay(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> tuple[np.ndarray, tuple[int, int, int]] | None:
        """Get reference spectrum overlay."""
        if not self.cal_state.overlay_visible:
            return None

        # Generate reference spectrum
        ref_intensity = get_reference_spectrum(
            self.cal_state.current_source,
            wavelengths,
        )

        # Scale to graph height
        scaled = scale_intensity_to_graph(ref_intensity, graph_height)

        # Color based on source
        colors = {
            ReferenceSource.FL: (255, 200, 0),  # Yellow
            ReferenceSource.HG: (255, 0, 255),  # Magenta
            ReferenceSource.D65: (100, 200, 255),  # Daylight blue
            ReferenceSource.LED: (200, 200, 200),  # White/gray
            ReferenceSource.LED2: (180, 220, 220),  # Light cyan
            ReferenceSource.LED3: (220, 200, 220),  # Light purple
        }
        color = colors.get(self.cal_state.current_source, (200, 200, 200))

        return (scaled, color)

    def select_source(self, source: ReferenceSource) -> None:
        """Select a reference source."""
        self.cal_state.current_source = source
        print(f"[Calibration] Selected reference: {get_reference_name(source)}")

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

        Hg reference has 5 lines (404, 436, 546, 577, 579 nm). FL has 9. Need 4+ measured
        peaks for reliable alignment; 3 peaks can give ambiguous mapping.

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
        ref_peaks = get_reference_peaks(self.cal_state.current_source)
        if len(ref_peaks) < 3:
            print(
                f"[Calibration] Reference '{get_reference_name(self.cal_state.current_source)}' "
                f"has {len(ref_peaks)} peaks; use Hg or FL for auto-calibration"
            )
            return []

        # Try peak-based ordered matching first (robust for slightly different spectra)
        result = self._auto_calibrate_peak_ordered(peak_indices)
        if result:
            return result
        # Fall back to correlation (only for sources with continuous spectrum)
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
            print(
                f"[Calibration] Peak matching: need {min_required}+ peaks, "
                f"have {n_meas} measured, {n_ref} reference"
            )
            return []

        n_use = min(n_ref, n_meas)
        best_points: list[tuple[int, float]] = []
        best_r2 = -1.0
        best_dispersion_score = -1.0

        # Typical spectrometer dispersion: 0.25–0.45 nm/px
        DISPERSION_LO, DISPERSION_HI = 0.2, 0.5

        def dispersion_score(pts: list[tuple[int, float]]) -> float:
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

        # Try shifts: measured can have extra peaks (skip leading); reference can
        # have extra (e.g. Hg 5 lines, we see 4). Pick mapping with best linearity.
        # For 3 peaks R² is always 1.0, so use dispersion as tiebreaker.
        meas_shift_max = max(0, n_meas - n_use)
        ref_offset_max = max(0, n_ref - n_use)
        for m_shift in range(meas_shift_max + 1):
            for r_off in range(ref_offset_max + 1):
                pts = [
                    (peak_pixels[m_shift + i], ref_sorted[r_off + i].wavelength)
                    for i in range(n_use)
                ]
                r2 = self._linearity_score(pts)
                dsc = dispersion_score(pts)
                if r2 > best_r2 or (r2 == best_r2 and dsc > best_dispersion_score):
                    best_r2 = r2
                    best_dispersion_score = dsc
                    best_points = pts

        calibration_points = best_points[: min(6, len(best_points))]
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
            if self.cal_state.sensitivity_correction_enabled:
                sens = self._interp_sensitivity(wl)
                if sens is not None:
                    ref = ref * np.maximum(sens, 1e-9)
            if ref.size < 2 or np.std(ref) < 1e-9 or np.std(measured_intensity) < 1e-9:
                return 1e6
            corr = np.corrcoef(measured_intensity, ref)[0, 1]
            if np.isnan(corr):
                return 1e6
            # Strong linearity penalty: mostly shift + scale, not drastic curvature
            linearity_penalty = 2.0 * (c2**2 + c3**2)
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

        # Generate 5 points for 3rd-order poly fit (existing recalibrate expects points)
        idx = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        calibration_points = [(int(i), float(wl_arr[int(i)])) for i in idx]
        self.cal_state.calibration_points = calibration_points

        ref_final = get_reference_spectrum(source, wl_arr)
        if self.cal_state.sensitivity_correction_enabled:
            sens = self._interp_sensitivity(wl_arr)
            if sens is not None:
                ref_final = ref_final * np.maximum(sens, 1e-9)
        corr_final = (
            np.corrcoef(measured_intensity, ref_final)[0, 1] if np.std(ref_final) > 1e-9 else 0
        )
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

        pixel_indices = [p.index if hasattr(p, "index") else p for p in peak_indices]
        peak_wavelengths = [wavelengths[min(idx, len(wavelengths) - 1)] for idx in pixel_indices]

        matches = self._match_peaks_to_reference(
            pixel_indices,
            peak_wavelengths,
            reference_peaks,
        )

        if len(matches) < 4:
            print(f"[Calibration] Could only match {len(matches)} peaks, need 4")
            return []

        matches = sorted(matches, key=lambda m: m[2])[: min(6, len(matches))]
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
            print(
                f"[Calibration] CMOS sensitivity file not found (tried: {[str(p) for p in candidates]})"
            )
            return

        try:
            data = np.loadtxt(csv_path, delimiter=",", skiprows=2)
            self._cmos_sensitivity_wl = data[:, 0]
            self._cmos_sensitivity_val = data[:, 1]
            print(
                f"[Calibration] Loaded CMOS sensitivity: {len(self._cmos_sensitivity_wl)} points, {self._cmos_sensitivity_wl[0]:.0f}-{self._cmos_sensitivity_wl[-1]:.0f} nm"
            )
        except Exception as e:
            print(f"[Calibration] Failed to load CMOS sensitivity: {e}")
            self._cmos_sensitivity_wl = None
            self._cmos_sensitivity_val = None

    def toggle_sensitivity_correction(self) -> bool:
        """Toggle CMOS sensitivity correction."""
        self.cal_state.sensitivity_correction_enabled = (
            not self.cal_state.sensitivity_correction_enabled
        )
        print(
            f"[Calibration] Sensitivity correction: {'ON' if self.cal_state.sensitivity_correction_enabled else 'OFF'}"
        )
        return self.cal_state.sensitivity_correction_enabled

    def _interp_sensitivity(self, wavelengths: np.ndarray) -> np.ndarray | None:
        """Interpolate CMOS sensitivity at wavelengths. Returns None if not loaded."""
        if self._cmos_sensitivity_wl is None or self._cmos_sensitivity_val is None:
            return None
        return np.interp(
            wavelengths,
            self._cmos_sensitivity_wl,
            self._cmos_sensitivity_val,
            left=0.0,
            right=0.0,
        )

    def get_sensitivity_curve(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> tuple[np.ndarray, tuple[int, int, int]] | None:
        """Get CMOS sensitivity curve for overlay."""
        if not self.cal_state.sensitivity_correction_enabled:
            return None
        sensitivity = self._interp_sensitivity(wavelengths)
        if sensitivity is None:
            return None
        scaled = scale_intensity_to_graph(sensitivity, graph_height)
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

        sensitivity = self._interp_sensitivity(wavelengths)
        if sensitivity is None:
            return intensity

        n_wl = len(wavelengths)
        n_int = len(intensity)
        n = min(n_wl, n_int)
        intensity = intensity[:n]
        sensitivity = sensitivity[:n]

        intensity_f = np.asarray(intensity, dtype=np.float32)
        corrected = np.zeros(n, dtype=np.float32)
        mask = sensitivity > 1e-6
        corrected[mask] = intensity_f[mask] / sensitivity[mask]
        corrected[~mask] = intensity_f[~mask]

        return corrected
