"""Calibration mode for wavelength calibration.

Workflow:
1. Select reference source (FL, Hg, D65, LED)
2. Point spectrometer at light source
3. Use auto-level and auto-shift to fit spectrum in view
4. Click AutoCal (works on live or frozen spectrum)
5. Correlation-based optimization finds polynomial that maximizes spectrum
   correlation with reference, regularized for linearity (prism dispersion)
6. Verify overlay aligns with measured spectrum
7. Save calibration manually when satisfied

Manual calibration (peak matching) via calibrate_from_peaks() for later use.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.spectrum import SpectrumData
import numpy as np

from ..core.mode_context import ModeContext
from ..data.reference_spectra import (
    ReferenceSource,
    get_reference_name,
    get_reference_spectrum,
)
from ..processing.auto_calibrator import AutoCalibrator
from ..processing.sensitivity_correction import SensitivityCorrection
from ..utils.graph_scale import scale_intensity_to_graph
from .base import BaseMode, ButtonDefinition, ModeType


@dataclass
class CalibrationState:
    """State specific to calibration mode."""

    # Current reference source
    current_source: ReferenceSource = ReferenceSource.FL12

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

    # Reference sources in cycle order: Hg, D65, FL1, FL2, FL3, FL12, LED1, LED2, LED3
    SOURCES = [
        ReferenceSource.HG,
        ReferenceSource.D65,
        ReferenceSource.FL1,
        ReferenceSource.FL2,
        ReferenceSource.FL3,
        ReferenceSource.FL12,
        ReferenceSource.LED,
        ReferenceSource.LED2,
        ReferenceSource.LED3,
    ]

    def __init__(self):
        """Initialize calibration mode."""
        super().__init__()
        self.cal_state = CalibrationState()
        self.state.auto_gain_enabled = True
        self._sensitivity = SensitivityCorrection()
        self._auto_calibrator = AutoCalibrator()

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
        ctx.display.set_button_active("toggle_overlay", self.cal_state.overlay_visible)
        ctx.display.set_button_active(
            "toggle_sensitivity", self.cal_state.sensitivity_correction_enabled
        )
        ctx.display.set_button_active("freeze", not ctx.frozen_spectrum)

    def on_start(self, ctx: ModeContext) -> None:
        """Select FL12 as default source and update source button states."""
        self.select_source(ReferenceSource.FL12)
        for s in self.SOURCES:
            ctx.display.set_button_active(f"source_{s.name.lower()}", s == ReferenceSource.FL12)

    def update_display(
        self,
        ctx: ModeContext,
        processed: "SpectrumData",
        graph_height: int,
    ) -> None:
        """Update calibration overlay and status."""
        self._sync_button_status(ctx)
        overlay = self.get_overlay(processed.wavelengths, graph_height)
        ctx.display.set_mode_overlay(overlay)
        sensitivity_overlay = (
            self._sensitivity.get_curve_for_display(processed.wavelengths, graph_height)
            if self.cal_state.sensitivity_correction_enabled
            else None
        )
        ctx.display.set_sensitivity_overlay(sensitivity_overlay)
        ctx.display.set_status("Ref", self.get_current_source_name())
        if ctx.frozen_spectrum:
            ctx.display.set_status("Status", "FROZEN")
        elif self.state.integration_mode != "none":
            ctx.display.set_status(
                "Status", f"{self.state.integration_mode.upper()}:{self.state.accumulated_frames}"
            )
        else:
            ctx.display.set_status("Status", "LIVE")

    def _sync_button_status(self, ctx: ModeContext) -> None:
        """Sync button active state from model to display (Peak, Avg, Acc, G, E, AG, AE, S, sources, Overlay, Play)."""
        for s in self.SOURCES:
            ctx.display.set_button_active(
                f"source_{s.name.lower()}", s == self.cal_state.current_source
            )
        ctx.display.set_button_active("capture_peak", ctx.display.state.hold_peaks)
        ctx.display.set_button_active("toggle_averaging", self.state.integration_mode == "avg")
        ctx.display.set_button_active("toggle_accumulation", self.state.integration_mode == "acc")
        ctx.display.set_button_active("show_gain_slider", ctx.display.is_gain_slider_visible())
        ctx.display.set_button_active(
            "show_exposure_slider", ctx.display.is_exposure_slider_visible()
        )
        ctx.display.set_button_active("auto_gain", ctx.auto_gain_enabled)
        ctx.display.set_button_active("auto_exposure", ctx.auto_exposure_enabled)
        ctx.display.set_button_active(
            "toggle_sensitivity", self.cal_state.sensitivity_correction_enabled
        )
        ctx.display.set_button_active("toggle_overlay", self.cal_state.overlay_visible)
        ctx.display.set_button_active("show_spectrum_bars", ctx.display.is_spectrum_bars_visible())
        ctx.display.set_button_active("freeze", not ctx.frozen_spectrum)

    def _on_select_source(self, ctx: ModeContext, source: ReferenceSource) -> None:
        """Handle reference source selection."""
        self.select_source(source)
        for s in self.SOURCES:
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
        measured = (
            ctx.last_intensity_pre_sensitivity
            if (
                self.cal_state.sensitivity_correction_enabled
                and ctx.last_intensity_pre_sensitivity is not None
                and len(ctx.last_intensity_pre_sensitivity) == len(ctx.last_data.intensity)
            )
            else ctx.last_data.intensity
        )
        cal_points = self._auto_calibrator.calibrate(
            measured_intensity=measured,
            wavelengths=ctx.last_data.wavelengths,
            peak_indices=ctx.last_data.peaks,
            source=self.cal_state.current_source,
            sensitivity=self._sensitivity,
            sensitivity_enabled=self.cal_state.sensitivity_correction_enabled,
        )
        if len(cal_points) >= 3:
            self.cal_state.calibration_points = cal_points
            print(f"[CAL] Auto-calibration found {len(cal_points)} points (4+ preferred)")
            for pixel, wl in cal_points:
                print(f"  Pixel {pixel} -> {wl:.1f} nm")
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
        saved_wl = False
        if len(pixels) >= 4:
            if ctx.calibration.save(pixels, wavelengths):
                print(f"[CAL] Wavelength calibration saved ({len(pixels)} points)")
                saved_wl = True
        ctx.calibration.save_extraction_params(
            rotation_angle=ctx.extractor.rotation_angle,
            spectrum_y_center=ctx.extractor.spectrum_y_center,
            perpendicular_width=ctx.extractor.perpendicular_width,
        )
        if not saved_wl:
            print(
                "[CAL] Extraction params (autolevel) saved. "
                "Run AutoCal for wavelength cal (need 4+ points)."
            )

    def _on_button_save(self, ctx: ModeContext) -> None:
        """Handle save spectrum button. In calibration mode, passes reference illuminant for special CSV format."""
        if ctx.last_data is None:
            print("[SAVE] No spectrum data available")
            return
        ref_intensity = get_reference_spectrum(
            self.cal_state.current_source,
            ctx.last_data.wavelengths,
        )
        ctx.save_snapshot(ctx.last_data, reference_intensity=ref_intensity)

    def _on_load_calibration(self, ctx: ModeContext) -> None:
        """Load calibration from file (calibration mode only)."""
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

    @property
    def mode_type(self) -> ModeType:
        return ModeType.CALIBRATION

    @property
    def name(self) -> str:
        return "Calibration"

    def get_buttons(self) -> list[ButtonDefinition]:
        """Get calibration mode buttons."""
        return [
            # Row 1: Source selection (Hg, D65, FL1, FL2, FL3, FL12, LED1, LED2, LED3)
            ButtonDefinition("Hg", "source_hg", is_toggle=True, row=1),
            ButtonDefinition("D65", "source_d65", is_toggle=True, row=1),
            ButtonDefinition("FL1", "source_fl1", is_toggle=True, row=1),
            ButtonDefinition("FL2", "source_fl2", is_toggle=True, row=1),
            ButtonDefinition("FL3", "source_fl3", is_toggle=True, row=1),
            ButtonDefinition("FL12", "source_fl12", is_toggle=True, row=1),
            ButtonDefinition("LED1", "source_led", is_toggle=True, row=1),
            ButtonDefinition("LED2", "source_led2", is_toggle=True, row=1),
            ButtonDefinition("LED3", "source_led3", is_toggle=True, row=1),
            ButtonDefinition("Overlay", "toggle_overlay", is_toggle=True, row=1),
            ButtonDefinition("__spacer__", "__spacer_left__", row=1),
            ButtonDefinition("S", "toggle_sensitivity", is_toggle=True, row=1),
            ButtonDefinition("CORR", "correct_sensitivity", row=1),
            ButtonDefinition("LEVEL", "auto_level", is_toggle=True, row=1),
            ButtonDefinition("CALIB", "auto_calibrate", row=1),
            ButtonDefinition("R", "reset_calibration", row=1),
            # Row 2: Playback (freeze), Peak, Avg, G, E, AG, AE, Prev, Save, CSV, Load
            ButtonDefinition("Play", "freeze", is_toggle=True, row=2, icon_type="playback"),
            ButtonDefinition("Peak", "capture_peak", is_toggle=True, row=2),
            ButtonDefinition("Avg", "toggle_averaging", is_toggle=True, row=2),
            ButtonDefinition("Acc", "toggle_accumulation", is_toggle=True, row=2),
            ButtonDefinition("Bars", "show_spectrum_bars", is_toggle=True, row=2),
            ButtonDefinition("G", "show_gain_slider", is_toggle=True, row=2),
            ButtonDefinition("E", "show_exposure_slider", is_toggle=True, row=2),
            ButtonDefinition("AG", "auto_gain", is_toggle=True, row=2),
            ButtonDefinition("AE", "auto_exposure", is_toggle=True, row=2),
            ButtonDefinition("Prev", "cycle_preview", shortcut="v", row=2),
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
            ReferenceSource.HG: (255, 0, 255),  # Magenta
            ReferenceSource.D65: (100, 200, 255),  # Daylight blue
            ReferenceSource.FL1: (255, 200, 0),  # Yellow
            ReferenceSource.FL2: (200, 255, 100),  # Light green
            ReferenceSource.FL3: (255, 220, 150),  # Light orange
            ReferenceSource.FL12: (255, 200, 0),  # Yellow
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
        """Perform automatic calibration. Delegates to AutoCalibrator."""
        return self._auto_calibrator.calibrate(
            measured_intensity=measured_intensity,
            wavelengths=wavelengths,
            peak_indices=peak_indices,
            source=self.cal_state.current_source,
            sensitivity=self._sensitivity,
            sensitivity_enabled=self.cal_state.sensitivity_correction_enabled,
        )

    def calibrate_from_peaks(
        self,
        measured_intensity: np.ndarray,
        wavelengths: np.ndarray,
        peak_indices: list[int],
    ) -> list[tuple[int, float]]:
        """Manual calibration by matching detected peaks to reference lines."""
        calibration_points = self._auto_calibrator.calibrate_from_peaks(
            measured_intensity, wavelengths, peak_indices, self.cal_state.current_source
        )
        if calibration_points:
            self.cal_state.calibration_points = calibration_points
            print(f"[Calibration] Peak-matched {len(calibration_points)} points:")
            for pixel, wl in calibration_points:
                print(f"  Pixel {pixel} -> {wl:.1f} nm")
        return calibration_points

    def get_calibration_points(self) -> list[tuple[int, float]]:
        """Get current calibration points for saving.

        Returns:
            List of (pixel, wavelength) tuples
        """
        return self.cal_state.calibration_points.copy()

    def get_current_source_name(self) -> str:
        """Get name of current reference source."""
        return get_reference_name(self.cal_state.current_source)

    def toggle_sensitivity_correction(self) -> bool:
        """Toggle CMOS sensitivity correction."""
        self.cal_state.sensitivity_correction_enabled = (
            not self.cal_state.sensitivity_correction_enabled
        )
        print(
            f"[Calibration] Sensitivity correction: {'ON' if self.cal_state.sensitivity_correction_enabled else 'OFF'}"
        )
        return self.cal_state.sensitivity_correction_enabled

    def apply_sensitivity_correction(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Apply CMOS sensitivity correction to linearize spectrum."""
        if not self.cal_state.sensitivity_correction_enabled:
            return intensity
        return self._sensitivity.apply(intensity, wavelengths)
