"""Measurement mode for general spectrum measurement.

Features:
- Save/Load spectrum
- Dark reference subtraction
- White reference normalization
- Spectrum averaging
- Auto gain control
- GPIO lamp control
"""

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.spectrum import SpectrumData
import numpy as np

from ..core.mode_context import ModeContext
from ..processing.reference_correction import apply_dark_white_correction
from ..utils.graph_scale import scale_intensity_to_graph
from .base import BaseMode, ButtonDefinition, ModeType


@dataclass
class MeasurementState:
    """State specific to measurement mode."""

    # Reference spectra
    dark_spectrum: np.ndarray | None = None
    white_spectrum: np.ndarray | None = None
    reference_spectrum: np.ndarray | None = None

    # Current captured spectrum (for saving)
    captured_spectrum: np.ndarray | None = None
    captured_wavelengths: np.ndarray | None = None

    # Display settings
    show_reference: bool = False
    normalize_to_reference: bool = False
    subtract_dark: bool = True

    # Loaded spectrum info
    loaded_spectrum_name: str = ""
    load_as: str = "overlay"  # overlay, black, white


class MeasurementMode(BaseMode):
    """Measurement mode for general spectrum analysis."""

    def __init__(self):
        """Initialize measurement mode."""
        super().__init__()
        self.meas_state = MeasurementState()

    def setup(self, ctx: ModeContext) -> None:
        """Register measurement-specific handlers."""
        super().setup(ctx)
        self._register_measurement_handlers(ctx)

    def _register_measurement_handlers(self, ctx: ModeContext) -> None:
        """Register measurement mode button handlers."""
        self.register_callback("capture", lambda: self._on_capture(ctx))
        self.register_callback("toggle_averaging", lambda: self._on_toggle_averaging(ctx))
        self.register_callback("toggle_accumulation", lambda: self._on_toggle_accumulation(ctx))
        self.register_callback("set_dark", lambda: self._on_set_dark(ctx))
        self.register_callback("set_white", lambda: self._on_set_white(ctx))
        self.register_callback("clear_refs", lambda: self._on_clear_refs(ctx))
        self.register_callback("save", lambda: self._on_save(ctx))
        self.register_callback("load", lambda: self._on_load(ctx))
        self.register_callback("cycle_load_as", lambda: self._on_cycle_load_as(ctx))
        self.register_callback("show_reference", lambda: self._on_toggle_show_reference(ctx))
        self.register_callback("normalize", lambda: self._on_toggle_normalize(ctx))
        self.register_callback("lamp_toggle", lambda: self._on_toggle_light(ctx))
        self.register_callback("snap_to_peaks", lambda: self._on_toggle_snap_to_peaks(ctx))
        self.register_callback("clear_markers", lambda: self._on_clear_markers(ctx))
        self.register_callback("clear_all", lambda: self._on_clear_all(ctx))
        self.register_callback("show_peak_delta", lambda: self._on_toggle_peak_delta(ctx))
        self.register_callback("export_csv", lambda: self._on_export_csv(ctx))

    def _on_toggle_peak_delta(self, ctx: ModeContext) -> None:
        """Toggle peak/marker width (FWHM) below wavelength label."""
        active = ctx.display.toggle_peak_delta_visible()
        ctx.display.set_button_active("show_peak_delta", active)
        print(f"[Pkd] Peak width display: {'ON' if active else 'OFF'}")

    def _on_toggle_snap_to_peaks(self, ctx: ModeContext) -> None:
        """Toggle snap-to-peaks for marker line placement (when peaks are off)."""
        active = ctx.display.toggle_snap_to_peaks()
        ctx.display.set_button_active("snap_to_peaks", active)
        print(f"[MARKERS] Snap to peaks: {'ON' if active else 'OFF'}")

    def _on_clear_markers(self, ctx: ModeContext) -> None:
        """Remove all vertical marker lines."""
        ctx.display.state.marker_lines.clear()
        print("[MARKERS] All marker lines cleared")

    def _on_clear_all(self, ctx: ModeContext) -> None:
        """Clear marker lines and peak-include region."""
        ctx.display.state.marker_lines.clear()
        ctx.display.clear_peak_include_region()
        print("[MARKERS] Markers and peak region cleared")

    def _on_export_csv(self, ctx: ModeContext) -> None:
        """Export current spectrum to CSV (same as Save: CSV + PNG)."""
        self._on_save(ctx)

    def _on_capture(self, ctx: ModeContext) -> None:
        """Toggle freeze (like calibration Play): active = live (red circle), inactive = frozen (gray square). Click when live → freeze; when frozen → unfreeze."""
        btn = ctx.display.control_bar.get_button("capture")
        if btn is not None and btn.is_active:
            # Currently live: user clicked to freeze
            if ctx.last_data is None:
                print("[CAPTURE] No spectrum data available")
                return
            frozen = self.toggle_freeze()
            assert frozen
            ctx.frozen_spectrum = True
            ctx.frozen_intensity = (
                ctx.last_raw_intensity.copy()
                if ctx.last_raw_intensity is not None
                else ctx.last_data.intensity.copy()
            )
            self.capture_current(ctx.last_data.intensity, ctx.last_data.wavelengths)
            self.set_reference_spectrum(ctx.last_data.intensity, "Captured")
            ctx.display.state.reference_spectrum = self.meas_state.reference_spectrum.copy()
            ctx.display.state.reference_name = "Captured"
            ctx.display.set_button_active("capture", False)
            print("[Measurement] Frozen, reference set")
        else:
            # Currently frozen: user clicked to unfreeze
            frozen = self.toggle_freeze()
            assert not frozen
            ctx.frozen_spectrum = False
            ctx.frozen_intensity = None
            self.meas_state.captured_spectrum = None
            self.meas_state.captured_wavelengths = None
            self.meas_state.reference_spectrum = None
            ctx.display.state.reference_spectrum = None
            ctx.display.state.reference_name = "None"
            ctx.display.set_button_active("capture", True)
            print("[Measurement] Unfrozen, reference cleared")

    def _on_toggle_averaging(self, ctx: ModeContext) -> None:
        """Toggle averaging (off Acc if on)."""
        enabled = self.toggle_averaging()
        ctx.display.set_button_active("toggle_averaging", enabled)
        ctx.display.set_button_active("toggle_accumulation", False)

    def _on_toggle_accumulation(self, ctx: ModeContext) -> None:
        """Toggle accumulation (off Avg if on)."""
        enabled = self.toggle_accumulation()
        ctx.display.set_button_active("toggle_accumulation", enabled)
        ctx.display.set_button_active("toggle_averaging", False)

    def _on_set_dark(self, ctx: ModeContext) -> None:
        """Toggle dark reference: set if unset, clear if already set."""
        if self.meas_state.dark_spectrum is not None:
            self.meas_state.dark_spectrum = None
            self.meas_state.subtract_dark = True
            ctx.display.set_button_active("set_dark", False)
            print("[DARK] Dark reference cleared")
            return

        raw = _get_raw(ctx)
        if raw is None:
            print("[DARK] No spectrum data available")
            return
        self.set_dark_reference(raw)
        ctx.display.set_button_active("set_dark", True)

    def _on_set_white(self, ctx: ModeContext) -> None:
        """Toggle white reference: set if unset, clear if already set."""
        if self.meas_state.white_spectrum is not None:
            self.meas_state.white_spectrum = None
            ctx.display.set_button_active("set_white", False)
            print("[WHITE] White reference cleared")
            return

        raw = _get_raw(ctx)
        if raw is None:
            print("[WHITE] No spectrum data available")
            return
        self.set_white_reference(raw)
        ctx.display.set_button_active("set_white", True)

    def _on_clear_refs(self, ctx: ModeContext) -> None:
        """Clear all references."""
        self.clear_references()
        ctx.display.set_button_active("show_reference", False)
        ctx.display.set_button_active("normalize", False)

    def _on_save(self, ctx: ModeContext) -> None:
        """Handle save button."""
        if ctx.last_data is None:
            print("[SAVE] No spectrum data available")
            return
        light_on = self.state.lamp_enabled
        led_pct = ctx.display.get_led_intensity_value() if light_on else None
        pwm_log = (
            f"{math.log10(0.01 + led_pct / 100):.4f}"
            if led_pct is not None and light_on
            else None
        )
        metadata = {
            "Mode": "Measurement",
            "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Light source": "Lamp" if light_on else "off",
            "PWM intensity %": f"{led_pct:.1f}" if led_pct is not None and light_on else None,
            "PWM intensity log10": pwm_log,
            "Gain": f"{ctx.camera.gain:.1f}",
            "Exposure": f"{getattr(ctx.camera, 'exposure', 0)}",
            "Note": "",
        }
        ctx.save_snapshot(
            ctx.last_data,
            dark_intensity=self.meas_state.dark_spectrum,
            white_intensity=self.meas_state.white_spectrum,
            metadata=metadata,
        )

    def _on_cycle_load_as(self, ctx: ModeContext) -> None:
        """Cycle load target: overlay -> black -> white -> overlay."""
        order = ["overlay", "black", "white"]
        idx = order.index(self.meas_state.load_as)
        self.meas_state.load_as = order[(idx + 1) % len(order)]
        ctx.display.set_status("LoadAs", self.meas_state.load_as.capitalize())
        print(f"[LOAD] Load as: {self.meas_state.load_as}")

    def _on_load(self, ctx: ModeContext) -> None:
        """Load spectrum into overlay, black, or white per load_as."""
        print(
            f"[LOAD] Load spectrum as {self.meas_state.load_as} (file dialog not implemented yet)"
        )

    def _on_toggle_show_reference(self, ctx: ModeContext) -> None:
        """Toggle reference overlay."""
        enabled = self.toggle_show_reference()
        ctx.display.set_button_active("show_reference", enabled)

    def _on_toggle_normalize(self, ctx: ModeContext) -> None:
        """Toggle normalization to reference."""
        enabled = self.toggle_normalize()
        ctx.display.set_button_active("normalize", enabled)

    def _on_toggle_light(self, ctx: ModeContext) -> None:
        """Toggle light control - placeholder."""
        print("[LIGHT] Toggle light (GPIO not implemented yet)")

    def update_display(
        self,
        ctx: ModeContext,
        processed: "SpectrumData",
        graph_height: int,
    ) -> None:
        """Update measurement overlay and status. Control graph click from mode: markers when peaks off, else default (pan/peak_region/spectrum_select)."""
        ctx.display.set_button_active("capture", not ctx.frozen_spectrum)
        ctx.display.state.graph_click_behavior = (
            "marker" if not ctx.display.state.peaks_visible else "default"
        )
        overlay = self.get_overlay(processed.wavelengths, graph_height)
        ctx.display.set_mode_overlay(overlay)
        ctx.display.set_sensitivity_overlay(None)
        for key, value in self.get_status().items():
            ctx.display.set_status(key, value)

    @property
    def mode_type(self) -> ModeType:
        return ModeType.MEASUREMENT

    @property
    def name(self) -> str:
        return "Measurement"

    @property
    def preview_modes(self) -> list[str]:
        """No full image preview in measurement; spectrum bar or none only."""
        return ["none", "spectrum"]

    def on_start(self, ctx: ModeContext) -> None:
        """Hide camera preview by default and sync Rec (freeze) button."""
        ctx.display.preview_mode = "none"
        ctx.display.set_button_active("capture", not ctx.frozen_spectrum)

    def get_buttons(self) -> list[ButtonDefinition]:
        """Get measurement mode buttons. Grouped with gaps; Quit right-aligned."""
        return [
            # Row 1: Rec (capture + progress) | Avg/Peak/Acc | Dark/White | Bars | ZX/ZY | VIEW
            ButtonDefinition("Rec", "capture", is_toggle=True, row=1, icon_type="playback"),
            ButtonDefinition("__gap__", "__gap__", row=1),
            ButtonDefinition("Avg", "toggle_averaging", is_toggle=True, row=1),
            ButtonDefinition("Peak", "capture_peak", is_toggle=True, shortcut="h", row=1),
            ButtonDefinition("Acc", "toggle_accumulation", is_toggle=True, row=1),
            ButtonDefinition("__gap__", "__gap__2", row=1),
            ButtonDefinition("Dark", "set_dark", is_toggle=True, row=1),
            ButtonDefinition("White", "set_white", is_toggle=True, row=1),
            ButtonDefinition("__gap__", "__gap__3", row=1),
            ButtonDefinition("Bars", "show_spectrum_bars", is_toggle=True, row=1),
            ButtonDefinition("__gap__", "__gap__4", row=1),
            ButtonDefinition("ZX", "show_zoom_x_slider", is_toggle=True, row=1),
            ButtonDefinition("ZY", "show_zoom_y_slider", is_toggle=True, row=1),
            ButtonDefinition("__gap__", "__gap__5", row=1),
            ButtonDefinition("VIEW", "cycle_preview", shortcut="v", row=1),
            # Row 2: Lamp | Peaks/Snap/Pkd/Clr | Ref | G/E/AG/AE | Save/Load | Export | spacer | Quit
            ButtonDefinition("Lamp", "lamp_toggle", is_toggle=True, row=2),
            ButtonDefinition("__gap__", "__gap__6", row=2),
            ButtonDefinition("Peaks", "show_peaks", is_toggle=True, row=2),
            ButtonDefinition("Snap", "snap_to_peaks", is_toggle=True, row=2),
            ButtonDefinition("Pkd", "show_peak_delta", is_toggle=True, row=2),
            ButtonDefinition("Clr", "clear_all", shortcut="z", row=2),
            ButtonDefinition("__gap__", "__gap__7", row=2),
            ButtonDefinition("Ref", "show_reference", is_toggle=True, row=2),
            ButtonDefinition("__gap__", "__gap__8", row=2),
            ButtonDefinition("G", "show_gain_slider", is_toggle=True, row=2),
            ButtonDefinition("E", "show_exposure_slider", is_toggle=True, row=2),
            ButtonDefinition("AG", "auto_gain", is_toggle=True, row=2),
            ButtonDefinition("AE", "auto_exposure", is_toggle=True, row=2),
            ButtonDefinition("__gap__", "__gap__9", row=2),
            ButtonDefinition("Save", "save", shortcut="s", row=2),
            ButtonDefinition("Load", "load", row=2),
            ButtonDefinition("__gap__", "__gap__10", row=2),
            ButtonDefinition("Export", "export_csv", row=2),
            ButtonDefinition("__spacer__", "__spacer_right__", row=2),
            ButtonDefinition("Quit", "quit", row=2),
        ]

    def process_spectrum(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Process spectrum with dark/white correction and normalization.

        Returns 0-1 normalized intensity for pipeline consistency.
        """
        result = intensity.astype(np.float64)

        if self.state.integration_mode != "none":
            result = self.accumulate_spectrum(result)

        # Apply dark/white reference correction (shared logic)
        dark = self.meas_state.dark_spectrum if self.meas_state.subtract_dark else None
        result = apply_dark_white_correction(
            result,
            dark,
            self.meas_state.white_spectrum,
        )

        # Normalize to reference spectrum if enabled
        if (
            self.meas_state.normalize_to_reference
            and self.meas_state.reference_spectrum is not None
        ):
            ref = np.maximum(
                np.asarray(self.meas_state.reference_spectrum, dtype=np.float64),
                1,
            )
            result = np.clip(result.astype(np.float64) / ref, 0, 1).astype(np.float32)

        return result

    def get_overlay(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> tuple[np.ndarray, tuple[int, int, int]] | None:
        """Get reference spectrum overlay if enabled."""
        if not self.meas_state.show_reference:
            return None

        ref = self.meas_state.reference_spectrum
        if ref is None:
            return None

        # Scale to graph height
        max_val = max(ref.max(), 1)
        scaled = scale_intensity_to_graph(ref / max_val, graph_height)

        # Light gray color for reference
        return (scaled, (150, 150, 150))

    def capture_current(self, intensity: np.ndarray, wavelengths: np.ndarray) -> None:
        """Capture current spectrum for later use."""
        self.meas_state.captured_spectrum = intensity.copy()
        self.meas_state.captured_wavelengths = wavelengths.copy()
        print("[Measurement] Spectrum captured")

    def set_dark_reference(self, intensity: np.ndarray) -> None:
        """Set dark/black reference spectrum."""
        self.meas_state.dark_spectrum = intensity.copy()
        self.meas_state.subtract_dark = True
        print("[Measurement] Dark reference set")

    def set_white_reference(self, intensity: np.ndarray) -> None:
        """Set white reference spectrum."""
        self.meas_state.white_spectrum = intensity.copy()
        print("[Measurement] White reference set")

    def set_reference_spectrum(self, intensity: np.ndarray, name: str = "Current") -> None:
        """Set reference spectrum for normalization/overlay."""
        self.meas_state.reference_spectrum = intensity.copy()
        self.meas_state.loaded_spectrum_name = name
        print(f"[Measurement] Reference spectrum set: {name}")

    def clear_references(self) -> None:
        """Clear all reference spectra."""
        self.meas_state.dark_spectrum = None
        self.meas_state.white_spectrum = None
        self.meas_state.reference_spectrum = None
        self.meas_state.subtract_dark = True
        self.meas_state.normalize_to_reference = False
        self.meas_state.show_reference = False
        self.meas_state.loaded_spectrum_name = ""
        print("[Measurement] All references cleared")

    def toggle_show_reference(self) -> bool:
        """Toggle reference spectrum overlay."""
        self.meas_state.show_reference = not self.meas_state.show_reference
        print(f"[Measurement] Show reference: {'ON' if self.meas_state.show_reference else 'OFF'}")
        return self.meas_state.show_reference

    def toggle_normalize(self) -> bool:
        """Toggle normalization to reference."""
        self.meas_state.normalize_to_reference = not self.meas_state.normalize_to_reference
        print(
            f"[Measurement] Normalize: {'ON' if self.meas_state.normalize_to_reference else 'OFF'}"
        )
        return self.meas_state.normalize_to_reference

    def get_status(self) -> dict[str, str]:
        """Get status values for display."""
        status = {}

        if self.meas_state.dark_spectrum is not None:
            status["Dark"] = "SET"
        if self.meas_state.white_spectrum is not None:
            status["White"] = "SET"
        if self.meas_state.reference_spectrum is not None:
            status["Ref"] = self.meas_state.loaded_spectrum_name or "SET"
        status["LoadAs"] = self.meas_state.load_as.capitalize()

        if self.state.integration_mode != "none":
            status[self.state.integration_mode.capitalize()] = str(self.state.accumulated_frames)

        return status


def _get_raw(ctx: ModeContext) -> np.ndarray | None:
    """Get the most recent raw spectrum intensity."""
    if ctx.last_raw_intensity is not None:
        return ctx.last_raw_intensity
    if ctx.last_data is not None:
        return ctx.last_data.intensity
    return None
