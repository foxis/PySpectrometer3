"""Raman mode for Raman scattering spectroscopy.

MVP: Laser wavelength config, wavelength→wavenumber conversion, display in cm⁻¹,
dark reference, averaging, save/load.
"""

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.spectrum import SpectrumData
import numpy as np

from ..core.calibration import GraticuleData, graticule_from_x_axis
from ..core.mode_context import ModeContext
from ..core.spectrum import SpectrumData
from ..export.csv_exporter import build_markers_peaks_metadata
from ..processing.raman import wavelengths_to_wavenumbers
from ..processing.reference_correction import apply_dark_white_correction
from ..utils.graph_scale import scale_intensity_to_graph
from .base import BaseMode, ButtonDefinition, ModeType


@dataclass
class RamanState:
    """State specific to Raman mode."""

    dark_spectrum: np.ndarray | None = None
    dark_acq: str = ""
    reference_spectrum: np.ndarray | None = None


class RamanMode(BaseMode):
    """Raman mode: display spectrum in cm⁻¹, dark reference, save/load."""

    def __init__(self, laser_nm: float = 785.0):
        """Initialize Raman mode.

        Args:
            laser_nm: Excitation laser wavelength in nm
        """
        super().__init__()
        self.raman_state = RamanState()
        self.laser_nm = laser_nm

    @property
    def mode_type(self) -> ModeType:
        return ModeType.RAMAN

    @property
    def name(self) -> str:
        return "Raman"

    @property
    def preview_modes(self) -> list[str]:
        """No full image preview; spectrum bar or none only, like Measurement."""
        return ["none", "spectrum"]

    def on_start(self, ctx: ModeContext) -> None:
        """Hide camera preview by default and sync Play (freeze) button."""
        ctx.display.preview_mode = "none"
        ctx.display.set_button_active("freeze", not ctx.frozen_spectrum)
        self._sync_lamp_control_bar(ctx)

    def get_buttons(self) -> list[ButtonDefinition]:
        """Layout aligned with Measurement: Play first, grouped with gaps, Quit right-aligned."""
        return [
            # Row 1: Play | Avg/Max/Acc | Dark/Ref | Bars | ZX/ZY | VIEW
            ButtonDefinition("Play", "freeze", is_toggle=True, row=1, icon_type="playback"),
            ButtonDefinition(
                "S", "toggle_sensitivity", is_toggle=True, row=1, icon_type="sensitivity"
            ),
            ButtonDefinition("__gap__", "__gap__", row=1),
            ButtonDefinition("Avg", "toggle_averaging", is_toggle=True, row=1, icon_type="avg"),
            ButtonDefinition(
                "Max", "capture_peak", is_toggle=True, shortcut="h", row=1, icon_type="peak_hold"
            ),
            ButtonDefinition("Acc", "toggle_accumulation", is_toggle=True, row=1, icon_type="acc"),
            ButtonDefinition("__gap__", "__gap__2", row=1),
            ButtonDefinition("Dark", "set_dark", is_toggle=True, row=1, icon_type="dark"),
            ButtonDefinition("Ref", "set_reference", row=1, icon_type="reference"),
            ButtonDefinition("__gap__", "__gap__3", row=1),
            ButtonDefinition("Bars", "show_spectrum_bars", is_toggle=True, row=1, icon_type="bars"),
            ButtonDefinition("__gap__", "__gap__4", row=1),
            ButtonDefinition("ZX", "show_zoom_x_slider", is_toggle=True, row=1, icon_type="zoom_x"),
            ButtonDefinition("ZY", "show_zoom_y_slider", is_toggle=True, row=1, icon_type="zoom_y"),
            ButtonDefinition("__gap__", "__gap__5", row=1),
            ButtonDefinition("VIEW", "cycle_preview", shortcut="v", row=1, icon_type="eye"),
            # Row 2: Lamp | Peaks | G/E/AG/AE | Save/Load | spacer | Quit
            ButtonDefinition("Lamp", "lamp_toggle", is_toggle=True, row=2, icon_type="lamp"),
            ButtonDefinition("__gap__", "__gap__6", row=2),
            ButtonDefinition("Peaks", "show_peaks", is_toggle=True, row=2, icon_type="peaks"),
            ButtonDefinition("__gap__", "__gap__7", row=2),
            ButtonDefinition("G", "show_gain_slider", is_toggle=True, row=2, icon_type="gain"),
            ButtonDefinition(
                "E", "show_exposure_slider", is_toggle=True, row=2, icon_type="exposure"
            ),
            ButtonDefinition("AG", "auto_gain", is_toggle=True, row=2, icon_type="auto_gain"),
            ButtonDefinition(
                "AE", "auto_exposure", is_toggle=True, row=2, icon_type="auto_exposure"
            ),
            ButtonDefinition("__gap__", "__gap__8", row=2),
            ButtonDefinition("Save", "save", shortcut="s", row=2, icon_type="save"),
            ButtonDefinition("Load", "load", row=2, icon_type="load"),
            ButtonDefinition("__spacer__", "__spacer_right__", row=2),
            ButtonDefinition("Quit", "quit", row=2, icon_type="quit"),
        ]

    def setup(self, ctx: ModeContext) -> None:
        """Register Raman-specific handlers."""
        super().setup(ctx)
        self.register_callback("freeze", lambda: self._on_toggle_freeze(ctx))
        self.register_callback("save", lambda: self._on_save(ctx))
        self.register_callback("load", lambda: self._on_load(ctx))
        self.register_callback("set_reference", lambda: self._on_set_reference(ctx))
        self.register_callback("set_dark", lambda: self._on_set_dark(ctx))
        self.register_callback("toggle_averaging", lambda: self._on_toggle_averaging(ctx))
        self.register_callback("toggle_accumulation", lambda: self._on_toggle_accumulation(ctx))

    def _on_toggle_freeze(self, ctx: ModeContext) -> None:
        """Toggle spectrum freeze (same behavior as Measurement/Color Science Play)."""
        frozen = self.toggle_freeze()
        ctx.frozen_spectrum = frozen
        if frozen and ctx.last_data is not None:
            ctx.frozen_intensity = ctx.last_data.intensity.copy()
        else:
            ctx.frozen_intensity = None
        ctx.display.set_button_active("freeze", not frozen)
        print(f"[RAMAN] Spectrum {'FROZEN' if frozen else 'LIVE'}")

    def _on_save(self, ctx: ModeContext) -> None:
        """Save spectrum (after transform to wavenumber)."""
        if ctx.last_data is None:
            print("[RAMAN] No spectrum data available")
            return
        measured_acq = self.acq_label(ctx.display.state.hold_peaks)
        metadata = {
            "Mode": "Raman",
            "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Base wavelength (nm)": f"{self.laser_nm:.1f}",
            "Gain": f"{ctx.camera.gain:.1f}",
            "Exposure": f"{getattr(ctx.camera, 'exposure', 0)}",
            "Measured_acq": measured_acq,
            "Dark_acq": self.raman_state.dark_acq or None,
            "Note": "",
        }
        markers_peaks = build_markers_peaks_metadata(
            ctx.display.state.marker_lines,
            ctx.last_data.wavelengths,
            ctx.last_data.intensity,
            ctx.last_data.peaks,
        )
        metadata.update(markers_peaks)
        measured_pre = getattr(self, "_last_measured_pre_correction", None)
        measured = measured_pre if measured_pre is not None else ctx.last_raw_intensity
        ctx.save_snapshot(
            ctx.last_data,
            metadata=metadata,
            measured_raw_intensity=measured,
        )

    def _on_load(self, ctx: ModeContext) -> None:
        """Load spectrum - placeholder."""
        print("[RAMAN] Load spectrum (file dialog not implemented yet)")

    def _on_set_reference(self, ctx: ModeContext) -> None:
        """Set current spectrum as reference overlay (displayed in cm⁻¹)."""
        if ctx.last_data is None:
            print("[RAMAN] No spectrum data available")
            return
        self.raman_state.reference_spectrum = ctx.last_data.intensity.copy()
        print("[RAMAN] Reference spectrum set")

    def _on_set_dark(self, ctx: ModeContext) -> None:
        """Toggle dark reference: set if unset, clear if already set."""
        if self.raman_state.dark_spectrum is not None:
            self.raman_state.dark_spectrum = None
            self.raman_state.dark_acq = ""
            ctx.display.set_button_active("set_dark", False)
            print("[RAMAN] Dark reference cleared")
            return

        raw = (
            ctx.last_raw_intensity
            if ctx.last_raw_intensity is not None
            else (ctx.last_data.intensity if ctx.last_data is not None else None)
        )
        if raw is None:
            print("[RAMAN] No spectrum data available")
            return
        self.raman_state.dark_acq = self.acq_label(ctx.display.state.hold_peaks)
        self.raman_state.dark_spectrum = raw.copy()
        ctx.display.set_button_active("set_dark", True)
        print("[RAMAN] Dark reference set")

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

    def process_spectrum(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Apply dark subtraction and averaging."""
        result = intensity.astype(np.float64)

        if self.state.integration_mode != "none":
            result = self.accumulate_spectrum(result)
        self._last_measured_pre_correction = result.copy()

        result = apply_dark_white_correction(
            result,
            self.raman_state.dark_spectrum,
            None,
        )
        return result

    def transform_spectrum_data(self, data: "SpectrumData") -> "SpectrumData":
        """Convert wavelength axis to Raman shift (cm⁻¹)."""
        wavenumbers = wavelengths_to_wavenumbers(data.wavelengths, self.laser_nm)
        return SpectrumData(
            intensity=data.intensity,
            wavelengths=wavenumbers,
            timestamp=data.timestamp,
            peaks=data.peaks,
            raw_frame=data.raw_frame,
            cropped_frame=data.cropped_frame,
            x_axis_label="Raman shift (cm⁻¹)",
            gain=data.gain,
            exposure_us=data.exposure_us,
        )

    def get_graticule(self, data: "SpectrumData") -> GraticuleData | None:
        """Graticule with wavenumber labels for Raman display."""
        return graticule_from_x_axis(
            data.wavelengths,
            step=500,
            unit_suffix=" cm⁻¹",
        )

    def get_overlay(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> tuple[np.ndarray, tuple[int, int, int]] | None:
        """Reference spectrum overlay if set."""
        ref = self.raman_state.reference_spectrum
        if ref is None:
            return None
        max_val = max(ref.max(), 1)
        scaled = scale_intensity_to_graph(ref / max_val, graph_height)
        return (scaled, (150, 150, 150))

    def update_display(
        self,
        ctx: ModeContext,
        processed: "SpectrumData",
        graph_height: int,
    ) -> None:
        """Update Raman overlay and status."""
        ctx.display.set_button_disabled("auto_calibrate", False)
        ctx.display.state.graph_click_behavior = "default"
        overlay = self.get_overlay(processed.wavelengths, graph_height)
        ctx.display.set_mode_overlay(overlay)
        ctx.display.set_sensitivity_overlay(None)
        ctx.display.set_status("Laser", f"{self.laser_nm:.0f} nm")
        ctx.display.set_status("Unit", "cm⁻¹")
        if self.raman_state.dark_spectrum is not None:
            ctx.display.set_status("Black", "SET")
        if self.state.integration_mode != "none":
            ctx.display.set_status(
                self.state.integration_mode.capitalize(), str(self.state.accumulated_frames)
            )
