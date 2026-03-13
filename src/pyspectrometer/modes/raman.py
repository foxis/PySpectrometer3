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
from ..processing.raman import wavelengths_to_wavenumbers
from ..processing.reference_correction import apply_dark_white_correction
from ..utils.graph_scale import scale_intensity_to_graph
from .base import BaseMode, ButtonDefinition, ModeType


@dataclass
class RamanState:
    """State specific to Raman mode."""

    dark_spectrum: np.ndarray | None = None
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

    def get_buttons(self) -> list[ButtonDefinition]:
        """Get Raman mode buttons per architecture."""
        return [
            ButtonDefinition("Save", "save", shortcut="s", row=1),
            ButtonDefinition("Load", "load", row=1),
            ButtonDefinition("Set Ref", "set_reference", row=1),
            ButtonDefinition("Avg", "toggle_averaging", is_toggle=True, row=1),
            ButtonDefinition("Acc", "toggle_accumulation", is_toggle=True, row=1),
            ButtonDefinition("Black", "set_dark", is_toggle=True, row=1),
            ButtonDefinition("Peak", "capture_peak", is_toggle=True, row=1),
            ButtonDefinition("Peaks", "show_peaks", is_toggle=True, row=2),
            ButtonDefinition("Bars", "show_spectrum_bars", is_toggle=True, row=2),
            ButtonDefinition("ZX", "show_zoom_x_slider", is_toggle=True, row=2),
            ButtonDefinition("ZY", "show_zoom_y_slider", is_toggle=True, row=2),
            ButtonDefinition("G", "show_gain_slider", is_toggle=True, row=2),
            ButtonDefinition("E", "show_exposure_slider", is_toggle=True, row=2),
            ButtonDefinition("AG", "auto_gain", is_toggle=True, row=2),
            ButtonDefinition("AE", "auto_exposure", is_toggle=True, row=2),
            ButtonDefinition("Prev", "cycle_preview", shortcut="v", row=2),
            ButtonDefinition("Quit", "quit", row=2),
            ButtonDefinition("__spacer__", "__spacer_right__", row=2),
        ]

    def setup(self, ctx: ModeContext) -> None:
        """Register Raman-specific handlers."""
        super().setup(ctx)
        self.register_callback("save", lambda: self._on_save(ctx))
        self.register_callback("load", lambda: self._on_load(ctx))
        self.register_callback("set_reference", lambda: self._on_set_reference(ctx))
        self.register_callback("set_dark", lambda: self._on_set_dark(ctx))
        self.register_callback("toggle_averaging", lambda: self._on_toggle_averaging(ctx))
        self.register_callback("toggle_accumulation", lambda: self._on_toggle_accumulation(ctx))

    def _on_save(self, ctx: ModeContext) -> None:
        """Save spectrum (after transform to wavenumber)."""
        if ctx.last_data is None:
            print("[RAMAN] No spectrum data available")
            return
        metadata = {
            "Mode": "Raman",
            "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Base wavelength (nm)": f"{self.laser_nm:.1f}",
            "Gain": f"{ctx.camera.gain:.1f}",
            "Exposure": f"{getattr(ctx.camera, 'exposure', 0)}",
            "Note": "",
        }
        ctx.save_snapshot(ctx.last_data, metadata=metadata)

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
