"""Color Science mode for colorimetry and spectrum.

MVP: Reflectance/Transmittance/Illumination sub-modes, XYZ/LAB display,
black/white references, save/load.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..core.spectrum import SpectrumData
import numpy as np

from .base import BaseMode, ModeType, ButtonDefinition
from ..core.mode_context import ModeContext
from ..utils.graph_scale import scale_intensity_to_graph
from ..processing.reference_correction import apply_dark_white_correction
from ..colorscience.xyz import calculate_XYZ, xyz_to_lab


class ColorMeasurementType(Enum):
    """Measurement type for color science."""

    REFLECTANCE = auto()
    TRANSMITTANCE = auto()
    ILLUMINATION = auto()


_MEASUREMENT_TO_XYZ_TYPE = {
    ColorMeasurementType.REFLECTANCE: "reflectance",
    ColorMeasurementType.TRANSMITTANCE: "transmittance",
    ColorMeasurementType.ILLUMINATION: "illumination",
}


@dataclass
class ColorScienceState:
    """State specific to Color Science mode."""

    measurement_type: ColorMeasurementType = ColorMeasurementType.ILLUMINATION
    dark_spectrum: Optional[np.ndarray] = None
    white_spectrum: Optional[np.ndarray] = None
    show_lab: bool = False  # False = XYZ, True = LAB


class ColorScienceMode(BaseMode):
    """Color Science mode: XYZ/LAB, reflectance/transmittance/illumination."""

    def __init__(self):
        """Initialize Color Science mode."""
        super().__init__()
        self.color_state = ColorScienceState()

    @property
    def mode_type(self) -> ModeType:
        return ModeType.COLORSCIENCE

    @property
    def name(self) -> str:
        return "Color Science"

    def get_buttons(self) -> list[ButtonDefinition]:
        """Get Color Science mode buttons per architecture."""
        return [
            ButtonDefinition("Refl", "type_reflectance", row=1),
            ButtonDefinition("Trans", "type_transmittance", row=1),
            ButtonDefinition("Illum", "type_illumination", row=1),
            ButtonDefinition("Black", "set_dark", row=1),
            ButtonDefinition("White", "set_white", row=1),
            ButtonDefinition("Save", "save", shortcut="s", row=1),
            ButtonDefinition("Load", "load", row=1),
            ButtonDefinition("XYZ/LAB", "toggle_xyz_lab", is_toggle=True, row=1),
            ButtonDefinition("Avg", "toggle_averaging", is_toggle=True, row=2),
            ButtonDefinition("Acc", "toggle_accumulation", is_toggle=True, row=2),
            ButtonDefinition("Peak", "capture_peak", is_toggle=True, row=2),
            ButtonDefinition("G", "show_gain_slider", is_toggle=True, row=2),
            ButtonDefinition("E", "show_exposure_slider", is_toggle=True, row=2),
            ButtonDefinition("AG", "auto_gain", is_toggle=True, row=2),
            ButtonDefinition("AE", "auto_exposure", is_toggle=True, row=2),
            ButtonDefinition("Prev", "cycle_preview", shortcut="v", row=2),
            ButtonDefinition("Lamp", "lamp_toggle", is_toggle=True, row=2),
            ButtonDefinition("Quit", "quit", row=2),
            ButtonDefinition("__spacer__", "__spacer_right__", row=2),
        ]

    def on_start(self, ctx: ModeContext) -> None:
        """Set initial measurement type and button states."""
        for t in ColorMeasurementType:
            ctx.display.set_button_active(f"type_{t.name.lower()}", t == self.color_state.measurement_type)
        ctx.display.set_button_active("toggle_xyz_lab", self.color_state.show_lab)

    def setup(self, ctx: ModeContext) -> None:
        """Register Color Science handlers."""
        super().setup(ctx)
        self.register_callback("type_reflectance", lambda: self._on_type(ctx, ColorMeasurementType.REFLECTANCE))
        self.register_callback("type_transmittance", lambda: self._on_type(ctx, ColorMeasurementType.TRANSMITTANCE))
        self.register_callback("type_illumination", lambda: self._on_type(ctx, ColorMeasurementType.ILLUMINATION))
        self.register_callback("set_dark", lambda: self._on_set_dark(ctx))
        self.register_callback("set_white", lambda: self._on_set_white(ctx))
        self.register_callback("save", lambda: self._on_save(ctx))
        self.register_callback("load", lambda: self._on_load(ctx))
        self.register_callback("toggle_xyz_lab", lambda: self._on_toggle_xyz_lab(ctx))
        self.register_callback("toggle_averaging", lambda: self._on_toggle_averaging(ctx))
        self.register_callback("toggle_accumulation", lambda: self._on_toggle_accumulation(ctx))
        self.register_callback("lamp_toggle", lambda: self._on_toggle_light(ctx))

    def _on_type(self, ctx: ModeContext, mtype: ColorMeasurementType) -> None:
        """Switch measurement type."""
        self.color_state.measurement_type = mtype
        for t in ColorMeasurementType:
            ctx.display.set_button_active(f"type_{t.name.lower()}", t == mtype)
        print(f"[COLOR] Mode: {mtype.name}")

    def _on_set_dark(self, ctx: ModeContext) -> None:
        """Set dark reference."""
        raw = ctx.last_raw_intensity if ctx.last_raw_intensity is not None else (ctx.last_data.intensity if ctx.last_data is not None else None)
        if raw is None:
            print("[COLOR] No spectrum data available")
            return
        self.color_state.dark_spectrum = raw.copy()
        print("[COLOR] Black reference set")

    def _on_set_white(self, ctx: ModeContext) -> None:
        """Set white reference."""
        raw = ctx.last_raw_intensity if ctx.last_raw_intensity is not None else (ctx.last_data.intensity if ctx.last_data is not None else None)
        if raw is None:
            print("[COLOR] No spectrum data available")
            return
        self.color_state.white_spectrum = raw.copy()
        print("[COLOR] White reference set")

    def _on_save(self, ctx: ModeContext) -> None:
        """Save spectrum."""
        if ctx.last_data is not None:
            ctx.save_snapshot(ctx.last_data)
        else:
            print("[COLOR] No spectrum data available")

    def _on_load(self, ctx: ModeContext) -> None:
        """Load spectrum - placeholder."""
        print("[COLOR] Load spectrum (file dialog not implemented yet)")

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

    def _on_toggle_xyz_lab(self, ctx: ModeContext) -> None:
        """Toggle XYZ/LAB display."""
        self.color_state.show_lab = not self.color_state.show_lab
        ctx.display.set_button_active("toggle_xyz_lab", self.color_state.show_lab)
        print(f"[COLOR] Display: {'LAB' if self.color_state.show_lab else 'XYZ'}")

    def _on_toggle_light(self, ctx: ModeContext) -> None:
        """Toggle lamp - placeholder."""
        print("[COLOR] Lamp toggle (GPIO not implemented yet)")

    def process_spectrum(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Apply dark/white correction per measurement type."""
        result = intensity.astype(np.float64)

        if self.state.integration_mode != "none":
            result = self.accumulate_spectrum(result)

        dark = self.color_state.dark_spectrum
        white = self.color_state.white_spectrum
        if self.color_state.measurement_type == ColorMeasurementType.ILLUMINATION:
            white = None
        result = apply_dark_white_correction(result, dark, white)

        return result

    def get_overlay(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> Optional[tuple[np.ndarray, tuple[int, int, int]]]:
        """No overlay for Color Science MVP."""
        return None

    def _compute_color_values(self, intensity: np.ndarray, wavelengths: np.ndarray) -> Optional[str]:
        """Compute XYZ or LAB using correct formula per measurement type.

        Reflectance/Transmittance: normalizes to white reference (our illuminant).
        Illumination: self-normalized (measures our light source).
        """
        try:
            mtype = _MEASUREMENT_TO_XYZ_TYPE[self.color_state.measurement_type]
            white = self.color_state.white_spectrum
            white_wl = wavelengths
            if mtype == "illumination":
                white = None
                white_wl = None
            elif white is None:
                return None

            X, Y, Z = calculate_XYZ(
                spectrum=intensity,
                wavelengths=wavelengths,
                measurement_type=mtype,
                illuminant_spectrum=white,
                illuminant_wavelengths=white_wl,
            )
            if self.color_state.show_lab:
                if mtype == "illumination":
                    ref_white = (X, Y, Z)
                else:
                    ref_white = tuple(
                        float(x)
                        for x in calculate_XYZ(
                            white, wavelengths, "illumination", None, None
                        )
                    )
                L, a, b = xyz_to_lab(X, Y, Z, reference_white_XYZ=ref_white)
                return f"L*={L:.1f} a*={a:.1f} b*={b:.1f}"
            return f"X={X:.1f} Y={Y:.1f} Z={Z:.1f}"
        except (FileNotFoundError, ValueError):
            return None

    def update_display(
        self,
        ctx: ModeContext,
        processed: "SpectrumData",
        graph_height: int,
    ) -> None:
        """Update Color Science status with XYZ/LAB."""
        ctx.display.set_mode_overlay(None)
        ctx.display.set_sensitivity_overlay(None)
        ctx.display.set_status("Mode", self.color_state.measurement_type.name[:4])
        if self.color_state.dark_spectrum is not None:
            ctx.display.set_status("Black", "SET")
        if self.color_state.white_spectrum is not None:
            ctx.display.set_status("White", "SET")
        color_str = self._compute_color_values(processed.intensity, processed.wavelengths)
        if color_str:
            ctx.display.set_status("Color", color_str)
