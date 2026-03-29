"""Color Science mode for colorimetry and spectrum.

Four PREV views cycle via the Prev button:
  cs_crop     — spectrum graph + camera crop  (default)
  cs_color    — spectrum graph + solid sRGB swatch  (XYZ → display colour)
  cs_xy       — CIE xy chromaticity diagram + spectrum-bar preview
  cs_swatches — colour-swatch grid + current-colour swatch

Swatches store XYZ/L*a*b*, measurement type, and the full spectrum so they
can be exported as extra columns when saving.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.spectrum import SpectrumData

import numpy as np

from ..colorscience.chromaticity import render_chromaticity
from ..colorscience.illumination_metrics import cct_from_xyz, cri_from_spectrum
from ..colorscience.swatches import (
    ColorSwatch,
    delta_e_cie76,
    render_color_preview,
    render_spectrum_strip,
    render_swatch_grid,
    swatch_index_at,
)
from ..colorscience.xyz import calculate_XYZ, xyz_to_lab
from ..core.mode_context import ModeContext
from ..export.csv_exporter import build_markers_peaks_metadata
from ..processing.reference_correction import apply_dark_white_correction
from .base import BaseMode, ButtonDefinition, ModeType


class ColorMeasurementType(Enum):
    REFLECTANCE = auto()
    TRANSMITTANCE = auto()
    ILLUMINATION = auto()


_TO_XYZ_TYPE = {
    ColorMeasurementType.REFLECTANCE: "reflectance",
    ColorMeasurementType.TRANSMITTANCE: "transmittance",
    ColorMeasurementType.ILLUMINATION: "illumination",
}

_MODE_LETTER = {
    ColorMeasurementType.REFLECTANCE: "R",
    ColorMeasurementType.TRANSMITTANCE: "T",
    ColorMeasurementType.ILLUMINATION: "I",
}

_PREVIEW_MODES = ["cs_color", "cs_xy", "cs_swatches"]
_VIS_RANGE = (400.0, 750.0)  # wavelength window for preview strip


@dataclass
class ColorScienceState:
    """Mutable state for Color Science mode."""

    measurement_type: ColorMeasurementType = ColorMeasurementType.ILLUMINATION
    dark_spectrum: np.ndarray | None = None
    # White reference: raw SPD. Refl/Trans use for R/T; Illumination for intensity scale by max(white).
    white_spectrum: np.ndarray | None = None
    # Illumination: (X_w, Y_w, Z_w) for L*a*b* reference (Y=100). Optional.
    white_level_xyz: tuple[float, float, float] | None = None
    # Illumination: max(white - dark) so XYZ scale matches normalized spectrum (sample/max_white).
    white_max: float = 1.0
    show_raw_overlay: bool = False
    show_absorption: bool = False
    # Acquisition mode labels recorded when each reference was captured
    dark_acq: str = ""
    white_acq: str = ""
    swatches: list[ColorSwatch] = field(default_factory=list)
    selected: set[int] = field(default_factory=set)


class ColorScienceMode(BaseMode):
    """Color Science mode: XYZ/LAB colorimetry, reflectance/transmittance/illumination."""

    def __init__(self):
        super().__init__()
        self.color_state = ColorScienceState()

    # ------------------------------------------------------------------
    # BaseMode interface
    # ------------------------------------------------------------------

    @property
    def mode_type(self) -> ModeType:
        return ModeType.COLORSCIENCE

    @property
    def name(self) -> str:
        return "Color Science"

    @property
    def preview_modes(self) -> list[str]:
        """Four PREV views cycled by the Prev button."""
        return _PREVIEW_MODES

    def get_buttons(self) -> list[ButtonDefinition]:
        """Row 1: Play | type | Dark/White | Add/Del/CLR | Save/Load. Row 2: grouped with gaps like Measurement."""
        return [
            # Row 1: Play | Refl/Trans/Illum | Dark/White | Add/Del/CLR | Save/Load
            ButtonDefinition("Play", "freeze", is_toggle=True, row=1, icon_type="playback"),
            ButtonDefinition(
                "S", "toggle_sensitivity", is_toggle=True, row=1, icon_type="sensitivity"
            ),
            ButtonDefinition("__gap__", "__gap__", row=1),
            ButtonDefinition("Refl", "type_reflectance", is_toggle=True, row=1),
            ButtonDefinition("Trans", "type_transmittance", is_toggle=True, row=1),
            ButtonDefinition("Illum", "type_illumination", is_toggle=True, row=1),
            ButtonDefinition("__gap__", "__gap__2", row=1),
            ButtonDefinition("Dark", "set_dark", is_toggle=True, row=1, icon_type="dark"),
            ButtonDefinition("White", "set_white", is_toggle=True, row=1, icon_type="white"),
            ButtonDefinition(
                "Overlay", "show_raw_overlay", is_toggle=True, row=1, icon_type="overlay"
            ),
            ButtonDefinition(
                "ABS", "show_absorption", is_toggle=True, row=1, icon_type="absorption"
            ),
            ButtonDefinition("__gap__", "__gap__3", row=1),
            ButtonDefinition("Add", "add_swatch", row=1),
            ButtonDefinition("Del", "del_swatch", row=1),
            ButtonDefinition("CLR", "clear_swatches", row=1, icon_type="clear"),
            ButtonDefinition("__gap__", "__gap__4", row=1),
            ButtonDefinition("Save", "save", shortcut="s", row=1, icon_type="save"),
            ButtonDefinition("SVG", "export_vector", shortcut="g", row=1),
            ButtonDefinition("Pdf", "export_pdf", shortcut="p", row=1),
            ButtonDefinition("Load", "load", row=1, icon_type="load"),
            # Row 2: Avg/Peak/Acc/Peaks | Bars | ZX/ZY | G/E/AG/AE | VIEW | Lamp | spacer | Quit
            ButtonDefinition("Avg", "toggle_averaging", is_toggle=True, row=2, icon_type="avg"),
            ButtonDefinition(
                "Max", "capture_peak", is_toggle=True, shortcut="h", row=2, icon_type="peak_hold"
            ),
            ButtonDefinition("Acc", "toggle_accumulation", is_toggle=True, row=2, icon_type="acc"),
            ButtonDefinition("Peaks", "show_peaks", is_toggle=True, row=2, icon_type="peaks"),
            ButtonDefinition("__gap__", "__gap__5", row=2),
            ButtonDefinition("Bars", "show_spectrum_bars", is_toggle=True, row=2, icon_type="bars"),
            ButtonDefinition("__gap__", "__gap__6", row=2),
            ButtonDefinition("ZX", "show_zoom_x_slider", is_toggle=True, row=2, icon_type="zoom_x"),
            ButtonDefinition("ZY", "show_zoom_y_slider", is_toggle=True, row=2, icon_type="zoom_y"),
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
            ButtonDefinition("VIEW", "cycle_preview", shortcut="v", row=2, icon_type="eye"),
            ButtonDefinition("__gap__", "__gap__9", row=2),
            ButtonDefinition("Lamp", "lamp_toggle", is_toggle=True, row=2, icon_type="lamp"),
            ButtonDefinition("__spacer__", "__spacer_right__", row=2),
            ButtonDefinition("Quit", "quit", row=2, icon_type="quit"),
        ]

    def on_start(self, ctx: ModeContext) -> None:
        """Restore button states and enter default view."""
        for t in ColorMeasurementType:
            ctx.display.set_button_active(
                f"type_{t.name.lower()}", t == self.color_state.measurement_type
            )
        ctx.display.set_button_active("set_dark", self.color_state.dark_spectrum is not None)
        ctx.display.set_button_active("set_white", self._is_white_set())
        ctx.display.set_button_active("show_raw_overlay", self.color_state.show_raw_overlay)
        refl_trans = self.color_state.measurement_type in (
            ColorMeasurementType.REFLECTANCE,
            ColorMeasurementType.TRANSMITTANCE,
        )
        white_set = self.color_state.white_spectrum is not None
        ctx.display.set_button_disabled("show_absorption", not (refl_trans and white_set))
        ctx.display.set_button_active("show_absorption", self.color_state.show_absorption)
        ctx.display.set_button_active("freeze", not ctx.frozen_spectrum)
        ctx.display.state.peaks_visible = False
        ctx.display.set_button_active("show_peaks", False)
        ctx.display.set_button_active("capture_peak", ctx.display.state.hold_peaks)
        ctx.display.preview_mode = "cs_crop"
        ctx.display.set_graph_click_callback(self._make_swatch_click_cb(ctx))
        self._sync_lamp_control_bar(ctx)

    def on_stop(self, ctx: ModeContext) -> None:
        """Clear overrides and callbacks when leaving the mode."""
        ctx.display.set_graph_override(None)
        ctx.display.set_preview_override(None)
        ctx.display.set_graph_click_callback(None)

    def setup(self, ctx: ModeContext) -> None:
        """Register all button callbacks."""
        super().setup(ctx)
        self.register_callback(
            "type_reflectance", lambda: self._on_type(ctx, ColorMeasurementType.REFLECTANCE)
        )
        self.register_callback(
            "type_transmittance", lambda: self._on_type(ctx, ColorMeasurementType.TRANSMITTANCE)
        )
        self.register_callback(
            "type_illumination", lambda: self._on_type(ctx, ColorMeasurementType.ILLUMINATION)
        )
        self.register_callback("set_dark", lambda: self._on_toggle_dark(ctx))
        self.register_callback("set_white", lambda: self._on_toggle_white(ctx))
        self.register_callback("show_raw_overlay", lambda: self._on_toggle_raw_overlay(ctx))
        self.register_callback("show_absorption", lambda: self._on_toggle_absorption(ctx))
        self.register_callback("save", lambda: self._on_save(ctx))
        self.register_callback("export_vector", lambda: self._on_export_vector(ctx))
        self.register_callback("export_pdf", lambda: self._on_export_pdf(ctx))
        self.register_callback("load", lambda: self._on_load(ctx))
        self.register_callback("add_swatch", lambda: self._on_add_swatch(ctx))
        self.register_callback("del_swatch", lambda: self._on_del_swatch(ctx))
        self.register_callback("clear_swatches", lambda: self._on_clear_swatches(ctx))
        self.register_callback("toggle_averaging", lambda: self._on_toggle_averaging(ctx))
        self.register_callback("toggle_accumulation", lambda: self._on_toggle_accumulation(ctx))
        self.register_callback("freeze", lambda: self._on_toggle_freeze(ctx))

    # ------------------------------------------------------------------
    # Spectrum processing
    # ------------------------------------------------------------------

    def process_spectrum(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Apply mode-specific correction. When overlay on: only dark subtract + scale to 0-1."""
        result = intensity.astype(np.float64)
        if self.state.integration_mode != "none":
            result = self.accumulate_spectrum(result)
        self._last_measured_pre_correction = result.copy()

        dark = self.color_state.dark_spectrum
        white = self.color_state.white_spectrum
        mtype = self.color_state.measurement_type

        if self.color_state.show_raw_overlay:
            if dark is not None:
                result = result - np.asarray(dark, dtype=np.float64)
                result = np.maximum(result, 0)
            m = float(np.max(result)) if result.size else 1.0
            m = max(m, 1e-10)
            return np.clip((result / m).astype(np.float32), 0, 1)

        if mtype == ColorMeasurementType.ILLUMINATION:
            if dark is not None:
                result = result - np.asarray(dark, dtype=np.float64)
                result = np.maximum(result, 0)
            if white is not None:
                white_arr = np.asarray(white, dtype=np.float64)
                if dark is not None:
                    white_arr = white_arr - np.asarray(dark, dtype=np.float64)
                white_max = float(np.max(np.maximum(white_arr, 0)))
                if white_max >= 1e-10:
                    result = result / white_max
            return result.astype(np.float32)

        if mtype in (ColorMeasurementType.REFLECTANCE, ColorMeasurementType.TRANSMITTANCE):
            result = apply_dark_white_correction(result, dark, white)
            if self.color_state.show_absorption and white is not None:
                r_t = np.maximum(result.astype(np.float64), 1e-10)
                absorption = -np.log10(r_t)
                result = np.clip(absorption / 4.0, 0, 1).astype(np.float32)
            else:
                result = result.astype(np.float32)
            return result

        return np.clip(result.astype(np.float32), 0, 1)

    def get_overlay(self, wavelengths: np.ndarray, graph_height: int):
        return None

    # ------------------------------------------------------------------
    # Display update (called every frame)
    # ------------------------------------------------------------------

    def update_display(
        self,
        ctx: ModeContext,
        processed: "SpectrumData",
        graph_height: int,
    ) -> None:
        """Update graph and preview strip according to current PREV view."""
        ctx.display.set_button_disabled("auto_calibrate", False)
        ctx.display.set_button_active("freeze", not ctx.frozen_spectrum)
        ctx.display.set_button_active("show_raw_overlay", self.color_state.show_raw_overlay)
        refl_trans = self.color_state.measurement_type in (
            ColorMeasurementType.REFLECTANCE,
            ColorMeasurementType.TRANSMITTANCE,
        )
        white_set = self.color_state.white_spectrum is not None
        ctx.display.set_button_disabled("show_absorption", not (refl_trans and white_set))
        if (not refl_trans or not white_set) and self.color_state.show_absorption:
            self.color_state.show_absorption = False
            ctx.display.set_button_active("show_absorption", False)
        ctx.display.set_button_active("show_absorption", self.color_state.show_absorption)
        ctx.display.state.graph_click_behavior = "default"

        if self.color_state.show_raw_overlay:
            ctx.display.set_raw_overlays(
                _build_raw_overlays(
                    ctx,
                    self.color_state.dark_spectrum,
                    self.color_state.white_spectrum,
                    processed.wavelengths,
                )
            )
        else:
            ctx.display.set_raw_overlays([])
        ctx.display.set_mode_overlay(None)
        ctx.display.set_sensitivity_overlay(None)

        xyz_lab = self._compute_xyz_lab(processed.intensity, processed.wavelengths)
        cat_white = self._display_cat_white_point(processed.wavelengths)
        power_ratio = None
        if refl_trans and white_set and not self.color_state.show_absorption:
            power_ratio = _power_ratio(
                processed.intensity,
                self.color_state.dark_spectrum,
                self.color_state.white_spectrum,
            )
        cct_duv = None
        cri_ra = None
        if (
            self.color_state.measurement_type == ColorMeasurementType.ILLUMINATION
            and xyz_lab is not None
        ):
            X, Y, Z = xyz_lab[0]
            cct_duv = cct_from_xyz(X, Y, Z)
            cri_ra = cri_from_spectrum(processed.wavelengths, processed.intensity)
        info_lines = _xyz_lab_lines(
            xyz_lab,
            white_set=white_set,
            cct_duv=cct_duv,
            cri_ra=cri_ra,
            power_ratio=power_ratio,
        )

        width = ctx.display.config.display.window_width
        p_height = ctx.display.config.display.preview_height

        pm = ctx.display.preview_mode

        match pm:
            case "cs_xy":
                # xy diagram fills the graph — no text overlay needed
                ctx.display.set_graph_info(None)
                xy = _xy_from_xyz(xyz_lab)
                diagram = render_chromaticity(width, graph_height, xy_current=xy)
                ctx.display.set_graph_override(diagram)
                ctx.display.set_preview_override(
                    render_spectrum_strip(
                        width,
                        p_height,
                        processed.wavelengths,
                        processed.intensity,
                        wl_range=_VIS_RANGE,
                    )
                )

            case "cs_swatches":
                # Swatch grid carries its own XYZ/LAB labels — no overlay
                ctx.display.set_graph_info(None)
                xyz = xyz_lab[0] if xyz_lab else None
                lab = xyz_lab[1] if xyz_lab else None
                current = (xyz, lab) if (xyz and lab) else None
                grid = render_swatch_grid(
                    width,
                    graph_height,
                    self.color_state.swatches,
                    current_xyz_lab=current,
                    current_wavelengths=processed.wavelengths,
                    current_spectrum=processed.intensity,
                    current_illuminant_xyz=cat_white,
                )
                ctx.display.set_graph_override(grid)
                ctx.display.set_preview_override(
                    render_spectrum_strip(
                        width,
                        p_height,
                        processed.wavelengths,
                        processed.intensity,
                        wl_range=_VIS_RANGE,
                    )
                )

            case _:
                # cs_color and any unknown mode: spectrum graph + XYZ/LAB overlay
                ctx.display.set_graph_override(None)
                xyz = xyz_lab[0] if xyz_lab else None
                ctx.display.set_preview_override(
                    render_color_preview(
                        width,
                        p_height,
                        xyz,
                        info_lines,
                        illuminant_xyz=cat_white,
                    )
                )

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_toggle_freeze(self, ctx: ModeContext) -> None:
        """Toggle spectrum freeze (same behavior as calibration Play)."""
        frozen = self.toggle_freeze()
        ctx.frozen_spectrum = frozen
        if frozen:
            raw = ctx.last_raw_intensity
            if raw is not None:
                ctx.frozen_intensity = raw.copy()
            elif ctx.last_data is not None:
                ctx.frozen_intensity = ctx.last_data.intensity.copy()
            else:
                ctx.frozen_intensity = None
        else:
            ctx.frozen_intensity = None
        ctx.display.set_button_active("freeze", not frozen)
        print(f"[FREEZE] Spectrum {'FROZEN' if frozen else 'LIVE'}")

    def _is_white_set(self) -> bool:
        """True when a white reference is set for the current type."""
        return self.color_state.white_spectrum is not None

    def _on_type(self, ctx: ModeContext, mtype: ColorMeasurementType) -> None:
        """Switch measurement type (radio-button)."""
        self.color_state.measurement_type = mtype
        for t in ColorMeasurementType:
            ctx.display.set_button_active(f"type_{t.name.lower()}", t == mtype)
        # White button reflects a different state for each type
        ctx.display.set_button_active("set_white", self._is_white_set())
        print(f"[COLOR] Mode: {mtype.name}")

    def _on_toggle_dark(self, ctx: ModeContext) -> None:
        """Set or clear the dark reference."""
        if self.color_state.dark_spectrum is not None:
            self.color_state.dark_spectrum = None
            self.color_state.dark_acq = ""
            ctx.display.set_button_active("set_dark", False)
            print("[COLOR] Dark reference cleared")
            return
        raw = _get_raw(ctx)
        if raw is None:
            return
        self.color_state.dark_acq = self.acq_label(ctx.display.state.hold_peaks)
        self.color_state.dark_spectrum = raw.copy()
        ctx.display.set_button_active("set_dark", True)
        print("[COLOR] Dark reference set")

    def _on_toggle_white(self, ctx: ModeContext) -> None:
        """Route to white-level (illumination) or white-spectrum (refl/trans)."""
        if self.color_state.measurement_type == ColorMeasurementType.ILLUMINATION:
            self._on_toggle_white_level(ctx)
        else:
            self._on_toggle_white_spectrum(ctx)

    def _on_toggle_white_level(self, ctx: ModeContext) -> None:
        """Set/clear white reference for illumination: scale by max(white-dark), L*a*b* ref from white XYZ."""
        if self.color_state.white_spectrum is not None:
            self.color_state.white_level_xyz = None
            self.color_state.white_spectrum = None
            self.color_state.white_acq = ""
            self.color_state.white_max = 1.0
            ctx.display.set_button_active("set_white", False)
            print("[COLOR] White level cleared")
            return

        raw = _get_raw(ctx)
        if raw is None:
            return
        self.color_state.white_acq = self.acq_label(ctx.display.state.hold_peaks)
        self.color_state.white_spectrum = raw.copy()
        dark = self.color_state.dark_spectrum
        white_arr = np.asarray(raw, dtype=np.float64)
        if dark is not None:
            white_arr = white_arr - np.asarray(dark, dtype=np.float64)
        white_arr = np.maximum(white_arr, 0)
        self.color_state.white_max = float(np.max(white_arr))
        if self.color_state.white_max < 1e-10:
            self.color_state.white_max = 1.0
        if ctx.last_data is not None:
            wl = ctx.last_data.wavelengths
            try:
                X_w, Y_w, Z_w = calculate_XYZ(
                    white_arr.astype(np.float32),
                    wl,
                    "illumination",
                    None,
                    None,
                )
                if Y_w >= 1e-10:
                    self.color_state.white_level_xyz = (float(X_w), float(Y_w), float(Z_w))
            except (FileNotFoundError, ValueError):
                pass
        ctx.display.set_button_active("set_white", True)
        print("[COLOR] White level set (intensity scale + L* ref)")

    def _on_toggle_white_spectrum(self, ctx: ModeContext) -> None:
        """Set or clear the white reference spectrum (Reflectance / Transmittance)."""
        if self.color_state.white_spectrum is not None:
            self.color_state.white_spectrum = None
            self.color_state.white_acq = ""
            self.color_state.show_absorption = False
            ctx.display.set_button_active("set_white", False)
            ctx.display.set_button_active("show_absorption", False)
            ctx.display.set_button_disabled("show_absorption", True)
            print("[COLOR] White reference cleared")
            return
        raw = _get_raw(ctx)
        if raw is None:
            return
        self.color_state.white_acq = self.acq_label(ctx.display.state.hold_peaks)
        self.color_state.white_spectrum = raw.copy()
        ctx.display.set_button_active("set_white", True)
        ctx.display.set_button_disabled("show_absorption", False)
        print("[COLOR] White reference set")

    def _on_toggle_raw_overlay(self, ctx: ModeContext) -> None:
        """Toggle overlay of dark, white, and measured SPDs (unmodified)."""
        self.color_state.show_raw_overlay = not self.color_state.show_raw_overlay
        ctx.display.set_button_active("show_raw_overlay", self.color_state.show_raw_overlay)
        print(f"[COLOR] Raw overlay: {'ON' if self.color_state.show_raw_overlay else 'OFF'}")

    def _on_toggle_absorption(self, ctx: ModeContext) -> None:
        """Toggle absorption spectrum (Refl/Trans only, requires white)."""
        if self.color_state.measurement_type not in (
            ColorMeasurementType.REFLECTANCE,
            ColorMeasurementType.TRANSMITTANCE,
        ):
            return
        if self.color_state.white_spectrum is None:
            return
        self.color_state.show_absorption = not self.color_state.show_absorption
        ctx.display.set_button_active("show_absorption", self.color_state.show_absorption)
        print(f"[COLOR] Absorption: {'ON' if self.color_state.show_absorption else 'OFF'}")

    def _on_add_swatch(self, ctx: ModeContext) -> None:
        """Capture current measurement as a new swatch."""
        if ctx.last_data is None:
            return
        data = ctx.last_data
        xyz_lab = self._compute_xyz_lab(data.intensity, data.wavelengths)
        if xyz_lab is None:
            print("[COLOR] Cannot add swatch: color computation failed")
            return
        (X, Y, Z), (L, a, b) = xyz_lab
        letter = _MODE_LETTER[self.color_state.measurement_type]
        ill_xyz = self._display_cat_white_point(data.wavelengths)
        swatch = ColorSwatch(
            X=X,
            Y=Y,
            Z=Z,
            L=L,
            a=a,
            b=b,
            mode=letter,
            wavelengths=data.wavelengths.copy(),
            spectrum=data.intensity.copy(),
            label=f"S{len(self.color_state.swatches) + 1}",
            illuminant_xyz=ill_xyz,
        )
        self.color_state.swatches.append(swatch)
        print(f"[COLOR] Swatch added: L*={L:.1f} a*={a:.1f} b*={b:.1f}  [{letter}]")

    def _on_del_swatch(self, ctx: ModeContext) -> None:
        """Delete the selected swatches."""
        if not self.color_state.selected:
            print("[COLOR] No swatch selected")
            return
        for idx in sorted(self.color_state.selected, reverse=True):
            if 0 <= idx < len(self.color_state.swatches):
                removed = self.color_state.swatches.pop(idx)
                print(f"[COLOR] Swatch {removed.label} deleted")
        self.color_state.selected.clear()
        # Reindex swatch labels
        for i, s in enumerate(self.color_state.swatches):
            s.selected = False
            s.label = f"S{i + 1}"

    def _on_clear_swatches(self, ctx: ModeContext) -> None:
        """Remove all swatches."""
        self.color_state.swatches.clear()
        self.color_state.selected.clear()
        print("[COLOR] All swatches cleared")

    def snapshot_metadata(self, ctx: ModeContext) -> dict:
        """CSV/PDF/SVG header fields (same keys as Save). Requires ``ctx.last_data``."""
        light_on = self.state.lamp_enabled
        led_pct = ctx.display.get_led_intensity_value() if light_on else None
        pwm_log = (
            f"{math.log10(0.01 + led_pct / 100):.4f}" if led_pct is not None and light_on else None
        )
        is_illum = self.color_state.measurement_type == ColorMeasurementType.ILLUMINATION
        metadata = {
            "Mode": "Color Science",
            "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Measurement": self.color_state.measurement_type.name,
            "Light": "Lamp" if light_on else "off",
            "PWM%": f"{led_pct:.1f}" if led_pct is not None and light_on else None,
            "PWM_log10": pwm_log,
            "Gain": f"{ctx.camera.gain:.1f}",
            "Exposure": f"{getattr(ctx.camera, 'exposure', 0)}",
            "Measured_acq": self.acq_label(ctx.display.state.hold_peaks),
            "Dark_acq": self.color_state.dark_acq or None,
            "White_acq": self.color_state.white_acq or None,
        }
        if ctx.last_data is None:
            return metadata

        xyz_lab = self._compute_xyz_lab(ctx.last_data.intensity, ctx.last_data.wavelengths)
        if xyz_lab is not None:
            (X, Y, Z), (L, a, b) = xyz_lab
            metadata["XYZ"] = f"X={X:.2f} Y={Y:.2f} Z={Z:.2f}"
            metadata["LAB"] = f"L*={L:.2f} a*={a:.2f} b*={b:.2f}"
            s = X + Y + Z
            if s > 0:
                metadata["xy"] = f"x={X / s:.4f} y={Y / s:.4f}"
            if is_illum:
                cct_duv = cct_from_xyz(X, Y, Z)
                if cct_duv is not None:
                    metadata["CCT_K"] = f"{cct_duv[0]:.0f}"
                    metadata["Duv"] = f"{cct_duv[1]:+.4f}"
                cri_ra = cri_from_spectrum(ctx.last_data.wavelengths, ctx.last_data.intensity)
                if cri_ra is not None:
                    metadata["CRI_Ra"] = f"{cri_ra:.1f}"

        wl = self.color_state.white_level_xyz
        if wl is not None:
            metadata["WhiteLevel"] = f"X={wl[0]:.2f} Y={wl[1]:.2f} Z={wl[2]:.2f}"

        for i, sw in enumerate(self.color_state.swatches):
            metadata[f"Swatch_{i + 1}"] = (
                f"{sw.label} {sw.mode} L*={sw.L:.1f} a*={sw.a:.1f} b*={sw.b:.1f}"
            )

        markers_peaks = build_markers_peaks_metadata(
            ctx.display.state.marker_lines,
            ctx.last_data.wavelengths,
            ctx.last_data.intensity,
            ctx.last_data.peaks,
        )
        metadata.update(markers_peaks)
        return metadata

    def _on_save(self, ctx: ModeContext) -> None:
        """Save spectrum + swatch spectra as columns."""
        if ctx.last_data is None:
            print("[COLOR] No spectrum data available")
            return
        is_illum = self.color_state.measurement_type == ColorMeasurementType.ILLUMINATION
        white = None if is_illum else self.color_state.white_spectrum
        metadata = self.snapshot_metadata(ctx)
        measured_pre = getattr(self, "_last_measured_pre_correction", None)
        measured = measured_pre if measured_pre is not None else ctx.last_raw_intensity
        ctx.save_snapshot(
            ctx.last_data,
            dark_intensity=self.color_state.dark_spectrum,
            white_intensity=white,
            metadata=metadata,
            measured_raw_intensity=measured,
            extra_spectra=[
                (sw.label, sw.wavelengths, sw.spectrum) for sw in self.color_state.swatches
            ],
        )

    def _on_export_vector(self, ctx: ModeContext) -> None:
        ctx.export_vector_graph()

    def _on_export_pdf(self, ctx: ModeContext) -> None:
        ctx.export_pdf_report()

    def _on_load(self, ctx: ModeContext) -> None:
        print("[COLOR] Load spectrum (file dialog not implemented yet)")

    def _on_toggle_averaging(self, ctx: ModeContext) -> None:
        enabled = self.toggle_averaging()
        ctx.display.set_button_active("toggle_averaging", enabled)
        ctx.display.set_button_active("toggle_accumulation", False)

    def _on_toggle_accumulation(self, ctx: ModeContext) -> None:
        enabled = self.toggle_accumulation()
        ctx.display.set_button_active("toggle_accumulation", enabled)
        ctx.display.set_button_active("toggle_averaging", False)

    # ------------------------------------------------------------------
    # Swatch click handling (registered as graph-click callback)
    # ------------------------------------------------------------------

    def _make_swatch_click_cb(self, ctx: ModeContext):
        """Return a closure that handles swatch-grid clicks."""

        def _click(x: int, rel_y: int) -> None:
            if ctx.display.preview_mode != "cs_swatches":
                return
            w = ctx.display.config.display.window_width
            h = ctx.display.config.display.graph_height
            idx = swatch_index_at(x, rel_y, w, h, len(self.color_state.swatches))
            if idx is None:
                return
            self._toggle_swatch_selection(idx)

        return _click

    def _toggle_swatch_selection(self, idx: int) -> None:
        """Toggle selection of a swatch; at most 2 selected at once."""
        swatches = self.color_state.swatches
        sel = self.color_state.selected
        if idx in sel:
            sel.discard(idx)
            swatches[idx].selected = False
        elif len(sel) < 2:
            sel.add(idx)
            swatches[idx].selected = True
        else:
            # Already 2 selected: replace oldest selection
            old = next(iter(sel))
            sel.discard(old)
            swatches[old].selected = False
            sel.add(idx)
            swatches[idx].selected = True

        if len(sel) == 2:
            idxs = list(sel)
            de = delta_e_cie76(swatches[idxs[0]].lab, swatches[idxs[1]].lab)
            print(
                f"[COLOR] ΔE76 between {swatches[idxs[0]].label} and "
                f"{swatches[idxs[1]].label}: {de:.2f}"
            )

    # ------------------------------------------------------------------
    # Colorimetry
    # ------------------------------------------------------------------

    def _display_cat_white_point(
        self,
        wavelengths: np.ndarray,
    ) -> tuple[float, float, float] | None:
        """Tristimulus of the reference white for CAT → D65.

        Reflectance/transmittance only. Illumination: no CAT — show physical
        chromaticity of the light.
        """
        m = self.color_state.measurement_type
        white = self.color_state.white_spectrum
        if white is None:
            return None
        if m in (
            ColorMeasurementType.REFLECTANCE,
            ColorMeasurementType.TRANSMITTANCE,
        ):
            return _white_point_xyz_reflective(white, wavelengths, m)
        return None

    def _compute_xyz_lab(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """Return ((X,Y,Z), (L*,a*,b*)) or None on failure."""
        try:
            mtype = _TO_XYZ_TYPE[self.color_state.measurement_type]
            white = self.color_state.white_spectrum
            white_wl = wavelengths

            if mtype == "illumination":
                if white is not None:
                    white_arr = np.asarray(white, dtype=np.float64)
                    dark = self.color_state.dark_spectrum
                    if dark is not None:
                        white_arr = white_arr - np.asarray(dark, dtype=np.float64)
                    white_arr = np.maximum(white_arr, 0)
                    w_max = self.color_state.white_max
                    raw_sample = (
                        intensity.astype(np.float64) * w_max
                        if w_max >= 1e-10
                        else intensity.astype(np.float64)
                    )
                    X, Y, Z = calculate_XYZ(
                        spectrum=raw_sample,
                        wavelengths=wavelengths,
                        measurement_type="illumination",
                        illuminant_spectrum=white_arr,
                        illuminant_wavelengths=white_wl,
                    )
                    ref_white = _illumination_ref_white_from_spectrum(white_arr, wavelengths)
                else:
                    X, Y, Z = calculate_XYZ(
                        spectrum=intensity,
                        wavelengths=wavelengths,
                        measurement_type="illumination",
                        illuminant_spectrum=None,
                        illuminant_wavelengths=None,
                    )
                    ref_white = _illumination_ref_white(
                        self.color_state.white_level_xyz,
                        self.color_state.white_max,
                    )
                    X, Y, Z = _apply_white_level(
                        X,
                        Y,
                        Z,
                        self.color_state.white_level_xyz,
                        self.color_state.white_max,
                    )
            else:
                if white is None:
                    return None
                X, Y, Z = calculate_XYZ(
                    spectrum=intensity,
                    wavelengths=wavelengths,
                    measurement_type=mtype,
                    illuminant_spectrum=white,
                    illuminant_wavelengths=white_wl,
                )
                ref_white = _white_point_xyz_reflective(
                    white, wavelengths, self.color_state.measurement_type
                )

            L, a, b = xyz_to_lab(X, Y, Z, reference_white_XYZ=ref_white)
            return ((X, Y, Z), (L, a, b))
        except (FileNotFoundError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _illumination_ref_white_from_spectrum(
    white_arr: np.ndarray,
    wavelengths: np.ndarray,
) -> tuple[float, float, float]:
    """Reference white for L*a*b* when using _xyz_illuminant_reference (Y=100)."""
    return tuple(
        float(v)
        for v in calculate_XYZ(
            white_arr,
            wavelengths,
            "illumination",
            illuminant_spectrum=white_arr,
            illuminant_wavelengths=wavelengths,
        )
    )


def _white_point_xyz_reflective(
    white: np.ndarray,
    wavelengths: np.ndarray,
    mtype: ColorMeasurementType,
) -> tuple[float, float, float]:
    """CIE XYZ for R=T=1 under the captured white reference (same basis as sample XYZ)."""
    key = _TO_XYZ_TYPE[mtype]
    ones = np.ones(len(wavelengths), dtype=np.float64)
    return tuple(float(v) for v in calculate_XYZ(ones, wavelengths, key, white, wavelengths))


def _get_raw(ctx: ModeContext) -> np.ndarray | None:
    if ctx.last_raw_intensity is not None:
        return ctx.last_raw_intensity
    if ctx.last_data is not None:
        return ctx.last_data.intensity
    return None


def _build_raw_overlays(
    ctx: ModeContext,
    dark: np.ndarray | None,
    white: np.ndarray | None,
    wavelengths: np.ndarray,
) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    """Build (intensity 0-1, BGR) list for dark, white, measured (raw) for overlay."""
    n = len(wavelengths)
    if n == 0:
        return []
    raw = _get_raw(ctx)
    spectra: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    if dark is not None:
        d = np.asarray(dark, dtype=np.float64)[:n]
        spectra.append((d, (40, 40, 40)))
    if white is not None:
        w = np.asarray(white, dtype=np.float64)[:n]
        spectra.append((w, (200, 120, 0)))
    if raw is not None:
        m = np.asarray(raw, dtype=np.float64)[:n]
        spectra.append((m, (0, 0, 200)))
    if not spectra:
        return []
    global_max = max(float(np.max(s[0])) for s in spectra)
    global_max = max(global_max, 1.0)
    return [(np.clip(arr / global_max, 0, 1).astype(np.float32), color) for arr, color in spectra]


def _power_ratio(
    processed: np.ndarray,
    dark: np.ndarray | None,
    white: np.ndarray | None,
) -> float | None:
    """Relative power (reflectance/transmittance): sum(sample-dark)/sum(white-dark). Returns None if not defined."""
    if dark is None or white is None or len(processed) == 0:
        return None
    n = min(len(processed), len(dark), len(white))
    white_sub = np.maximum(
        np.asarray(white, dtype=np.float64)[:n] - np.asarray(dark, dtype=np.float64)[:n],
        1e-10,
    )
    # processed is R or T = (sample-dark)/(white-dark), so sample-dark = processed * (white-dark)
    numer = np.sum(processed[:n].astype(np.float64) * white_sub)
    denom = float(np.sum(white_sub))
    if denom < 1e-20:
        return None
    return float(numer / denom)


def _apply_white_level(
    X: float,
    Y: float,
    Z: float,
    white_level_xyz: tuple[float, float, float] | None,
    white_max: float = 1.0,
) -> tuple[float, float, float]:
    """Scale XYZ so the white-level reference reads Y = 100.

    Illumination passes spectrum = (sample - dark) / max(white - dark), so its XYZ
    is (1/white_max) * XYZ(raw). We store white_level_xyz = XYZ(raw white - dark).
    So scale must be 100 * white_max / Y_w so that (Y_w/white_max) * scale = 100.
    """
    if white_level_xyz is None:
        return (X, Y, Z)
    _, Y_w, _ = white_level_xyz
    if Y_w < 1e-10:
        return (X, Y, Z)
    scale = 100.0 * white_max / Y_w
    return (X * scale, Y * scale, Z * scale)


def _illumination_ref_white(
    white_level_xyz: tuple[float, float, float] | None,
    white_max: float = 1.0,
) -> tuple[float, float, float]:
    """Return the L*a*b* reference white for an illumination measurement.

    Without a white level: equal-energy E → (100, 100, 100).
    With a white level: scale so the reference white reads Y = 100 (same scale
    as _apply_white_level so L* is correct).
    """
    if white_level_xyz is None:
        return (100.0, 100.0, 100.0)
    X_w, Y_w, Z_w = white_level_xyz
    if Y_w < 1e-10:
        return (100.0, 100.0, 100.0)
    scale = 100.0 * white_max / Y_w
    return (X_w * scale, 100.0, Z_w * scale)


def _xyz_lab_lines(
    xyz_lab: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    white_set: bool = False,
    cct_duv: tuple[float, float] | None = None,
    cri_ra: float | None = None,
    power_ratio: float | None = None,
) -> list[str]:
    if xyz_lab is None:
        lines = []
    else:
        (X, Y, Z), (L, a, b) = xyz_lab
        wl_tag = "  [WL]" if white_set else ""
        lines = [
            f"X={X:.1f}  Y={Y:.1f}  Z={Z:.1f}{wl_tag}",
            f"L*={L:.1f}  a*={a:.1f}  b*={b:.1f}",
        ]
        if cct_duv is not None:
            cct_k, duv = cct_duv
            lines.append(f"CCT: {cct_k:.0f} K  Δuv={duv:+.4f}")
        if cri_ra is not None:
            lines.append(f"CRI Ra: {cri_ra:.1f}")
    if power_ratio is not None:
        lines.append(f"P/P_w: {power_ratio:.3f}")
    return lines


def _xy_from_xyz(
    xyz_lab: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
) -> tuple[float, float] | None:
    if xyz_lab is None:
        return None
    X, Y, Z = xyz_lab[0]
    s = X + Y + Z
    return (X / s, Y / s) if s > 0 else None
