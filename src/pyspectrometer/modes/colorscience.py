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
from ..colorscience.swatches import (
    GRID_COLS,
    GRID_ROWS,
    ColorSwatch,
    delta_e_cie76,
    render_color_preview,
    render_spectrum_strip,
    render_swatch_grid,
    swatch_index_at,
    xyz_to_display_bgr,
)
from ..colorscience.xyz import calculate_XYZ, xyz_to_lab
from ..core.mode_context import ModeContext
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
_VIS_RANGE = (400.0, 750.0)   # wavelength window for preview strip


@dataclass
class ColorScienceState:
    """Mutable state for Color Science mode."""

    measurement_type: ColorMeasurementType = ColorMeasurementType.ILLUMINATION
    dark_spectrum: np.ndarray | None = None
    # Reflectance / Transmittance: raw white-reference spectrum
    white_spectrum: np.ndarray | None = None
    # Illumination: (X_w, Y_w, Z_w) of a reference white measurement.
    # All subsequent XYZ values are scaled by 100/Y_w so the reference reads
    # Y=100, and L*a*b* is computed against the chromatically-correct
    # reference white (X_w/Y_w·100, 100, Z_w/Y_w·100).
    white_level_xyz: tuple[float, float, float] | None = None
    swatches: list[ColorSwatch] = field(default_factory=list)
    # Indices into swatches list that are currently selected (max 2)
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
        """Row 1: type switch | XYZ/LAB | Dark/White | Add/Del/CLR | Save/Load."""
        return [
            # Measurement type (radio-switch)
            ButtonDefinition("Refl",  "type_reflectance",   is_toggle=True, row=1),
            ButtonDefinition("Trans", "type_transmittance", is_toggle=True, row=1),
            ButtonDefinition("Illum", "type_illumination",  is_toggle=True, row=1),
            ButtonDefinition("", "__gap__", row=1),
            # References
            ButtonDefinition("Dark",  "set_dark",  is_toggle=True, row=1),
            ButtonDefinition("White", "set_white", is_toggle=True, row=1),
            ButtonDefinition("", "__gap__", row=1),
            # Swatch management
            ButtonDefinition("Add", "add_swatch",   row=1),
            ButtonDefinition("Del", "del_swatch",   row=1),
            ButtonDefinition("CLR", "clear_swatches", row=1),
            ButtonDefinition("", "__gap__", row=1),
            ButtonDefinition("Save", "save", shortcut="s", row=1),
            ButtonDefinition("Load", "load", row=1),
            # Row 2: display and camera controls
            ButtonDefinition("Avg",  "toggle_averaging",    is_toggle=True, row=2),
            ButtonDefinition("Acc",  "toggle_accumulation", is_toggle=True, row=2),
            ButtonDefinition("Bars", "show_spectrum_bars",  is_toggle=True, row=2),
            ButtonDefinition("ZX",   "show_zoom_x_slider",  is_toggle=True, row=2),
            ButtonDefinition("ZY",   "show_zoom_y_slider",  is_toggle=True, row=2),
            ButtonDefinition("G",    "show_gain_slider",    is_toggle=True, row=2),
            ButtonDefinition("E",    "show_exposure_slider",is_toggle=True, row=2),
            ButtonDefinition("AG",   "auto_gain",           is_toggle=True, row=2),
            ButtonDefinition("AE",   "auto_exposure",       is_toggle=True, row=2),
            ButtonDefinition("Prev", "cycle_preview", shortcut="v", row=2),
            ButtonDefinition("Lamp", "lamp_toggle",   is_toggle=True, row=2),
            ButtonDefinition("Quit", "quit", row=2),
            ButtonDefinition("__spacer__", "__spacer_right__", row=2),
        ]

    def on_start(self, ctx: ModeContext) -> None:
        """Restore button states and enter default view."""
        for t in ColorMeasurementType:
            ctx.display.set_button_active(
                f"type_{t.name.lower()}", t == self.color_state.measurement_type
            )
        ctx.display.set_button_active("set_dark",  self.color_state.dark_spectrum is not None)
        ctx.display.set_button_active("set_white", self._is_white_set())
        ctx.display.state.peaks_visible = False
        # Start in plain spectrum + crop view
        ctx.display.preview_mode = "cs_crop"
        ctx.display.set_graph_click_callback(self._make_swatch_click_cb(ctx))

    def on_stop(self, ctx: ModeContext) -> None:
        """Clear overrides and callbacks when leaving the mode."""
        ctx.display.set_graph_override(None)
        ctx.display.set_preview_override(None)
        ctx.display.set_graph_click_callback(None)

    def setup(self, ctx: ModeContext) -> None:
        """Register all button callbacks."""
        super().setup(ctx)
        self.register_callback("type_reflectance",
                               lambda: self._on_type(ctx, ColorMeasurementType.REFLECTANCE))
        self.register_callback("type_transmittance",
                               lambda: self._on_type(ctx, ColorMeasurementType.TRANSMITTANCE))
        self.register_callback("type_illumination",
                               lambda: self._on_type(ctx, ColorMeasurementType.ILLUMINATION))
        self.register_callback("set_dark",           lambda: self._on_toggle_dark(ctx))
        self.register_callback("set_white",          lambda: self._on_toggle_white(ctx))
        self.register_callback("save",               lambda: self._on_save(ctx))
        self.register_callback("load",               lambda: self._on_load(ctx))
        self.register_callback("add_swatch",         lambda: self._on_add_swatch(ctx))
        self.register_callback("del_swatch",         lambda: self._on_del_swatch(ctx))
        self.register_callback("clear_swatches",     lambda: self._on_clear_swatches(ctx))
        self.register_callback("toggle_averaging",   lambda: self._on_toggle_averaging(ctx))
        self.register_callback("toggle_accumulation",lambda: self._on_toggle_accumulation(ctx))
        self.register_callback("lamp_toggle",        lambda: self._on_toggle_light(ctx))

    # ------------------------------------------------------------------
    # Spectrum processing
    # ------------------------------------------------------------------

    def process_spectrum(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Apply dark/white correction per measurement type."""
        result = intensity.astype(np.float64)
        if self.state.integration_mode != "none":
            result = self.accumulate_spectrum(result)

        dark  = self.color_state.dark_spectrum
        white = None if self.color_state.measurement_type == ColorMeasurementType.ILLUMINATION \
                     else self.color_state.white_spectrum
        return apply_dark_white_correction(result, dark, white)

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
        ctx.display.set_mode_overlay(None)
        ctx.display.set_sensitivity_overlay(None)

        xyz_lab = self._compute_xyz_lab(processed.intensity, processed.wavelengths)
        info_lines = _xyz_lab_lines(xyz_lab, white_set=self._is_white_set())

        width  = ctx.display.config.display.window_width
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
                    render_spectrum_strip(width, p_height,
                                         processed.wavelengths, processed.intensity,
                                         wl_range=_VIS_RANGE)
                )

            case "cs_swatches":
                # Swatch grid carries its own XYZ/LAB labels — no overlay
                ctx.display.set_graph_info(None)
                xyz = xyz_lab[0] if xyz_lab else None
                lab = xyz_lab[1] if xyz_lab else None
                current = (xyz, lab) if (xyz and lab) else None
                grid = render_swatch_grid(
                    width, graph_height,
                    self.color_state.swatches,
                    current_xyz_lab=current,
                    current_wavelengths=processed.wavelengths,
                    current_spectrum=processed.intensity,
                )
                ctx.display.set_graph_override(grid)
                ctx.display.set_preview_override(
                    render_spectrum_strip(width, p_height,
                                         processed.wavelengths, processed.intensity,
                                         wl_range=_VIS_RANGE)
                )

            case _:
                # cs_color and any unknown mode: spectrum graph + XYZ/LAB overlay
                ctx.display.set_graph_override(None)
                xyz = xyz_lab[0] if xyz_lab else None
                ctx.display.set_preview_override(
                    render_color_preview(width, p_height, xyz, info_lines)
                )

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _is_white_set(self) -> bool:
        """True when a white reference of any kind is active for the current type."""
        if self.color_state.measurement_type == ColorMeasurementType.ILLUMINATION:
            return self.color_state.white_level_xyz is not None
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
            ctx.display.set_button_active("set_dark", False)
            print("[COLOR] Dark reference cleared")
            return
        raw = _get_raw(ctx)
        if raw is None:
            return
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
        """Set/clear the white-level scalar for illumination mode.

        Captures the current measurement's XYZ as a reference white point.
        All subsequent XYZ values are scaled so this reference reads Y = 100,
        making L*a*b* meaningful relative to the scene white.
        """
        if self.color_state.white_level_xyz is not None:
            self.color_state.white_level_xyz = None
            ctx.display.set_button_active("set_white", False)
            print("[COLOR] White level cleared")
            return

        if ctx.last_data is None:
            return
        # white_level_xyz is None here, so _compute_xyz_lab returns equal-energy
        # normalised raw values — exactly what we want to store as the reference.
        xyz_lab = self._compute_xyz_lab(ctx.last_data.intensity, ctx.last_data.wavelengths)
        if xyz_lab is None:
            print("[COLOR] Cannot set white level: color computation failed")
            return
        X_w, Y_w, Z_w = xyz_lab[0]
        if Y_w < 1e-10:
            print("[COLOR] Cannot set white level: Y ≈ 0")
            return
        self.color_state.white_level_xyz = (X_w, Y_w, Z_w)
        ctx.display.set_button_active("set_white", True)
        print(f"[COLOR] White level set: X={X_w:.2f} Y={Y_w:.2f} Z={Z_w:.2f}")

    def _on_toggle_white_spectrum(self, ctx: ModeContext) -> None:
        """Set or clear the white reference spectrum (Reflectance / Transmittance)."""
        if self.color_state.white_spectrum is not None:
            self.color_state.white_spectrum = None
            ctx.display.set_button_active("set_white", False)
            print("[COLOR] White reference cleared")
            return
        raw = _get_raw(ctx)
        if raw is None:
            return
        self.color_state.white_spectrum = raw.copy()
        ctx.display.set_button_active("set_white", True)
        print("[COLOR] White reference set")

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
        swatch = ColorSwatch(
            X=X, Y=Y, Z=Z, L=L, a=a, b=b,
            mode=letter,
            wavelengths=data.wavelengths.copy(),
            spectrum=data.intensity.copy(),
            label=f"S{len(self.color_state.swatches) + 1}",
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

    def _on_save(self, ctx: ModeContext) -> None:
        """Save spectrum + swatch spectra as columns."""
        if ctx.last_data is None:
            print("[COLOR] No spectrum data available")
            return
        light_on = self.state.lamp_enabled
        led_pct  = ctx.display.get_led_intensity_value() if light_on else None
        pwm_log  = (
            f"{math.log10(0.01 + led_pct / 100):.4f}"
            if led_pct is not None and light_on else None
        )
        is_illum = self.color_state.measurement_type == ColorMeasurementType.ILLUMINATION
        white    = None if is_illum else self.color_state.white_spectrum

        xyz_lab = self._compute_xyz_lab(ctx.last_data.intensity, ctx.last_data.wavelengths)
        metadata = {
            "Mode":        "Color Science",
            "Date":        time.strftime("%Y-%m-%d %H:%M:%S"),
            "Measurement": self.color_state.measurement_type.name,
            "Light":       "Lamp" if light_on else "off",
            "PWM%":        f"{led_pct:.1f}" if led_pct is not None and light_on else None,
            "PWM_log10":   pwm_log,
            "Gain":        f"{ctx.camera.gain:.1f}",
            "Exposure":    f"{getattr(ctx.camera, 'exposure', 0)}",
        }
        if xyz_lab is not None:
            (X, Y, Z), (L, a, b) = xyz_lab
            metadata["XYZ"] = f"X={X:.2f} Y={Y:.2f} Z={Z:.2f}"
            metadata["LAB"] = f"L*={L:.2f} a*={a:.2f} b*={b:.2f}"
            s = X + Y + Z
            if s > 0:
                metadata["xy"] = f"x={X/s:.4f} y={Y/s:.4f}"

        wl = self.color_state.white_level_xyz
        if wl is not None:
            metadata["WhiteLevel"] = f"X={wl[0]:.2f} Y={wl[1]:.2f} Z={wl[2]:.2f}"

        for i, sw in enumerate(self.color_state.swatches):
            metadata[f"Swatch_{i+1}"] = (
                f"{sw.label} {sw.mode} L*={sw.L:.1f} a*={sw.a:.1f} b*={sw.b:.1f}"
            )

        ctx.save_snapshot(
            ctx.last_data,
            dark_intensity=self.color_state.dark_spectrum,
            white_intensity=white,
            metadata=metadata,
            extra_spectra=[(sw.label, sw.wavelengths, sw.spectrum)
                           for sw in self.color_state.swatches],
        )

    def _on_load(self, ctx: ModeContext) -> None:
        print("[COLOR] Load spectrum (file dialog not implemented yet)")

    def _on_toggle_averaging(self, ctx: ModeContext) -> None:
        enabled = self.toggle_averaging()
        ctx.display.set_button_active("toggle_averaging",   enabled)
        ctx.display.set_button_active("toggle_accumulation", False)

    def _on_toggle_accumulation(self, ctx: ModeContext) -> None:
        enabled = self.toggle_accumulation()
        ctx.display.set_button_active("toggle_accumulation", enabled)
        ctx.display.set_button_active("toggle_averaging",    False)

    def _on_toggle_light(self, ctx: ModeContext) -> None:
        print("[COLOR] Lamp toggle (GPIO not implemented yet)")

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
        sel      = self.color_state.selected
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
            print(f"[COLOR] ΔE76 between {swatches[idxs[0]].label} and "
                  f"{swatches[idxs[1]].label}: {de:.2f}")

    # ------------------------------------------------------------------
    # Colorimetry
    # ------------------------------------------------------------------

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
                white = white_wl = None
            elif white is None:
                return None

            X, Y, Z = calculate_XYZ(
                spectrum=intensity,
                wavelengths=wavelengths,
                measurement_type=mtype,
                illuminant_spectrum=white,
                illuminant_wavelengths=white_wl,
            )
            if mtype == "illumination":
                ref_white = _illumination_ref_white(
                    X, Y, Z, self.color_state.white_level_xyz
                )
                X, Y, Z = _apply_white_level(X, Y, Z, self.color_state.white_level_xyz)
            else:
                ref_white = tuple(
                    float(v) for v in
                    calculate_XYZ(white, wavelengths, "illumination", None, None)
                )
            L, a, b = xyz_to_lab(X, Y, Z, reference_white_XYZ=ref_white)
            return ((X, Y, Z), (L, a, b))
        except (FileNotFoundError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _get_raw(ctx: ModeContext) -> np.ndarray | None:
    if ctx.last_raw_intensity is not None:
        return ctx.last_raw_intensity
    if ctx.last_data is not None:
        return ctx.last_data.intensity
    return None


def _apply_white_level(
    X: float, Y: float, Z: float,
    white_level_xyz: tuple[float, float, float] | None,
) -> tuple[float, float, float]:
    """Scale XYZ so the white-level reference reads Y = 100."""
    if white_level_xyz is None:
        return (X, Y, Z)
    _, Y_w, _ = white_level_xyz
    if Y_w < 1e-10:
        return (X, Y, Z)
    scale = 100.0 / Y_w
    return (X * scale, Y * scale, Z * scale)


def _illumination_ref_white(
    X: float, Y: float, Z: float,
    white_level_xyz: tuple[float, float, float] | None,
) -> tuple[float, float, float]:
    """Return the L*a*b* reference white for an illumination measurement.

    Without a white level: equal-energy E → (100, 100, 100).
    With a white level: scale the captured white XYZ so Y_w → 100, preserving
    the chromaticity of the scene white (e.g. D65-ish daylight).
    """
    if white_level_xyz is None:
        return (100.0, 100.0, 100.0)
    X_w, Y_w, Z_w = white_level_xyz
    if Y_w < 1e-10:
        return (100.0, 100.0, 100.0)
    scale = 100.0 / Y_w
    return (X_w * scale, 100.0, Z_w * scale)


def _xyz_lab_lines(
    xyz_lab: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
    white_set: bool = False,
) -> list[str]:
    if xyz_lab is None:
        return []
    (X, Y, Z), (L, a, b) = xyz_lab
    wl_tag = "  [WL]" if white_set else ""
    return [
        f"X={X:.1f}  Y={Y:.1f}  Z={Z:.1f}{wl_tag}",
        f"L*={L:.1f}  a*={a:.1f}  b*={b:.1f}",
    ]


def _xy_from_xyz(
    xyz_lab: tuple[tuple[float, float, float], tuple[float, float, float]] | None,
) -> tuple[float, float] | None:
    if xyz_lab is None:
        return None
    X, Y, Z = xyz_lab[0]
    s = X + Y + Z
    return (X / s, Y / s) if s > 0 else None
