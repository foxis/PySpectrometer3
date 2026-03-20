"""CSV viewer mode: display a loaded spectrum CSV as if in measurement mode.

Static data — no capture, no freeze, no averaging.
Supports dark/white correction, absorption, normalization, zoom/pan, peaks, markers.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..data.reference_spectra import ReferenceSource, get_reference_spectrum
from ..export.csv_exporter import build_absorption_metadata, build_markers_peaks_metadata
from ..processing.reference_correction import apply_dark_white_correction
from ..utils.graph_scale import scale_intensity_to_graph
from .loader import LoadedCsv

if TYPE_CHECKING:
    from ..core.mode_context import ModeContext
    from ..core.spectrum import SpectrumData
from ..modes.base import BaseMode, ButtonDefinition, ModeType


# Same source list as CalibrationMode — all have data via colour-science or CSV files
SOURCES: list[ReferenceSource] = [
    ReferenceSource.HG,
    ReferenceSource.D65,
    ReferenceSource.A,
    ReferenceSource.FL1,
    ReferenceSource.FL2,
    ReferenceSource.FL3,
    ReferenceSource.FL12,
    ReferenceSource.LED,
    ReferenceSource.LED2,
    ReferenceSource.LED3,
]

# BGR overlay colour per illuminant
_ILLUM_COLOR: dict[ReferenceSource, tuple[int, int, int]] = {
    ReferenceSource.HG:   (220,  60,  60),   # blue  — mercury blue lines
    ReferenceSource.D65:  ( 50, 210, 255),   # yellow — daylight
    ReferenceSource.A:    (  0, 120, 255),   # orange — tungsten warmth
    ReferenceSource.FL1:  (255, 200,  80),   # sky blue
    ReferenceSource.FL2:  ( 80, 220, 100),   # green
    ReferenceSource.FL3:  ( 40, 160,  80),   # dark green
    ReferenceSource.FL12: (200,  80, 220),   # violet
    ReferenceSource.LED:  (255,  80, 180),   # pink
    ReferenceSource.LED2: (180,  60, 255),   # magenta
    ReferenceSource.LED3: (120,  40, 200),   # purple
}


@dataclass
class CsvViewerState:
    """Mutable view state for CSV viewer (static data, no camera)."""

    loaded: LoadedCsv | None = None
    dark: np.ndarray | None = None
    white: np.ndarray | None = None
    sensitivity: np.ndarray | None = None   # sensitivity curve stored in CSV
    show_absorption: bool = False
    refs_active: bool = True
    show_raw_overlay: bool = False
    show_sensitivity_curve: bool = False    # show sensor sensitivity as overlay
    show_csv_sensitivity: bool = False      # show sensitivity stored in CSV
    active_sources: set[ReferenceSource] = field(default_factory=set)


class CsvViewerMode(BaseMode):
    """Measurement-like viewer for a loaded spectrum CSV.

    Does not depend on camera, extractor, or auto-gain/exposure.
    """

    def __init__(self, loaded: LoadedCsv):
        super().__init__()
        self._view = CsvViewerState(
            loaded=loaded,
            dark=loaded.dark.copy() if loaded.dark is not None else None,
            white=loaded.white.copy() if loaded.white is not None else None,
            sensitivity=loaded.sensitivity.copy() if loaded.sensitivity is not None else None,
        )

    # ------------------------------------------------------------------
    # BaseMode interface
    # ------------------------------------------------------------------

    @property
    def mode_type(self) -> ModeType:
        return ModeType.MEASUREMENT

    @property
    def name(self) -> str:
        return "CSV Viewer"

    @property
    def preview_modes(self) -> list[str]:
        """No camera strip — locked to 'none'."""
        return ["none"]

    def on_start(self, ctx: "ModeContext") -> None:
        ctx.display.preview_mode = "none"
        self._sync_button_states(ctx)

    def setup(self, ctx: "ModeContext") -> None:
        super().setup(ctx)
        self._register_viewer_handlers(ctx)

    def process_spectrum(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Apply dark/white correction and optional absorption transform."""
        if not self._view.refs_active:
            return np.clip(intensity, 0, 1).astype(np.float32)

        dark = self._view.dark
        white = self._view.white

        result = apply_dark_white_correction(
            np.asarray(intensity, dtype=np.float64),
            dark,
            white,
        )

        if self._view.show_absorption and white is not None:
            transmission = np.maximum(result, 1e-10)
            absorption = -np.log10(transmission)
            result = np.clip(absorption / 4.0, 0, 1).astype(np.float32)

        return np.clip(result, 0, 1).astype(np.float32)

    def get_overlay(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> tuple[np.ndarray, tuple[int, int, int]] | None:
        """Return single overlay (white > dark > None) when raw-overlay is off."""
        if self._view.show_raw_overlay:
            return None
        n = len(wavelengths)
        if n == 0:
            return None

        if self._view.white is not None:
            arr = np.asarray(self._view.white, dtype=np.float64)[:n]
            max_val = max(float(arr.max()), 1e-10)
            return (scale_intensity_to_graph(arr / max_val, graph_height), (200, 120, 0))

        if self._view.dark is not None:
            arr = np.asarray(self._view.dark, dtype=np.float64)[:n]
            max_val = max(float(arr.max()), 1e-10)
            return (scale_intensity_to_graph(arr / max_val, graph_height), (60, 60, 60))

        return None

    def get_buttons(self) -> list[ButtonDefinition]:
        """Buttons for CSV viewer — fixed layout, no camera controls."""
        row1: list[ButtonDefinition] = [
            ButtonDefinition("S", "toggle_sensitivity", is_toggle=True, row=1),
            ButtonDefinition("SensOv", "show_sens_curve", is_toggle=True, row=1),
            ButtonDefinition("SensRef", "show_csv_sens", is_toggle=True, row=1),
            ButtonDefinition("__gap__", "__gap__illum", row=1),
        ]
        for source in SOURCES:
            row1.append(
                ButtonDefinition(source.name, f"source_{source.name.lower()}", is_toggle=True, row=1)
            )
        row1 += [
            ButtonDefinition("__gap__", "__gap__refs", row=1),
            ButtonDefinition("Overlay", "show_raw_overlay", is_toggle=True, row=1),
            ButtonDefinition("ABS", "show_absorption", is_toggle=True, row=1),
            ButtonDefinition("Refs", "toggle_refs", is_toggle=True, row=1),
            ButtonDefinition("ClearRef", "clear_refs", row=1),
            ButtonDefinition("__spacer__", "__spacer_r1__", row=1),
        ]

        row2: list[ButtonDefinition] = [
            ButtonDefinition("Peaks", "show_peaks", is_toggle=True, row=2),
            ButtonDefinition("Snap", "snap_to_peaks", is_toggle=True, row=2),
            ButtonDefinition("Pkd", "show_peak_delta", is_toggle=True, row=2),
            ButtonDefinition("Clr", "clear_all", shortcut="z", row=2),
            ButtonDefinition("__gap__", "__gap__r1", row=2),
            ButtonDefinition("ZX", "show_zoom_x_slider", is_toggle=True, row=2),
            ButtonDefinition("ZY", "show_zoom_y_slider", is_toggle=True, row=2),
            ButtonDefinition("__gap__", "__gap__r2", row=2),
            ButtonDefinition("Export", "export_csv", shortcut="s", row=2),
            ButtonDefinition("Load", "load_csv", row=2),
            ButtonDefinition("__spacer__", "__spacer_right__", row=2),
            ButtonDefinition("Quit", "quit", row=2),
        ]

        return row1 + row2

    def update_display(
        self,
        ctx: "ModeContext",
        processed: "SpectrumData",
        graph_height: int,
    ) -> None:
        ctx.display.state.graph_click_behavior = (
            "marker" if not ctx.display.is_peaks_visible() else "default"
        )
        if not self._view.white and self._view.show_absorption:
            self._view.show_absorption = False
        self._sync_button_states(ctx)

        if self._view.show_raw_overlay:
            ctx.display.set_raw_overlays(
                self._build_raw_overlays(processed.wavelengths, graph_height)
            )
            ctx.display.set_mode_overlay(None)
        else:
            ctx.display.set_raw_overlays(
                self._build_illuminant_overlays(processed.wavelengths, graph_height)
            )
            overlay = self.get_overlay(processed.wavelengths, graph_height)
            ctx.display.set_mode_overlay(overlay)

        sens_overlay = None
        if self._view.show_sensitivity_curve and ctx.sensitivity_engine is not None:
            sens_overlay = ctx.sensitivity_engine.get_curve_for_display(
                processed.wavelengths, graph_height
            )
        ctx.display.set_sensitivity_overlay(sens_overlay)
        ctx.display.state.reference_spectrum = None

        for key, value in self._status(ctx).items():
            ctx.display.set_status(key, value)

    # ------------------------------------------------------------------
    # Runtime reload (Load button swaps data in-place)
    # ------------------------------------------------------------------

    def reload(self, loaded: LoadedCsv, ctx: "ModeContext") -> None:
        """Replace loaded data in the running loop without restarting."""
        self._view.loaded = loaded
        self._view.dark = loaded.dark.copy() if loaded.dark is not None else None
        self._view.white = loaded.white.copy() if loaded.white is not None else None
        self._view.sensitivity = loaded.sensitivity.copy() if loaded.sensitivity is not None else None
        self._view.show_absorption = False
        self._view.refs_active = True
        self._view.show_raw_overlay = False
        self._view.show_csv_sensitivity = False
        self._view.active_sources.clear()
        self._sync_button_states(ctx)

    @property
    def loaded(self) -> LoadedCsv | None:
        return self._view.loaded

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _register_viewer_handlers(self, ctx: "ModeContext") -> None:
        self.register_callback("show_absorption", lambda: self._on_toggle_absorption(ctx))
        self.register_callback("toggle_refs", lambda: self._on_toggle_refs(ctx))
        self.register_callback("clear_refs", lambda: self._on_clear_refs(ctx))
        self.register_callback("show_raw_overlay", lambda: self._on_toggle_raw_overlay(ctx))
        self.register_callback("show_sens_curve", lambda: self._on_toggle_sens_curve(ctx))
        self.register_callback("show_csv_sens", lambda: self._on_toggle_csv_sens(ctx))
        for source in SOURCES:
            self.register_callback(
                f"source_{source.name.lower()}",
                lambda s=source: self._on_toggle_source(ctx, s),
            )
        self.register_callback("export_csv", lambda: self._on_export(ctx))
        self.register_callback("load_csv", lambda: self._on_load(ctx))
        self.register_callback("snap_to_peaks", lambda: self._on_snap_to_peaks(ctx))
        self.register_callback("clear_all", lambda: self._on_clear_all(ctx))
        self.register_callback("show_peak_delta", lambda: self._on_peak_delta(ctx))

    def _on_toggle_absorption(self, ctx: "ModeContext") -> None:
        if self._view.white is None:
            return
        self._view.show_absorption = not self._view.show_absorption
        ctx.display.set_button_active("show_absorption", self._view.show_absorption)

    def _on_toggle_refs(self, ctx: "ModeContext") -> None:
        self._view.refs_active = not self._view.refs_active
        ctx.display.set_button_active("toggle_refs", self._view.refs_active)
        if not self._view.refs_active:
            self._view.show_absorption = False
            ctx.display.set_button_active("show_absorption", False)

    def _on_clear_refs(self, ctx: "ModeContext") -> None:
        self._view.dark = None
        self._view.white = None
        self._view.show_absorption = False
        self._view.refs_active = True
        self._sync_button_states(ctx)

    def _on_toggle_raw_overlay(self, ctx: "ModeContext") -> None:
        self._view.show_raw_overlay = not self._view.show_raw_overlay
        ctx.display.set_button_active("show_raw_overlay", self._view.show_raw_overlay)

    def _on_snap_to_peaks(self, ctx: "ModeContext") -> None:
        active = ctx.display.toggle_snap_to_peaks()
        ctx.display.set_button_active("snap_to_peaks", active)

    def _on_clear_all(self, ctx: "ModeContext") -> None:
        ctx.display.state.marker_lines.clear()
        ctx.display.clear_peak_include_region()

    def _on_peak_delta(self, ctx: "ModeContext") -> None:
        active = ctx.display.toggle_peak_delta_visible()
        ctx.display.set_button_active("show_peak_delta", active)

    def _on_export(self, ctx: "ModeContext") -> None:
        if ctx.last_data is None:
            print("[CSV Viewer] No data to export")
            return
        data = ctx.last_data
        metadata: dict = {
            "Mode": "Measurement",
            "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Source": "CSV Viewer",
        }
        if self._view.loaded and self._view.loaded.metadata:
            src = self._view.loaded.metadata
            for k in ("Date", "Gain", "Exposure", "Note"):
                if k in src:
                    metadata[f"Original_{k}"] = src[k]

        markers_peaks = build_markers_peaks_metadata(
            ctx.display.state.marker_lines,
            data.wavelengths,
            data.intensity,
            data.peaks,
        )
        metadata.update(markers_peaks)

        if self._view.white is not None:
            metadata.update(
                build_absorption_metadata(
                    data.intensity, self._view.dark, self._view.white
                )
            )

        ctx.save_snapshot(
            data,
            dark_intensity=self._view.dark,
            white_intensity=self._view.white,
            metadata=metadata,
        )

    def _on_toggle_source(self, ctx: "ModeContext", source: ReferenceSource) -> None:
        if source in self._view.active_sources:
            self._view.active_sources.discard(source)
        else:
            self._view.active_sources.add(source)
        ctx.display.set_button_active(f"source_{source.name.lower()}", source in self._view.active_sources)

    def _on_toggle_sens_curve(self, ctx: "ModeContext") -> None:
        self._view.show_sensitivity_curve = not self._view.show_sensitivity_curve
        ctx.display.set_button_active("show_sens_curve", self._view.show_sensitivity_curve)

    def _on_toggle_csv_sens(self, ctx: "ModeContext") -> None:
        if self._view.sensitivity is None:
            return
        self._view.show_csv_sensitivity = not self._view.show_csv_sensitivity
        ctx.display.set_button_active("show_csv_sens", self._view.show_csv_sensitivity)

    def _on_load(self, ctx: "ModeContext") -> None:
        """Signal the spectrometer loop to prompt for a new file."""
        # The CsvViewerSpectrometer watches ctx for this flag
        ctx.pending_csv_reload = True  # type: ignore[attr-defined]

    def _sync_button_states(self, ctx: "ModeContext") -> None:
        """Sync all button active/disabled states to current view state."""
        has_refs = self._view.dark is not None or self._view.white is not None
        white_set = self._view.white is not None
        has_csv_sens = self._view.sensitivity is not None

        ctx.display.set_button_active("toggle_refs", self._view.refs_active)
        ctx.display.set_button_disabled("toggle_refs", not has_refs)
        ctx.display.set_button_active("show_raw_overlay", self._view.show_raw_overlay)
        ctx.display.set_button_disabled("show_raw_overlay", not has_refs)
        ctx.display.set_button_active("show_absorption", self._view.show_absorption)
        ctx.display.set_button_disabled("show_absorption", not white_set)
        ctx.display.set_button_disabled("clear_refs", not has_refs)

        ctx.display.set_button_active("show_csv_sens", self._view.show_csv_sensitivity)
        ctx.display.set_button_disabled("show_csv_sens", not has_csv_sens)

        ctx.display.set_button_active("show_sens_curve", self._view.show_sensitivity_curve)
        ctx.display.set_button_active("show_peaks", ctx.display.is_peaks_visible())

        for source in SOURCES:
            ctx.display.set_button_active(
                f"source_{source.name.lower()}", source in self._view.active_sources
            )

    def _build_raw_overlays(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
        """Dark/white/CSV-sensitivity overlays when Overlay mode is active."""
        n = len(wavelengths)
        spectra: list[tuple[np.ndarray, tuple[int, int, int]]] = []
        if self._view.dark is not None:
            spectra.append((np.asarray(self._view.dark, dtype=np.float64)[:n], (60, 60, 60)))
        if self._view.white is not None:
            spectra.append((np.asarray(self._view.white, dtype=np.float64)[:n], (200, 120, 0)))
        if self._view.show_csv_sensitivity and self._view.sensitivity is not None:
            spectra.append((np.asarray(self._view.sensitivity, dtype=np.float64)[:n], (80, 200, 80)))
        if not spectra:
            return []
        global_max = max(float(np.max(s[0])) for s in spectra)
        global_max = max(global_max, 1.0)
        return [
            (np.clip(arr / global_max, 0, 1).astype(np.float32), color)
            for arr, color in spectra
        ]

    def _build_illuminant_overlays(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
        """SPD overlays for each active illuminant, each normalised to its own peak."""
        overlays: list[tuple[np.ndarray, tuple[int, int, int]]] = []
        for source in SOURCES:
            if source not in self._view.active_sources:
                continue
            spd = get_reference_spectrum(source, wavelengths)
            if spd is None:
                continue
            spd = np.asarray(spd, dtype=np.float32)
            peak = float(np.max(spd))
            if peak > 1e-6:
                spd = spd / peak
            color = _ILLUM_COLOR.get(source, (200, 200, 200))
            overlays.append((np.clip(spd, 0, 1), color))
        return overlays

    def _status(self, ctx: "ModeContext | None" = None) -> dict[str, str]:
        s: dict[str, str] = {}
        if self._view.dark is not None:
            s["Dark"] = "CSV"
        if self._view.white is not None:
            s["White"] = "CSV"
        if not self._view.refs_active:
            s["Refs"] = "OFF"
        if self._view.show_absorption:
            s["ABS"] = "ON"
        if self._view.show_sensitivity_curve:
            s["Sens"] = "OV"
        if self._view.show_csv_sensitivity:
            s["SensRef"] = "CSV"
        if self._view.active_sources:
            names = "+".join(src.name for src in SOURCES if src in self._view.active_sources)
            s["Illum"] = names
        return s
