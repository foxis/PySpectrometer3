"""CSV viewer mode: display a loaded spectrum CSV as if in measurement mode.

Static data — no capture, no freeze, no averaging.
Supports dark/white correction, absorption, normalization, zoom/pan, peaks, markers.
Includes a calibration sub-mode (Cal button) for pixel→wavelength calibration using a
reference spectrum. Right-click in the graph to manually pair measured and reference peaks.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..data.reference_spectra import ReferenceSource, get_reference_name, get_reference_spectrum
from ..core.spectrum_overlay import (
    SpectrumOverlay,
    overlay_display_intensity,
    overlay_from_loaded_csv,
)
from ..export.csv_exporter import build_absorption_metadata, build_markers_peaks_metadata
from ..export.csv_exporter import _slugify as export_slugify
from ..modes.base import BaseMode, ButtonDefinition, ModeType
from ..processing.reference_correction import apply_dark_white_correction
from ..utils.graph_scale import scale_intensity_to_graph
from .calibration import CsvCalibrationHandler
from .loader import LoadedCsv

if TYPE_CHECKING:
    from ..core.mode_context import ModeContext
    from ..core.spectrum import SpectrumData
    from ..data.reference_loader import ReferenceFileLoader

# Same source list as CalibrationMode — all have data via colour-science or CSV files
SOURCES: list[ReferenceSource] = [
    ReferenceSource.HG,
    ReferenceSource.D65,
    ReferenceSource.A,
    ReferenceSource.XE_HID,
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
    ReferenceSource.XE_HID: (255, 220, 200), # cool white-blue — xenon daylight
    ReferenceSource.FL1:  (255, 200,  80),   # sky blue
    ReferenceSource.FL2:  ( 80, 220, 100),   # green
    ReferenceSource.FL3:  ( 40, 160,  80),   # dark green
    ReferenceSource.FL12: (200,  80, 220),   # violet
    ReferenceSource.LED:  (255,  80, 180),   # pink
    ReferenceSource.LED2: (180,  60, 255),   # magenta
    ReferenceSource.LED3: (120,  40, 200),   # purple
}

# Extra CSV traces (Load+): distinct from illuminant palette
_EXTRA_OVERLAY_BGR: tuple[tuple[int, int, int], ...] = (
    (255, 255, 100),   # cyan
    (255, 180, 100),   # light blue
    (180, 255, 255),   # yellow
    (200, 130, 255),   # pink
    (100, 255, 100),   # green
    (255, 100, 200),   # orchid
)


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
    # Load+ overlays: (label, intensity 0–1 on calibration axis; resampled per frame if length changes)
    extra_overlays: list[tuple[str, np.ndarray]] = field(default_factory=list)


class CsvViewerMode(BaseMode):
    """Measurement-like viewer for a loaded spectrum CSV.

    Does not depend on camera, extractor, or auto-gain/exposure.
    Includes a calibration sub-mode for pixel→wavelength calibration.
    """

    def __init__(self, loaded: LoadedCsv):
        super().__init__()
        self._view = CsvViewerState(
            loaded=loaded,
            dark=loaded.dark.copy() if loaded.dark is not None else None,
            white=loaded.white.copy() if loaded.white is not None else None,
            sensitivity=loaded.sensitivity.copy() if loaded.sensitivity is not None else None,
        )
        self._cal = CsvCalibrationHandler()
        self._ctx: ModeContext | None = None

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
        self._ctx = ctx
        self._register_viewer_handlers(ctx)
        ctx.display.set_graph_right_click(self._on_right_click)

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
        """Return single overlay (cal > white > dark > None) when raw-overlay is off."""
        if self._cal.active:
            fl = self._ctx.reference_file_loader if self._ctx else None
            return self._cal.reference_overlay(wavelengths, graph_height, file_loader=fl)

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
            ButtonDefinition("S", "toggle_sensitivity", is_toggle=True, row=1, icon_type="sensitivity"),
            ButtonDefinition("SensOv", "show_sens_curve", is_toggle=True, row=1),
            ButtonDefinition("SensRef", "show_csv_sens", is_toggle=True, row=1),
            ButtonDefinition("__gap__", "__gap__illum", row=1),
        ]
        for source in SOURCES:
            row1.append(
                ButtonDefinition(source.name, f"source_{source.name.lower()}", is_toggle=True, row=1)
            )
        row1 += [
            ButtonDefinition("CalcSens", "calc_sens", row=1),
            ButtonDefinition("__gap__", "__gap__refs", row=1),
            ButtonDefinition("Overlay", "show_raw_overlay", is_toggle=True, row=1, icon_type="overlay"),
            ButtonDefinition("ABS", "show_absorption", is_toggle=True, row=1, icon_type="absorption"),
            ButtonDefinition("Refs", "toggle_refs", is_toggle=True, row=1),
            ButtonDefinition("ClearRef", "clear_refs", row=1, icon_type="clear"),
            ButtonDefinition("__spacer__", "__spacer_r1__", row=1),
        ]

        row2: list[ButtonDefinition] = [
            ButtonDefinition("Peaks", "show_peaks", is_toggle=True, row=2, icon_type="peaks"),
            ButtonDefinition("Snap", "snap_to_peaks", is_toggle=True, row=2, icon_type="snap"),
            ButtonDefinition("Pkd", "show_peak_delta", is_toggle=True, row=2, icon_type="delta"),
            ButtonDefinition("Clr", "clear_all", shortcut="z", row=2, icon_type="clear"),
            ButtonDefinition("__gap__", "__gap__r1", row=2),
            ButtonDefinition("ZX", "show_zoom_x_slider", is_toggle=True, row=2, icon_type="zoom_x"),
            ButtonDefinition("ZY", "show_zoom_y_slider", is_toggle=True, row=2, icon_type="zoom_y"),
            ButtonDefinition("__gap__", "__gap__r2", row=2),
            ButtonDefinition("Export", "export_csv", shortcut="s", row=2, icon_type="save"),
            ButtonDefinition("Pdf", "export_pdf", shortcut="p", row=2, icon_type="pdf"),
            ButtonDefinition("Load", "load_csv", row=2, icon_type="load"),
            ButtonDefinition("Load+", "load_overlay_csv", row=2, icon_type="load_plus"),
            ButtonDefinition("ClrOv", "clear_overlay_csv", row=2, icon_type="clear"),
            ButtonDefinition("__gap__", "__gap__cal", row=2),
            # Calibration sub-mode buttons
            ButtonDefinition("Cal", "cal_toggle", is_toggle=True, row=2, icon_type="calibrate"),
            ButtonDefinition("AutoCal", "cal_auto", row=2, icon_type="calibrate"),
            ButtonDefinition("R", "cal_reset", row=2, icon_type="reset"),
            ButtonDefinition("SaveCal", "cal_save", row=2, icon_type="save"),
            ButtonDefinition("LoadCal", "cal_load", row=2, icon_type="load"),
            ButtonDefinition("ClrCal", "cal_clear", row=2, icon_type="clear"),
            ButtonDefinition("__spacer__", "__spacer_right__", row=2),
            ButtonDefinition("Quit", "quit", row=2, icon_type="quit"),
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

        if self._cal.active:
            self._update_cal_display(ctx, processed, graph_height)
        elif self._view.show_raw_overlay:
            raw_list = self._build_raw_overlays(processed.wavelengths, graph_height)
            raw_list.extend(self._build_extra_overlay_polylines(processed.wavelengths, ctx))
            ctx.display.set_raw_overlays(raw_list)
            ctx.display.set_mode_overlay(None)
            ctx.display.set_graph_post_draw(None)
        else:
            illum = self._build_illuminant_overlays(
                processed.wavelengths, graph_height, file_loader=ctx.reference_file_loader
            )
            illum.extend(self._build_extra_overlay_polylines(processed.wavelengths, ctx))
            ctx.display.set_raw_overlays(illum)
            overlay = self.get_overlay(processed.wavelengths, graph_height)
            ctx.display.set_mode_overlay(overlay)
            ctx.display.set_graph_post_draw(None)

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
        self._view.extra_overlays.clear()
        self._sync_button_states(ctx)

    @property
    def loaded(self) -> LoadedCsv | None:
        return self._view.loaded

    # ------------------------------------------------------------------
    # Private — calibration sub-mode
    # ------------------------------------------------------------------

    def _update_cal_display(
        self,
        ctx: "ModeContext",
        processed: "SpectrumData",
        graph_height: int,
    ) -> None:
        """Set overlays and post-draw hook while calibration sub-mode is active."""
        # Load+ traces (0–1 polylines); reference SPD (orange) via mode_overlay
        ctx.display.set_raw_overlays(
            self._build_extra_overlay_polylines(processed.wavelengths, ctx)
        )
        overlay = self._cal.reference_overlay(processed.wavelengths, graph_height)
        ctx.display.set_mode_overlay(overlay)

        # Post-draw: reference peak lines, pair connections, pending marker
        width = ctx.display.runtime.display.window_width
        viewport = ctx.display.viewport

        def _post_draw(graph_img: "np.ndarray", data: "SpectrumData") -> None:
            self._cal.draw_on_graph(
                graph_img, data.wavelengths, data.peaks, viewport, width
            )

        ctx.display.set_graph_post_draw(_post_draw)

    def _on_right_click(self, data_x: float) -> None:
        """Handle right-click in graph area for manual peak pairing."""
        if not self._cal.active or self._ctx is None or self._ctx.last_data is None:
            return
        data = self._ctx.last_data
        formed = self._cal.handle_right_click(
            data_x,
            data.peaks,
            data.wavelengths,
            self._ctx.display.viewport,
            self._ctx.display.runtime.display.window_width,
        )
        if formed:
            self._apply_calibration(self._ctx)

    def _apply_calibration(self, ctx: "ModeContext") -> None:
        """Apply current cal points to the live calibration object."""
        points = self._cal.all_cal_points()
        if len(points) < 4:
            print(f"[Cal] Need at least 4 points (have {len(points)})")
            return
        pixels = [p for p, _ in points]
        wavelengths = [w for _, w in points]
        ctx.calibration.recalibrate(pixels, wavelengths)
        print(f"[Cal] Applied {len(points)} calibration points")

    # ------------------------------------------------------------------
    # Private — button handlers
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
        self.register_callback("calc_sens", lambda: self._on_calc_sensitivity(ctx))
        self.register_callback("export_csv", lambda: self._on_export(ctx))
        self.register_callback("export_pdf", lambda: self._on_export_pdf(ctx))
        self.register_callback("load_csv", lambda: self._on_load(ctx))
        self.register_callback("load_overlay_csv", lambda: self._on_load_overlay(ctx))
        self.register_callback("clear_overlay_csv", lambda: self._on_clear_overlays(ctx))
        self.register_callback("snap_to_peaks", lambda: self._on_snap_to_peaks(ctx))
        self.register_callback("clear_all", lambda: self._on_clear_all(ctx))
        self.register_callback("show_peak_delta", lambda: self._on_peak_delta(ctx))
        # Calibration sub-mode
        self.register_callback("cal_toggle", lambda: self._on_cal_toggle(ctx))
        self.register_callback("cal_auto", lambda: self._on_cal_auto(ctx))
        self.register_callback("cal_reset", lambda: self._on_cal_reset(ctx))
        self.register_callback("cal_save", lambda: self._on_cal_save(ctx))
        self.register_callback("cal_load", lambda: self._on_cal_load(ctx))
        self.register_callback("cal_clear", lambda: self._on_cal_clear(ctx))

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

    def metadata_for_export(self, ctx: "ModeContext") -> dict[str, Any]:
        """Metadata shared by CSV and PDF export (caller must have ``ctx.last_data``)."""
        data = ctx.last_data
        assert data is not None
        metadata: dict[str, Any] = {
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
        return metadata

    def _on_export(self, ctx: "ModeContext") -> None:
        if ctx.last_data is None:
            print("[CSV Viewer] No data to export")
            return
        metadata = self.metadata_for_export(ctx)
        ctx.save_snapshot(
            ctx.last_data,
            dark_intensity=self._view.dark,
            white_intensity=self._view.white,
            metadata=metadata,
        )

    def _on_export_pdf(self, ctx: "ModeContext") -> None:
        """Multi-page PDF report; path chosen by spectrometer (same folder as CSV)."""
        ctx.export_pdf_report()

    def _on_toggle_source(self, ctx: "ModeContext", source: ReferenceSource) -> None:
        """Dual-purpose: calibration reference select (cal active) or illuminant toggle."""
        if self._cal.active:
            self._cal.select_source(source)
            for s in SOURCES:
                ctx.display.set_button_active(f"source_{s.name.lower()}", s == source)
        else:
            if source in self._view.active_sources:
                self._view.active_sources.discard(source)
            else:
                self._view.active_sources.add(source)
            ctx.display.set_button_active(
                f"source_{source.name.lower()}", source in self._view.active_sources
            )

    def _on_calc_sensitivity(self, ctx: "ModeContext") -> None:
        """Fit a new sensitivity curve from the loaded CSV vs a known reference illuminant.

        Uses dark-corrected raw intensity so white-reference normalisation and any
        prior sensitivity correction do not interfere with the ratio calculation.
        """
        if ctx.sensitivity_engine is None:
            print("[CalcSens] Sensitivity engine not available")
            return
        if self._view.loaded is None or ctx.last_data is None:
            print("[CalcSens] No spectrum loaded")
            return

        source = self._active_source_for_calc()
        if source is None:
            print("[CalcSens] Select exactly one reference illuminant first")
            return

        wl = ctx.last_data.wavelengths
        raw = np.asarray(self._view.loaded.intensity, dtype=np.float64)
        measured = apply_dark_white_correction(raw[: len(wl)], self._view.dark, None)

        ref = get_reference_spectrum(source, wl)
        if ref is None:
            print(f"[CalcSens] No reference data for {source.name}")
            return

        pair = ctx.sensitivity_engine.recalibrate_from_measurement(measured, wl, ref)
        if pair is None:
            print("[CalcSens] Fit failed (no CMOS datasheet curve available?)")
            return

        wl_out, sens = pair
        ctx.sensitivity_engine.set_custom_curve(wl_out, sens)
        ref_name = get_reference_name(source)
        ctx.calibration.save_sensitivity_curve(
            list(map(float, wl_out)),
            list(map(float, sens)),
            calibration_reference=ref_name,
        )
        print(
            f"[CalcSens] Sensitivity fitted vs {ref_name} "
            f"({len(wl_out)} pts). Toggle S to apply."
        )

    def _active_source_for_calc(self) -> "ReferenceSource | None":
        """Return the reference source to use for sensitivity calculation.

        Cal mode takes precedence; otherwise the single active illuminant is used.
        Multiple simultaneous selections are ambiguous — the user must pick one.
        """
        if self._cal.active:
            return self._cal.state.current_source
        if len(self._view.active_sources) == 1:
            return next(iter(self._view.active_sources))
        if len(self._view.active_sources) > 1:
            print("[CalcSens] Multiple illuminants selected — select exactly one")
        return None

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
        ctx.pending_csv_reload = True  # type: ignore[attr-defined]

    def _on_load_overlay(self, ctx: "ModeContext") -> None:
        """Signal the loop to prompt for an additional CSV (does not replace primary)."""
        ctx.pending_csv_overlay = True  # type: ignore[attr-defined]

    def _on_clear_overlays(self, ctx: "ModeContext") -> None:
        """Remove all Load+ traces; primary spectrum unchanged."""
        self._view.extra_overlays.clear()
        self._sync_button_states(ctx)

    def append_overlay_from_loaded(self, loaded: LoadedCsv, label: str) -> None:
        """Append a Load+ overlay from parsed CSV (keeps dark/white/sensitivity columns)."""
        self._view.extra_overlays.append(overlay_from_loaded_csv(loaded, label))

    # ------------------------------------------------------------------
    # Private — calibration button handlers
    # ------------------------------------------------------------------

    def _on_cal_toggle(self, ctx: "ModeContext") -> None:
        active = self._cal.toggle()
        ctx.display.set_button_active("cal_toggle", active)
        if not active:
            ctx.display.set_graph_post_draw(None)
        self._sync_button_states(ctx)
        print(f"[Cal] Calibration mode {'ON' if active else 'OFF'}")

    def _on_cal_auto(self, ctx: "ModeContext") -> None:
        if not self._cal.active or ctx.last_data is None:
            return
        self._cal.run_auto_cal(
            ctx.last_data.intensity,
            ctx.last_data.wavelengths,
            file_loader=ctx.reference_file_loader,
        )
        self._apply_calibration(ctx)

    def _on_cal_reset(self, ctx: "ModeContext") -> None:
        if ctx.last_data is None:
            return
        n = len(ctx.last_data.wavelengths)
        self._cal.reset_linear(n)
        self._apply_calibration(ctx)

    def _on_cal_save(self, ctx: "ModeContext") -> None:
        points = self._cal.all_cal_points()
        if len(points) < 4:
            print(f"[Cal] Need at least 4 points to save (have {len(points)})")
            return
        pixels = [p for p, _ in points]
        wavelengths = [w for _, w in points]
        ctx.calibration.save(pixels, wavelengths)
        print(f"[Cal] Saved {len(points)} calibration points to config")

    def _on_cal_load(self, ctx: "ModeContext") -> None:
        ctx.calibration.load()
        # Sync the in-memory cal_points from loaded config
        pixels = ctx.calibration.cal_pixels
        wls = ctx.calibration.cal_wavelengths
        if len(pixels) >= 4:
            self._cal.state.auto_cal_points = list(zip(pixels, wls))
            self._cal.state.manual_pairs.clear()
        print(f"[Cal] Loaded {len(pixels)} calibration points from config")

    def _on_cal_clear(self, ctx: "ModeContext") -> None:
        self._cal.clear_pairs()
        print("[Cal] Cleared all calibration pairs")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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

        # CalcSens: needs a sensitivity engine AND a single unambiguous reference source
        has_source = (
            self._cal.active
            or len(self._view.active_sources) == 1
        )
        ctx.display.set_button_disabled(
            "calc_sens",
            ctx.sensitivity_engine is None or not has_source or self._view.loaded is None,
        )

        # Source buttons: single-select in cal mode, multi-select otherwise
        if self._cal.active:
            for source in SOURCES:
                ctx.display.set_button_active(
                    f"source_{source.name.lower()}",
                    source == self._cal.state.current_source,
                )
        else:
            for source in SOURCES:
                ctx.display.set_button_active(
                    f"source_{source.name.lower()}", source in self._view.active_sources
                )

        # Calibration sub-mode buttons
        cal_active = self._cal.active
        has_points = len(self._cal.all_cal_points()) >= 4
        ctx.display.set_button_active("cal_toggle", cal_active)
        ctx.display.set_button_disabled("cal_auto", not cal_active)
        ctx.display.set_button_disabled("cal_reset", not cal_active)
        ctx.display.set_button_disabled("cal_save", not cal_active or not has_points)
        ctx.display.set_button_disabled("cal_clear", not cal_active)

        has_extra = len(self._view.extra_overlays) > 0
        ctx.display.set_button_disabled("clear_overlay_csv", not has_extra)

    def _build_extra_overlay_polylines(
        self,
        wavelengths: np.ndarray,
        ctx: "ModeContext",
    ) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
        """Load+ traces in the same intensity units as the main line (no per-trace peak norm)."""
        n = len(wavelengths)
        out: list[tuple[np.ndarray, tuple[int, int, int]]] = []
        sens_on = ctx.sensitivity_correction_enabled
        dark_s = self._view.dark
        white_s = self._view.white
        for i, ov in enumerate(self._view.extra_overlays):
            row = overlay_display_intensity(
                ov,
                np.asarray(wavelengths, dtype=np.float64),
                ctx,
                sensitivity_enabled=sens_on,
                dark_session=dark_s,
                white_session=white_s,
            )
            if len(row) != n:
                row = np.asarray(row, dtype=np.float32).ravel()[:n]
                if len(row) < n:
                    row = np.pad(row, (0, n - len(row)))
            color = _EXTRA_OVERLAY_BGR[i % len(_EXTRA_OVERLAY_BGR)]
            out.append((row.astype(np.float32), color))
        return out

    def overlay_export_columns(
        self,
        ctx: "ModeContext",
        wavelengths: np.ndarray,
    ) -> list[tuple[str, np.ndarray]]:
        """Extra CSV columns: overlay intensities matching current S toggle (same as graph)."""
        rows: list[tuple[str, np.ndarray]] = []
        sens_on = ctx.sensitivity_correction_enabled
        dark_s = self._view.dark
        white_s = self._view.white
        wl = np.asarray(wavelengths, dtype=np.float64)
        for ov in self._view.extra_overlays:
            col = overlay_display_intensity(
                ov,
                wl,
                ctx,
                sensitivity_enabled=sens_on,
                dark_session=dark_s,
                white_session=white_s,
            )
            slug = export_slugify(ov.label) or "overlay"
            rows.append((f"Overlay_{slug}", np.asarray(col, dtype=np.float64)))
        return rows

    def _build_raw_overlays(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
        """Dark/white/CSV-sensitivity overlays when Overlay mode is active.

        Dark and white are jointly normalised so their relative levels are
        preserved (dark at ~5% when white is at 100%, for example).
        Sensitivity is a correction-factor curve — potentially >> 1.0 — so it
        gets its own independent peak normalisation to avoid suppressing the
        dark/white lines into invisibility.
        """
        n = len(wavelengths)
        result: list[tuple[np.ndarray, tuple[int, int, int]]] = []

        ref_pairs: list[tuple[np.ndarray, tuple[int, int, int]]] = []
        if self._view.dark is not None:
            ref_pairs.append((np.asarray(self._view.dark, dtype=np.float64)[:n], (60, 60, 60)))
        if self._view.white is not None:
            ref_pairs.append((np.asarray(self._view.white, dtype=np.float64)[:n], (200, 120, 0)))
        if ref_pairs:
            ref_max = max(float(np.max(arr)) for arr, _ in ref_pairs)
            ref_max = max(ref_max, 1e-6)
            result.extend(
                (np.clip(arr / ref_max, 0, 1).astype(np.float32), color)
                for arr, color in ref_pairs
            )

        if self._view.show_csv_sensitivity and self._view.sensitivity is not None:
            sens = np.asarray(self._view.sensitivity, dtype=np.float64)[:n]
            sens_max = max(float(np.max(sens)), 1e-6)
            result.append((np.clip(sens / sens_max, 0, 1).astype(np.float32), (80, 200, 80)))

        return result

    def _build_illuminant_overlays(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
        *,
        file_loader: "ReferenceFileLoader | None" = None,
    ) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
        """SPD overlays for each active illuminant, each normalised to its own peak."""
        overlays: list[tuple[np.ndarray, tuple[int, int, int]]] = []
        for source in SOURCES:
            if source not in self._view.active_sources:
                continue
            spd = get_reference_spectrum(source, wavelengths, file_loader=file_loader)
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
        n_extra = len(self._view.extra_overlays)
        if n_extra:
            s["Ov+"] = str(n_extra)
        if self._cal.active:
            n_pts = len(self._cal.all_cal_points())
            s["Cal"] = f"{self._cal.state.current_source.name} {n_pts}pts"
            if self._cal.state.pending_pixel is not None:
                s["Cal"] += " →ref?"
        elif self._view.active_sources:
            names = "+".join(src.name for src in SOURCES if src in self._view.active_sources)
            s["Illum"] = names
        return s
