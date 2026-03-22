"""Waterfall mode: standalone waterfall display only.

Single window showing control bar + preview + waterfall.
No spectrograph window. Save/Export write raw measured SPD rows with dark/white in comments.
REC appends raw SPD rows to CSV indefinitely.
"""

import time
from collections import deque
from typing import TYPE_CHECKING

from ..export.csv_exporter import (
    build_absorption_metadata,
    build_markers_peaks_metadata,
    build_sensitivity_for_export,
    waterfall_rec_trim_bundle,
)
from .base import ButtonDefinition, ModeType
from .measurement import MeasurementMode

if TYPE_CHECKING:
    from ..core.mode_context import ModeContext


class WaterfallMode(MeasurementMode):
    """Waterfall mode: measurement processing with waterfall-only display."""

    def __init__(self) -> None:
        super().__init__()
        self._waterfall_rec_writer = None

    @property
    def mode_type(self) -> ModeType:
        return ModeType.MEASUREMENT

    @property
    def name(self) -> str:
        return "Waterfall"

    def setup(self, ctx: "ModeContext") -> None:
        """Apply waterfall-specific display defaults and buffer for export."""
        super().setup(ctx)
        height = ctx.display.config.display.graph_height
        ctx.waterfall_raw_buffer = deque(maxlen=3 * height)
        self.register_callback("save", lambda: self._on_save_waterfall(ctx))
        self.register_callback("export_csv", lambda: self._on_export_csv_waterfall(ctx))
        self.register_callback("waterfall_rec", lambda: self._on_toggle_waterfall_rec(ctx))
        ctx.display.show_spectrum_bars(True)
        ctx.display.state.peaks_visible = False
        ctx.display.state.graph_click_behavior = "marker"
        ctx.display.set_button_active("show_peaks", False)
        self.register_callback("snap_to_peaks", lambda: self._on_toggle_snap_to_peaks())
        self.register_callback("clear_markers", lambda: self._on_clear_markers())
        ctx.display.state.snap_to_peaks = True
        ctx.display.set_button_active("snap_to_peaks", True)

    def _on_toggle_snap_to_peaks(self) -> None:
        """Toggle snap-to-peaks for marker line placement."""
        ctx = self._ctx
        if ctx is None:
            return
        active = ctx.display.toggle_snap_to_peaks()
        ctx.display.set_button_active("snap_to_peaks", active)
        print(f"[MARKERS] Snap to peaks: {'ON' if active else 'OFF'}")

    def _on_clear_markers(self) -> None:
        """Remove all vertical marker lines."""
        ctx = self._ctx
        if ctx is None:
            return
        ctx.display.state.marker_lines.clear()
        print("[MARKERS] All marker lines cleared")

    def _on_save_waterfall(self, ctx: "ModeContext") -> None:
        """Export current waterfall buffer (raw SPD rows, limited to 3× height) with dark/white in comments."""
        buf = getattr(ctx, "waterfall_raw_buffer", None)
        if buf is None or len(buf) == 0:
            print("[WATERFALL] No data to save")
            return
        if ctx.last_data is None:
            print("[WATERFALL] No wavelength calibration")
            return
        rows = list(buf)
        wl = ctx.last_data.wavelengths
        metadata = {
            "Mode": "Waterfall",
            "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Rows": str(len(rows)),
            "Dark_acq": self.meas_state.dark_acq or None,
            "White_acq": self.meas_state.white_acq or None,
        }
        markers_peaks = build_markers_peaks_metadata(
            ctx.display.state.marker_lines,
            ctx.last_data.wavelengths,
            ctx.last_data.intensity,
            ctx.last_data.peaks,
        )
        metadata.update(markers_peaks)
        if self.meas_state.white_spectrum is not None:
            last_measured = rows[-1] if rows else None
            if last_measured is not None:
                absorption_meta = build_absorption_metadata(
                    last_measured,
                    self.meas_state.dark_spectrum,
                    self.meas_state.white_spectrum,
                )
                metadata.update(absorption_meta)
        ctx.save_waterfall_snapshot(
            rows,
            wl,
            dark_intensity=self.meas_state.dark_spectrum,
            white_intensity=self.meas_state.white_spectrum,
            metadata=metadata,
        )

    def _on_export_csv_waterfall(self, ctx: "ModeContext") -> None:
        """Same as Save: export waterfall buffer to CSV."""
        self._on_save_waterfall(ctx)

    def _on_toggle_waterfall_rec(self, ctx: "ModeContext") -> None:
        """Toggle REC: when on, append each frame's raw SPD to CSV indefinitely; when off, stop and close file."""
        if self._waterfall_rec_writer is not None:
            self._waterfall_rec_writer.close()
            self._waterfall_rec_writer = None
            ctx.waterfall_rec_writer = None
            ctx.display.set_button_active("waterfall_rec", False)
            print("[WATERFALL] Recording stopped")
            return
        if ctx.last_data is None:
            print("[WATERFALL] No wavelength calibration; start capture first")
            return
        path = ctx.exporter.generate_filename("Waterfall-Rec")
        wl = ctx.last_data.wavelengths
        rec_meta = {
            "Mode": "Waterfall-Rec",
            "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Dark_acq": self.meas_state.dark_acq or None,
            "White_acq": self.meas_state.white_acq or None,
        }
        sens_col, sens_meta = build_sensitivity_for_export(
            correction_enabled=ctx.sensitivity_correction_enabled,
            engine=ctx.sensitivity_engine,
            wavelengths=wl,
            sens_cfg=ctx.calibration.sensitivity_settings,
        )
        rec_meta.update(sens_meta)
        wl_rec, dark_rec, white_rec, sens_rec, src_n, row_mask = waterfall_rec_trim_bundle(
            wl,
            ctx.min_wavelength,
            self.meas_state.dark_spectrum,
            self.meas_state.white_spectrum,
            sens_col,
        )
        self._waterfall_rec_writer = ctx.exporter.start_waterfall_rec(
            path,
            wl_rec,
            dark_intensity=dark_rec,
            white_intensity=white_rec,
            sensitivity_intensity=sens_rec,
            metadata=rec_meta,
            row_source_length=src_n,
            row_trim_mask=row_mask,
        )
        ctx.waterfall_rec_writer = self._waterfall_rec_writer
        ctx.display.set_button_active("waterfall_rec", True)
        print(f"[WATERFALL] Recording to {path}")

    @property
    def preview_modes(self) -> list[str]:
        """Waterfall-relevant preview modes.

        "window" shows the camera crop strip; "spectrum" shows the spectrum
        graph scaled into the same strip; "none" hides the strip entirely.
        "full" is omitted — the waterfall already occupies the full graph area.
        """
        return ["window", "spectrum", "none"]

    def get_buttons(self) -> list[ButtonDefinition]:
        """Waterfall mode buttons — layout like Measurement; ZY omitted (no intensity zoom)."""
        return [
            # Row 1: Play | Avg/Max/Acc | Dark/White | Bars | ZX | VIEW
            ButtonDefinition("Play", "capture", is_toggle=True, row=1, icon_type="playback"),
            ButtonDefinition("S", "toggle_sensitivity", is_toggle=True, row=1, icon_type="sensitivity"),
            ButtonDefinition("__gap__", "__gap__", row=1),
            ButtonDefinition("Avg", "toggle_averaging", is_toggle=True, row=1, icon_type="avg"),
            ButtonDefinition("Max", "capture_peak", is_toggle=True, shortcut="h", row=1, icon_type="peak_hold"),
            ButtonDefinition("Acc", "toggle_accumulation", is_toggle=True, row=1, icon_type="acc"),
            ButtonDefinition("__gap__", "__gap__2", row=1),
            ButtonDefinition("Dark", "set_dark", is_toggle=True, row=1, icon_type="dark"),
            ButtonDefinition("White", "set_white", is_toggle=True, row=1, icon_type="white"),
            ButtonDefinition("ABS", "show_absorption", is_toggle=True, row=1, icon_type="absorption"),
            ButtonDefinition("__gap__", "__gap__3", row=1),
            ButtonDefinition("Bars", "show_spectrum_bars", is_toggle=True, row=1, icon_type="bars"),
            ButtonDefinition("__gap__", "__gap__4", row=1),
            ButtonDefinition("ZX", "show_zoom_x_slider", is_toggle=True, row=1, icon_type="zoom_x"),
            ButtonDefinition("__gap__", "__gap__5", row=1),
            ButtonDefinition("VIEW", "cycle_preview", shortcut="v", row=1, icon_type="eye"),
            # Row 2: Lamp | Peaks/Snap/Pkd/Clr | Ref | G/E/AG/AE | Save/Load/Export | spacer | Quit
            ButtonDefinition("Lamp", "lamp_toggle", is_toggle=True, row=2, icon_type="lamp"),
            ButtonDefinition("__gap__", "__gap__6", row=2),
            ButtonDefinition("Peaks", "show_peaks", is_toggle=True, row=2, icon_type="peaks"),
            ButtonDefinition("Snap", "snap_to_peaks", is_toggle=True, row=2, icon_type="snap"),
            ButtonDefinition("Pkd", "show_peak_delta", is_toggle=True, row=2, icon_type="delta"),
            ButtonDefinition("Clr", "clear_all", shortcut="z", row=2, icon_type="clear"),
            ButtonDefinition("__gap__", "__gap__7", row=2),
            ButtonDefinition("Ref", "show_reference", is_toggle=True, row=2, icon_type="reference"),
            ButtonDefinition("__gap__", "__gap__8", row=2),
            ButtonDefinition("G", "show_gain_slider", is_toggle=True, row=2, icon_type="gain"),
            ButtonDefinition("E", "show_exposure_slider", is_toggle=True, row=2, icon_type="exposure"),
            ButtonDefinition("AG", "auto_gain", is_toggle=True, row=2, icon_type="auto_gain"),
            ButtonDefinition("AE", "auto_exposure", is_toggle=True, row=2, icon_type="auto_exposure"),
            ButtonDefinition("__gap__", "__gap__9", row=2),
            ButtonDefinition("REC", "waterfall_rec", is_toggle=True, row=2, icon_type="playback"),
            ButtonDefinition("Save", "save", shortcut="s", row=2, icon_type="save"),
            ButtonDefinition("Load", "load", row=2, icon_type="load"),
            ButtonDefinition("__gap__", "__gap__10", row=2),
            ButtonDefinition("Export", "export_csv", row=2, icon_type="save"),
            ButtonDefinition("__spacer__", "__spacer_right__", row=2),
            ButtonDefinition("Quit", "quit", row=2, icon_type="quit"),
        ]
