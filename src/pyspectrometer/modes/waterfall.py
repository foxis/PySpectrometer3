"""Waterfall mode: standalone waterfall display only.

Single window showing control bar + preview + waterfall.
No spectrograph window.
"""

from typing import TYPE_CHECKING

from .base import BaseMode, ButtonDefinition, ModeType
from .measurement import MeasurementMode

if TYPE_CHECKING:
    from ..core.mode_context import ModeContext


class WaterfallMode(MeasurementMode):
    """Waterfall mode: measurement processing with waterfall-only display."""

    @property
    def mode_type(self) -> ModeType:
        return ModeType.MEASUREMENT

    @property
    def name(self) -> str:
        return "Waterfall"

    def setup(self, ctx: "ModeContext") -> None:
        """Apply waterfall-specific display defaults after common setup."""
        super().setup(ctx)
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
            # Row 1: Play | Avg/Peak/Acc | Dark/White | Bars | ZX | VIEW
            ButtonDefinition("Play", "capture", is_toggle=True, row=1, icon_type="playback"),
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
            ButtonDefinition("__gap__", "__gap__5", row=1),
            ButtonDefinition("VIEW", "cycle_preview", shortcut="v", row=1),
            # Row 2: Lamp | Peaks/Snap/Pkd/Clr | Ref | G/E/AG/AE | Save/Load/Export | spacer | Quit
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
