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
        """Waterfall mode buttons — ZY omitted (no intensity axis to zoom)."""
        return [
            ButtonDefinition("Peak", "capture_peak", is_toggle=True, shortcut="h", row=1),
            ButtonDefinition("SnapPk", "snap_to_peaks", is_toggle=True, row=1),
            ButtonDefinition("ClrMk", "clear_markers", row=1),
            ButtonDefinition("PkΔ", "show_peak_delta", is_toggle=True, row=1),
            ButtonDefinition("Dark", "set_dark", row=1),
            ButtonDefinition("White", "set_white", row=1),
            ButtonDefinition("ClrRef", "clear_refs", row=1),
            ButtonDefinition("Peaks", "show_peaks", is_toggle=True, row=2),
            ButtonDefinition("ZX", "show_zoom_x_slider", is_toggle=True, row=2),
            ButtonDefinition("G", "show_gain_slider", is_toggle=True, row=2),
            ButtonDefinition("E", "show_exposure_slider", is_toggle=True, row=2),
            ButtonDefinition("AG", "auto_gain", is_toggle=True, row=2),
            ButtonDefinition("AE", "auto_exposure", is_toggle=True, row=2),
            ButtonDefinition("Prev", "cycle_preview", shortcut="v", row=2),
            ButtonDefinition("Save", "save", shortcut="s", row=2),
            ButtonDefinition("Quit", "quit", row=2),
            ButtonDefinition("__spacer__", "__spacer_right__", row=2),
        ]
