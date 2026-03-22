"""Measurement mode for general spectrum measurement.

Features:
- Save spectrum; Load+ / Cap+ overlays to compare SPDs
- Dark / white references (reflectance, transmittance, absorption)
- Spectrum averaging
- Auto gain control
- GPIO lamp control
"""

import math
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.spectrum import SpectrumData
import numpy as np

from ..core.mode_context import ModeContext
from ..core.spectrum_overlay import (
    SpectrumOverlay,
    overlay_display_intensity,
    overlay_from_loaded_csv,
)
from ..csv_viewer.loader import load_csv
from ..export.csv_exporter import build_absorption_metadata, build_markers_peaks_metadata
from ..export.csv_exporter import _slugify as export_slugify
from ..processing.reference_correction import apply_dark_white_correction
from .base import BaseMode, ButtonDefinition, ModeType

# Load+ overlay line colours (BGR)
_EXTRA_OVERLAY_BGR: tuple[tuple[int, int, int], ...] = (
    (255, 255, 100),
    (255, 180, 100),
    (180, 255, 255),
    (200, 130, 255),
    (100, 255, 100),
    (255, 100, 200),
)


@dataclass
class MeasurementState:
    """State specific to measurement mode."""

    # Dark / white references (session)
    dark_spectrum: np.ndarray | None = None
    white_spectrum: np.ndarray | None = None

    # Current captured spectrum (for saving)
    captured_spectrum: np.ndarray | None = None
    captured_wavelengths: np.ndarray | None = None

    # Display settings
    show_raw_overlay: bool = False
    show_absorption: bool = False
    subtract_dark: bool = True

    # Acquisition mode labels recorded when each reference was captured
    dark_acq: str = ""
    white_acq: str = ""

    # Load+ overlays (raw SPD columns; display follows S toggle)
    extra_overlays: list[SpectrumOverlay] = field(default_factory=list)


class MeasurementMode(BaseMode):
    """Measurement mode for general spectrum analysis."""

    def __init__(self):
        """Initialize measurement mode."""
        super().__init__()
        self.meas_state = MeasurementState()

    def setup(self, ctx: ModeContext) -> None:
        """Register measurement-specific handlers."""
        super().setup(ctx)
        self._register_measurement_handlers(ctx)

    def _register_measurement_handlers(self, ctx: ModeContext) -> None:
        """Register measurement mode button handlers."""
        self.register_callback("capture", lambda: self._on_capture(ctx))
        self.register_callback("toggle_averaging", lambda: self._on_toggle_averaging(ctx))
        self.register_callback("toggle_accumulation", lambda: self._on_toggle_accumulation(ctx))
        self.register_callback("set_dark", lambda: self._on_set_dark(ctx))
        self.register_callback("set_white", lambda: self._on_set_white(ctx))
        self.register_callback("clear_refs", lambda: self._on_clear_refs(ctx))
        self.register_callback("save", lambda: self._on_save(ctx))
        self.register_callback("export_vector", lambda: self._on_export_vector(ctx))
        self.register_callback("export_pdf", lambda: self._on_export_pdf(ctx))
        self.register_callback("load_overlay_csv", lambda: self._on_load_overlay(ctx))
        self.register_callback("push_overlay", lambda: self._on_push_overlay(ctx))
        self.register_callback("clear_overlay_csv", lambda: self._on_clear_overlay_csv(ctx))
        self.register_callback("show_raw_overlay", lambda: self._on_toggle_raw_overlay(ctx))
        self.register_callback("show_absorption", lambda: self._on_toggle_absorption(ctx))
        self.register_callback("lamp_toggle", lambda: self._on_toggle_light(ctx))
        self.register_callback("snap_to_peaks", lambda: self._on_toggle_snap_to_peaks(ctx))
        self.register_callback("clear_markers", lambda: self._on_clear_markers(ctx))
        self.register_callback("clear_all", lambda: self._on_clear_all(ctx))
        self.register_callback("show_peak_delta", lambda: self._on_toggle_peak_delta(ctx))

    def _on_toggle_peak_delta(self, ctx: ModeContext) -> None:
        """Toggle peak/marker width (FWHM) below wavelength label."""
        active = ctx.display.toggle_peak_delta_visible()
        ctx.display.set_button_active("show_peak_delta", active)
        print(f"[Pkd] Peak width display: {'ON' if active else 'OFF'}")

    def _on_toggle_snap_to_peaks(self, ctx: ModeContext) -> None:
        """Toggle snap-to-peaks for marker line placement (when peaks are off)."""
        active = ctx.display.toggle_snap_to_peaks()
        ctx.display.set_button_active("snap_to_peaks", active)
        print(f"[MARKERS] Snap to peaks: {'ON' if active else 'OFF'}")

    def _on_clear_markers(self, ctx: ModeContext) -> None:
        """Remove all vertical marker lines."""
        ctx.display.state.marker_lines.clear()
        print("[MARKERS] All marker lines cleared")

    def _on_clear_all(self, ctx: ModeContext) -> None:
        """Clear marker lines and peak-include region."""
        ctx.display.state.marker_lines.clear()
        ctx.display.clear_peak_include_region()
        print("[MARKERS] Markers and peak region cleared")

    def _on_capture(self, ctx: ModeContext) -> None:
        """Toggle freeze (like calibration Play): active = live (red circle), inactive = frozen (gray square). Click when live → freeze; when frozen → unfreeze."""
        btn = ctx.display.control_bar.get_button("capture")
        if btn is not None and btn.is_active:
            # Currently live: user clicked to freeze
            if ctx.last_data is None:
                print("[CAPTURE] No spectrum data available")
                return
            frozen = self.toggle_freeze()
            assert frozen
            ctx.frozen_spectrum = True
            ctx.frozen_intensity = (
                ctx.last_intensity_accumulated.copy()
                if ctx.last_intensity_accumulated is not None
                else ctx.last_raw_intensity.copy()
                if ctx.last_raw_intensity is not None
                else ctx.last_data.intensity.copy()
            )
            self.capture_current(ctx.last_data.intensity, ctx.last_data.wavelengths)
            ctx.display.set_button_active("capture", False)
            print("[Measurement] Frozen")
        else:
            # Currently frozen: user clicked to unfreeze
            frozen = self.toggle_freeze()
            assert not frozen
            ctx.frozen_spectrum = False
            ctx.frozen_intensity = None
            self.meas_state.captured_spectrum = None
            self.meas_state.captured_wavelengths = None
            ctx.display.set_button_active("capture", True)
            print("[Measurement] Unfrozen")

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

    def _on_set_dark(self, ctx: ModeContext) -> None:
        """Toggle dark reference: set if unset, clear if already set."""
        if self.meas_state.dark_spectrum is not None:
            self.meas_state.dark_spectrum = None
            self.meas_state.dark_acq = ""
            self.meas_state.subtract_dark = True
            ctx.display.set_button_active("set_dark", False)
            print("[DARK] Dark reference cleared")
            return

        raw = _get_raw(ctx)
        if raw is None:
            print("[DARK] No spectrum data available")
            return
        self.meas_state.dark_acq = self.acq_label(ctx.display.state.hold_peaks)
        self.set_dark_reference(raw)
        ctx.display.set_button_active("set_dark", True)

    def _on_set_white(self, ctx: ModeContext) -> None:
        """Toggle white reference: set if unset, clear if already set."""
        if self.meas_state.white_spectrum is not None:
            self.meas_state.white_spectrum = None
            self.meas_state.white_acq = ""
            self.meas_state.show_absorption = False
            ctx.display.set_button_active("set_white", False)
            ctx.display.set_button_active("show_absorption", False)
            ctx.display.set_button_disabled("show_absorption", True)
            print("[WHITE] White reference cleared")
            return

        raw = _get_raw(ctx)
        if raw is None:
            print("[WHITE] No spectrum data available")
            return
        self.meas_state.white_acq = self.acq_label(ctx.display.state.hold_peaks)
        self.set_white_reference(raw)
        ctx.display.set_button_active("set_white", True)

    def _on_clear_refs(self, ctx: ModeContext) -> None:
        """Clear all references."""
        self.clear_references()
        ctx.display.set_button_active("set_dark", False)
        ctx.display.set_button_active("set_white", False)
        ctx.display.set_button_active("show_absorption", False)
        ctx.display.set_button_disabled("show_absorption", True)
        ctx.display.set_button_active("show_raw_overlay", False)

    def snapshot_metadata(self, ctx: ModeContext) -> dict:
        """CSV/PDF/graph header fields (same keys as spectrum save). Note may be empty."""
        light_on = self.state.lamp_enabled
        led_pct = ctx.display.get_led_intensity_value() if light_on else None
        pwm_log = (
            f"{math.log10(0.01 + led_pct / 100):.4f}"
            if led_pct is not None and light_on
            else None
        )
        measured_acq = self.acq_label(ctx.display.state.hold_peaks)
        metadata = {
            "Mode": "Measurement",
            "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Light source": "Lamp" if light_on else "off",
            "PWM intensity %": f"{led_pct:.1f}" if led_pct is not None and light_on else None,
            "PWM intensity log10": pwm_log,
            "Gain": f"{ctx.camera.gain:.1f}",
            "Exposure": f"{getattr(ctx.camera, 'exposure', 0)}",
            "Measured_acq": measured_acq,
            "Dark_acq": self.meas_state.dark_acq or None,
            "White_acq": self.meas_state.white_acq or None,
            "Note": "",
        }
        if ctx.last_data is None:
            return metadata
        markers_peaks = build_markers_peaks_metadata(
            ctx.display.state.marker_lines,
            ctx.last_data.wavelengths,
            ctx.last_data.intensity,
            ctx.last_data.peaks,
        )
        metadata.update(markers_peaks)
        if (
            self.meas_state.white_spectrum is not None
            and ctx.last_intensity_accumulated is not None
        ):
            absorption_meta = build_absorption_metadata(
                ctx.last_intensity_accumulated,
                self.meas_state.dark_spectrum,
                self.meas_state.white_spectrum,
            )
            metadata.update(absorption_meta)
        return metadata

    def _on_save(self, ctx: ModeContext) -> None:
        """Handle save button."""
        if ctx.last_data is None:
            print("[SAVE] No spectrum data available")
            return
        metadata = self.snapshot_metadata(ctx)
        wl = ctx.last_data.wavelengths
        extras = self.overlay_export_columns(ctx, wl)
        extra_spectra = [(name, wl, sp) for name, sp in extras] if extras else None
        ctx.save_snapshot(
            ctx.last_data,
            dark_intensity=self.meas_state.dark_spectrum,
            white_intensity=self.meas_state.white_spectrum,
            metadata=metadata,
            extra_spectra=extra_spectra,
        )

    def _on_export_vector(self, ctx: ModeContext) -> None:
        """SVG/EPS export of the current graph view."""
        ctx.export_vector_graph()

    def _on_export_pdf(self, ctx: ModeContext) -> None:
        """Multi-page PDF report (metadata + curves)."""
        ctx.export_pdf_report()

    def _on_load_overlay(self, ctx: ModeContext) -> None:
        """Add a CSV trace as Load+."""
        path_str = _pick_csv_file()
        if not path_str:
            return
        try:
            loaded = load_csv(Path(path_str))
        except Exception as exc:
            print(f"[LOAD+] Failed to read {path_str}: {exc}")
            return
        if ctx.last_data is None:
            print("[LOAD+] No current wavelength range available")
            return
        label = Path(path_str).stem
        self.meas_state.extra_overlays.append(overlay_from_loaded_csv(loaded, label))
        ctx.display.set_button_disabled("clear_overlay_csv", False)
        print(f"[LOAD+] Added overlay: {Path(path_str).name}")

    def _on_push_overlay(self, ctx: ModeContext) -> None:
        """Snapshot current live SPD as Load+ (same ``SpectrumOverlay`` storage as Load+ CSV)."""
        if ctx.last_data is None:
            print("[Cap+] No current spectrum")
            return
        acc = ctx.last_intensity_accumulated
        if acc is None:
            print("[Cap+] No accumulated intensity")
            return
        wl = np.asarray(ctx.last_data.wavelengths, dtype=np.float64)
        n = len(wl)
        measured = np.asarray(acc, dtype=np.float64).ravel()[:n]
        if measured.size < n:
            measured = np.pad(measured, (0, n - int(measured.size)))
        stamp = datetime.now(timezone.utc).strftime("%H%M%S_%f")
        label = f"Cap_{stamp}"
        self.meas_state.extra_overlays.append(
            SpectrumOverlay(
                label=label,
                source_wavelengths=wl.copy(),
                measured=measured.copy(),
                dark=None,
                white=None,
                sensitivity=None,
            )
        )
        ctx.display.set_button_disabled("clear_overlay_csv", False)
        print(f"[Cap+] Added overlay: {label}")

    def _on_clear_overlay_csv(self, ctx: ModeContext) -> None:
        """Clear Load+ traces only."""
        self.meas_state.extra_overlays.clear()
        ctx.display.set_button_disabled("clear_overlay_csv", True)

    def _on_toggle_raw_overlay(self, ctx: ModeContext) -> None:
        """Toggle overlay of dark, white, and measured spectra (unmodified)."""
        self.meas_state.show_raw_overlay = not self.meas_state.show_raw_overlay
        ctx.display.set_button_active("show_raw_overlay", self.meas_state.show_raw_overlay)
        print(f"[Measurement] Raw overlay: {'ON' if self.meas_state.show_raw_overlay else 'OFF'}")

    def _on_toggle_absorption(self, ctx: ModeContext) -> None:
        """Toggle absorption spectrum (only effective when white reference is set)."""
        if self.meas_state.white_spectrum is None:
            return
        self.meas_state.show_absorption = not self.meas_state.show_absorption
        ctx.display.set_button_active("show_absorption", self.meas_state.show_absorption)
        print(f"[Measurement] Absorption: {'ON' if self.meas_state.show_absorption else 'OFF'}")

    def _on_toggle_light(self, ctx: ModeContext) -> None:
        """Toggle light control - placeholder."""
        print("[LIGHT] Toggle light (GPIO not implemented yet)")

    def update_display(
        self,
        ctx: ModeContext,
        processed: "SpectrumData",
        graph_height: int,
    ) -> None:
        """Update measurement overlay and status. Control graph click from mode: markers when peaks off, else default (pan/peak_region/spectrum_select)."""
        ctx.display.set_button_active("capture", not ctx.frozen_spectrum)
        ctx.display.set_button_active("show_raw_overlay", self.meas_state.show_raw_overlay)
        white_set = self.meas_state.white_spectrum is not None
        ctx.display.set_button_disabled("show_absorption", not white_set)
        if not white_set and self.meas_state.show_absorption:
            self.meas_state.show_absorption = False
            ctx.display.set_button_active("show_absorption", False)
        ctx.display.set_button_active("show_absorption", self.meas_state.show_absorption)
        ctx.display.set_button_disabled(
            "clear_overlay_csv",
            len(self.meas_state.extra_overlays) == 0,
        )
        ctx.display.state.graph_click_behavior = (
            "marker" if not ctx.display.state.peaks_visible else "default"
        )
        wl = processed.wavelengths
        if self.meas_state.show_raw_overlay:
            raw_list = self._build_raw_overlays(ctx, wl, graph_height)
            raw_list.extend(self._build_extra_overlay_polylines(wl, ctx))
            ctx.display.set_raw_overlays(raw_list)
            ctx.display.set_mode_overlay(None)
        else:
            ctx.display.set_raw_overlays(self._build_extra_overlay_polylines(wl, ctx))
            ctx.display.set_mode_overlay(None)
        ctx.display.set_sensitivity_overlay(None)
        for key, value in self.get_status().items():
            ctx.display.set_status(key, value)

    @property
    def mode_type(self) -> ModeType:
        return ModeType.MEASUREMENT

    @property
    def name(self) -> str:
        return "Measurement"

    @property
    def preview_modes(self) -> list[str]:
        """No full image preview in measurement; spectrum bar or none only."""
        return ["none", "spectrum"]

    def on_start(self, ctx: ModeContext) -> None:
        """Hide camera preview by default and sync Rec (freeze), Overlay and ABS buttons."""
        ctx.display.preview_mode = "none"
        ctx.display.state.reference_spectrum = None
        ctx.display.state.reference_name = "None"
        ctx.display.set_button_active("capture", not ctx.frozen_spectrum)
        ctx.display.set_button_active("show_raw_overlay", self.meas_state.show_raw_overlay)
        white_set = self.meas_state.white_spectrum is not None
        ctx.display.set_button_disabled("show_absorption", not white_set)
        ctx.display.set_button_active("show_absorption", self.meas_state.show_absorption)
        ctx.display.set_button_disabled(
            "clear_overlay_csv",
            len(self.meas_state.extra_overlays) == 0,
        )

    def get_buttons(self) -> list[ButtonDefinition]:
        """Get measurement mode buttons. Grouped with gaps; Quit right-aligned."""
        return [
            # Row 1: Rec (capture + progress) | Avg/Max/Acc | Dark/White | Bars | ZX/ZY | VIEW
            ButtonDefinition("Rec", "capture", is_toggle=True, row=1, icon_type="playback"),
            ButtonDefinition("S", "toggle_sensitivity", is_toggle=True, row=1, icon_type="sensitivity"),
            ButtonDefinition("__gap__", "__gap__", row=1),
            ButtonDefinition("Avg", "toggle_averaging", is_toggle=True, row=1, icon_type="avg"),
            ButtonDefinition("Max", "capture_peak", is_toggle=True, shortcut="h", row=1, icon_type="peak_hold"),
            ButtonDefinition("Acc", "toggle_accumulation", is_toggle=True, row=1, icon_type="acc"),
            ButtonDefinition("__gap__", "__gap__2", row=1),
            ButtonDefinition("Drk", "set_dark", is_toggle=True, row=1, icon_type="dark"),
            ButtonDefinition("Wht", "set_white", is_toggle=True, row=1, icon_type="white"),
            ButtonDefinition("ABS", "show_absorption", is_toggle=True, row=1, icon_type="absorption"),
            ButtonDefinition("__gap__", "__gap__3", row=1),
            ButtonDefinition("Brs", "show_spectrum_bars", is_toggle=True, row=1, icon_type="bars"),
            ButtonDefinition("__gap__", "__gap__4", row=1),
            ButtonDefinition("ZX", "show_zoom_x_slider", is_toggle=True, row=1, icon_type="zoom_x"),
            ButtonDefinition("ZY", "show_zoom_y_slider", is_toggle=True, row=1, icon_type="zoom_y"),
            ButtonDefinition("__gap__", "__gap__5", row=1),
            ButtonDefinition("VIEW", "cycle_preview", shortcut="v", row=1, icon_type="eye"),
            ButtonDefinition("__spacer__", "__spacer_right__", row=1),
            ButtonDefinition("Quit", "quit", row=1, icon_type="quit"),
            # Row 2: Lamp | Peaks/Snap/Pkd/Clr | G/E/AG/AE | Save / Load+ / Export | Quit
            ButtonDefinition("Lamp", "lamp_toggle", is_toggle=True, row=2, icon_type="lamp"),
            ButtonDefinition("__gap__", "__gap__6", row=2),
            ButtonDefinition("Pks", "show_peaks", is_toggle=True, row=2, icon_type="peaks"),
            ButtonDefinition("Snp", "snap_to_peaks", is_toggle=True, row=2, icon_type="snap"),
            ButtonDefinition("Pkd", "show_peak_delta", is_toggle=True, row=2, icon_type="delta"),
            ButtonDefinition("Clr", "clear_all", shortcut="z", row=2, icon_type="clear"),
            ButtonDefinition("__gap__", "__gap__7", row=2),
            ButtonDefinition("Ovrl", "show_raw_overlay", is_toggle=True, row=2, icon_type="overlay"),
            ButtonDefinition("__gap__", "__gap__8", row=2),
            ButtonDefinition("G", "show_gain_slider", is_toggle=True, row=2, icon_type="gain"),
            ButtonDefinition("E", "show_exposure_slider", is_toggle=True, row=2, icon_type="exposure"),
            ButtonDefinition("AG", "auto_gain", is_toggle=True, row=2, icon_type="auto_gain"),
            ButtonDefinition("AE", "auto_exposure", is_toggle=True, row=2, icon_type="auto_exposure"),
            ButtonDefinition("__gap__", "__gap__9", row=2),
            ButtonDefinition("Save", "save", shortcut="s", row=2, icon_type="save"),
            ButtonDefinition("SVG", "export_vector", shortcut="g", row=2),
            ButtonDefinition("Pdf", "export_pdf", shortcut="p", row=2, icon_type="pdf"),
            ButtonDefinition("Load+", "load_overlay_csv", row=2, icon_type="load_plus"),
            ButtonDefinition("Cap+", "push_overlay", row=2, icon_type="acc"),
            ButtonDefinition("ClrOv", "clear_overlay_csv", row=2, icon_type="clear"),
        ]

    def process_spectrum(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Process spectrum with dark/white correction (and optional absorption view).

        Returns 0-1 normalized intensity for pipeline consistency.
        """
        result = intensity.astype(np.float64)

        if self.state.integration_mode != "none":
            result = self.accumulate_spectrum(result)
        self._last_measured_pre_correction = result.copy()

        # Apply dark/white reference correction (shared logic)
        dark = self.meas_state.dark_spectrum if self.meas_state.subtract_dark else None
        result = apply_dark_white_correction(
            result,
            dark,
            self.meas_state.white_spectrum,
        )

        # Absorption: A = -log10(T), with T the transmission (result). Scale to 0-1 for display.
        if self.meas_state.show_absorption and self.meas_state.white_spectrum is not None:
            transmission = np.maximum(result, 1e-10)
            absorption = -np.log10(transmission)
            max_abs = 4.0
            result = np.clip(absorption / max_abs, 0, 1).astype(np.float32)

        return result

    def _build_raw_overlays(
        self,
        ctx: ModeContext,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
        """Build list of (intensity 0-1, BGR) for dark, white, measured (unmodified)."""
        n = len(wavelengths)
        if n == 0:
            return []
        raw = _get_raw(ctx)
        spectra: list[tuple[np.ndarray, tuple[int, int, int]]] = []
        if self.meas_state.dark_spectrum is not None:
            d = np.asarray(self.meas_state.dark_spectrum, dtype=np.float64)[:n]
            spectra.append((d, (40, 40, 40)))
        if self.meas_state.white_spectrum is not None:
            w = np.asarray(self.meas_state.white_spectrum, dtype=np.float64)[:n]
            spectra.append((w, (200, 120, 0)))
        if raw is not None:
            m = np.asarray(raw, dtype=np.float64)[:n]
            spectra.append((m, (0, 0, 200)))
        if not spectra:
            return []
        global_max = max(
            float(np.max(s[0])) for s in spectra
        )
        global_max = max(global_max, 1.0)
        out: list[tuple[np.ndarray, tuple[int, int, int]]] = []
        for arr, color in spectra:
            norm = np.clip(arr / global_max, 0, 1).astype(np.float32)
            out.append((norm, color))
        return out

    def overlay_export_columns(
        self,
        ctx: ModeContext,
        wavelengths: np.ndarray,
    ) -> list[tuple[str, np.ndarray]]:
        """Extra CSV columns for Load+ traces (matches graph for current S toggle)."""
        wl = np.asarray(wavelengths, dtype=np.float64)
        rows: list[tuple[str, np.ndarray]] = []
        sens_on = ctx.sensitivity_correction_enabled
        dk = self.meas_state.dark_spectrum
        wh = self.meas_state.white_spectrum
        for ov in self.meas_state.extra_overlays:
            col = overlay_display_intensity(
                ov,
                wl,
                ctx,
                sensitivity_enabled=sens_on,
                dark_session=dk,
                white_session=wh,
            )
            slug = export_slugify(ov.label) or "overlay"
            rows.append((f"Overlay_{slug}", np.asarray(col, dtype=np.float64)))
        return rows

    def _build_extra_overlay_polylines(
        self,
        wavelengths: np.ndarray,
        ctx: ModeContext,
    ) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
        """Load+ traces in the same units as the main measurement line."""
        n = len(wavelengths)
        out: list[tuple[np.ndarray, tuple[int, int, int]]] = []
        sens_on = ctx.sensitivity_correction_enabled
        dk = self.meas_state.dark_spectrum
        wh = self.meas_state.white_spectrum
        wl = np.asarray(wavelengths, dtype=np.float64)
        for i, ov in enumerate(self.meas_state.extra_overlays):
            row = overlay_display_intensity(
                ov,
                wl,
                ctx,
                sensitivity_enabled=sens_on,
                dark_session=dk,
                white_session=wh,
            )
            if len(row) != n:
                row = np.asarray(row, dtype=np.float32).ravel()[:n]
                if len(row) < n:
                    row = np.pad(row, (0, n - len(row)))
            color = _EXTRA_OVERLAY_BGR[i % len(_EXTRA_OVERLAY_BGR)]
            out.append((row.astype(np.float32), color))
        return out

    def get_overlay(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> tuple[np.ndarray, tuple[int, int, int]] | None:
        """No dedicated mode overlay line; comparisons use Load+ / Cap+ via ``set_raw_overlays``."""
        return None

    def capture_current(self, intensity: np.ndarray, wavelengths: np.ndarray) -> None:
        """Capture current spectrum for later use."""
        self.meas_state.captured_spectrum = intensity.copy()
        self.meas_state.captured_wavelengths = wavelengths.copy()
        print("[Measurement] Spectrum captured")

    def set_dark_reference(self, intensity: np.ndarray) -> None:
        """Set dark/black reference spectrum."""
        self.meas_state.dark_spectrum = intensity.copy()
        self.meas_state.subtract_dark = True
        print("[Measurement] Dark reference set")

    def set_white_reference(self, intensity: np.ndarray) -> None:
        """Set white reference spectrum."""
        self.meas_state.white_spectrum = intensity.copy()
        print("[Measurement] White reference set")

    def clear_references(self) -> None:
        """Clear dark/white references and related display flags."""
        self.meas_state.dark_spectrum = None
        self.meas_state.dark_acq = ""
        self.meas_state.white_spectrum = None
        self.meas_state.white_acq = ""
        self.meas_state.subtract_dark = True
        self.meas_state.show_raw_overlay = False
        self.meas_state.show_absorption = False
        print("[Measurement] All references cleared")

    def get_status(self) -> dict[str, str]:
        """Get status values for display."""
        status = {}

        if self.meas_state.dark_spectrum is not None:
            status["Dark"] = "SET"
        if self.meas_state.white_spectrum is not None:
            status["White"] = "SET"
        if self.meas_state.show_absorption:
            status["ABS"] = "ON"

        if self.state.integration_mode != "none":
            status[self.state.integration_mode.capitalize()] = str(self.state.accumulated_frames)

        n_extra = len(self.meas_state.extra_overlays)
        if n_extra:
            status["Ov+"] = str(n_extra)

        return status


def _pick_csv_file() -> str:
    """Show an OS file-open dialog and return the chosen path, or empty string."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Load spectrum overlay",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        parent=root,
    )
    root.destroy()
    return path


def _get_raw(ctx: ModeContext) -> np.ndarray | None:
    """Return post-accumulation, pre-dark/white intensity for dark/white capture."""
    if ctx.last_intensity_accumulated is not None:
        return ctx.last_intensity_accumulated
    if ctx.last_raw_intensity is not None:
        return ctx.last_raw_intensity
    return None
