"""Calibration sub-mode for CSV viewer: pixel→wavelength calibration without a camera.

Handles automatic peak-matching and manual right-click peak pairing.
Draws reference peaks, match connections, and pending-pair markers onto the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..data.reference_spectra import ReferenceSource, get_reference_spectrum
from ..processing.auto_calibrator import calibrate as run_auto_calibration
from ..processing.calibration import get_reference_peaks
from ..utils.graph_scale import scale_intensity_to_graph

if TYPE_CHECKING:
    from ..display.viewport import Viewport

# Screen-pixel radius within which snap activates on right-click
_SNAP_RADIUS_PX = 20
# Screen-pixel radius to count as "near an existing manual pair" for removal
_REMOVE_RADIUS_PX = 12


@dataclass
class CsvCalibrationState:
    """Mutable state for the CSV calibration sub-mode."""

    active: bool = False
    current_source: ReferenceSource = ReferenceSource.HG
    overlay_visible: bool = True
    auto_cal_points: list[tuple[int, float]] = field(default_factory=list)
    manual_pairs: list[tuple[int, float]] = field(default_factory=list)
    # First right-click sets this; second right-click completes the pair
    pending_pixel: int | None = None
    # Future: show only reference peak vertical lines (no SPD overlay)
    ref_peaks_only: bool = False


class CsvCalibrationHandler:
    """Auto-calibration + manual peak-pairing for the CSV viewer.

    Right-click workflow (two steps):
      1. Right-click near a measured peak → stores pending_pixel
      2. Right-click near a reference peak line → completes (pixel, wavelength) pair
    Pairs are combined with auto-calibration results for recalibrate().
    """

    def __init__(self) -> None:
        self._state = CsvCalibrationState()

    @property
    def state(self) -> CsvCalibrationState:
        return self._state

    @property
    def active(self) -> bool:
        return self._state.active

    def toggle(self) -> bool:
        """Toggle cal sub-mode on/off. Returns new active state."""
        self._state.active = not self._state.active
        if not self._state.active:
            self._state.pending_pixel = None
        return self._state.active

    def select_source(self, source: ReferenceSource) -> None:
        self._state.current_source = source

    def toggle_overlay(self) -> bool:
        """Toggle reference SPD overlay. Returns new visibility."""
        self._state.overlay_visible = not self._state.overlay_visible
        return self._state.overlay_visible

    def run_auto_cal(
        self,
        measured: np.ndarray,
        wavelengths: np.ndarray,
        *,
        file_loader: "ReferenceFileLoader | None" = None,
    ) -> list[tuple[int, float]]:
        """Run RANSAC peak-matching auto-calibration against current reference source."""
        ref_wl = np.linspace(380.0, 750.0, 500)
        ref_int = get_reference_spectrum(self._state.current_source, ref_wl, file_loader=file_loader)
        points = run_auto_calibration(measured, ref_wl, ref_int)
        self._state.auto_cal_points = points
        print(f"[Cal] Auto-cal found {len(points)} points for {self._state.current_source.name}")
        return points

    def handle_right_click(
        self,
        data_x: float,
        peaks: list,
        wavelengths: np.ndarray,
        viewport: "Viewport",
        width: int,
    ) -> bool:
        """Process a right-click in the graph area.

        Step 1 (pending_pixel is None): snap to nearest measured peak.
        Step 2 (pending_pixel set): snap to nearest reference peak → form pair.
        Returns True when a new pair is formed and recalibration should run.
        """
        ref_peaks = get_reference_peaks(self._state.current_source)

        if self._state.pending_pixel is None:
            self._state.pending_pixel = self._snap_measured(data_x, peaks, viewport, width)
            print(f"[Cal] Pending measured pixel: {self._state.pending_pixel}")
            return False

        ref_wl = self._snap_ref_peak(data_x, ref_peaks, wavelengths, viewport, width)
        if ref_wl is None:
            # Cancels if no reference peak is close enough
            print("[Cal] No reference peak nearby — cancelled pairing")
            self._state.pending_pixel = None
            return False

        pixel = self._state.pending_pixel
        self._drop_nearby_manual(pixel, ref_wl)
        self._state.manual_pairs.append((pixel, ref_wl))
        self._state.pending_pixel = None
        print(f"[Cal] Manual pair: pixel {pixel} → {ref_wl:.2f} nm")
        return True

    def cancel_pending(self) -> None:
        """Discard a half-completed right-click pair."""
        self._state.pending_pixel = None

    def all_cal_points(self) -> list[tuple[int, float]]:
        """Merged auto + manual points; manual supersedes nearby auto points."""
        manual_pixels = {p for p, _ in self._state.manual_pairs}
        auto_filtered = [
            (p, w) for p, w in self._state.auto_cal_points
            if all(abs(p - mp) > 5 for mp in manual_pixels)
        ]
        return auto_filtered + list(self._state.manual_pairs)

    def clear_pairs(self) -> None:
        """Clear both auto and manual calibration pairs."""
        self._state.auto_cal_points.clear()
        self._state.manual_pairs.clear()
        self._state.pending_pixel = None

    def reset_linear(self, n: int) -> list[tuple[int, float]]:
        """Reset to a linear 380–750 nm mapping across n pixels."""
        pixels = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        wl_min, wl_max = 380.0, 750.0
        wls = [wl_min + (wl_max - wl_min) * p / max(n - 1, 1) for p in pixels]
        self._state.auto_cal_points = list(zip(pixels, wls))
        self._state.manual_pairs.clear()
        self._state.pending_pixel = None
        return list(zip(pixels, wls))

    def reference_overlay(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
        *,
        file_loader: "ReferenceFileLoader | None" = None,
    ) -> tuple[np.ndarray, tuple[int, int, int]] | None:
        """Return reference SPD as (scaled_intensity, BGR_color) for the mode overlay."""
        if not self._state.overlay_visible:
            return None
        ref = get_reference_spectrum(self._state.current_source, wavelengths, file_loader=file_loader)
        if ref is None:
            return None
        ref = np.asarray(ref, dtype=np.float32)
        peak = float(np.max(ref))
        if peak > 1e-6:
            ref = ref / peak
        return (scale_intensity_to_graph(ref, graph_height), (0, 150, 255))

    def draw_on_graph(
        self,
        graph: np.ndarray,
        wavelengths: np.ndarray,
        peaks: list,
        viewport: "Viewport",
        width: int,
    ) -> None:
        """Draw calibration markers on the graph image in-place.

        Draws reference peak lines, auto-pair connections (blue),
        manual-pair connections (green), and the pending pixel marker (red).
        """
        graph_h = graph.shape[0]
        ref_peaks = get_reference_peaks(self._state.current_source)
        n = len(wavelengths)

        mid_y = graph_h // 2
        upper_y = graph_h // 4

        def wl_to_sx(wl: float) -> int:
            px = int(np.searchsorted(wavelengths, wl))
            px = min(max(0, px), n - 1)
            return viewport.data_x_to_screen(float(px), width)

        def px_to_sx(px: int) -> int:
            return viewport.data_x_to_screen(float(px), width)

        self._draw_ref_peaks(graph, ref_peaks, wl_to_sx, graph_h, width)
        self._draw_auto_pairs(graph, wl_to_sx, px_to_sx, mid_y, upper_y, width)
        self._draw_manual_pairs(graph, wl_to_sx, px_to_sx, mid_y, upper_y, width)
        self._draw_pending(graph, px_to_sx, mid_y, width)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_ref_peaks(self, graph, ref_peaks, wl_to_sx, graph_h, width) -> None:
        for ref in ref_peaks:
            sx = wl_to_sx(ref.wavelength)
            if 0 <= sx < width:
                cv2.line(graph, (sx, 0), (sx, graph_h), (0, 130, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    graph, f"{ref.wavelength:.0f}",
                    (sx + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 100, 200), 1,
                )

    def _draw_auto_pairs(self, graph, wl_to_sx, px_to_sx, mid_y, upper_y, width) -> None:
        for meas_px, ref_wl in self._state.auto_cal_points:
            sx_m = px_to_sx(meas_px)
            sx_r = wl_to_sx(ref_wl)
            if 0 <= sx_m < width:
                cv2.circle(graph, (sx_m, mid_y), 4, (200, 80, 0), -1)
            if 0 <= sx_m < width and 0 <= sx_r < width:
                cv2.line(graph, (sx_m, mid_y), (sx_r, upper_y), (200, 80, 0), 1, cv2.LINE_AA)

    def _draw_manual_pairs(self, graph, wl_to_sx, px_to_sx, mid_y, upper_y, width) -> None:
        for meas_px, ref_wl in self._state.manual_pairs:
            sx_m = px_to_sx(meas_px)
            sx_r = wl_to_sx(ref_wl)
            if 0 <= sx_m < width:
                cv2.circle(graph, (sx_m, mid_y), 6, (0, 200, 0), 2)
                cv2.putText(
                    graph, f"{ref_wl:.1f}",
                    (sx_m + 4, mid_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1,
                )
            if 0 <= sx_m < width and 0 <= sx_r < width:
                cv2.line(graph, (sx_m, mid_y), (sx_r, upper_y), (0, 200, 0), 2, cv2.LINE_AA)

    def _draw_pending(self, graph, px_to_sx, mid_y, width) -> None:
        if self._state.pending_pixel is None:
            return
        sx = px_to_sx(self._state.pending_pixel)
        if 0 <= sx < width:
            cv2.circle(graph, (sx, mid_y), 8, (0, 0, 220), 2)
            cv2.putText(
                graph, "? ref",
                (sx + 6, mid_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 220), 1,
            )

    def _snap_measured(
        self, data_x: float, peaks: list, viewport: "Viewport", width: int
    ) -> int:
        screen_x = viewport.data_x_to_screen(data_x, width)
        best_px = int(round(data_x))
        best_dist = _SNAP_RADIUS_PX + 1
        for peak in peaks:
            sx = viewport.data_x_to_screen(float(peak.index), width)
            dist = abs(sx - screen_x)
            if dist < best_dist:
                best_dist = dist
                best_px = peak.index
        return best_px

    def _snap_ref_peak(
        self,
        data_x: float,
        ref_peaks: list,
        wavelengths: np.ndarray,
        viewport: "Viewport",
        width: int,
    ) -> float | None:
        screen_x = viewport.data_x_to_screen(data_x, width)
        n = len(wavelengths)
        best_wl = None
        best_dist = _SNAP_RADIUS_PX + 1
        for ref in ref_peaks:
            px = int(np.searchsorted(wavelengths, ref.wavelength))
            px = min(max(0, px), n - 1)
            sx = viewport.data_x_to_screen(float(px), width)
            dist = abs(sx - screen_x)
            if dist < best_dist:
                best_dist = dist
                best_wl = ref.wavelength
        return best_wl

    def _drop_nearby_manual(self, pixel: int, ref_wl: float) -> None:
        """Remove any existing manual pair close to the given pixel or ref_wl."""
        self._state.manual_pairs = [
            (p, w) for p, w in self._state.manual_pairs
            if abs(p - pixel) > 5 and abs(w - ref_wl) > 3.0
        ]
