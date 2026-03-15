"""Main display manager for spectrum visualization."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..config import Config
from ..core.calibration import Calibration
from ..core.spectrum import SpectrumData
from ..gui.control_bar import ControlBar, ControlBarConfig
from ..gui.sliders import (
    HorizontalSlider,
    SliderMode,
    SliderPanel,
    SliderStyle,
    VerticalSlider,
)
from ..modes.base import BaseMode
from ..processing.spectrum_transform import (
    get_crop_corners_rotated,
    params_from_saved,
    transform_bbox_rotated_to_original,
)
from ..utils.display import scale_to_uint8
from .graticule import GraticuleRenderer
from .markers import MarkersRenderer
from .overlay_utils import render_polyline_overlay
from .peaks import PeaksRenderer, _spectrum_screen_y
from .spectrum import SpectrogramRenderer
from .viewport import Viewport
from .waterfall import WaterfallDisplay

def _scale_to_fit(
    img: np.ndarray,
    target_width: int,
    target_height: int,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Scale image to fit within target area, maintaining aspect ratio. Letterbox/pillarbox as needed."""
    if img.size == 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    src_h, src_w = img.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)
    scale = min(target_width / src_w, target_height / src_h)
    out_w = int(src_w * scale)
    out_h = int(src_h * scale)
    if out_w <= 0 or out_h <= 0:
        return np.full((target_height, target_width, 3), bg_color, dtype=np.uint8)
    scaled = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    if scaled.ndim == 2:
        scaled = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    out = np.full((target_height, target_width, 3), bg_color, dtype=np.uint8)
    y0 = (target_height - out_h) // 2
    x0 = (target_width - out_w) // 2
    out[y0 : y0 + out_h, x0 : x0 + out_w] = scaled
    return out


@dataclass
class CursorState:
    """State for cursor display."""

    x: int = 0
    y: int = 0
    measure_mode: bool = False
    pixel_mode: bool = False


# Max pixel movement to treat as click (not drag)
_CLICK_DRAG_THRESHOLD = 5
# Screen-pixel radius to grab an existing marker line for drag/remove
_MARKER_GRAB_RADIUS = 10
# Screen-pixel radius within which snap-to-peak activates while dragging
_MARKER_SNAP_RADIUS = 15


@dataclass
class DisplayState:
    """Mutable display state."""

    hold_peaks: bool = False
    held_intensity: np.ndarray | None = None
    cursor: CursorState = None
    save_message: str = "No data saved"
    reference_spectrum: np.ndarray | None = None
    reference_name: str = "None"
    spectrum_bars_visible: bool = False
    selected_spectrum_index: int | None = None
    # Click-to-include: single (center_index, half_width) in data coordinates, or None
    peak_include_region: tuple[int, int] | None = None
    peaks_visible: bool = True
    # Vertical marker lines: list of data indices (max 10)
    marker_lines: list[int] = field(default_factory=list)
    snap_to_peaks: bool = False
    # Show peak/dip width (FWHM) below wavelength on peaks and markers when on a peak
    peak_delta_visible: bool = False

    def __post_init__(self):
        if self.cursor is None:
            self.cursor = CursorState()


class DisplayManager:
    """Manages all spectrum display rendering.

    This class handles the OpenCV window management and rendering
    of spectrum graphs, camera preview, messages, and optional
    waterfall display.
    """

    def __init__(
        self,
        config: Config,
        calibration: Calibration,
        mode: str = "measurement",
        mode_instance: Optional["BaseMode"] = None,
    ):
        """Initialize display manager.

        Args:
            config: Application configuration
            calibration: Wavelength calibration data
            mode: Operating mode (calibration, measurement, raman, colorscience)
            mode_instance: Mode instance for button definitions (single source of truth)
        """
        self.config = config
        self.calibration = calibration
        self.state = DisplayState()
        self.mode = mode

        self._graticule = GraticuleRenderer(
            font=config.display.graph_label_font,
            font_scale=config.display.graph_label_font_scale,
        )
        self._spectrogram = SpectrogramRenderer()
        self._peaks_renderer = PeaksRenderer(
            font=config.display.graph_label_font,
            font_scale=config.display.graph_label_font_scale,
        )
        self._markers_renderer = MarkersRenderer(
            font=config.display.graph_label_font,
            font_scale=config.display.graph_label_font_scale,
        )
        self._waterfall: WaterfallDisplay | None = None

        display_width = config.display.window_width
        if config.display.waterfall_enabled or mode == "waterfall":
            self._waterfall = WaterfallDisplay(
                width=display_width,
                height=config.display.graph_height,
                contrast=config.waterfall.contrast,
                brightness=config.waterfall.brightness,
            )

        self._mode_instance = mode_instance
        buttons = mode_instance.get_buttons() if mode_instance is not None else None
        self._control_bar = ControlBar(
            width=display_width,
            config=ControlBarConfig(
                height=config.display.message_height,
                font_scale=config.display.font_scale,
            ),
            mode=mode,
            buttons=buttons,
            mode_instance=mode_instance,
        )

        # Slider panel for gain/exposure/LED (positioned on right side of graph area)
        slider_x = display_width - 170
        slider_y = config.display.message_height + config.display.preview_height + 10
        slider_height = config.display.graph_height - 30
        self._slider_panel = SliderPanel(
            x=slider_x,
            y=slider_y,
            height=slider_height,
        )

        # Mode overlay (intensity array, color)
        self._mode_overlay: tuple[np.ndarray, tuple[int, int, int]] | None = None
        # Sensitivity curve overlay (intensity array, color)
        self._sensitivity_overlay: tuple[np.ndarray, tuple[int, int, int]] | None = None
        # Autolevel non-blocking overlay
        self._autolevel_overlay: np.ndarray | None = None
        self._on_autolevel_click: Callable[[], None] | None = None
        # Full graph area override (replaces spectrum rendering when set)
        self._graph_override: np.ndarray | None = None
        # Preview strip override (replaces camera crop when set)
        self._preview_override: np.ndarray | None = None
        # Prominent info text lines rendered on graph (e.g. XYZ/LAB values)
        self._graph_info: list[str] = []
        # Optional callback for left-clicks in the graph area: fn(x, rel_y)
        self._on_graph_click: Callable[[int, int], None] | None = None

        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._graph_font = config.display.graph_label_font
        self._graph_font_scale = config.display.graph_label_font_scale
        self._windows_created = False

        # Preview display mode: "full", "window", "none"
        self._preview_mode = "window"

        # Track mouse button state for slider dragging
        self._mouse_down = False

        # Zoom/pan viewport
        self._viewport = Viewport()

        # Zoom sliders (visible when user clicks on left/bottom edge, auto-hide after 1s)
        style = SliderStyle()
        self._zoom_vertical = VerticalSlider(
            x=0,
            y=config.display.message_height + config.display.preview_height + 20,
            height=config.display.graph_height - 50,
            min_val=1.0,
            max_val=20.0,
            value=1.0,
            label="V",
            style=style,
            mode=SliderMode.RELATIVE,
        )
        self._zoom_horizontal = HorizontalSlider(
            x=20,
            y=config.display.message_height
            + config.display.preview_height
            + config.display.graph_height
            - 25,
            width=display_width - 200,
            min_val=1.0,
            max_val=20.0,
            value=1.0,
            label="H",
            style=style,
            mode=SliderMode.RELATIVE,
        )
        self._zoom_vertical_prev = 1.0
        self._zoom_horizontal_prev = 1.0

        def _on_zoom_vertical(val: float) -> None:
            if val <= self._zoom_vertical.min_val:
                self._viewport.y_min = 0.0
                self._viewport.y_max = 1.0
            elif self._zoom_vertical_prev > 0:
                self._viewport.zoom_y(val / self._zoom_vertical_prev)
            self._zoom_vertical_prev = val

        def _on_zoom_horizontal(val: float) -> None:
            if val <= self._zoom_horizontal.min_val:
                self._viewport.x_start = 0.0
                if self._last_data_width > 0:
                    self._viewport.x_end = float(self._last_data_width - 1)
            elif self._zoom_horizontal_prev > 0:
                self._viewport.zoom_x(val / self._zoom_horizontal_prev)
            self._zoom_horizontal_prev = val

        self._zoom_vertical.on_change = _on_zoom_vertical
        self._zoom_horizontal.on_change = _on_zoom_horizontal

        # Pan state
        self._panning = False
        self._pan_last_x = 0
        self._pan_last_y = 0
        self._last_data_width = 0

        # Click vs drag for peak-include regions
        self._click_for_region = False
        self._click_down_x = 0
        self._click_down_y = 0

        # Marker line interaction state
        self._dragging_marker_idx: int | None = None
        self._drag_is_existing: bool = False
        self._drag_start_x: int = 0
        # Last known spectrum peaks for snap-to-peak
        self._last_peaks: list = []

    def set_frame_dimensions(self, width: int, height: int) -> None:
        """Update dimensions (e.g. after camera reports actual size).

        Display stays at window_width. Only used for waterfall when enabled.
        """
        display_width = self.config.display.window_width
        self._control_bar.set_width(display_width)
        if self._waterfall is not None:
            self._waterfall.set_dimensions(display_width, self.config.display.graph_height)
        self._slider_panel.x = display_width - 170

    def setup_windows(self) -> None:
        """Create and configure OpenCV windows.

        Uses WINDOW_GUI_NORMAL to disable Qt/GTK toolbar and statusbar.
        If toolbar still appears, OpenCV was built with Qt and ignores this flag.
        In that case, rebuild OpenCV with -DWITH_QT=OFF or use GTK backend.
        """
        if self._windows_created:
            return

        title1 = self.config.spectrograph_title
        title2 = self.config.waterfall_title
        waterfall_mode = self.mode == "waterfall"

        # WINDOW_GUI_NORMAL (0x10 = 16) disables the Qt/GTK toolbar and statusbar
        WINDOW_GUI_NORMAL = getattr(cv2, "WINDOW_GUI_NORMAL", 0x00000010)
        win_w = self.config.display.window_width
        win_h = self.config.display.stack_height

        if waterfall_mode:
            # Waterfall mode: single window (waterfall only)
            cv2.namedWindow(title2, cv2.WINDOW_NORMAL | WINDOW_GUI_NORMAL)
            cv2.resizeWindow(title2, win_w, win_h)
            cv2.moveWindow(title2, 0, 0)
            cv2.setMouseCallback(title2, self._handle_mouse)
        else:
            if self.config.display.waterfall_enabled:
                cv2.namedWindow(title2, cv2.WINDOW_NORMAL | WINDOW_GUI_NORMAL)
                cv2.resizeWindow(title2, win_w, win_h)
                cv2.moveWindow(title2, win_w + 10, 0)

            if self.config.display.fullscreen:
                cv2.namedWindow(title1, cv2.WINDOW_NORMAL | WINDOW_GUI_NORMAL)
                cv2.setWindowProperty(
                    title1,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )
            else:
                cv2.namedWindow(title1, cv2.WINDOW_NORMAL | WINDOW_GUI_NORMAL)
                cv2.resizeWindow(title1, win_w, win_h)
                cv2.moveWindow(title1, 0, 0)

            cv2.setMouseCallback(title1, self._handle_mouse)

        self._windows_created = True

    def cycle_preview_mode(self) -> str:
        """Cycle through preview modes defined by the active mode instance."""
        modes = (
            self._mode_instance.preview_modes
            if self._mode_instance is not None
            else ["window", "spectrum", "full", "none"]
        )
        current_idx = modes.index(self._preview_mode) if self._preview_mode in modes else 0
        self._preview_mode = modes[(current_idx + 1) % len(modes)]
        print(f"[PREVIEW] Mode: {self._preview_mode}")
        return self._preview_mode

    def set_autolevel_overlay(self, img: np.ndarray | None) -> None:
        """Set autolevel overlay image (None to clear). Non-blocking."""
        self._autolevel_overlay = img

    def set_autolevel_click_dismiss(self, callback: Callable[[], None] | None) -> None:
        """Set callback for click-to-dismiss on autolevel overlay."""
        self._on_autolevel_click = callback

    def clear_peak_include_region(self) -> None:
        """Clear the click-to-include peak region."""
        self.state.peak_include_region = None

    def _handle_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events on the spectrum window (pixel-perfect, no scaling)."""
        control_bar_height = self.config.display.message_height
        preview_height = self.config.display.preview_height
        graph_height = self.config.display.graph_height
        width = self.config.display.window_width
        mouse_y_offset = control_bar_height + preview_height
        graph_rel_y = y - mouse_y_offset

        # Graph area bounds
        in_graph = 0 <= graph_rel_y < graph_height and 0 <= x < width

        if event == cv2.EVENT_MOUSEMOVE:
            self.state.cursor.x = x
            self.state.cursor.y = y

            if y < control_bar_height:
                self._control_bar.handle_mouse_move(x, y)

            if self._mouse_down:
                self._slider_panel.handle_mouse_move(x, y)
                if self._zoom_vertical.visible:
                    self._zoom_vertical.handle_mouse_move(x, y)
                if self._zoom_horizontal.visible:
                    self._zoom_horizontal.handle_mouse_move(x, y)
                if self._dragging_marker_idx is not None:
                    self._update_marker_drag(x, width)
                if self._panning:
                    x_zoomed = self._last_data_width > 0 and (
                        self._viewport.x_end - self._viewport.x_start < self._last_data_width - 1
                    )
                    y_zoomed = self._viewport.y_max - self._viewport.y_min < 0.99

                    if x_zoomed:
                        dx = self._viewport.screen_x_to_data(
                            self._pan_last_x, width
                        ) - self._viewport.screen_x_to_data(x, width)
                        self._viewport.pan_x(dx)
                    if y_zoomed:
                        dy = self._viewport.screen_y_to_intensity(
                            self._pan_last_y, graph_height
                        ) - self._viewport.screen_y_to_intensity(
                            graph_rel_y, graph_height
                        )
                        self._viewport.pan_y(dy)
                    self._pan_last_x = x
                    self._pan_last_y = graph_rel_y

        elif event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down = True
            self._click_for_region = False

            if y < control_bar_height:
                self._control_bar.handle_click(x, y)
            elif self._slider_panel.handle_mouse_down(x, y):
                pass
            elif self._zoom_vertical.visible and self._zoom_vertical.handle_mouse_down(x, y):
                pass
            elif self._zoom_horizontal.visible and self._zoom_horizontal.handle_mouse_down(x, y):
                pass
            elif self._on_autolevel_click is not None:
                self._on_autolevel_click()
            elif in_graph:
                if self._on_graph_click is not None:
                    self._on_graph_click(x, graph_rel_y)
                elif self._is_zoomed():
                    self._panning = True
                    self._pan_last_x = x
                    self._pan_last_y = graph_rel_y
                elif self.mode == "waterfall" or (
                    self.mode == "measurement" and not self.state.peaks_visible
                ):
                    self._start_marker_interaction(x, width)
                elif self.state.spectrum_bars_visible and self._last_data_width > 0:
                    data_x = self._viewport.screen_x_to_data(x, width)
                    idx = int(round(data_x))
                    idx = max(0, min(self._last_data_width - 1, idx))
                    self.state.selected_spectrum_index = idx
                else:
                    self._click_for_region = True
                    self._click_down_x = x
                    self._click_down_y = y

        elif event == cv2.EVENT_LBUTTONUP:
            if self._click_for_region and not self._panning:
                dx = abs(x - self._click_down_x)
                dy = abs(y - self._click_down_y)
                if dx < _CLICK_DRAG_THRESHOLD and dy < _CLICK_DRAG_THRESHOLD:
                    data_x = self._viewport.screen_x_to_data(x, width)
                    idx = int(round(data_x))
                    idx = max(0, min(self._last_data_width - 1, idx))
                    half_width = self.config.processing.peak_include_region_half_width
                    self.state.peak_include_region = (idx, half_width)
            self._finalize_marker_drag(x)
            self._click_for_region = False
            self._mouse_down = False
            self._slider_panel.handle_mouse_up()
            self._zoom_vertical.handle_mouse_up()
            self._zoom_horizontal.handle_mouse_up()
            self._panning = False

    def _is_zoomed(self) -> bool:
        """True if viewport is zoomed (not full view)."""
        if self._last_data_width <= 0:
            return False
        x_span = self._viewport.x_end - self._viewport.x_start
        y_span = self._viewport.y_max - self._viewport.y_min
        return x_span < self._last_data_width - 1 or y_span < 0.99

    # ------------------------------------------------------------------ markers

    def _find_nearby_marker(self, screen_x: int, width: int) -> int | None:
        """Return marker_lines list index of the closest marker within grab radius, or None."""
        best_idx = None
        best_dist = _MARKER_GRAB_RADIUS + 1
        for i, data_idx in enumerate(self.state.marker_lines):
            sx = self._viewport.data_x_to_screen(float(data_idx), width)
            dist = abs(sx - screen_x)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return best_idx

    def _resolve_marker_position(self, data_x: float, screen_x: int, width: int) -> int:
        """Convert data x float to marker data index, snapping to nearest peak when enabled."""
        n = max(1, self._last_data_width)
        raw_idx = max(0, min(n - 1, int(round(data_x))))
        if not self.state.snap_to_peaks or not self._last_peaks:
            return raw_idx
        best_idx = raw_idx
        best_dist = _MARKER_SNAP_RADIUS + 1
        for peak in self._last_peaks:
            peak_sx = self._viewport.data_x_to_screen(float(peak.index), width)
            dist = abs(peak_sx - screen_x)
            if dist < best_dist:
                best_dist = dist
                best_idx = peak.index
        return best_idx

    def _start_marker_interaction(self, screen_x: int, width: int) -> None:
        """Begin marker add-or-drag interaction on mouse-down in waterfall area."""
        existing = self._find_nearby_marker(screen_x, width)
        if existing is not None:
            self._dragging_marker_idx = existing
            self._drag_is_existing = True
        elif len(self.state.marker_lines) < 10:
            data_x = self._viewport.screen_x_to_data(screen_x, width)
            idx = self._resolve_marker_position(data_x, screen_x, width)
            self.state.marker_lines.append(idx)
            self._dragging_marker_idx = len(self.state.marker_lines) - 1
            self._drag_is_existing = False
        self._drag_start_x = screen_x

    def _update_marker_drag(self, screen_x: int, width: int) -> None:
        """Update position of the marker being dragged."""
        if self._dragging_marker_idx is None:
            return
        data_x = self._viewport.screen_x_to_data(screen_x, width)
        idx = self._resolve_marker_position(data_x, screen_x, width)
        self.state.marker_lines[self._dragging_marker_idx] = idx

    def _finalize_marker_drag(self, screen_x: int) -> None:
        """On mouse-up: remove existing marker if it was only tapped (not dragged)."""
        if self._dragging_marker_idx is None:
            return
        dx = abs(screen_x - self._drag_start_x)
        if self._drag_is_existing and dx < _CLICK_DRAG_THRESHOLD:
            self.state.marker_lines.pop(self._dragging_marker_idx)
        self._dragging_marker_idx = None
        self._drag_is_existing = False

    def toggle_snap_to_peaks(self) -> bool:
        """Toggle peak snapping for marker line placement. Returns new state."""
        self.state.snap_to_peaks = not self.state.snap_to_peaks
        return self.state.snap_to_peaks

    def toggle_peak_delta_visible(self) -> bool:
        """Toggle peak/marker width (FWHM) below wavelength label. Returns new state."""
        self.state.peak_delta_visible = not self.state.peak_delta_visible
        return self.state.peak_delta_visible

    def render(
        self,
        data: SpectrumData,
        camera_gain: float,
        savgol_poly: int,
        peak_min_dist: int,
        peak_threshold: int,
        extraction_method: str = "Weighted Sum",
        rotation_angle: float = 0.0,
        perp_width: int = 20,
        spectrum_y_center: int = 0,
    ) -> None:
        """Render the complete spectrum display.

        Args:
            data: Processed spectrum data to display
            camera_gain: Current camera gain for status display
            savgol_poly: Current Savitzky-Golay polynomial order
            peak_min_dist: Current peak minimum distance
            peak_threshold: Current peak threshold
            extraction_method: Current extraction method name
            rotation_angle: Spectrum rotation angle in degrees
            perp_width: Perpendicular sampling width
            spectrum_y_center: Y coordinate of spectrum center (for full view indicator)
        """
        self._spectrum_y_center = spectrum_y_center
        width = self.config.display.window_width
        graph_height = self.config.display.graph_height
        data_width = len(data.intensity)
        self._last_data_width = data_width
        self._last_peaks = list(data.peaks)

        # Update selected bar status and validate index
        sel = self.state.selected_spectrum_index
        if sel is not None and (sel < 0 or sel >= data_width):
            self.state.selected_spectrum_index = None
            sel = None
        if sel is not None:
            wl = data.wavelengths[sel]
            int_val = data.intensity[sel]
            self.set_status("Sel", f"{wl:.0f}nm {int_val:.2f}")
        else:
            self.set_status("Sel", "-")

        if len(self.state.marker_lines) >= 2 and data_width > 0:
            i1, i2 = sorted(self.state.marker_lines[:2])
            i1 = max(0, min(i1, data_width - 1))
            i2 = max(0, min(i2, data_width - 1))
            wl1 = data.wavelengths[i1]
            wl2 = data.wavelengths[i2]
            d_wl = abs(wl2 - wl1)
            d_i = abs(float(data.intensity[i2]) - float(data.intensity[i1]))
            self.set_status("Δ", f"Δλ={d_wl:.1f}nm ΔI={d_i:.2f}")
        else:
            self.set_status("Δ", "-")

        self._viewport.init_if_needed(data_width)
        self._viewport.clamp_x(data_width)

        if self._graph_override is not None:
            src = self._graph_override
            if src.shape[0] != graph_height or src.shape[1] != width:
                graph = cv2.resize(src, (width, graph_height))
            else:
                graph = src.copy()
        else:
            graph = np.zeros([graph_height, width, 3], dtype=np.uint8)
            graph.fill(255)

            graticule = None
            if self._mode_instance is not None and hasattr(self._mode_instance, "get_graticule"):
                graticule = self._mode_instance.get_graticule(data)
            if graticule is None:
                graticule = self.calibration.graticule
            self._graticule.render_lines_on_graph(graph, graticule, self._viewport)
            self._graticule.render_horizontal_lines(graph)

            self._render_reference_spectrum(graph, data)
            self._spectrogram.filled = self.state.spectrum_bars_visible
            self._spectrogram.selected_index = self.state.selected_spectrum_index
            self._spectrogram.render(graph, data, self._viewport)
            self._render_mode_overlay(graph)
            self._render_sensitivity_overlay(graph)
            if self.state.peaks_visible:
                self._peaks_renderer.show_width = self.state.peak_delta_visible
                self._peaks_renderer.render(graph, data, self._viewport)

            # Graticule labels drawn last; need peak screen-x positions for spacing
            peak_screen_x = [
                self._viewport.data_x_to_screen(float(p.index), width)
                for p in data.peaks
                if self._viewport.x_start <= p.index <= self._viewport.x_end
                and self._viewport.y_min <= p.intensity <= self._viewport.y_max
            ]
            spectrum_screen_y = _spectrum_screen_y(data, self._viewport, width, graph_height)
            self._graticule.render_labels_on_graph(
                graph,
                graticule,
                self._viewport,
                peak_screen_x=peak_screen_x,
                spectrum_screen_y=spectrum_screen_y,
            )
            self._render_cursor(graph, data)
            if self.mode == "measurement" and self.state.marker_lines:
                self._markers_renderer.show_width = self.state.peak_delta_visible
                self._markers_renderer.render(
                    graph,
                    self.state.marker_lines,
                    data.wavelengths,
                    self._viewport,
                    y_offset=0,
                    dragging_idx=self._dragging_marker_idx,
                    intensity=data.intensity,
                )

        if self._graph_info:
            self._render_graph_info(graph)

        message_height = self.config.display.message_height
        preview_height = self.config.display.preview_height

        # Always use control bar - no legacy banner
        self._update_control_bar_status(
            camera_gain,
            savgol_poly,
            peak_min_dist,
            peak_threshold,
            extraction_method,
            rotation_angle,
            perp_width,
        )
        messages = self._control_bar.render()

        # Handle preview mode
        if self._preview_override is not None:
            # Mode has supplied a custom strip (e.g. solid color, spectrum bar)
            src = self._preview_override
            if src.shape[1] != width or src.shape[0] != preview_height:
                cropped = cv2.resize(src, (width, preview_height))
            else:
                cropped = src.copy()
        else:
            cropped = data.cropped_frame
            # Scale camera crop to fit preview area, then apply horizontal viewport zoom
            if cropped is not None:
                if cropped.shape[1] != width or cropped.shape[0] != preview_height:
                    cropped = _scale_to_fit(cropped, width, preview_height)
                cropped = self._apply_viewport_to_preview(cropped, width)
            else:
                cropped = np.zeros([preview_height, width, 3], dtype=np.uint8)

        # Defensive: resize messages if control bar width differs (e.g. before sync took effect)
        if messages.shape[1] != width:
            messages = cv2.resize(messages, (width, messages.shape[0]))

        # Get status segments for overlay (supports colored "Calibrate" in non-calibration modes)
        status_segments = self._control_bar.get_status_segments()

        match self._preview_mode:
            case "none":
                # No preview - just control bar with spectrum graph
                # Draw status at bottom of graph
                self._draw_status_bar(graph, status_segments)
                spectrum_vertical = np.vstack((messages, graph))
            case "window":
                # Windowed preview (cropped bands) above spectrum graph
                self._draw_status_overlay(cropped, status_segments)
                spectrum_vertical = np.vstack((messages, cropped, graph))
            case "spectrum":
                # Wavelength-colored spectrum bar replacing the camera preview strip
                spec_bar = np.zeros((preview_height, width, 3), dtype=np.uint8)
                self._spectrogram.render_bar(spec_bar, data, self._viewport)
                self._draw_status_overlay(spec_bar, status_segments)
                spectrum_vertical = np.vstack((messages, spec_bar, graph))
            case "full":
                # Full camera view REPLACES the spectrum graph entirely
                # Show UNROTATED frame with a ROTATED crop box
                raw_frame = data.raw_frame
                if raw_frame is not None:
                    # Convert raw frame to displayable BGR
                    if raw_frame.ndim == 2:
                        # Monochrome - scale to 8-bit for display only (preserve 10/16-bit in data)
                        if raw_frame.dtype == np.uint16:
                            display = scale_to_uint8(raw_frame)
                        else:
                            display = raw_frame.astype(np.uint8)
                        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
                    else:
                        display = raw_frame.copy()

                    original_height = display.shape[0]
                    original_width = display.shape[1]
                    camera_area_height = preview_height + graph_height

                    # Scale to fit maintaining aspect ratio, then draw crop box on scaled image
                    scale = min(width / original_width, camera_area_height / original_height)
                    out_w = int(original_width * scale)
                    out_h = int(original_height * scale)
                    scaled = cv2.resize(
                        display, (out_w, out_h), interpolation=cv2.INTER_AREA
                    )
                    self._draw_rotated_crop_box(
                        scaled,
                        rotation_angle,
                        perp_width,
                        self._spectrum_y_center,
                        original_width,
                        original_height,
                    )
                    display = _scale_to_fit(scaled, width, camera_area_height)
                    self._draw_status_bar(display, status_segments)

                    spectrum_vertical = np.vstack((messages, display))
                else:
                    # No raw frame available, fall back to windowed
                    self._draw_status_overlay(cropped, status_segments)
                    spectrum_vertical = np.vstack((messages, cropped, graph))
            case _:
                # Default to windowed - no lines, just clean preview
                self._draw_status_overlay(cropped, status_segments)
                spectrum_vertical = np.vstack((messages, cropped, graph))

        # Non-blocking autolevel overlay (replaces content below control bar)
        if self._autolevel_overlay is not None:
            content_height = preview_height + graph_height
            img = self._autolevel_overlay
            if img.shape[1] != width or img.shape[0] != content_height:
                img = _scale_to_fit(img, width, content_height)
            autolevel_segments = status_segments + [
                ("  [Click or 'a' to close]", (0, 255, 255))
            ]
            self._draw_status_bar(img, autolevel_segments)
            spectrum_vertical = np.vstack((messages, img))

        # Pixel-perfect: canvas built at (window_width, stack_height), no scaling
        waterfall_mode = self.mode == "waterfall"

        if self._waterfall is not None and (
            self.config.display.waterfall_enabled or waterfall_mode
        ):
            self._waterfall.update(data)
            waterfall_img = self._waterfall.render()

            # Defensive: ensure waterfall matches width for vstack
            if waterfall_img.shape[1] != width:
                waterfall_img = cv2.resize(waterfall_img, (width, waterfall_img.shape[0]))

            waterfall_img = self._apply_viewport_to_waterfall(waterfall_img, width)

            waterfall_vertical = self._build_waterfall_frame(
                messages, graph, cropped, waterfall_img,
                preview_height, message_height, width,
            )
            graticule_y_offset = waterfall_vertical.shape[0] - waterfall_img.shape[0]
            self._graticule.render_on_waterfall(
                waterfall_vertical,
                self.calibration.graticule,
                y_offset=graticule_y_offset + 2,
                viewport=self._viewport,
            )
            if waterfall_mode:
                if self.state.marker_lines:
                    self._markers_renderer.show_width = self.state.peak_delta_visible
                    self._markers_renderer.render(
                        waterfall_vertical,
                        self.state.marker_lines,
                        data.wavelengths,
                        self._viewport,
                        y_offset=graticule_y_offset,
                        dragging_idx=self._dragging_marker_idx,
                        intensity=data.intensity,
                    )
                self._render_overlays(waterfall_vertical)
                cv2.imshow(self.config.waterfall_title, waterfall_vertical)
            else:
                self._render_overlays(spectrum_vertical)
                cv2.imshow(self.config.spectrograph_title, spectrum_vertical)
                cv2.imshow(self.config.waterfall_title, waterfall_vertical)
        else:
            self._render_overlays(spectrum_vertical)
            cv2.imshow(self.config.spectrograph_title, spectrum_vertical)

    @property
    def preview_mode(self) -> str:
        """Current preview mode string (set by cycle_preview_mode or directly)."""
        return self._preview_mode

    @preview_mode.setter
    def preview_mode(self, value: str) -> None:
        self._preview_mode = value

    def set_preview_override(self, img: np.ndarray | None) -> None:
        """Replace the camera-crop preview strip with a custom image.

        Pass None to restore the live camera crop.
        """
        self._preview_override = img

    def set_graph_click_callback(
        self, callback: Callable[[int, int], None] | None
    ) -> None:
        """Register a callback for left-clicks in the graph area.

        Signature: fn(x, rel_y) where rel_y is relative to the graph area top.
        Pass None to deregister.
        """
        self._on_graph_click = callback

    def set_graph_override(self, img: np.ndarray | None) -> None:
        """Set a full-graph replacement image (e.g. chromaticity diagram).

        When set, spectrum rendering is skipped and this image is shown instead.
        Pass None to restore normal spectrum rendering.
        """
        self._graph_override = img

    def set_graph_info(self, lines: list[str] | None) -> None:
        """Set prominent info text lines rendered in the graph area.

        Shown as large overlay text (e.g. XYZ/LAB colorimetry values).
        Pass None or [] to clear.
        """
        self._graph_info = list(lines) if lines else []

    def _render_graph_info(self, graph: np.ndarray) -> None:
        """Render prominent info text in the top-left of the graph."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        y = 22
        for line in self._graph_info:
            cv2.putText(graph, line, (8, y + 1), font, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(graph, line, (8, y), font, 0.55, (0, 220, 220), 1, cv2.LINE_AA)
            y += 24

    def set_mode_overlay(
        self,
        overlay: tuple[np.ndarray, tuple[int, int, int]] | None,
    ) -> None:
        """Set mode-specific overlay to render on graph.

        Args:
            overlay: Tuple of (intensity array, BGR color) or None to clear
        """
        self._mode_overlay = overlay

    def set_sensitivity_overlay(
        self,
        overlay: tuple[np.ndarray, tuple[int, int, int]] | None,
    ) -> None:
        """Set CMOS sensitivity curve overlay to render on graph.

        Args:
            overlay: Tuple of (intensity array, BGR color) or None to clear
        """
        self._sensitivity_overlay = overlay

    def _render_mode_overlay(self, graph: np.ndarray) -> None:
        """Render mode-specific overlay (e.g., reference spectrum for calibration)."""
        if self._mode_overlay is None:
            return
        intensity, color = self._mode_overlay
        height = graph.shape[0]
        eff = max(1, height - 1)
        intensity_01 = np.asarray(intensity, dtype=np.float32) / eff
        render_polyline_overlay(
            graph, intensity_01, color, thickness=1, viewport=self._viewport
        )

    def _render_sensitivity_overlay(self, graph: np.ndarray) -> None:
        """Render CMOS sensitivity curve overlay.
        Resamples overlay to graph width so it always matches the spectrum axis.
        """
        if self._sensitivity_overlay is None:
            return
        intensity, color = self._sensitivity_overlay
        width = graph.shape[1]
        height = graph.shape[0]
        eff = max(1, height - 1)
        intensity_01 = np.asarray(intensity, dtype=np.float32) / eff
        render_polyline_overlay(
            graph,
            intensity_01,
            color,
            resample_to_width=width,
            thickness=1,
            viewport=self._viewport,
        )

    def _render_reference_spectrum(self, graph: np.ndarray, data: SpectrumData) -> None:
        """Render the reference spectrum overlay line. Both measured and reference are 0-1."""
        ref_intensity = self.state.reference_spectrum
        if ref_intensity is None:
            return

        render_polyline_overlay(
            graph,
            ref_intensity,
            (180, 180, 180),
            thickness=1,
            viewport=self._viewport,
        )

    def _render_cursor(self, graph: np.ndarray, data: SpectrumData) -> None:
        """Render measurement cursor if active."""
        cursor = self.state.cursor

        if cursor.measure_mode:
            x = cursor.x
            y = cursor.y

            cv2.line(graph, (x, y - 140), (x, y - 180), (0, 0, 0), 1)
            cv2.line(graph, (x - 20, y - 160), (x + 20, y - 160), (0, 0, 0), 1)

            if 0 <= x < len(data.wavelengths):
                wavelength = round(data.wavelengths[x], 2)
                cv2.putText(
                    graph,
                    f"{wavelength}nm",
                    (x + 5, y - 165),
                    self._font,
                    0.4,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        elif cursor.pixel_mode:
            x = cursor.x
            y = cursor.y

            cv2.line(graph, (x, y - 140), (x, y - 180), (0, 0, 0), 1)
            cv2.line(graph, (x - 20, y - 160), (x + 20, y - 160), (0, 0, 0), 1)
            cv2.putText(
                graph,
                f"{x}px",
                (x + 5, y - 165),
                self._font,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    def _draw_rotated_crop_box(
        self,
        frame: np.ndarray,
        rotation_angle: float,
        perp_width: int,
        spectrum_y_center: int,
        original_width: int,
        original_height: int,
    ) -> None:
        """Draw crop box on UNROTATED full camera view using spectrum_transform.

        Uses spectrum_transform to transform bbox from corrected (rotated) coords
        to original image coords. Uses cv2.getRotationMatrix2D (no manual cos/sin).
        """
        display_height, display_width = frame.shape[:2]
        if original_width <= 0 or original_height <= 0:
            return

        params = params_from_saved(rotation_angle, spectrum_y_center, original_height)
        crop_height = self.config.display.preview_height

        corners_rotated = get_crop_corners_rotated(original_width, spectrum_y_center, crop_height)
        corners_original = transform_bbox_rotated_to_original(
            corners_rotated, params, original_width, original_height
        )

        scale_x = display_width / original_width
        scale_y = display_height / original_height
        corners_display = np.array(
            [[int(x * scale_x), int(y * scale_y)] for x, y in corners_original], dtype=np.int32
        )

        cv2.polylines(frame, [corners_display], True, (0, 255, 255), 2)

        center_line_rot = np.array(
            [[0, spectrum_y_center], [original_width, spectrum_y_center]],
            dtype=np.float32,
        )
        center_line_orig = transform_bbox_rotated_to_original(
            center_line_rot, params, original_width, original_height
        )
        p0 = (int(center_line_orig[0, 0] * scale_x), int(center_line_orig[0, 1] * scale_y))
        p1 = (int(center_line_orig[1, 0] * scale_x), int(center_line_orig[1, 1] * scale_y))
        cv2.line(frame, p0, p1, (255, 255, 0), 1)

    def _draw_status_segments(
        self,
        image: np.ndarray,
        segments: list[tuple[str, tuple[int, int, int]]],
        x: int,
        y: int,
        shadow: bool = True,
    ) -> None:
        """Draw status segments with per-segment colors."""
        font = self._font
        font_scale = self.config.display.font_scale
        thickness = 1
        for text, color in segments:
            if not text:
                continue
            if shadow:
                cv2.putText(
                    image, text, (x + 1, y + 1), font, font_scale,
                    (0, 0, 0), thickness, cv2.LINE_AA
                )
            cv2.putText(
                image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA
            )
            (w, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x += w

    def _draw_status_overlay(
        self,
        image: np.ndarray,
        segments: list[tuple[str, tuple[int, int, int]]],
    ) -> None:
        """Draw status segments overlaid on the preview strip (semi-transparent background)."""
        if not segments:
            return

        full_text = "".join(s[0] for s in segments)
        font = self._font
        font_scale = self.config.display.font_scale
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            full_text, font, font_scale, thickness
        )

        padding = 3
        x = padding
        y = image.shape[0] - padding

        bg_x1 = x - padding
        bg_y1 = y - text_height - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + baseline + padding

        overlay = image.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        self._draw_status_segments(image, segments, x, y, shadow=False)

    def _draw_status_bar(
        self,
        image: np.ndarray,
        segments: list[tuple[str, tuple[int, int, int]]],
    ) -> None:
        """Draw status segments at the bottom of the image."""
        if not segments:
            return

        x = 5
        y = image.shape[0] - 5
        self._draw_status_segments(image, segments, x, y, shadow=True)

    def get_waterfall_image(self) -> np.ndarray | None:
        """Get the current waterfall display image."""
        if self._waterfall is not None:
            return self._waterfall.render()
        return None

    def register_button_callback(
        self,
        action_name: str,
        callback: Callable[[], None],
    ) -> bool:
        """Register a callback for a control bar button.

        Args:
            action_name: Name of the button action
            callback: Function to call when button is clicked

        Returns:
            True if button was found and callback registered
        """
        return self._control_bar.register_callback(action_name, callback)

    def set_button_active(self, action_name: str, active: bool) -> bool:
        """Set the active state of a toggle button.

        Args:
            action_name: Name of the button action
            active: Whether button should appear active

        Returns:
            True if button was found
        """
        return self._control_bar.set_button_active(action_name, active)

    def handle_key(self, key_char: str) -> bool:
        """Invoke button callback for shortcut. Returns True if handled."""
        return self._control_bar.handle_key(key_char)

    def set_status(self, key: str, value: str) -> None:
        """Set a status value displayed in the control bar.

        Prefer this over accessing _control_bar directly.

        Args:
            key: Status key (e.g., "Ref", "Pts", "Status", "Dark")
            value: Value to display
        """
        self._control_bar.set_status(key, value)

    def _update_control_bar_status(
        self,
        camera_gain: float,
        savgol_poly: int,
        peak_min_dist: int,
        peak_threshold: int,
        extraction_method: str,
        rotation_angle: float,
        perp_width: int,
    ) -> None:
        """Update status values displayed in the control bar."""
        cal_result = self.calibration.result
        cal_label = "Calibrate" if self.mode != "calibration" else "Cal"
        self._control_bar.set_calibrate_red(self.mode != "calibration")
        self.set_status(cal_label, cal_result.status_message[:12])
        self.set_status("Gain", f"{camera_gain:.0f}")
        self.set_status("SG", str(savgol_poly))
        self.set_status("Ext", extraction_method[:3].upper())
        self.set_status("Ang", f"{rotation_angle:.1f}")
        self.set_status("Wid", str(perp_width))

        if self.state.reference_name != "None":
            self.set_status("Ref", self.state.reference_name[:8])

    @property
    def control_bar(self) -> ControlBar:
        """Get the control bar instance."""
        return self._control_bar

    def register_slider_callbacks(
        self,
        gain_cb: Callable[[float], None] | None = None,
        exposure_cb: Callable[[float], None] | None = None,
        led_intensity_cb: Callable[[float], None] | None = None,
    ) -> None:
        """Register callbacks for slider value changes.

        Args:
            gain_cb: Called when gain slider value changes
            exposure_cb: Called when exposure slider value changes
            led_intensity_cb: Called when LED intensity slider value changes (Measurement, Color Science)
        """
        if gain_cb is not None:
            self._slider_panel.gain_slider.on_change = gain_cb
        if exposure_cb is not None:
            self._slider_panel.exposure_slider.on_change = exposure_cb
        if led_intensity_cb is not None:
            self._slider_panel.led_intensity_slider.on_change = led_intensity_cb

    def toggle_gain_slider(self) -> bool:
        """Toggle gain slider visibility. Returns new visibility state."""
        s = self._slider_panel.gain_slider
        s.visible = not s.visible
        return s.visible

    def toggle_exposure_slider(self) -> bool:
        """Toggle exposure slider visibility. Returns new visibility state."""
        s = self._slider_panel.exposure_slider
        s.visible = not s.visible
        return s.visible

    def toggle_zoom_x_slider(self) -> bool:
        """Toggle horizontal (X) zoom slider visibility. Returns new visibility state."""
        self._zoom_horizontal.visible = not self._zoom_horizontal.visible
        return self._zoom_horizontal.visible

    def toggle_zoom_y_slider(self) -> bool:
        """Toggle vertical (Y) zoom slider visibility. Returns new visibility state."""
        self._zoom_vertical.visible = not self._zoom_vertical.visible
        return self._zoom_vertical.visible

    def toggle_led_slider(self) -> bool:
        """Toggle LED intensity slider visibility. Returns new visibility state."""
        s = self._slider_panel.led_intensity_slider
        s.visible = not s.visible
        return s.visible

    def is_gain_slider_visible(self) -> bool:
        """Return whether the gain slider is visible."""
        return self._slider_panel.gain_slider.visible

    def is_exposure_slider_visible(self) -> bool:
        """Return whether the exposure slider is visible."""
        return self._slider_panel.exposure_slider.visible

    def is_zoom_x_slider_visible(self) -> bool:
        """Return whether the horizontal zoom slider is visible."""
        return self._zoom_horizontal.visible

    def is_zoom_y_slider_visible(self) -> bool:
        """Return whether the vertical zoom slider is visible."""
        return self._zoom_vertical.visible

    def show_gain_slider(self, visible: bool) -> None:
        """Show or hide the gain slider."""
        self._slider_panel.gain_slider.visible = visible

    def show_exposure_slider(self, visible: bool) -> None:
        """Show or hide the exposure slider."""
        self._slider_panel.exposure_slider.visible = visible

    def show_led_slider(self, visible: bool) -> None:
        """Show or hide the LED intensity slider (Measurement, Color Science)."""
        self._slider_panel.led_intensity_slider.visible = visible

    def toggle_spectrum_bars(self) -> bool:
        """Toggle vertical colored bars on spectrum. Returns new visibility state."""
        self.state.spectrum_bars_visible = not self.state.spectrum_bars_visible
        if not self.state.spectrum_bars_visible:
            self.state.selected_spectrum_index = None
        return self.state.spectrum_bars_visible

    def show_spectrum_bars(self, visible: bool) -> None:
        """Show or hide vertical colored bars on spectrum."""
        self.state.spectrum_bars_visible = visible
        if not visible:
            self.state.selected_spectrum_index = None

    def is_spectrum_bars_visible(self) -> bool:
        """Return whether spectrum bars are visible."""
        return self.state.spectrum_bars_visible

    def toggle_peaks_visible(self) -> bool:
        """Toggle peak labels and vertical lines visibility. Returns new state."""
        self.state.peaks_visible = not self.state.peaks_visible
        return self.state.peaks_visible

    def is_peaks_visible(self) -> bool:
        """Return whether peaks are visible."""
        return self.state.peaks_visible

    def set_gain_value(self, value: float) -> None:
        """Set the gain slider value."""
        self._slider_panel.gain_slider.value = value

    def set_exposure_value(self, value: float) -> None:
        """Set the exposure slider value."""
        self._slider_panel.exposure_slider.value = value

    def set_led_intensity_value(self, value: float) -> None:
        """Set the LED intensity slider value (0–100% PWM duty cycle)."""
        self._slider_panel.led_intensity_slider.value = value

    def get_gain_value(self) -> float:
        """Get current gain slider value."""
        return self._slider_panel.gain_slider.value

    def get_exposure_value(self) -> float:
        """Get current exposure slider value (microseconds)."""
        return self._slider_panel.exposure_slider.value

    def get_led_intensity_value(self) -> float:
        """Get LED intensity slider value (0–100% PWM)."""
        return self._slider_panel.led_intensity_slider.value

    def _apply_viewport_to_preview(self, img: np.ndarray, width: int) -> np.ndarray:
        """Crop the (already display-scaled) preview strip to the current horizontal viewport.

        The strip has been scaled to `width` pixels wide, covering the full data
        range [0, data_width-1]. We crop to the viewport x range and resize back
        so the preview stays aligned with the spectrum graph below it.
        """
        data_width = max(1, self._last_data_width)
        col_start = int(self._viewport.x_start / data_width * width)
        col_end = int(self._viewport.x_end / data_width * width) + 1
        col_start = max(0, col_start)
        col_end = min(width, col_end)
        if col_end - col_start < 2:
            return img
        region = img[:, col_start:col_end, :]
        return cv2.resize(region, (width, img.shape[0]), interpolation=cv2.INTER_AREA)

    def _render_overlays(self, frame: np.ndarray) -> None:
        """Render all visible control overlays (sliders) on the final composed frame.

        Sliders store absolute window y-coordinates (including message and preview
        heights). When the frame is shorter than the full window (e.g. "none"
        preview mode omits the preview strip), we shift slider positions by the
        height difference so they stay in the graph/waterfall area.
        """
        expected_h = (
            self.config.display.message_height
            + self.config.display.preview_height
            + self.config.display.graph_height
        )
        dy = frame.shape[0] - expected_h  # 0 normally; -preview_height in "none" mode

        if self._slider_panel.any_visible():
            sliders = self._slider_panel._sliders
            for s in sliders:
                s.y += dy
            self._slider_panel.render(frame)
            for s in sliders:
                s.y -= dy

        if self._zoom_vertical.visible:
            self._zoom_vertical.y += dy
            self._zoom_vertical.render(frame)
            self._zoom_vertical.y -= dy

        if self._zoom_horizontal.visible:
            self._zoom_horizontal.y += dy
            self._zoom_horizontal.render(frame)
            self._zoom_horizontal.y -= dy

    def _apply_viewport_to_waterfall(self, img: np.ndarray, width: int) -> np.ndarray:
        """Crop and stretch waterfall image to match the current horizontal viewport."""
        x_start = int(self._viewport.x_start)
        x_end = int(self._viewport.x_end)
        if x_start <= 0 and x_end >= img.shape[1] - 1:
            return img
        x_start = max(0, x_start)
        x_end = min(img.shape[1] - 1, x_end)
        if x_end <= x_start:
            return img
        cropped = img[:, x_start : x_end + 1, :]
        return cv2.resize(cropped, (width, img.shape[0]), interpolation=cv2.INTER_AREA)

    def _build_waterfall_frame(
        self,
        messages: np.ndarray,
        graph: np.ndarray,
        cropped: np.ndarray | None,
        waterfall_img: np.ndarray,
        preview_height: int,
        message_height: int,
        width: int,
    ) -> np.ndarray:
        """Build the waterfall window frame respecting current preview mode.

        "window" → camera crop strip above waterfall (default).
        "spectrum" → spectrum graph scaled to preview_height above waterfall.
        "none" → waterfall only, no preview strip.
        """
        match self._preview_mode:
            case "spectrum":
                # Squash height only — preserve full width so x-axis aligns with waterfall
                preview = cv2.resize(graph, (width, preview_height), interpolation=cv2.INTER_AREA)
                return np.vstack((messages, preview, waterfall_img))
            case "none":
                return np.vstack((messages, waterfall_img))
            case _:
                # "window" and anything else: camera crop
                strip = (
                    cropped
                    if cropped is not None
                    else np.zeros((preview_height, width, 3), dtype=np.uint8)
                )
                return np.vstack((messages, strip, waterfall_img))

    def destroy(self) -> None:
        """Destroy all OpenCV windows."""
        cv2.destroyAllWindows()
        self._windows_created = False
