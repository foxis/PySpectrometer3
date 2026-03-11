"""Main display manager for spectrum visualization."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..config import Config
from ..core.calibration import Calibration
from ..core.spectrum import SpectrumData
from ..gui.control_bar import ControlBar, ControlBarConfig
from ..gui.sliders import HorizontalSlider, SliderPanel, SliderStyle, VerticalSlider
from ..modes.base import BaseMode
from ..processing.spectrum_transform import (
    get_crop_corners_rotated,
    params_from_saved,
    transform_bbox_rotated_to_original,
)
from ..utils.color import rgb_to_bgr, wavelength_to_rgb
from ..utils.display import scale_to_uint8
from ..utils.graph_scale import scale_intensity_to_graph
from .graticule import GraticuleRenderer
from .overlay_utils import render_polyline_overlay
from .viewport import Viewport
from .waterfall import WaterfallDisplay

# Hot zone width for zoom slider toggle (click to show/hide)
ZOOM_HOTZONE = 12


@dataclass
class CursorState:
    """State for cursor display."""

    x: int = 0
    y: int = 0
    measure_mode: bool = False
    pixel_mode: bool = False


@dataclass
class DisplayState:
    """Mutable display state."""

    hold_peaks: bool = False
    held_intensity: np.ndarray | None = None
    cursor: CursorState = None
    save_message: str = "No data saved"
    reference_spectrum: np.ndarray | None = None
    reference_name: str = "None"

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

        self._graticule = GraticuleRenderer()
        self._waterfall: WaterfallDisplay | None = None

        if config.display.waterfall_enabled:
            self._waterfall = WaterfallDisplay(
                width=config.camera.frame_width,
                height=config.display.graph_height,
                contrast=config.waterfall.contrast,
                brightness=config.waterfall.brightness,
            )

        self._mode_instance = mode_instance
        buttons = mode_instance.get_buttons() if mode_instance is not None else None
        self._control_bar = ControlBar(
            width=config.camera.frame_width,
            config=ControlBarConfig(
                height=config.display.message_height,
                font_scale=config.display.font_scale,
            ),
            mode=mode,
            buttons=buttons,
            mode_instance=mode_instance,
        )

        # Slider panel for gain/exposure/LED (positioned on right side of graph area)
        # Need room for three sliders: gain, exposure, LED intensity (~165 px total)
        slider_x = config.camera.frame_width - 170
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

        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._windows_created = False

        # Preview display mode: "full", "window", "none"
        self._preview_mode = "window"

        # Track mouse button state for slider dragging
        self._mouse_down = False

        # Zoom/pan viewport
        self._viewport = Viewport()

        # Zoom sliders (visible when user clicks on left/bottom edge)
        style = SliderStyle()
        self._zoom_vertical = VerticalSlider(
            x=2,
            y=config.display.message_height + config.display.preview_height + 20,
            height=config.display.graph_height - 50,
            min_val=1.0,
            max_val=20.0,
            value=1.0,
            label="V",
            style=style,
        )
        self._zoom_horizontal = HorizontalSlider(
            x=20,
            y=config.display.message_height
            + config.display.preview_height
            + config.display.graph_height
            - 25,
            width=config.camera.frame_width - 200,
            min_val=1.0,
            max_val=20.0,
            value=1.0,
            label="H",
            style=style,
        )

        self._zoom_vertical_prev = 1.0
        self._zoom_horizontal_prev = 1.0

        def _on_zoom_vertical(val: float) -> None:
            if self._zoom_vertical_prev > 0:
                self._viewport.zoom_y(val / self._zoom_vertical_prev)
            self._zoom_vertical_prev = val

        def _on_zoom_horizontal(val: float) -> None:
            if self._zoom_horizontal_prev > 0:
                self._viewport.zoom_x(val / self._zoom_horizontal_prev)
            self._zoom_horizontal_prev = val

        self._zoom_vertical.on_change = _on_zoom_vertical
        self._zoom_horizontal.on_change = _on_zoom_horizontal

        # Pan state
        self._panning = False
        self._pan_last_x = 0
        self._pan_last_y = 0
        self._last_data_width = 0

    def set_frame_dimensions(self, width: int, height: int) -> None:
        """Update dimensions (e.g. after camera reports actual size)."""
        self._control_bar.set_width(width)
        if self._waterfall is not None:
            self._waterfall.set_dimensions(width, self.config.display.graph_height)
        self._slider_panel.x = width - 170

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

        # WINDOW_GUI_NORMAL (0x10 = 16) disables the Qt/GTK toolbar and statusbar
        # WINDOW_GUI_EXPANDED (0x00) shows toolbar (default)
        # We want NORMAL (no toolbar)
        WINDOW_GUI_NORMAL = getattr(cv2, "WINDOW_GUI_NORMAL", 0x00000010)

        if self.config.display.waterfall_enabled:
            cv2.namedWindow(title2, cv2.WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL)
            cv2.moveWindow(title2, 200, 200)

        if self.config.display.fullscreen:
            # True fullscreen - fills entire screen, no toolbar
            cv2.namedWindow(title1, cv2.WINDOW_NORMAL | WINDOW_GUI_NORMAL)
            cv2.setWindowProperty(
                title1,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        else:
            # Windowed mode - auto-size, no toolbar/statusbar
            cv2.namedWindow(title1, cv2.WINDOW_AUTOSIZE | WINDOW_GUI_NORMAL)
            cv2.moveWindow(title1, 0, 0)

        cv2.setMouseCallback(title1, self._handle_mouse)

        self._windows_created = True

    def cycle_preview_mode(self) -> str:
        """Cycle through preview display modes.

        Returns:
            Current mode name after cycling
        """
        modes = ["window", "full", "none"]
        current_idx = modes.index(self._preview_mode) if self._preview_mode in modes else 0
        next_idx = (current_idx + 1) % len(modes)
        self._preview_mode = modes[next_idx]
        print(f"[PREVIEW] Mode: {self._preview_mode}")
        return self._preview_mode

    def set_autolevel_overlay(self, img: np.ndarray | None) -> None:
        """Set autolevel overlay image (None to clear). Non-blocking."""
        self._autolevel_overlay = img

    def set_autolevel_click_dismiss(self, callback: Callable[[], None] | None) -> None:
        """Set callback for click-to-dismiss on autolevel overlay."""
        self._on_autolevel_click = callback

    def _handle_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events on the spectrum window."""
        control_bar_height = self.config.display.message_height
        preview_height = self.config.display.preview_height
        graph_height = self.config.display.graph_height
        width = self.config.camera.frame_width
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
                if self._panning:
                    dx = self._viewport.screen_x_to_data(
                        self._pan_last_x, width
                    ) - self._viewport.screen_x_to_data(x, width)
                    dy = self._viewport.screen_y_to_intensity(
                        self._pan_last_y, graph_height
                    ) - self._viewport.screen_y_to_intensity(
                        graph_rel_y, graph_height
                    )
                    self._viewport.pan_x(dx)
                    self._viewport.pan_y(dy)
                    self._pan_last_x = x
                    self._pan_last_y = graph_rel_y

        elif event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down = True

            if y < control_bar_height:
                self._control_bar.handle_click(x, y)
            elif self._on_autolevel_click is not None:
                self._on_autolevel_click()
            elif self._slider_panel.handle_mouse_down(x, y):
                pass
            elif self._zoom_vertical.visible and self._zoom_vertical.handle_mouse_down(x, y):
                pass
            elif self._zoom_horizontal.visible and self._zoom_horizontal.handle_mouse_down(x, y):
                pass
            elif in_graph:
                if x < ZOOM_HOTZONE:
                    self._zoom_vertical.visible = not self._zoom_vertical.visible
                elif graph_rel_y >= graph_height - ZOOM_HOTZONE:
                    self._zoom_horizontal.visible = not self._zoom_horizontal.visible
                elif self._is_zoomed():
                    self._panning = True
                    self._pan_last_x = x
                    self._pan_last_y = graph_rel_y

        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_down = False
            self._slider_panel.handle_mouse_up()
            if self._zoom_vertical.handle_mouse_up():
                pass
            if self._zoom_horizontal.handle_mouse_up():
                pass
            self._panning = False

    def _is_zoomed(self) -> bool:
        """True if viewport is zoomed (not full view)."""
        if self._last_data_width <= 0:
            return False
        x_span = self._viewport.x_end - self._viewport.x_start
        y_span = self._viewport.y_max - self._viewport.y_min
        return x_span < self._last_data_width - 1 or y_span < 0.99

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
        width = self.config.camera.frame_width
        graph_height = self.config.display.graph_height
        data_width = len(data.intensity)
        self._last_data_width = data_width

        self._viewport.init_if_needed(data_width)
        self._viewport.clamp_x(data_width)

        graph = np.zeros([graph_height, width, 3], dtype=np.uint8)
        graph.fill(255)

        graticule = None
        if self._mode_instance is not None and hasattr(self._mode_instance, "get_graticule"):
            graticule = self._mode_instance.get_graticule(data)
        if graticule is None:
            graticule = self.calibration.graticule
        self._graticule.render_on_graph(graph, graticule, self._viewport)
        self._graticule.render_horizontal_lines(graph)

        self._render_reference_spectrum(graph, data)
        self._render_spectrum(graph, data)
        self._render_mode_overlay(graph)
        self._render_sensitivity_overlay(graph)
        self._render_peaks(graph, data)
        self._render_cursor(graph, data)

        if self._slider_panel.any_visible():
            self._render_sliders_on_graph(graph)
        if self._zoom_vertical.visible or self._zoom_horizontal.visible:
            self._render_zoom_sliders_on_graph(graph)

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
        cropped = data.cropped_frame

        # Ensure all arrays match (width, height) for vstack - camera may report different size than config
        if cropped is not None:
            if cropped.shape[1] != width or cropped.shape[0] != preview_height:
                cropped = cv2.resize(cropped, (width, preview_height))
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
                # Windowed preview (cropped) above spectrum graph
                # NO lines drawn - just show the clean cropped spectrum image
                # Draw status on preview strip
                self._draw_status_overlay(cropped, status_segments)
                spectrum_vertical = np.vstack((messages, cropped, graph))
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

                    # Scale to fill the area below control bar (preview + graph height)
                    camera_area_height = preview_height + graph_height
                    display = cv2.resize(display, (width, camera_area_height))

                    # Draw rotated crop box on the UNROTATED display
                    # The crop box shows where extraction happens in the rotated space
                    self._draw_rotated_crop_box(
                        display,
                        rotation_angle,
                        perp_width,
                        self._spectrum_y_center,
                        original_width,
                        original_height,
                    )
                    # Draw status at bottom of camera view
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
                img = cv2.resize(img, (width, content_height))
            autolevel_segments = status_segments + [
                ("  [Click or 'a' to close]", (0, 255, 255))
            ]
            self._draw_status_bar(img, autolevel_segments)
            spectrum_vertical = np.vstack((messages, img))

        # Show the composed image
        cv2.imshow(self.config.spectrograph_title, spectrum_vertical)

        if self._waterfall is not None and self.config.display.waterfall_enabled:
            self._waterfall.update(data)
            waterfall_img = self._waterfall.render()

            # Use cropped preview for waterfall if available
            if cropped is None:
                cropped = np.zeros([preview_height, width, 3], dtype=np.uint8)
            # Defensive: ensure waterfall matches width for vstack
            if waterfall_img.shape[1] != width:
                waterfall_img = cv2.resize(waterfall_img, (width, waterfall_img.shape[0]))
            waterfall_vertical = np.vstack((messages, cropped, waterfall_img))

            self._graticule.render_on_waterfall(
                waterfall_vertical,
                self.calibration.graticule,
                y_offset=message_height + preview_height + 2,
            )

            cv2.imshow(self.config.waterfall_title, waterfall_vertical)

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

    def _render_spectrum(self, graph: np.ndarray, data: SpectrumData) -> None:
        """Render the spectrum intensity curve.

        Intensity is float32 0-1. Scale to fit graph height.
        """
        height = graph.shape[0]
        width = graph.shape[1]

        for i, intensity in enumerate(data.intensity):
            if i < self._viewport.x_start or i > self._viewport.x_end:
                continue

            rgb = wavelength_to_rgb(round(data.wavelengths[i]))
            bgr = rgb_to_bgr(rgb)

            sx = self._viewport.data_x_to_screen(float(i), width)
            sy = self._viewport.intensity_to_screen_y(intensity, height)

            cv2.line(graph, (sx, height - 1), (sx, sy), bgr, 1)
            cv2.line(
                graph,
                (sx, sy - 1),
                (sx, sy),
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    def _render_peaks(self, graph: np.ndarray, data: SpectrumData) -> None:
        """Render peak labels on the graph. Peak intensity is float 0-1."""
        height = graph.shape[0]
        width = graph.shape[1]
        text_offset = 12

        for peak in data.peaks:
            if peak.index < self._viewport.x_start or peak.index > self._viewport.x_end:
                continue
            if peak.intensity < self._viewport.y_min or peak.intensity > self._viewport.y_max:
                continue

            sx = self._viewport.data_x_to_screen(float(peak.index), width)
            label_height = self._viewport.intensity_to_screen_y(peak.intensity, height)

            cv2.rectangle(
                graph,
                (sx - text_offset - 2, label_height),
                (sx - text_offset + 60, label_height - 15),
                (0, 255, 255),
                -1,
            )
            cv2.rectangle(
                graph,
                (sx - text_offset - 2, label_height),
                (sx - text_offset + 60, label_height - 15),
                (0, 0, 0),
                1,
            )
            cv2.putText(
                graph,
                f"{peak.wavelength}nm",
                (sx - text_offset, label_height - 3),
                self._font,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                graph,
                (sx, label_height),
                (sx, label_height + 10),
                (0, 0, 0),
                1,
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

    def _render_zoom_sliders_on_graph(self, graph: np.ndarray) -> None:
        """Render zoom sliders on graph with graph-relative coordinates."""
        message_height = self.config.display.message_height
        preview_height = self.config.display.preview_height
        offset_y = message_height + preview_height

        if self._zoom_vertical.visible:
            orig_y = self._zoom_vertical.y
            self._zoom_vertical.y = orig_y - offset_y
            self._zoom_vertical.render(graph)
            self._zoom_vertical.y = orig_y

        if self._zoom_horizontal.visible:
            orig_y = self._zoom_horizontal.y
            self._zoom_horizontal.y = graph.shape[0] - 25
            self._zoom_horizontal.render(graph)
            self._zoom_horizontal.y = orig_y

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

    def show_gain_slider(self, visible: bool) -> None:
        """Show or hide the gain slider."""
        self._slider_panel.gain_slider.visible = visible

    def show_exposure_slider(self, visible: bool) -> None:
        """Show or hide the exposure slider."""
        self._slider_panel.exposure_slider.visible = visible

    def show_led_slider(self, visible: bool) -> None:
        """Show or hide the LED intensity slider (Measurement, Color Science)."""
        self._slider_panel.led_intensity_slider.visible = visible

    def set_gain_value(self, value: float) -> None:
        """Set the gain slider value."""
        self._slider_panel.gain_slider.value = value

    def set_exposure_value(self, value: float) -> None:
        """Set the exposure slider value."""
        self._slider_panel.exposure_slider.value = value

    def set_led_intensity_value(self, value: float) -> None:
        """Set the LED intensity slider value (0–100% PWM duty cycle)."""
        self._slider_panel.led_intensity_slider.value = value

    def _render_sliders_on_graph(self, graph: np.ndarray) -> None:
        """Render sliders directly on the graph area."""
        # Sliders are positioned relative to the full window, but we render
        # them on the graph portion. Adjust slider coordinates temporarily.
        panel = self._slider_panel

        # Calculate offset from slider position to graph-relative position
        message_height = self.config.display.message_height
        preview_height = self.config.display.preview_height

        # Sliders render directly on graph, so we adjust their Y positions
        # to be graph-relative (subtract the offset above graph)
        offset_y = message_height + preview_height

        # Temporarily adjust slider positions for graph rendering
        originals = [(s, s.y) for s in panel._sliders]
        for s, _ in originals:
            s.y = s.y - offset_y

        panel.render(graph)

        # Restore original positions for mouse handling
        for s, y_orig in originals:
            s.y = y_orig

    def destroy(self) -> None:
        """Destroy all OpenCV windows."""
        cv2.destroyAllWindows()
        self._windows_created = False
