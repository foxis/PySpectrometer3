"""Main display manager for spectrum visualization."""

from dataclasses import dataclass
from typing import Callable, Optional
import cv2
import numpy as np

from ..config import Config
from ..core.spectrum import SpectrumData
from ..core.calibration import Calibration
from ..utils.color import wavelength_to_rgb, rgb_to_bgr
from ..gui.control_bar import ControlBar, ControlBarConfig
from ..gui.sliders import SliderPanel
from .graticule import GraticuleRenderer
from .waterfall import WaterfallDisplay


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
    held_intensity: Optional[np.ndarray] = None
    cursor: CursorState = None
    click_points: list[tuple[int, int]] = None
    save_message: str = "No data saved"
    reference_spectrum: Optional[np.ndarray] = None
    reference_name: str = "None"
    
    def __post_init__(self):
        if self.cursor is None:
            self.cursor = CursorState()
        if self.click_points is None:
            self.click_points = []


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
    ):
        """Initialize display manager.
        
        Args:
            config: Application configuration
            calibration: Wavelength calibration data
            mode: Operating mode (calibration, measurement, raman, colorscience)
        """
        self.config = config
        self.calibration = calibration
        self.state = DisplayState()
        self.mode = mode
        
        self._graticule = GraticuleRenderer()
        self._waterfall: Optional[WaterfallDisplay] = None
        
        if config.display.waterfall_enabled:
            self._waterfall = WaterfallDisplay(
                width=config.camera.frame_width,
                height=config.display.graph_height,
                contrast=config.waterfall.contrast,
                brightness=config.waterfall.brightness,
            )
        
        self._control_bar = ControlBar(
            width=config.camera.frame_width,
            config=ControlBarConfig(
                height=config.display.message_height,
                font_scale=config.display.font_scale,
            ),
            mode=mode,
        )
        
        # Slider panel for gain/exposure (positioned on right side of graph area)
        slider_x = config.camera.frame_width - 60
        slider_y = config.display.message_height + config.display.preview_height + 10
        slider_height = config.display.graph_height - 30
        self._slider_panel = SliderPanel(
            x=slider_x,
            y=slider_y,
            height=slider_height,
        )
        
        # Mode overlay (intensity array, color)
        self._mode_overlay: Optional[tuple[np.ndarray, tuple[int, int, int]]] = None
        
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._windows_created = False
        
        # Preview display mode: "full", "window", "none"
        self._preview_mode = "window"
        
        # Track mouse button state for slider dragging
        self._mouse_down = False
    
    def setup_windows(self) -> None:
        """Create and configure OpenCV windows."""
        if self._windows_created:
            return
        
        title1 = self.config.spectrograph_title
        title2 = self.config.waterfall_title
        
        if self.config.display.waterfall_enabled:
            cv2.namedWindow(title2, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(title2, 200, 200)
        
        if self.config.display.fullscreen:
            # True fullscreen - fills entire screen
            cv2.namedWindow(title1, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(
                title1,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        else:
            # Windowed mode - auto-size to content, minimal GUI
            cv2.namedWindow(title1, cv2.WINDOW_AUTOSIZE)
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
    
    def set_preview_mode(self, mode: str) -> None:
        """Set preview display mode.
        
        Args:
            mode: One of "window", "full", "none"
        """
        if mode in ("window", "full", "none"):
            self._preview_mode = mode
    
    def _handle_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events on the spectrum window."""
        control_bar_height = self.config.display.message_height
        preview_height = self.config.display.preview_height
        mouse_y_offset = control_bar_height + preview_height
        
        if event == cv2.EVENT_MOUSEMOVE:
            self.state.cursor.x = x
            self.state.cursor.y = y
            
            if y < control_bar_height:
                self._control_bar.handle_mouse_move(x, y)
            
            # Handle slider dragging
            if self._mouse_down:
                self._slider_panel.handle_mouse_move(x, y)
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_down = True
            
            if y < control_bar_height:
                self._control_bar.handle_click(x, y)
            elif self._slider_panel.handle_mouse_down(x, y):
                pass  # Slider handled it
            else:
                mouse_y = y - mouse_y_offset
                self.state.click_points.append((x, mouse_y))
        
        elif event == cv2.EVENT_LBUTTONUP:
            self._mouse_down = False
            self._slider_panel.handle_mouse_up()
    
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
        """
        width = self.config.camera.frame_width
        graph_height = self.config.display.graph_height
        
        graph = np.zeros([graph_height, width, 3], dtype=np.uint8)
        graph.fill(255)
        
        self._graticule.render_on_graph(graph, self.calibration.graticule)
        self._graticule.render_horizontal_lines(graph)
        
        self._render_reference_spectrum(graph, data)
        self._render_mode_overlay(graph)
        self._render_spectrum(graph, data)
        self._render_peaks(graph, data)
        self._render_cursor(graph, data)
        self._render_click_points(graph)
        
        # Render sliders directly on graph (if any are visible)
        if self._slider_panel.any_visible():
            self._render_sliders_on_graph(graph)
        
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
        
        # Ensure cropped frame is exactly preview_height for windowed mode
        if cropped is not None:
            if cropped.shape[0] != preview_height:
                cropped = cv2.resize(cropped, (width, preview_height))
        else:
            cropped = np.zeros([preview_height, width, 3], dtype=np.uint8)
        
        # Get status text for overlay
        status_text = self._control_bar.get_status_text()
        
        match self._preview_mode:
            case "none":
                # No preview - just control bar with spectrum graph
                # Draw status at bottom of graph
                self._draw_status_bar(graph, status_text)
                spectrum_vertical = np.vstack((messages, graph))
            case "window":
                # Windowed preview (cropped) above spectrum graph
                self._draw_sampling_lines(cropped, rotation_angle, perp_width)
                # Draw status on preview strip
                self._draw_status_overlay(cropped, status_text)
                spectrum_vertical = np.vstack((messages, cropped, graph))
            case "full":
                # Full camera view REPLACES the spectrum graph entirely
                raw_frame = data.raw_frame
                if raw_frame is not None:
                    # Convert raw frame to displayable BGR
                    if raw_frame.ndim == 2:
                        # Monochrome - convert to BGR
                        if raw_frame.dtype == np.uint16:
                            display = (raw_frame / 4).astype(np.uint8)
                        else:
                            display = raw_frame.astype(np.uint8)
                        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
                    else:
                        display = raw_frame.copy()
                    
                    # Scale to fill the area below control bar (preview + graph height)
                    camera_area_height = preview_height + graph_height
                    display = cv2.resize(display, (width, camera_area_height))
                    
                    # Draw sampling lines on full camera view
                    self._draw_sampling_lines_full(display, rotation_angle, perp_width)
                    # Draw status at bottom of camera view
                    self._draw_status_bar(display, status_text)
                    
                    spectrum_vertical = np.vstack((messages, display))
                else:
                    # No raw frame available, fall back to windowed
                    self._draw_sampling_lines(cropped, rotation_angle, perp_width)
                    self._draw_status_overlay(cropped, status_text)
                    spectrum_vertical = np.vstack((messages, cropped, graph))
            case _:
                # Default to windowed
                self._draw_sampling_lines(cropped, rotation_angle, perp_width)
                self._draw_status_overlay(cropped, status_text)
                spectrum_vertical = np.vstack((messages, cropped, graph))
        
        # Show the composed image
        cv2.imshow(self.config.spectrograph_title, spectrum_vertical)
        
        if self._waterfall is not None and self.config.display.waterfall_enabled:
            self._waterfall.update(data)
            waterfall_img = self._waterfall.render()
            
            # Use cropped preview for waterfall if available
            if cropped is None:
                cropped = np.zeros([preview_height, width, 3], dtype=np.uint8)
            waterfall_vertical = np.vstack((messages, cropped, waterfall_img))
            
            self._graticule.render_on_waterfall(
                waterfall_vertical,
                self.calibration.graticule,
                y_offset=message_height + preview_height + 2,
            )
            
            cv2.imshow(self.config.waterfall_title, waterfall_vertical)
    
    def set_mode_overlay(
        self,
        overlay: Optional[tuple[np.ndarray, tuple[int, int, int]]],
    ) -> None:
        """Set mode-specific overlay to render on graph.
        
        Args:
            overlay: Tuple of (intensity array, BGR color) or None to clear
        """
        self._mode_overlay = overlay
    
    def _render_mode_overlay(self, graph: np.ndarray) -> None:
        """Render mode-specific overlay (e.g., reference spectrum for calibration)."""
        if self._mode_overlay is None:
            return
        
        intensity, color = self._mode_overlay
        height = graph.shape[0]
        
        # Draw connected line for overlay
        points = []
        for i, val in enumerate(intensity):
            y = height - int(min(val, height - 1))
            points.append((i, max(0, y)))
        
        if len(points) > 1:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(graph, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
    
    def _render_reference_spectrum(self, graph: np.ndarray, data: SpectrumData) -> None:
        """Render the reference spectrum overlay line."""
        ref_intensity = self.state.reference_spectrum
        if ref_intensity is None:
            return
        
        height = graph.shape[0]
        ref_color = (180, 180, 180)  # Light gray for reference
        
        # Draw connected line for reference spectrum
        points = []
        for i, intensity in enumerate(ref_intensity):
            if i < len(data.wavelengths):
                y = height - int(min(intensity, height - 1))
                points.append((i, max(0, y)))
        
        if len(points) > 1:
            for j in range(len(points) - 1):
                cv2.line(graph, points[j], points[j + 1], ref_color, 1, cv2.LINE_AA)
    
    def _render_spectrum(self, graph: np.ndarray, data: SpectrumData) -> None:
        """Render the spectrum intensity curve."""
        height = graph.shape[0]
        
        for i, intensity in enumerate(data.intensity):
            rgb = wavelength_to_rgb(round(data.wavelengths[i]))
            bgr = rgb_to_bgr(rgb)
            
            cv2.line(graph, (i, height), (i, height - int(intensity)), bgr, 1)
            cv2.line(
                graph,
                (i, height - 1 - int(intensity)),
                (i, height - int(intensity)),
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    
    def _render_peaks(self, graph: np.ndarray, data: SpectrumData) -> None:
        """Render peak labels on the graph."""
        text_offset = 12
        
        for peak in data.peaks:
            label_height = 310 - peak.intensity
            
            cv2.rectangle(
                graph,
                (peak.index - text_offset - 2, label_height),
                (peak.index - text_offset + 60, label_height - 15),
                (0, 255, 255),
                -1,
            )
            cv2.rectangle(
                graph,
                (peak.index - text_offset - 2, label_height),
                (peak.index - text_offset + 60, label_height - 15),
                (0, 0, 0),
                1,
            )
            cv2.putText(
                graph,
                f"{peak.wavelength}nm",
                (peak.index - text_offset, label_height - 3),
                self._font,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                graph,
                (peak.index, label_height),
                (peak.index, label_height + 10),
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
    
    def _render_click_points(self, graph: np.ndarray) -> None:
        """Render clicked calibration points."""
        for x, y in self.state.click_points:
            cv2.circle(graph, (x, y), 5, (0, 0, 0), -1)
            cv2.putText(
                graph,
                str(x),
                (x + 5, y),
                self._font,
                0.4,
                (0, 0, 0),
            )
    
    def _draw_sampling_lines(
        self,
        cropped: np.ndarray,
        rotation_angle: float = 0.0,
        perp_width: int = 20,
    ) -> None:
        """Draw sampling region indicator lines on cropped preview.
        
        Since the image is already rotated to straighten the spectrum,
        we only draw simple horizontal lines showing the sampling bounds.
        """
        width = cropped.shape[1]
        rows = cropped.shape[0]
        halfway = rows // 2
        half_perp = perp_width // 2
        
        # Draw horizontal lines at sampling bounds (image is already rotated)
        y_top = max(0, halfway - half_perp)
        y_bottom = min(rows - 1, halfway + half_perp)
        cv2.line(cropped, (0, y_top), (width, y_top), (0, 255, 255), 1)
        cv2.line(cropped, (0, y_bottom), (width, y_bottom), (0, 255, 255), 1)
    
    def _draw_sampling_lines_full(
        self,
        frame: np.ndarray,
        rotation_angle: float = 0.0,
        perp_width: int = 20,
    ) -> None:
        """Draw sampling region indicator lines on full camera view.
        
        Shows the sampling region in the center of the full frame.
        """
        width = frame.shape[1]
        height = frame.shape[0]
        
        # Get spectrum Y center from config (or use frame center)
        spectrum_y = self.config.extraction.spectrum_y_center
        if spectrum_y == 0:
            spectrum_y = height // 2
        
        half_perp = perp_width // 2
        
        # Draw horizontal lines at sampling bounds
        y_top = max(0, spectrum_y - half_perp)
        y_bottom = min(height - 1, spectrum_y + half_perp)
        
        # Draw with thicker lines for visibility on full view
        cv2.line(frame, (0, y_top), (width, y_top), (0, 255, 255), 2)
        cv2.line(frame, (0, y_bottom), (width, y_bottom), (0, 255, 255), 2)
        
        # Draw center line (thinner, different color)
        cv2.line(frame, (0, spectrum_y), (width, spectrum_y), (255, 255, 0), 1)
    
    def _draw_status_overlay(self, image: np.ndarray, status_text: str) -> None:
        """Draw status text overlaid on the preview strip (semi-transparent background)."""
        if not status_text:
            return
        
        font = self._font
        font_scale = self.config.display.font_scale
        thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            status_text, font, font_scale, thickness
        )
        
        # Position at bottom-left with small padding
        padding = 3
        x = padding
        y = image.shape[0] - padding
        
        # Draw semi-transparent background rectangle
        bg_x1 = x - padding
        bg_y1 = y - text_height - padding
        bg_x2 = x + text_width + padding
        bg_y2 = y + baseline + padding
        
        # Create overlay for transparency effect
        overlay = image.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        # Draw text
        cv2.putText(
            image, status_text, (x, y),
            font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA
        )
    
    def _draw_status_bar(self, image: np.ndarray, status_text: str) -> None:
        """Draw status text at the bottom of the image."""
        if not status_text:
            return
        
        font = self._font
        font_scale = self.config.display.font_scale
        thickness = 1
        
        # Position at bottom-left
        x = 5
        y = image.shape[0] - 5
        
        # Draw with shadow for readability
        cv2.putText(
            image, status_text, (x + 1, y + 1),
            font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
        )
        cv2.putText(
            image, status_text, (x, y),
            font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA
        )
    
    def get_waterfall_image(self) -> Optional[np.ndarray]:
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
        
        self._control_bar.set_status("Cal", cal_result.status_message[:12])
        self._control_bar.set_status("Gain", f"{camera_gain:.0f}")
        self._control_bar.set_status("SG", str(savgol_poly))
        self._control_bar.set_status("Ext", extraction_method[:3].upper())
        self._control_bar.set_status("Ang", f"{rotation_angle:.1f}")
        self._control_bar.set_status("Wid", str(perp_width))
        
        if self.state.reference_name != "None":
            self._control_bar.set_status("Ref", self.state.reference_name[:8])
    
    @property
    def control_bar(self) -> ControlBar:
        """Get the control bar instance."""
        return self._control_bar
    
    @property
    def slider_panel(self) -> SliderPanel:
        """Get the slider panel instance."""
        return self._slider_panel
    
    def show_gain_slider(self, visible: bool) -> None:
        """Show or hide the gain slider."""
        self._slider_panel.gain_slider.visible = visible
    
    def show_exposure_slider(self, visible: bool) -> None:
        """Show or hide the exposure slider."""
        self._slider_panel.exposure_slider.visible = visible
    
    def set_gain_value(self, value: float) -> None:
        """Set the gain slider value."""
        self._slider_panel.gain_slider.value = value
    
    def set_exposure_value(self, value: float) -> None:
        """Set the exposure slider value."""
        self._slider_panel.exposure_slider.value = value
    
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
        original_gain_y = panel.gain_slider.y
        original_exp_y = panel.exposure_slider.y
        
        panel.gain_slider.y = panel.gain_slider.y - offset_y
        panel.exposure_slider.y = panel.exposure_slider.y - offset_y
        
        panel.render(graph)
        
        # Restore original positions for mouse handling
        panel.gain_slider.y = original_gain_y
        panel.exposure_slider.y = original_exp_y
    
    def destroy(self) -> None:
        """Destroy all OpenCV windows."""
        cv2.destroyAllWindows()
        self._windows_created = False
