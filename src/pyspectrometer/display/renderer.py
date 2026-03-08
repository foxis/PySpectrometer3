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
    ):
        """Initialize display manager.
        
        Args:
            config: Application configuration
            calibration: Wavelength calibration data
        """
        self.config = config
        self.calibration = calibration
        self.state = DisplayState()
        
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
        )
        
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._windows_created = False
        self._use_control_bar = True
    
    def setup_windows(self) -> None:
        """Create and configure OpenCV windows."""
        if self._windows_created:
            return
        
        title1 = self.config.spectrograph_title
        title2 = self.config.waterfall_title
        
        if self.config.display.waterfall_enabled:
            cv2.namedWindow(title2, cv2.WINDOW_NORMAL)
            cv2.moveWindow(title2, 200, 200)
        
        if self.config.display.fullscreen:
            cv2.namedWindow(title1, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                title1,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        else:
            # Use WINDOW_NORMAL to allow window to resize and fit content
            cv2.namedWindow(title1, cv2.WINDOW_NORMAL)
            cv2.moveWindow(title1, 0, 0)
        
        cv2.setMouseCallback(title1, self._handle_mouse)
        
        self._windows_created = True
    
    def _handle_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        """Handle mouse events on the spectrum window."""
        control_bar_height = self.config.display.message_height
        preview_height = self.config.display.preview_height
        mouse_y_offset = control_bar_height + preview_height
        
        if event == cv2.EVENT_MOUSEMOVE:
            self.state.cursor.x = x
            self.state.cursor.y = y
            
            if self._use_control_bar and y < control_bar_height:
                self._control_bar.handle_mouse_move(x, y)
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self._use_control_bar and y < control_bar_height:
                self._control_bar.handle_click(x, y)
            else:
                mouse_y = y - mouse_y_offset
                self.state.click_points.append((x, mouse_y))
    
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
        self._render_spectrum(graph, data)
        self._render_peaks(graph, data)
        self._render_cursor(graph, data)
        self._render_click_points(graph)
        
        if self._use_control_bar:
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
        else:
            messages = self._create_message_banner()
        
        cropped = data.cropped_frame
        if cropped is not None:
            self._draw_sampling_lines(cropped, rotation_angle, perp_width)
        else:
            cropped = np.zeros([80, width, 3], dtype=np.uint8)
        
        spectrum_vertical = np.vstack((messages, cropped, graph))
        
        cv2.line(spectrum_vertical, (0, 80), (width, 80), (255, 255, 255), 1)
        cv2.line(spectrum_vertical, (0, 160), (width, 160), (255, 255, 255), 1)
        
        self._render_status_text(
            spectrum_vertical,
            camera_gain,
            savgol_poly,
            peak_min_dist,
            peak_threshold,
            extraction_method,
            rotation_angle,
            perp_width,
        )
        
        cv2.imshow(self.config.spectrograph_title, spectrum_vertical)
        
        if self._waterfall is not None and self.config.display.waterfall_enabled:
            self._waterfall.update(data)
            waterfall_img = self._waterfall.render()
            
            waterfall_vertical = np.vstack((messages, cropped, waterfall_img))
            cv2.line(waterfall_vertical, (0, 80), (width, 80), (255, 255, 255), 1)
            cv2.line(waterfall_vertical, (0, 160), (width, 160), (255, 255, 255), 1)
            
            self._graticule.render_on_waterfall(
                waterfall_vertical,
                self.calibration.graticule,
                y_offset=162,
            )
            
            self._render_waterfall_status(waterfall_vertical, camera_gain)
            
            cv2.imshow(self.config.waterfall_title, waterfall_vertical)
    
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
    
    def _create_message_banner(self) -> np.ndarray:
        """Create the banner/message area image."""
        width = self.config.camera.frame_width
        height = self.config.display.message_height
        font_scale = self.config.display.font_scale
        
        banner = np.zeros([height, width, 3], dtype=np.uint8)
        
        title_scale = min(0.8, font_scale * 2.0)
        subtitle_scale = font_scale
        
        y_title = min(30, int(height * 0.4))
        y_subtitle = min(55, int(height * 0.7))
        y_hint = min(75, int(height * 0.95))
        
        cv2.putText(
            banner,
            "PySpectrometer 3",
            (5, y_title),
            self._font,
            title_scale,
            (0, 200, 255),
            max(1, int(title_scale * 2)),
            cv2.LINE_AA,
        )
        cv2.putText(
            banner,
            "github.com/leswright1977",
            (5, y_subtitle),
            self._font,
            subtitle_scale,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            banner,
            "Press 'q' to quit",
            (5, y_hint),
            self._font,
            max(0.25, subtitle_scale * 0.85),
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        
        return banner
    
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
    
    def _render_status_text(
        self,
        image: np.ndarray,
        camera_gain: float,
        savgol_poly: int,
        peak_min_dist: int,
        peak_threshold: int,
        extraction_method: str = "Weighted Sum",
        rotation_angle: float = 0.0,
        perp_width: int = 20,
    ) -> None:
        """Render status text on the spectrum display."""
        cal_result = self.calibration.result
        font_scale = self.config.display.font_scale
        thickness = self.config.display.text_thickness
        col1_x = self.config.display.status_col1_x
        col2_x = self.config.display.status_col2_x
        
        width = self.config.camera.frame_width
        col3_x = max(col2_x + 80, width - 120)
        
        line_height = int(18 * (font_scale / 0.4))
        
        cv2.putText(
            image,
            cal_result.status_message,
            (col1_x, line_height),
            self._font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            cal_result.detail_message,
            (col1_x, line_height * 2),
            self._font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            self.state.save_message,
            (col1_x, line_height * 3),
            self._font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Gain: {camera_gain:.0f}",
            (col1_x, line_height * 4),
            self._font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        
        hold_msg = "Hold ON" if self.state.hold_peaks else "Hold OFF"
        cv2.putText(
            image,
            hold_msg,
            (col2_x, line_height),
            self._font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"SG: {savgol_poly}",
            (col2_x, line_height * 2),
            self._font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"PkW: {peak_min_dist}",
            (col2_x, line_height * 3),
            self._font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Thr: {peak_threshold}",
            (col2_x, line_height * 4),
            self._font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        
        method_short = extraction_method[:3].upper()
        cv2.putText(
            image,
            f"Ext: {method_short}",
            (col3_x, line_height),
            self._font,
            font_scale,
            (0, 200, 200),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Ang: {rotation_angle:.1f}",
            (col3_x, line_height * 2),
            self._font,
            font_scale,
            (0, 200, 200),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Wid: {perp_width}",
            (col3_x, line_height * 3),
            self._font,
            font_scale,
            (0, 200, 200),
            thickness,
            cv2.LINE_AA,
        )
        
        # Reference spectrum indicator
        if self.state.reference_name != "None":
            ref_short = self.state.reference_name[:8]
            cv2.putText(
                image,
                f"Ref: {ref_short}",
                (col3_x, line_height * 4),
                self._font,
                font_scale,
                (180, 180, 180),
                thickness,
                cv2.LINE_AA,
            )
    
    def _render_waterfall_status(
        self,
        image: np.ndarray,
        camera_gain: float,
    ) -> None:
        """Render status text on the waterfall display."""
        cal_result = self.calibration.result
        font_scale = self.config.display.font_scale
        thickness = self.config.display.text_thickness
        col1_x = self.config.display.status_col1_x
        col2_x = self.config.display.status_col2_x
        
        line_height = int(18 * (font_scale / 0.4))
        
        cv2.putText(
            image,
            cal_result.status_message,
            (col1_x, line_height),
            self._font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            cal_result.detail_message,
            (col1_x, line_height * 2),
            self._font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            self.state.save_message,
            (col1_x, line_height * 3),
            self._font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Gain: {camera_gain:.0f}",
            (col1_x, line_height * 4),
            self._font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        
        hold_msg = "Hold ON" if self.state.hold_peaks else "Hold OFF"
        cv2.putText(
            image,
            hold_msg,
            (col2_x, line_height),
            self._font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
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
    
    def destroy(self) -> None:
        """Destroy all OpenCV windows."""
        cv2.destroyAllWindows()
        self._windows_created = False
