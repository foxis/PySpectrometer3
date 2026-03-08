"""Main spectrometer application orchestrator."""

import time
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

from .config import Config
from .core.spectrum import SpectrumData
from .core.calibration import Calibration
from .core.reference_spectrum import ReferenceSpectrumManager
from .capture.base import CameraInterface
from .capture.picamera import PicameraCapture
from .processing.pipeline import ProcessingPipeline
from .processing.filters import SavitzkyGolayFilter
from .processing.peak_detection import PeakDetector
from .processing.extraction import SpectrumExtractor, ExtractionMethod
from .display.renderer import DisplayManager
from .export.csv_exporter import CSVExporter
from .input.keyboard import KeyboardHandler, Action


class Spectrometer:
    """Main spectrometer application orchestrator.
    
    This class wires together all components (camera, processing, display,
    input, export) and manages the main application loop.
    """
    
    # Valid operating modes
    VALID_MODES = ("calibration", "measurement", "raman", "colorscience")
    
    def __init__(
        self,
        config: Optional[Config] = None,
        camera: Optional[CameraInterface] = None,
        mode: str = "measurement",
        laser_nm: float = 785.0,
    ):
        """Initialize spectrometer application.
        
        Args:
            config: Application configuration (uses defaults if None)
            camera: Camera backend (creates PicameraCapture if None)
            mode: Operating mode (calibration, measurement, raman, colorscience)
            laser_nm: Raman laser wavelength in nm (for raman mode)
        """
        self.config = config or Config()
        self.mode = mode if mode in self.VALID_MODES else "measurement"
        self.laser_nm = laser_nm
        
        self._camera = camera or PicameraCapture(
            width=self.config.camera.frame_width,
            height=self.config.camera.frame_height,
            gain=self.config.camera.gain,
            fps=self.config.camera.fps,
        )
        
        self._calibration = Calibration(
            width=self.config.camera.frame_width,
            height=self.config.camera.frame_height,
            cal_file=self.config.calibration.data_file,
            default_pixels=self.config.calibration.default_pixels,
            default_wavelengths=self.config.calibration.default_wavelengths,
        )
        
        method_map = {
            "median": ExtractionMethod.MEDIAN,
            "weighted_sum": ExtractionMethod.WEIGHTED_SUM,
            "gaussian": ExtractionMethod.GAUSSIAN,
        }
        extraction_method = method_map.get(
            self.config.extraction.method,
            ExtractionMethod.WEIGHTED_SUM
        )
        
        spectrum_y = self.config.extraction.spectrum_y_center
        if spectrum_y == 0:
            spectrum_y = self.config.camera.frame_height // 2
        
        self._extractor = SpectrumExtractor(
            frame_width=self.config.camera.frame_width,
            frame_height=self.config.camera.frame_height,
            method=extraction_method,
            rotation_angle=self.config.extraction.rotation_angle,
            perpendicular_width=self.config.extraction.perpendicular_width,
            spectrum_y_center=spectrum_y,
            crop_height=self.config.display.preview_height,
            background_percentile=self.config.extraction.background_percentile,
        )
        
        self._savgol_filter = SavitzkyGolayFilter(
            window_size=self.config.processing.savgol_window,
            poly_order=self.config.processing.savgol_poly,
            poly_order_min=self.config.processing.savgol_poly_min,
            poly_order_max=self.config.processing.savgol_poly_max,
        )
        
        self._peak_detector = PeakDetector(
            min_distance=self.config.processing.peak_min_distance,
            threshold=self.config.processing.peak_threshold,
            min_distance_min=self.config.processing.peak_min_distance_min,
            min_distance_max=self.config.processing.peak_min_distance_max,
            threshold_min=self.config.processing.peak_threshold_min,
            threshold_max=self.config.processing.peak_threshold_max,
        )
        
        self._pipeline = ProcessingPipeline([
            self._savgol_filter,
            self._peak_detector,
        ])
        
        self._display = DisplayManager(self.config, self._calibration)
        self._exporter = CSVExporter(output_dir=self.config.export.output_dir)
        self._keyboard = KeyboardHandler()
        self._reference_manager = ReferenceSpectrumManager()
        
        self._held_intensity: Optional[np.ndarray] = None
        self._running = False
        self._last_frame: Optional[np.ndarray] = None
        
        self._register_key_callbacks()
        self._register_button_callbacks()
    
    def _register_key_callbacks(self) -> None:
        """Register keyboard action callbacks."""
        self._keyboard.register(Action.QUIT, self._on_quit)
        self._keyboard.register(Action.TOGGLE_HOLD_PEAKS, self._on_toggle_hold)
        self._keyboard.register(Action.SAVE, self._on_save)
        self._keyboard.register(Action.CALIBRATE, self._on_calibrate)
        self._keyboard.register(Action.CLEAR_CLICKS, self._on_clear_clicks)
        self._keyboard.register(Action.TOGGLE_MEASURE, self._on_toggle_measure)
        self._keyboard.register(Action.TOGGLE_PIXEL_MODE, self._on_toggle_pixel_mode)
        self._keyboard.register(Action.SAVGOL_UP, self._on_savgol_up)
        self._keyboard.register(Action.SAVGOL_DOWN, self._on_savgol_down)
        self._keyboard.register(Action.PEAK_WIDTH_UP, self._on_peak_width_up)
        self._keyboard.register(Action.PEAK_WIDTH_DOWN, self._on_peak_width_down)
        self._keyboard.register(Action.THRESHOLD_UP, self._on_threshold_up)
        self._keyboard.register(Action.THRESHOLD_DOWN, self._on_threshold_down)
        self._keyboard.register(Action.GAIN_UP, self._on_gain_up)
        self._keyboard.register(Action.GAIN_DOWN, self._on_gain_down)
        self._keyboard.register(Action.CYCLE_EXTRACTION_METHOD, self._on_cycle_extraction)
        self._keyboard.register(Action.AUTO_DETECT_ANGLE, self._on_auto_detect_angle)
        self._keyboard.register(Action.PERP_WIDTH_UP, self._on_perp_width_up)
        self._keyboard.register(Action.PERP_WIDTH_DOWN, self._on_perp_width_down)
        self._keyboard.register(Action.SAVE_EXTRACTION_PARAMS, self._on_save_extraction)
        self._keyboard.register(Action.CYCLE_REFERENCE_SPECTRUM, self._on_cycle_reference)
    
    def _register_button_callbacks(self) -> None:
        """Register callbacks for control bar buttons."""
        display = self._display
        
        display.register_button_callback("quit", self._on_quit)
        display.register_button_callback("save", self._on_button_save)
        display.register_button_callback("capture_peak", self._on_toggle_hold)
        display.register_button_callback("capture_current", self._on_capture_current)
        display.register_button_callback("load_reference", self._on_load_reference)
        display.register_button_callback("use_as_reference", self._on_use_as_reference)
        display.register_button_callback("capture_dark", self._on_capture_dark)
        display.register_button_callback("gain_up", self._on_gain_up)
        display.register_button_callback("gain_down", self._on_gain_down)
        display.register_button_callback("auto_gain", self._on_toggle_auto_gain)
        display.register_button_callback("light_toggle", self._on_toggle_light)
        display.register_button_callback("fit_xyz", self._on_fit_xyz)
        display.register_button_callback("toggle_pixel_mode", self._on_toggle_pixel_mode)
        display.register_button_callback("toggle_measure", self._on_toggle_measure)
        display.register_button_callback("calibrate", self._on_calibrate)
        display.register_button_callback("clear_clicks", self._on_clear_clicks)
        display.register_button_callback("cycle_extraction", self._on_cycle_extraction)
        display.register_button_callback("auto_detect_angle", self._on_auto_detect_angle)
    
    def _on_button_save(self) -> None:
        """Handle save button click."""
        if self._last_data is not None:
            self._save_snapshot(self._last_data)
        else:
            print("[SAVE] No spectrum data available")
    
    def _on_capture_current(self) -> None:
        """Handle capture current spectrum."""
        print("[CAPTURE] Capturing current spectrum (not implemented yet)")
    
    def _on_load_reference(self) -> None:
        """Handle load reference button."""
        print("[LOAD] Load reference spectrum (not implemented yet)")
    
    def _on_use_as_reference(self) -> None:
        """Handle use current spectrum as reference."""
        if self._last_data is not None:
            self._display.state.reference_spectrum = self._last_data.intensity.copy()
            self._display.state.reference_name = "Current"
            print("[REF] Set current spectrum as reference")
        else:
            print("[REF] No spectrum data available")
    
    def _on_capture_dark(self) -> None:
        """Handle capture dark spectrum."""
        print("[DARK] Capture dark spectrum (not implemented yet)")
    
    def _on_toggle_auto_gain(self) -> None:
        """Handle toggle auto gain."""
        print("[AUTO_GAIN] Toggle auto gain (not implemented yet)")
    
    def _on_toggle_light(self) -> None:
        """Handle toggle light control."""
        print("[LIGHT] Toggle light (GPIO not implemented yet)")
    
    def _on_fit_xyz(self) -> None:
        """Handle XYZ curve fitting."""
        print("[XYZ] Fit XYZ curves (not implemented yet)")
    
    def _on_quit(self) -> None:
        """Handle quit action."""
        self._running = False
    
    def _on_toggle_hold(self) -> None:
        """Handle toggle hold peaks action."""
        self._display.state.hold_peaks = not self._display.state.hold_peaks
        self._display.set_button_active("capture_peak", self._display.state.hold_peaks)
        if not self._display.state.hold_peaks:
            self._held_intensity = None
        print(f"[HOLD] Peak hold {'ON' if self._display.state.hold_peaks else 'OFF'}")
    
    def _on_save(self) -> None:
        """Handle save action."""
        pass
    
    def _on_calibrate(self) -> None:
        """Handle calibration action."""
        clicks = self._display.state.click_points
        if len(clicks) < 3:
            print(f"Need at least 3 calibration points (have {len(clicks)})")
            return
        
        print("Enter known wavelengths for observed pixels!")
        pixel_data = []
        wavelength_data = []
        
        for x, y in clicks:
            try:
                wavelength_str = input(f"Enter wavelength for: {x}px: ")
                wavelength = float(wavelength_str)
                pixel_data.append(x)
                wavelength_data.append(wavelength)
            except ValueError:
                print("Only numbers are allowed!")
                print("Calibration aborted!")
                return
        
        if self._calibration.save(pixel_data, wavelength_data):
            self._display.state.click_points.clear()
    
    def _on_clear_clicks(self) -> None:
        """Handle clear clicks action."""
        self._display.state.click_points.clear()
    
    def _on_toggle_measure(self) -> None:
        """Handle toggle measure mode action."""
        self._display.state.cursor.pixel_mode = False
        self._display.state.cursor.measure_mode = not self._display.state.cursor.measure_mode
        self._display.set_button_active("toggle_measure", self._display.state.cursor.measure_mode)
        self._display.set_button_active("toggle_pixel_mode", False)
        print(f"[MODE] Measure mode {'ON' if self._display.state.cursor.measure_mode else 'OFF'}")
    
    def _on_toggle_pixel_mode(self) -> None:
        """Handle toggle pixel mode action."""
        self._display.state.cursor.measure_mode = False
        self._display.state.cursor.pixel_mode = not self._display.state.cursor.pixel_mode
        self._display.set_button_active("toggle_pixel_mode", self._display.state.cursor.pixel_mode)
        self._display.set_button_active("toggle_measure", False)
        if not self._display.state.cursor.pixel_mode:
            self._display.state.click_points.clear()
        print(f"[MODE] Pixel mode {'ON' if self._display.state.cursor.pixel_mode else 'OFF'}")
    
    def _on_savgol_up(self) -> None:
        """Handle Savitzky-Golay order increase."""
        self._savgol_filter.increase_poly_order()
    
    def _on_savgol_down(self) -> None:
        """Handle Savitzky-Golay order decrease."""
        self._savgol_filter.decrease_poly_order()
    
    def _on_peak_width_up(self) -> None:
        """Handle peak width increase."""
        self._peak_detector.increase_min_distance()
    
    def _on_peak_width_down(self) -> None:
        """Handle peak width decrease."""
        self._peak_detector.decrease_min_distance()
    
    def _on_threshold_up(self) -> None:
        """Handle threshold increase."""
        self._peak_detector.increase_threshold()
    
    def _on_threshold_down(self) -> None:
        """Handle threshold decrease."""
        self._peak_detector.decrease_threshold()
    
    def _on_gain_up(self) -> None:
        """Handle camera gain increase."""
        self._camera.gain = self._camera.gain + 1
        print(f"Camera Gain: {self._camera.gain}")
    
    def _on_gain_down(self) -> None:
        """Handle camera gain decrease."""
        self._camera.gain = self._camera.gain - 1
        print(f"Camera Gain: {self._camera.gain}")
    
    def _on_cycle_extraction(self) -> None:
        """Handle extraction method cycling."""
        new_method = self._extractor.cycle_method()
        print(f"Extraction Method: {new_method}")
    
    def _on_auto_detect_angle(self) -> None:
        """Handle auto-detect rotation angle and Y center."""
        if self._last_frame is None:
            print("No frame available for angle detection")
            return
        
        print("Detecting spectrum rotation angle and position...")
        angle, y_center, vis_image = self._extractor.detect_angle(self._last_frame, visualize=True)
        
        if vis_image is not None:
            cv2.imshow("Angle Detection", vis_image)
            cv2.waitKey(2000)
            cv2.destroyWindow("Angle Detection")
        
        # Apply both angle and Y center
        self._extractor.set_rotation_angle(angle)
        self._extractor.set_spectrum_y_center(y_center)
        print(f"Rotation Angle: {angle:.2f}°, Y Center: {y_center}")
        
        # Auto-save extraction parameters
        self._calibration.save_extraction_params(
            rotation_angle=self._extractor.rotation_angle,
            spectrum_y_center=self._extractor.spectrum_y_center,
            perpendicular_width=self._extractor.perpendicular_width,
        )
        print("Extraction parameters saved.")
    
    def _on_perp_width_up(self) -> None:
        """Handle perpendicular width increase."""
        width = self._extractor.increase_perpendicular_width()
        print(f"Perpendicular Width: {width}")
    
    def _on_perp_width_down(self) -> None:
        """Handle perpendicular width decrease."""
        width = self._extractor.decrease_perpendicular_width()
        print(f"Perpendicular Width: {width}")
    
    def _on_save_extraction(self) -> None:
        """Handle saving extraction parameters."""
        self._calibration.save_extraction_params(
            rotation_angle=self._extractor.rotation_angle,
            spectrum_y_center=self._extractor.spectrum_y_center,
            perpendicular_width=self._extractor.perpendicular_width,
        )
    
    def _on_cycle_reference(self) -> None:
        """Handle cycling through reference spectrums."""
        new_name = self._reference_manager.cycle_next()
        self._display.state.reference_name = new_name
        print(f"Reference Spectrum: {new_name}")
    
    def run(self) -> None:
        """Run the main spectrometer application loop."""
        mode_names = {
            "calibration": "Calibration",
            "measurement": "Measurement",
            "raman": "Raman",
            "colorscience": "Color Science",
        }
        print(f"Starting PySpectrometer 3 - {mode_names.get(self.mode, self.mode)} Mode...")
        print(self._keyboard.get_help_text())
        print()
        
        self._calibration.load()
        
        if self._calibration.rotation_angle != 0.0:
            self._extractor.set_rotation_angle(self._calibration.rotation_angle)
        if self._calibration.spectrum_y_center != 0:
            self._extractor.set_spectrum_y_center(self._calibration.spectrum_y_center)
        if self._calibration.perpendicular_width != 20:
            self._extractor.perpendicular_width = self._calibration.perpendicular_width
            self._extractor._precompute_sampling_coords()
        
        self._camera.start()
        self._display.setup_windows()
        
        self._running = True
        self._last_data: Optional[SpectrumData] = None
        
        try:
            while self._running:
                frame = self._camera.capture()
                self._last_frame = frame
                
                extraction_result = self._extractor.extract(frame)
                cropped = extraction_result.cropped_frame
                intensity = extraction_result.intensity
                
                if self._display.state.hold_peaks:
                    if self._held_intensity is None:
                        self._held_intensity = intensity.copy()
                    else:
                        self._held_intensity = np.maximum(self._held_intensity, intensity)
                    intensity = self._held_intensity
                    self._savgol_filter.enabled = False
                else:
                    self._savgol_filter.enabled = True
                
                data = SpectrumData(
                    intensity=intensity,
                    wavelengths=self._calibration.wavelengths,
                    raw_frame=frame,
                    cropped_frame=cropped,
                )
                
                processed = self._pipeline.run(data)
                
                self._last_data = processed
                
                # Update reference spectrum overlay
                ref_intensity = self._reference_manager.get_interpolated(
                    processed.wavelengths,
                    processed.intensity,
                )
                self._display.state.reference_spectrum = ref_intensity
                
                self._display.render(
                    processed,
                    camera_gain=self._camera.gain,
                    savgol_poly=self._savgol_filter.poly_order,
                    peak_min_dist=self._peak_detector.min_distance,
                    peak_threshold=self._peak_detector.threshold,
                    extraction_method=str(self._extractor.method),
                    rotation_angle=self._extractor.rotation_angle,
                    perp_width=self._extractor.perpendicular_width,
                )
                
                action = self._keyboard.poll()
                
                if action == Action.SAVE and self._last_data is not None:
                    self._save_snapshot(processed)
                    
        finally:
            self._camera.stop()
            self._display.destroy()
            print("PySpectrometer 3 stopped.")
    
    def _save_snapshot(self, data: SpectrumData) -> None:
        """Save current spectrum snapshot."""
        timestamp = time.strftime(self.config.export.timestamp_format)
        time_display = time.strftime(self.config.export.time_format)
        
        csv_path = self._exporter.generate_filename("Spectrum")
        self._exporter.export(data, csv_path)
        
        spectrum_path = Path(f"spectrum-{timestamp}.png")
        
        waterfall_img = self._display.get_waterfall_image()
        if waterfall_img is not None:
            waterfall_path = Path(f"waterfall-{timestamp}.png")
            cv2.imwrite(str(waterfall_path), waterfall_img)
        
        self._display.state.save_message = f"Last Save: {time_display}"
        print(f"Saved: {csv_path}")
    
    @property
    def camera(self) -> CameraInterface:
        """Get the camera backend."""
        return self._camera
    
    @property
    def calibration(self) -> Calibration:
        """Get the calibration manager."""
        return self._calibration
    
    @property
    def pipeline(self) -> ProcessingPipeline:
        """Get the processing pipeline."""
        return self._pipeline
    
    @property
    def display(self) -> DisplayManager:
        """Get the display manager."""
        return self._display
