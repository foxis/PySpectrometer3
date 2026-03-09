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
from .capture.picamera import Capture
from .processing.pipeline import ProcessingPipeline
from .processing.filters import SavitzkyGolayFilter
from .processing.auto_controls import AutoGainController, AutoExposureController
from .processing.peak_detection import PeakDetector
from .processing.extraction import SpectrumExtractor, ExtractionMethod
from .display.renderer import DisplayManager
from .export.csv_exporter import CSVExporter
from .input.keyboard import KeyboardHandler, Action
from .modes.base import BaseMode
from .modes.calibration import CalibrationMode
from .modes.measurement import MeasurementMode
from .data.reference_spectra import ReferenceSource


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
            camera: Camera backend (creates Pi Capture if None)
            mode: Operating mode (calibration, measurement, raman, colorscience)
            laser_nm: Raman laser wavelength in nm (for raman mode)
        """
        self.config = config or Config()
        self.mode = mode if mode in self.VALID_MODES else "measurement"
        self.laser_nm = laser_nm
        
        self._camera = camera or Capture(
            width=self.config.camera.frame_width,
            height=self.config.camera.frame_height,
            gain=self.config.camera.gain,
            fps=self.config.camera.fps,
            monochrome=self.config.camera.monochrome,
            bit_depth=self.config.camera.bit_depth,
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
        self._savgol_filter.enabled = False  # Disabled for now (clipping artifacts)
        
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
        
        # Mode instance for button definitions (single source of truth)
        mode_instance: Optional[BaseMode] = None
        match mode:
            case "calibration":
                self._calibration_mode = CalibrationMode()
                mode_instance = self._calibration_mode
            case "measurement":
                self._measurement_mode = MeasurementMode()
                mode_instance = self._measurement_mode
            case _:
                mode_instance = MeasurementMode()  # Fallback for raman/colorscience
        self._display = DisplayManager(self.config, self._calibration, mode=mode, mode_instance=mode_instance)
        self._exporter = CSVExporter(output_dir=self.config.export.output_dir)
        self._keyboard = KeyboardHandler()
        self._reference_manager = ReferenceSpectrumManager()
        
        # Mode-specific handlers (modes created above)
        
        self._held_intensity: Optional[np.ndarray] = None
        self._running = False
        self._last_frame: Optional[np.ndarray] = None
        self._frozen_spectrum: bool = False
        self._frozen_intensity: Optional[np.ndarray] = None  # Stored frozen spectrum (pre-sensitivity)
        self._last_intensity_pre_sensitivity: Optional[np.ndarray] = None  # For freeze capture
        self._autolevel_overlay: Optional[np.ndarray] = None  # Non-blocking autolevel display

        # Auto-gain/exposure state
        self._auto_gain_enabled: bool = False
        self._auto_exposure_enabled: bool = False
        self._auto_gain = AutoGainController()
        self._auto_exposure = AutoExposureController()

        # AutoCal debounce (touch screens can send repeat events)
        self._last_auto_calibrate_time: float = 0.0
        self._AUTO_CAL_DEBOUNCE_SEC: float = 1.5

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
        
        # Common buttons
        display.register_button_callback("quit", self._on_quit)
        display.register_button_callback("capture_peak", self._on_toggle_hold)
        display.register_button_callback("cycle_preview", self._on_cycle_preview)
        
        # Gain/Exposure slider toggles
        display.register_button_callback("show_gain_slider", self._on_toggle_gain_slider)
        display.register_button_callback("show_exposure_slider", self._on_toggle_exposure_slider)
        display.register_button_callback("auto_gain", self._on_toggle_auto_gain)
        display.register_button_callback("auto_exposure", self._on_toggle_auto_exposure)
        
        # Set up slider change callbacks
        self._display.register_slider_callbacks(
            gain_cb=self._on_gain_slider_change,
            exposure_cb=self._on_exposure_slider_change,
        )
        
        # Initialize slider values from camera
        self._display.set_gain_value(self._camera.gain)
        self._display.set_exposure_value(self._camera.exposure)
        
        if self.mode == "calibration":
            self._register_calibration_callbacks()
        else:
            self._register_measurement_callbacks()
    
    def _register_measurement_callbacks(self) -> None:
        """Register callbacks for measurement mode buttons."""
        display = self._display
        
        # Capture and references
        display.register_button_callback("capture", self._on_capture_current)
        display.register_button_callback("toggle_averaging", self._on_toggle_averaging_meas)
        display.register_button_callback("set_dark", self._on_set_dark)
        display.register_button_callback("set_white", self._on_set_white)
        display.register_button_callback("clear_refs", self._on_clear_refs)
        display.register_button_callback("save", self._on_button_save)
        display.register_button_callback("load", self._on_load_spectrum)
        
        # Display and control
        display.register_button_callback("show_reference", self._on_toggle_show_reference)
        display.register_button_callback("normalize", self._on_toggle_normalize)
        display.register_button_callback("lamp_toggle", self._on_toggle_light)
    
    def _register_calibration_callbacks(self) -> None:
        """Register callbacks for calibration mode buttons."""
        display = self._display
        
        # Source selection
        display.register_button_callback("source_fl", lambda: self._on_select_source(ReferenceSource.FL))
        display.register_button_callback("source_fl2", lambda: self._on_select_source(ReferenceSource.FL2))
        display.register_button_callback("source_fl3", lambda: self._on_select_source(ReferenceSource.FL3))
        display.register_button_callback("source_fl4", lambda: self._on_select_source(ReferenceSource.FL4))
        display.register_button_callback("source_fl5", lambda: self._on_select_source(ReferenceSource.FL5))
        display.register_button_callback("source_a", lambda: self._on_select_source(ReferenceSource.A))
        display.register_button_callback("source_hg", lambda: self._on_select_source(ReferenceSource.HG))
        display.register_button_callback("source_d65", lambda: self._on_select_source(ReferenceSource.D65))
        display.register_button_callback("source_led", lambda: self._on_select_source(ReferenceSource.LED))
        display.register_button_callback("source_led2", lambda: self._on_select_source(ReferenceSource.LED2))
        display.register_button_callback("source_led3", lambda: self._on_select_source(ReferenceSource.LED3))
        
        # Calibration actions
        display.register_button_callback("toggle_overlay", self._on_toggle_overlay)
        display.register_button_callback("toggle_sensitivity", self._on_toggle_sensitivity)
        display.register_button_callback("correct_sensitivity", self._on_correct_sensitivity)
        display.register_button_callback("auto_level", self._on_auto_level)
        display.register_button_callback("auto_calibrate", self._on_auto_calibrate)
        display.register_button_callback("reset_calibration", self._on_reset_calibration)
        display.register_button_callback("save_cal", self._on_save_calibration)
        display.register_button_callback("save_spectrum", self._on_button_save)
        display.register_button_callback("load_cal", self._on_load_calibration)
        
        # Display and control
        display.register_button_callback("freeze", self._on_toggle_freeze)
        display.register_button_callback("toggle_averaging", self._on_toggle_averaging)
        display.register_button_callback("clear_points", self._on_clear_cal_points)
        
        # Set initial button states
        if self._calibration_mode is not None:
            display.set_button_active("toggle_overlay", self._calibration_mode.cal_state.overlay_visible)
            display.set_button_active("toggle_sensitivity", self._calibration_mode.cal_state.sensitivity_correction_enabled)
            # Freeze button: active = record (live), inactive = stopped (frozen). Default: live = record
            display.set_button_active("freeze", not self._frozen_spectrum)

    def _on_correct_sensitivity(self) -> None:
        """Handle CORR button - placeholder for future sensitivity curve correction."""
        print("[CORR] Correct sensitivity curve (placeholder, noop)")

    def _on_reset_calibration(self) -> None:
        """Handle R button - reset calibration to linear 1:1 pixels to spectrum."""
        n = self._calibration.width
        wl_min, wl_max = 380.0, 750.0
        pixels = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        wavelengths = [wl_min + (wl_max - wl_min) * p / max(n - 1, 1) for p in pixels]
        self._calibration.recalibrate(pixels, wavelengths)
        if self._calibration_mode is not None:
            self._calibration_mode.cal_state.calibration_points = list(zip(pixels, wavelengths))
        print(f"[R] Calibration reset to linear 1:1: {wl_min:.0f}-{wl_max:.0f} nm over {n} pixels")
    
    def _on_button_save(self) -> None:
        """Handle save button click."""
        if self._last_data is not None:
            self._save_snapshot(self._last_data)
        else:
            print("[SAVE] No spectrum data available")
    
    def _on_capture_current(self) -> None:
        """Handle capture current spectrum."""
        if self._last_data is None:
            print("[CAPTURE] No spectrum data available")
            return
        
        if self._measurement_mode is not None:
            self._measurement_mode.capture_current(
                self._last_data.intensity,
                self._last_data.wavelengths,
            )
            # Also set as reference for overlay/normalization
            self._measurement_mode.set_reference_spectrum(
                self._last_data.intensity,
                "Captured",
            )
        else:
            print("[CAPTURE] Spectrum captured")
    
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
        self._auto_gain_enabled = not getattr(self, "_auto_gain_enabled", False)
        self._display.set_button_active("auto_gain", self._auto_gain_enabled)
        print(f"[AUTO_GAIN] Auto gain: {'ON' if self._auto_gain_enabled else 'OFF'}")
    
    def _on_toggle_auto_exposure(self) -> None:
        """Handle toggle auto exposure."""
        self._auto_exposure_enabled = not getattr(self, "_auto_exposure_enabled", False)
        self._display.set_button_active("auto_exposure", self._auto_exposure_enabled)
        print(f"[AUTO_EXPOSURE] Auto exposure: {'ON' if self._auto_exposure_enabled else 'OFF'}")
    
    def _on_toggle_gain_slider(self) -> None:
        """Handle toggle gain slider visibility."""
        visible = self._display.toggle_gain_slider()
        self._display.set_button_active("show_gain_slider", visible)
        print(f"[GAIN] Gain slider: {'VISIBLE' if visible else 'HIDDEN'}")
    
    def _on_toggle_exposure_slider(self) -> None:
        """Handle toggle exposure slider visibility."""
        visible = self._display.toggle_exposure_slider()
        self._display.set_button_active("show_exposure_slider", visible)
        print(f"[EXPOSURE] Exposure slider: {'VISIBLE' if visible else 'HIDDEN'}")
    
    def _on_gain_slider_change(self, value: float) -> None:
        """Handle gain slider value change."""
        self._camera.gain = value
    
    def _on_exposure_slider_change(self, value: float) -> None:
        """Handle exposure slider value change."""
        self._camera.exposure = int(value)
    
    def _handle_auto_gain_exposure(self, data: SpectrumData) -> None:
        """Handle auto-gain and auto-exposure based on spectrum peak."""
        if not self._auto_gain_enabled and not self._auto_exposure_enabled:
            return
        if self._frozen_spectrum:
            return

        def get_gain() -> float:
            return self._camera.gain

        def set_gain(v: float) -> None:
            self._camera.gain = v

        def get_exposure() -> int:
            return getattr(self._camera, "exposure", 10000)

        def set_exposure(v: int) -> None:
            if hasattr(self._camera, "exposure"):
                self._camera.exposure = v

        if self._auto_exposure_enabled:
            self._auto_exposure.adjust(
                data,
                get_exposure,
                set_exposure,
                self._display.set_exposure_value,
            )
        if self._auto_gain_enabled:
            self._auto_gain.adjust(
                data,
                get_gain,
                set_gain,
                self._display.set_gain_value,
            )
    
    def _on_toggle_light(self) -> None:
        """Handle toggle light control."""
        print("[LIGHT] Toggle light (GPIO not implemented yet)")
    
    def _on_fit_xyz(self) -> None:
        """Handle XYZ curve fitting."""
        print("[XYZ] Fit XYZ curves (not implemented yet)")
    
    # Measurement mode specific handlers
    
    def _on_toggle_averaging_meas(self) -> None:
        """Handle toggle averaging in measurement mode."""
        if self._measurement_mode is None:
            return
        
        enabled = self._measurement_mode.toggle_averaging()
        self._display.set_button_active("toggle_averaging", enabled)
    
    def _on_set_dark(self) -> None:
        """Handle set dark reference."""
        if self._measurement_mode is None or self._last_data is None:
            print("[DARK] No spectrum data available")
            return
        
        self._measurement_mode.set_dark_reference(self._last_data.intensity)
    
    def _on_set_white(self) -> None:
        """Handle set white reference."""
        if self._measurement_mode is None or self._last_data is None:
            print("[WHITE] No spectrum data available")
            return
        
        self._measurement_mode.set_white_reference(self._last_data.intensity)
    
    def _on_clear_refs(self) -> None:
        """Handle clear all references."""
        if self._measurement_mode is None:
            return
        
        self._measurement_mode.clear_references()
        self._display.set_button_active("show_reference", False)
        self._display.set_button_active("normalize", False)
    
    def _on_load_spectrum(self) -> None:
        """Handle load spectrum from file."""
        print("[LOAD] Load spectrum (file dialog not implemented yet)")
    
    def _on_toggle_show_reference(self) -> None:
        """Handle toggle show reference overlay."""
        if self._measurement_mode is None:
            return
        
        enabled = self._measurement_mode.toggle_show_reference()
        self._display.set_button_active("show_reference", enabled)
    
    def _on_toggle_normalize(self) -> None:
        """Handle toggle normalization to reference."""
        if self._measurement_mode is None:
            return
        
        enabled = self._measurement_mode.toggle_normalize()
        self._display.set_button_active("normalize", enabled)
    
    def _on_toggle_auto_gain_meas(self) -> None:
        """Handle toggle auto gain in measurement mode."""
        if self._measurement_mode is None:
            return
        
        self._measurement_mode.state.auto_gain_enabled = not self._measurement_mode.state.auto_gain_enabled
        enabled = self._measurement_mode.state.auto_gain_enabled
        self._display.set_button_active("auto_gain", enabled)
        print(f"[AUTO_GAIN] {'ON' if enabled else 'OFF'}")
    
    # Calibration mode specific handlers
    
    def _on_select_source(self, source: ReferenceSource) -> None:
        """Handle reference source selection."""
        if self._calibration_mode is None:
            return
        
        self._calibration_mode.select_source(source)
        
        # Update button active states (show which source is selected)
        for s in ReferenceSource:
            action_name = f"source_{s.name.lower()}"
            self._display.set_button_active(action_name, source == s)
    
    def _on_toggle_overlay(self) -> None:
        """Handle toggle reference overlay."""
        if self._calibration_mode is None:
            return
        
        visible = self._calibration_mode.toggle_overlay()
        self._display.set_button_active("toggle_overlay", visible)
    
    def _on_toggle_sensitivity(self) -> None:
        """Handle toggle CMOS sensitivity correction."""
        if self._calibration_mode is None:
            return
        
        enabled = self._calibration_mode.toggle_sensitivity_correction()
        self._display.set_button_active("toggle_sensitivity", enabled)
    
    def _on_auto_level(self) -> None:
        """Handle auto-level button - detects rotation, shift, centers spectrum."""
        # This calls the same function as pressing "a" key
        self._on_auto_detect_angle()
    
    def _on_toggle_freeze(self) -> None:
        """Handle toggle spectrum freeze."""
        if self._calibration_mode is None:
            return
        
        frozen = self._calibration_mode.toggle_freeze()
        self._frozen_spectrum = frozen
        
        if frozen and self._last_intensity_pre_sensitivity is not None:
            # Store pre-sensitivity intensity so S toggle applies correctly when frozen
            self._frozen_intensity = self._last_intensity_pre_sensitivity.copy()
        elif not frozen:
            # Clear frozen spectrum when unfreezing
            self._frozen_intensity = None
        
        # Playback icon: active=red (live), inactive=stop (frozen)
        self._display.set_button_active("freeze", not frozen)
        print(f"[FREEZE] Spectrum {'FROZEN' if frozen else 'LIVE'}")
    
    def _on_toggle_averaging(self) -> None:
        """Handle toggle averaging."""
        if self._calibration_mode is None:
            return
        
        enabled = self._calibration_mode.toggle_averaging()
        self._display.set_button_active("toggle_averaging", enabled)
        print(f"[AVG] Averaging {'ON' if enabled else 'OFF'}")
    
    def _on_toggle_auto_gain_cal(self) -> None:
        """Handle toggle auto gain in calibration mode."""
        if self._calibration_mode is None:
            return
        
        self._calibration_mode.state.auto_gain_enabled = not self._calibration_mode.state.auto_gain_enabled
        enabled = self._calibration_mode.state.auto_gain_enabled
        self._display.set_button_active("auto_gain", enabled)
        print(f"[AUTO_GAIN] {'ON' if enabled else 'OFF'}")
    
    def _on_auto_calibrate(self) -> None:
        """Handle auto-calibrate action (correlation-based optimization)."""
        if self._calibration_mode is None or self._last_data is None:
            print("[CAL] No data available for calibration")
            return

        # Debounce: ignore rapid repeat (touch screens)
        now = time.monotonic()
        if now - self._last_auto_calibrate_time < self._AUTO_CAL_DEBOUNCE_SEC:
            return
        self._last_auto_calibrate_time = now

        if not self._frozen_spectrum:
            print("[CAL] Freeze spectrum first (Play button), then click AutoCal")
            return

        # Use raw (pre-sensitivity) measured when S is on so correlation uses ref*sens vs raw (both sensor space)
        measured = (
            self._last_intensity_pre_sensitivity
            if (
                self._calibration_mode.cal_state.sensitivity_correction_enabled
                and self._last_intensity_pre_sensitivity is not None
                and len(self._last_intensity_pre_sensitivity) == len(self._last_data.intensity)
            )
            else self._last_data.intensity
        )
        cal_points = self._calibration_mode.auto_calibrate(
            measured_intensity=measured,
            wavelengths=self._last_data.wavelengths,
            peak_indices=self._last_data.peaks,
        )
        
        if len(cal_points) >= 3:
            print(f"[CAL] Auto-calibration found {len(cal_points)} points (4+ preferred)")
            print("[CAL] Click 'Save' to save, or Freeze+AutoCal again to retry")
            
            # Apply calibration to see result
            pixels = [p for p, w in cal_points]
            wavelengths = [w for p, w in cal_points]
            self._calibration.recalibrate(pixels, wavelengths)
        else:
            print("[CAL] Auto-calibration failed to find enough matches")
    
    def _on_save_calibration(self) -> None:
        """Handle save calibration - saves both wavelength cal and extraction params."""
        saved_something = False
        
        # Save wavelength calibration points (if in calibration mode)
        if self._calibration_mode is not None:
            cal_points = self._calibration_mode.get_calibration_points()
            if len(cal_points) < 4:
                # Use current calibration data if no auto-cal points
                pixels = self._calibration.cal_pixels
                wavelengths = self._calibration.cal_wavelengths
            else:
                pixels = [p for p, w in cal_points]
                wavelengths = [w for p, w in cal_points]
            
            if len(pixels) >= 4:
                if self._calibration.save(pixels, wavelengths):
                    print(f"[CAL] Wavelength calibration saved ({len(pixels)} points)")
                    saved_something = True
                else:
                    print("[CAL] Failed to save wavelength calibration")
        
        # Always save extraction parameters (rotation, y center, perp width)
        self._calibration.save_extraction_params(
            rotation_angle=self._extractor.rotation_angle,
            spectrum_y_center=self._extractor.spectrum_y_center,
            perpendicular_width=self._extractor.perpendicular_width,
        )
        print(f"[CAL] Extraction params saved (angle: {self._extractor.rotation_angle:.2f}°, Y: {self._extractor.spectrum_y_center})")
        saved_something = True
        
        if not saved_something:
            print("[CAL] Nothing to save - need at least 4 calibration points")
    
    def _on_load_calibration(self) -> None:
        """Handle load calibration - load file and sync extractor with saved params."""
        if self._calibration.load():
            self._extractor.set_rotation_angle(self._calibration.rotation_angle)
            self._extractor.set_spectrum_y_center(self._calibration.spectrum_y_center)
            self._extractor.set_perpendicular_width(self._calibration.perpendicular_width)
            print(
                f"[CAL] Calibration loaded (angle: {self._calibration.rotation_angle:.2f}°, "
                f"Y: {self._calibration.spectrum_y_center})"
            )
        else:
            print("[CAL] Failed to load calibration (file may not exist)")
    
    def _on_clear_cal_points(self) -> None:
        """Handle clear calibration points."""
        if self._calibration_mode is not None:
            self._calibration_mode.clear_calibration_points()
        self._display.state.click_points.clear()
        print("[CAL] Points cleared")
    
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
    
    def _on_cycle_preview(self) -> None:
        """Handle preview mode cycling."""
        self._display.cycle_preview_mode()
    
    def _on_cycle_extraction(self) -> None:
        """Handle extraction method cycling."""
        new_method = self._extractor.cycle_method()
        print(f"Extraction Method: {new_method}")
    
    def _on_auto_detect_angle(self) -> None:
        """Handle auto-detect rotation angle and Y center (non-blocking overlay)."""
        if self._autolevel_overlay is not None:
            self._autolevel_overlay = None
            self._display.set_autolevel_overlay(None)
            self._display.set_autolevel_click_dismiss(None)
            print("[AutoLevel] Overlay dismissed")
            return

        if self._last_frame is None:
            print("[AutoLevel] No frame available for angle detection")
            return

        print("[AutoLevel] Detecting spectrum rotation angle and position...")
        angle, y_center, vis_image = self._extractor.detect_angle(self._last_frame, visualize=True)

        self._extractor.set_rotation_angle(angle)
        self._extractor.set_spectrum_y_center(y_center)
        print(f"[AutoLevel] Rotation: {angle:.2f}°  Y Center: {y_center}")
        # AutoLevel only sets values. Click Save to persist.

        if vis_image is not None:
            self._autolevel_overlay = vis_image
            self._display.set_autolevel_overlay(vis_image)
            self._display.set_autolevel_click_dismiss(self._clear_autolevel_overlay)
            print("[AutoLevel] Click or press 'a' to close overlay")

    def _clear_autolevel_overlay(self) -> None:
        """Clear autolevel overlay (called on click or key)."""
        self._autolevel_overlay = None
        self._display.set_autolevel_overlay(None)
        self._display.set_autolevel_click_dismiss(None)
    
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
        
        # Always apply extraction params from calibration file
        self._extractor.set_rotation_angle(self._calibration.rotation_angle)
        self._extractor.set_spectrum_y_center(self._calibration.spectrum_y_center)
        self._extractor.set_perpendicular_width(self._calibration.perpendicular_width)

        self._camera.start()
        self._display.setup_windows()
        
        self._running = True
        self._last_data: Optional[SpectrumData] = None
        
        # Calibration mode: select FL by default
        if self._calibration_mode is not None:
            self._on_select_source(ReferenceSource.FL)
        
        try:
            while self._running:
                frame = self._camera.capture()
                self._last_frame = frame
                
                bit_depth = getattr(self._camera, "bit_depth", self.config.camera.bit_depth)
                max_val = float((1 << bit_depth) - 1)
                extraction_result = self._extractor.extract(frame, max_val=max_val)
                cropped = extraction_result.cropped_frame
                intensity = extraction_result.intensity.astype(np.float32)
                
                # Handle calibration mode freeze - use stored frozen intensity
                if self._calibration_mode is not None and self._frozen_spectrum:
                    if self._frozen_intensity is not None:
                        intensity = self._frozen_intensity.copy()
                
                if self._display.state.hold_peaks:
                    if self._held_intensity is None:
                        self._held_intensity = intensity.copy()
                    else:
                        self._held_intensity = np.maximum(self._held_intensity, intensity)
                    intensity = self._held_intensity
                
                # Mode-specific spectrum processing (skip if frozen)
                if self._calibration_mode is not None and not self._frozen_spectrum:
                    intensity = self._calibration_mode.accumulate_spectrum(intensity)
                elif self._measurement_mode is not None:
                    intensity = self._measurement_mode.process_spectrum(
                        intensity,
                        self._calibration.wavelengths,
                    )

                if self._calibration_mode is not None:
                    self._last_intensity_pre_sensitivity = intensity.copy()
                
                # Apply CMOS sensitivity correction if enabled (calibration mode)
                if self._calibration_mode is not None:
                    intensity = self._calibration_mode.apply_sensitivity_correction(
                        intensity,
                        self._calibration.wavelengths,
                    )

                # Align intensity and wavelengths length (extractor width may differ from calibration width)
                n_wl = len(self._calibration.wavelengths)
                n_int = len(intensity)
                n = min(n_wl, n_int)
                if n < n_int:
                    intensity = intensity[:n]
                wl = self._calibration.wavelengths[:n]

                data = SpectrumData(
                    intensity=intensity,
                    wavelengths=wl,
                    raw_frame=frame,
                    cropped_frame=cropped,
                )
                
                processed = self._pipeline.run(data)
                
                self._last_data = processed
                
                # Handle auto-gain/auto-exposure (95% threshold algorithm)
                self._handle_auto_gain_exposure(processed)
                
                # Update mode overlay
                if self._calibration_mode is not None:
                    overlay = self._calibration_mode.get_overlay(
                        processed.wavelengths,
                        self.config.display.graph_height,
                    )
                    self._display.set_mode_overlay(overlay)
                    
                    # Add sensitivity curve overlay if enabled
                    sensitivity_overlay = self._calibration_mode.get_sensitivity_curve(
                        processed.wavelengths,
                        self.config.display.graph_height,
                    )
                    if sensitivity_overlay is not None:
                        self._display.set_sensitivity_overlay(sensitivity_overlay)
                    else:
                        self._display.set_sensitivity_overlay(None)
                    
                    # Update status display
                    self._display.set_status("Ref", self._calibration_mode.get_current_source_name())
                    self._display.set_status("Pts", str(len(self._calibration_mode.cal_state.calibration_points)))
                    if self._frozen_spectrum:
                        self._display.set_status("Status", "FROZEN")
                    elif self._calibration_mode.state.averaging_enabled:
                        self._display.set_status("Status", f"AVG:{self._calibration_mode.state.accumulated_frames}")
                    else:
                        self._display.set_status("Status", "LIVE")
                elif self._measurement_mode is not None:
                    overlay = self._measurement_mode.get_overlay(
                        processed.wavelengths,
                        self.config.display.graph_height,
                    )
                    self._display.set_mode_overlay(overlay)
                    self._display.set_sensitivity_overlay(None)

                    # Update status display
                    for key, value in self._measurement_mode.get_status().items():
                        self._display.set_status(key, value)
                else:
                    # Legacy mode: update reference spectrum overlay
                    self._display.set_mode_overlay(None)
                    self._display.set_sensitivity_overlay(None)
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
                    spectrum_y_center=self._extractor.spectrum_y_center,
                )
                
                action = self._keyboard.poll()
                if action == Action.SAVE and self._last_data is not None:
                    self._save_snapshot(processed)

                # Exit when user closes window (X button)
                try:
                    if cv2.getWindowProperty(self.config.spectrograph_title, cv2.WND_PROP_VISIBLE) < 1:
                        self._running = False
                except cv2.error:
                    self._running = False
                    
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
