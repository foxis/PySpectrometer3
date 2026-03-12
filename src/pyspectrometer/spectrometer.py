"""Main spectrometer application orchestrator.

Thin orchestrator: creates components, builds ModeContext, delegates
button layout and mode logic to mode classes.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from .capture.base import CameraInterface
from .capture.picamera import Capture
from .config import Config
from .core.calibration import Calibration
from .core.mode_context import ModeContext
from .core.reference_spectrum import ReferenceSpectrumManager
from .core.spectrum import SpectrumData
from .display.renderer import DisplayManager
from .export.csv_exporter import CSVExporter
from .modes.base import BaseMode
from .modes.calibration import CalibrationMode
from .modes.colorscience import ColorScienceMode
from .modes.measurement import MeasurementMode
from .modes.raman import RamanMode
from .modes.waterfall import WaterfallMode
from .processing.auto_controls import AutoExposureController, AutoGainController
from .processing.extraction import ExtractionMethod, SpectrumExtractor
from .processing.filters import SavitzkyGolayFilter
from .processing.peak_detection import PeakDetector, detect_peaks_in_region
from .processing.pipeline import ProcessingPipeline
from .utils.dialog import prompt_calibrate


class Spectrometer:
    """Thin orchestrator: wires components, passes context to mode, runs loop."""

    VALID_MODES = ("calibration", "measurement", "raman", "colorscience", "waterfall")

    def __init__(
        self,
        config: Config | None = None,
        camera: CameraInterface | None = None,
        mode: str = "measurement",
        laser_nm: float = 785.0,
        load_calibration: bool = True,
        config_path: Path | None = None,
    ):
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
            config=self.config,
            config_path=config_path,
            height=self.config.camera.frame_height,
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
            ExtractionMethod.WEIGHTED_SUM,
        )
        spectrum_y = (
            self.config.extraction.spectrum_y_center or self.config.camera.frame_height // 2
        )
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
        self._savgol_filter.enabled = False
        self._peak_detector = PeakDetector(
            min_distance=self.config.processing.peak_min_distance,
            threshold=self.config.processing.peak_threshold,
            max_count=self.config.processing.peak_max_count,
            min_distance_min=self.config.processing.peak_min_distance_min,
            min_distance_max=self.config.processing.peak_min_distance_max,
            threshold_min=self.config.processing.peak_threshold_min,
            threshold_max=self.config.processing.peak_threshold_max,
            max_count_min=self.config.processing.peak_max_count_min,
            max_count_max=self.config.processing.peak_max_count_max,
        )
        self._pipeline = ProcessingPipeline([self._savgol_filter, self._peak_detector])

        self._auto_gain = AutoGainController()
        self._auto_exposure = AutoExposureController()
        from .data.reference_loader import set_reference_dirs

        set_reference_dirs(self.config.export.reference_dirs)
        output_dir = self.config.export.output_dir
        if str(output_dir) == ".":
            output_dir = Path("output")
        self._exporter = CSVExporter(output_dir=output_dir)
        self._reference_manager = ReferenceSpectrumManager()

        self._mode_instance: BaseMode
        self._calibration_mode: CalibrationMode | None = None
        self._measurement_mode: MeasurementMode | None = None
        match mode:
            case "calibration":
                self._calibration_mode = CalibrationMode()
                self._mode_instance = self._calibration_mode
            case "measurement":
                self._measurement_mode = MeasurementMode()
                self._mode_instance = self._measurement_mode
            case "raman":
                self._mode_instance = RamanMode(laser_nm=self.laser_nm)
            case "colorscience":
                self._mode_instance = ColorScienceMode()
            case "waterfall":
                self._mode_instance = WaterfallMode()
                self.config.display.waterfall_enabled = True
            case _:
                self._measurement_mode = MeasurementMode()
                self._mode_instance = self._measurement_mode

        self._display = DisplayManager(
            self.config, self._calibration, mode=mode, mode_instance=self._mode_instance
        )
        self._ctx = self._build_context()
        self._mode_instance.setup(self._ctx)

    def _build_context(self) -> ModeContext:
        """Build mode context with services and callbacks."""
        ctx = ModeContext(
            camera=self._camera,
            calibration=self._calibration,
            display=self._display,
            exporter=self._exporter,
            extractor=self._extractor,
            auto_gain=self._auto_gain,
            auto_exposure=self._auto_exposure,
            reference_manager=self._reference_manager,
        )
        ctx.quit_app = lambda: setattr(ctx, "running", False)
        ctx.save_snapshot = self._save_snapshot
        ctx.clear_autolevel_overlay = self._clear_autolevel_overlay
        ctx.auto_calibrate_debounce_sec = 1.5
        return ctx

    def _capture_frame(self) -> np.ndarray:
        """Capture frame and update context."""
        frame = self._camera.capture()
        self._ctx.last_frame = frame
        return frame

    def _process_intensity(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract intensity, apply freeze/hold, mode processing, sensitivity correction. Returns (intensity, cropped)."""
        bit_depth = getattr(self._camera, "bit_depth", self.config.camera.bit_depth)
        max_val = float((1 << bit_depth) - 1)
        extraction_result = self._extractor.extract(frame, max_val=max_val)
        cropped = extraction_result.cropped_frame
        intensity = extraction_result.intensity.astype(np.float32)

        if self._calibration_mode is not None and self._ctx.frozen_spectrum:
            if self._ctx.frozen_intensity is not None:
                intensity = self._ctx.frozen_intensity.copy()

        if self._display.state.hold_peaks:
            if self._ctx.held_intensity is None:
                self._ctx.held_intensity = intensity.copy()
            else:
                self._ctx.held_intensity = np.maximum(self._ctx.held_intensity, intensity)
            intensity = self._ctx.held_intensity

        if self._calibration_mode is not None and not self._ctx.frozen_spectrum:
            intensity = self._calibration_mode.accumulate_spectrum(intensity)
        else:
            self._ctx.last_raw_intensity = intensity.copy()
            intensity = self._mode_instance.process_spectrum(
                intensity, self._calibration.wavelengths
            )

        if self._calibration_mode is not None:
            self._ctx.last_intensity_pre_sensitivity = intensity.copy()

        if self._calibration_mode is not None:
            intensity = self._calibration_mode.apply_sensitivity_correction(
                intensity, self._calibration.wavelengths
            )

        n_wl = len(self._calibration.wavelengths)
        n_int = len(intensity)
        n = min(n_wl, n_int)
        if n < n_int:
            intensity = intensity[:n]
        return intensity, cropped

    def _run_pipeline(
        self, frame: np.ndarray, intensity: np.ndarray, cropped: np.ndarray
    ) -> SpectrumData:
        """Build SpectrumData and run processing pipeline."""
        n = min(len(self._calibration.wavelengths), len(intensity))
        wl = self._calibration.wavelengths[:n]
        data = SpectrumData(
            intensity=intensity[:n],
            wavelengths=wl,
            raw_frame=frame,
            cropped_frame=cropped,
        )
        return self._pipeline.run(data)

    def _expand_peaks_with_regions(self, processed: SpectrumData) -> SpectrumData:
        """Add peaks from click-to-include region, merged with existing peaks."""
        region = self._display.state.peak_include_region
        if region is None:
            return processed

        intensity = processed.intensity.astype(np.float64)
        threshold = self._peak_detector.threshold / 100.0
        min_dist = self._peak_detector.min_distance

        seen: set[int] = {p.index for p in processed.peaks}
        merged = list(processed.peaks)

        center, half_width = region
        region_peaks = detect_peaks_in_region(
            intensity,
            processed.wavelengths,
            center,
            half_width,
            threshold=threshold,
            min_dist=min_dist,
        )
        for p in region_peaks:
            if p.index not in seen:
                seen.add(p.index)
                merged.append(p)

        return processed.with_peaks(merged)

    def _render_frame(self, processed: SpectrumData) -> None:
        """Render spectrum display."""
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

    def _check_window_closed(self) -> None:
        """Check if window was closed by user and stop if so."""
        try:
            title = (
                self.config.waterfall_title
                if self.mode == "waterfall"
                else self.config.spectrograph_title
            )
            visible = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
            if visible < 1:  # 0 or -1 when closed
                self._ctx.running = False
        except Exception:
            self._ctx.running = False

    def _clear_autolevel_overlay(self) -> None:
        """Clear autolevel overlay (used by calibration mode)."""
        self._display.set_autolevel_overlay(None)
        self._display.set_autolevel_click_dismiss(None)

    def _save_snapshot(
        self,
        data: SpectrumData,
        *,
        reference_intensity: np.ndarray | None = None,
        dark_intensity: np.ndarray | None = None,
        white_intensity: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Save spectrum to CSV and optionally waterfall image."""
        csv_path = self._exporter.generate_filename("Spectrum")
        self._exporter.export(
            data,
            csv_path,
            reference_intensity=reference_intensity,
            dark_intensity=dark_intensity,
            white_intensity=white_intensity,
            metadata=metadata,
        )
        time_display = time.strftime(self.config.export.time_format)
        self._display.state.save_message = f"Last Save: {time_display}"
        print(f"Saved: {csv_path}")
        waterfall_img = self._display.get_waterfall_image()
        if waterfall_img is not None:
            timestamp = time.strftime(self.config.export.timestamp_format)
            cv2.imwrite(str(Path(f"waterfall-{timestamp}.png")), waterfall_img)

    def run(self) -> None:
        """Run the main loop."""
        mode_names = {
            "calibration": "Calibration",
            "measurement": "Measurement",
            "raman": "Raman",
            "colorscience": "Color Science",
        }
        print(f"Starting PySpectrometer 3 - {mode_names.get(self.mode, self.mode)} Mode...")

        result = self._calibration.load()
        self._extractor.set_rotation_angle(self._calibration.rotation_angle)
        self._extractor.set_spectrum_y_center(self._calibration.spectrum_y_center)
        self._extractor.set_perpendicular_width(self._calibration.perpendicular_width)

        if self.mode != "calibration" and result.order == 0:
            prompt_calibrate()

        self._camera.start()
        self._display.setup_windows()
        self._ctx.running = True
        self._ctx.last_data = None
        self._mode_instance.on_start(self._ctx)

        try:
            while self._ctx.running:
                frame = self._capture_frame()
                intensity, cropped = self._process_intensity(frame)
                processed = self._run_pipeline(frame, intensity, cropped)
                processed = self._mode_instance.transform_spectrum_data(processed)
                processed = self._expand_peaks_with_regions(processed)
                self._ctx.last_data = processed
                self._ctx.handle_auto_gain_exposure(processed)
                self._mode_instance.update_display(
                    self._ctx, processed, self.config.display.graph_height
                )
                self._render_frame(processed)
                key = cv2.waitKey(1)
                if key != -1:
                    self._display.handle_key(chr(key & 0xFF))
                self._check_window_closed()
        finally:
            self._camera.stop()
            self._display.destroy()
            print("PySpectrometer 3 stopped.")

    @property
    def camera(self) -> CameraInterface:
        return self._camera

    @property
    def calibration(self) -> Calibration:
        return self._calibration

    @property
    def pipeline(self) -> ProcessingPipeline:
        return self._pipeline

    @property
    def display(self) -> DisplayManager:
        return self._display
