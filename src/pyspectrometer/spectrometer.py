"""Main spectrometer application orchestrator.

Thin orchestrator: creates components, builds ModeContext, delegates
button layout and mode logic to mode classes.
"""

import queue
import threading
import time
from pathlib import Path
from queue import Empty

import cv2
import numpy as np

from .capture.base import CameraInterface
from .capture.picamera import Capture
from .config import Config
from .hardware.led import apply_led_config_from_values
from .core.calibration import Calibration
from .core.mode_context import ModeContext
from .core.reference_spectrum import ReferenceSpectrumManager
from .core.spectrum import SpectrumData
from .display.renderer import DisplayManager
from .export.csv_exporter import (
    CSVExporter,
    build_sensitivity_for_export,
    export_wavelength_mask,
    trim_optional_intensity_row,
    trim_spectrum_data_for_export_min_wavelength,
    trim_waterfall_export_arrays,
)
from .export.graph_export import (
    ColorSciencePdfBundle,
    MeasurementPdfBundle,
    ViewExportRequest,
    export_colorscience_pdf,
    export_measurement_pdf,
    export_view_vector,
)
from .modes.base import BaseMode
from .modes.calibration import CalibrationMode
from .modes.colorscience import ColorMeasurementType, ColorScienceMode
from .modes.measurement import MeasurementMode
from .modes.raman import RamanMode
from .modes.waterfall import WaterfallMode
from .processing.auto_controls import AutoExposureController, AutoGainController
from .processing.extraction import ExtractionMethod, SpectrumExtractor
from .processing.filters import SavitzkyGolayFilter
from .processing.peak_detection import PeakDetector, detect_peaks_in_region
from .processing.pipeline import ProcessingPipeline
from .processing.sensitivity_correction import SensitivityCorrection
from .utils.dialog import prompt_calibrate, prompt_label, prompt_save_file_path


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
        apply_led_config_from_values(
            self.config.hardware.led_pin,
            self.config.hardware.led_pwm_frequency_hz,
        )
        self.mode = mode if mode in self.VALID_MODES else "measurement"
        self.laser_nm = laser_nm

        self._camera = camera or Capture(self.config.camera)
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
            "max": ExtractionMethod.MAX,
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
            frame_black_strip_height=self.config.extraction.frame_black_strip_height,
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

        smoothing = self.config.auto.peak_smoothing_period_sec
        rate_hz = self.config.auto.max_adjust_rate_hz
        self._auto_gain = AutoGainController(
            peak_smoothing_period_sec=smoothing,
            max_adjust_rate_hz=rate_hz,
        )
        self._auto_exposure = AutoExposureController(
            peak_smoothing_period_sec=smoothing,
            max_adjust_rate_hz=rate_hz,
        )
        from .data.reference_loader import set_reference_dirs

        set_reference_dirs(self.config.export.reference_dirs)
        output_dir = self.config.export.output_dir
        if str(output_dir) == ".":
            output_dir = Path("output")
        self._exporter = CSVExporter(output_dir=output_dir)
        self._reference_manager = ReferenceSpectrumManager()
        self._sensitivity = SensitivityCorrection(config=self.config.sensitivity)

        self._mode_instance: BaseMode
        self._calibration_mode: CalibrationMode | None = None
        self._measurement_mode: MeasurementMode | None = None
        self._colorscience_mode: ColorScienceMode | None = None
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
                self._colorscience_mode = ColorScienceMode()
                self._mode_instance = self._colorscience_mode
            case "waterfall":
                self._mode_instance = WaterfallMode()
                self.config.waterfall.enabled = True
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
        ctx.save_waterfall_snapshot = self._save_waterfall_snapshot
        ctx.export_vector_graph = self._export_vector_graph
        ctx.export_pdf_report = self._export_measurement_pdf
        ctx.clear_autolevel_overlay = self._clear_autolevel_overlay
        ctx.auto_calibrate_debounce_sec = 1.5
        ctx.sensitivity_engine = self._sensitivity
        if self.mode == "calibration":
            ctx.sensitivity_correction_enabled = False
        ctx.min_wavelength = self._min_wavelength_for_mode()
        return ctx

    def _min_wavelength_for_mode(self) -> float:
        """CSV wavelength floor (nm) for the active mode; 0 keeps the full calibrated axis."""
        if self.mode in ("measurement", "waterfall"):
            return float(self.config.export.min_wavelength)
        return 0.0

    def _capture_frame(self) -> np.ndarray:
        """Capture frame and update context (blocking; used when not using threaded capture)."""
        frame = self._camera.capture()
        self._ctx.last_frame = frame
        return frame

    def _capture_loop(self) -> None:
        """Run in thread: capture frames and put on queue so main loop does not block on long exposure."""
        while self._ctx.running and self._camera.is_running:
            try:
                frame = self._camera.capture()
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait(frame)
            except Exception:
                if not self._ctx.running:
                    break

    def _process_intensity(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract intensity, apply freeze/hold, mode processing, sensitivity correction. Returns (intensity, cropped)."""
        bit_depth = getattr(self._camera, "bit_depth", self.config.camera.bit_depth)
        max_val = float((1 << bit_depth) - 1)
        extraction_result = self._extractor.extract(frame, max_val=max_val)
        cropped = extraction_result.cropped_frame
        intensity = extraction_result.intensity.astype(np.float32)
        self._ctx.last_raw_extraction_max = float(extraction_result.max_in_roi)

        if self._ctx.frozen_spectrum and self._ctx.frozen_intensity is not None:
            intensity = self._ctx.frozen_intensity.copy()

        if self._display.state.hold_peaks:
            if self._ctx.held_intensity is None:
                self._ctx.held_intensity = intensity.copy()
            else:
                self._ctx.held_intensity = np.maximum(self._ctx.held_intensity, intensity)
            intensity = self._ctx.held_intensity

        if (
            self._calibration_mode is not None
            and not self._ctx.frozen_spectrum
        ):
            intensity = self._calibration_mode.accumulate_spectrum(intensity)
            self._ctx.last_intensity_accumulated = intensity.copy()
        else:
            self._ctx.last_raw_intensity = intensity.copy()
            intensity = self._mode_instance.process_spectrum(
                intensity, self._calibration.wavelengths
            )
            # Grab the post-accumulation, pre-dark/white value the mode stored internally.
            accumulated = getattr(self._mode_instance, "_last_measured_pre_correction", None)
            self._ctx.last_intensity_accumulated = (
                accumulated.copy() if accumulated is not None else self._ctx.last_raw_intensity
            )

        self._ctx.last_intensity_pre_sensitivity = intensity.copy()

        if self._ctx.sensitivity_correction_enabled:
            intensity = self._sensitivity.apply(intensity, self._calibration.wavelengths)

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
        gain = getattr(self._camera, "gain", None)
        exposure_us = getattr(self._camera, "exposure", None)
        pre_sens = self._ctx.last_intensity_pre_sensitivity
        if pre_sens is not None and len(pre_sens) >= n:
            y_axis = np.asarray(pre_sens[:n], dtype=np.float32)
        else:
            y_axis = np.asarray(intensity[:n], dtype=np.float32)
        data = SpectrumData(
            intensity=intensity[:n],
            wavelengths=wl,
            raw_frame=frame,
            cropped_frame=cropped,
            gain=float(gain) if gain is not None else None,
            exposure_us=int(exposure_us) if exposure_us is not None else None,
            y_axis_intensity=y_axis,
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
            spectrum_y_center=self._extractor.spectrum_y_center,
            frozen_spectrum=self._ctx.frozen_spectrum,
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

    _MODE_PREFIX: dict[str, str] = {
        "measurement": "spectrum",
        "raman": "raman",
        "colorscience": "colors",
        "waterfall": "waterfall",
        "calibration": "calibration",
    }

    def _save_snapshot(
        self,
        data: SpectrumData,
        *,
        reference_intensity: np.ndarray | None = None,
        dark_intensity: np.ndarray | None = None,
        white_intensity: np.ndarray | None = None,
        metadata: dict | None = None,
        extra_spectra: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
        measured_raw_intensity: np.ndarray | None = None,
    ) -> None:
        """Save spectrum to CSV and optionally waterfall image.

        All SPD columns (Measured, Dark, White, extra) are written as raw values.
        extra_spectra: list of (column_name, wavelengths, spectrum) tuples, one
        per swatch.  Each swatch spectrum is interpolated to the main wavelength
        grid inside the exporter.
        """
        meta = dict(metadata) if metadata else {}
        if not meta.get("Note"):
            meta["Note"] = prompt_label("Save spectrum", "Label for this measurement (optional):")
        prefix = self._MODE_PREFIX.get(self.mode, "spectrum")
        csv_path = self._exporter.generate_filename(prefix, label=meta.get("Note", ""))
        extra_columns = (
            [(name, (wl, sp)) for name, wl, sp in extra_spectra]
            if extra_spectra else None
        )
        data_e = data
        dark_e = dark_intensity
        white_e = white_intensity
        # Fall back to pre-sensitivity intensity so modes that don't explicitly
        # pass raw data (e.g. calibration) still export uncorrected SPD.
        measured_e = (
            measured_raw_intensity
            if measured_raw_intensity is not None
            else self._ctx.last_intensity_pre_sensitivity
        )
        min_wl = self._ctx.min_wavelength
        if min_wl > 0:
            data_e = trim_spectrum_data_for_export_min_wavelength(data, min_wl)
            n = min(len(data.wavelengths), len(data.intensity))
            mask = export_wavelength_mask(data.wavelengths, n, min_wl)
            if not np.all(mask):
                measured_e = trim_optional_intensity_row(measured_e, n, mask)
                dark_e = trim_optional_intensity_row(dark_intensity, n, mask)
                white_e = trim_optional_intensity_row(white_intensity, n, mask)
        sens_col, sens_meta = build_sensitivity_for_export(
            correction_enabled=self._ctx.sensitivity_correction_enabled,
            engine=self._ctx.sensitivity_engine,
            wavelengths=data_e.wavelengths,
            sens_cfg=self.config.sensitivity,
        )
        meta.update(sens_meta)
        self._exporter.export(
            data_e,
            csv_path,
            reference_intensity=reference_intensity,
            dark_intensity=dark_e,
            white_intensity=white_e,
            metadata=meta,
            extra_columns=extra_columns,
            measured_raw_intensity=measured_e,
            sensitivity_intensity=sens_col,
        )
        time_display = time.strftime(self.config.export.time_format)
        self._display.state.save_message = f"Last Save: {time_display}"
        print(f"Saved: {csv_path}")
        waterfall_img = self._display.get_waterfall_image()
        if waterfall_img is not None:
            cv2.imwrite(str(csv_path.with_suffix(".png")), waterfall_img)

    def _save_waterfall_snapshot(
        self,
        rows: list[np.ndarray],
        wavelengths: np.ndarray,
        *,
        dark_intensity: np.ndarray | None = None,
        white_intensity: np.ndarray | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Save waterfall buffer to CSV (rows = raw measured SPD; dark/white in comments)."""
        meta = dict(metadata) if metadata else {}
        if not meta.get("Note"):
            meta["Note"] = prompt_label("Save waterfall", "Label for this waterfall (optional):")
        csv_path = self._exporter.generate_filename("waterfall", label=meta.get("Note", ""))
        min_wl = self._ctx.min_wavelength
        sens_col, sens_meta = build_sensitivity_for_export(
            correction_enabled=self._ctx.sensitivity_correction_enabled,
            engine=self._ctx.sensitivity_engine,
            wavelengths=wavelengths,
            sens_cfg=self.config.sensitivity,
        )
        meta.update(sens_meta)
        rows_e, wl_e, dark_e, white_e, sens_e, _ = trim_waterfall_export_arrays(
            rows,
            wavelengths,
            min_wl,
            dark_intensity,
            white_intensity,
            sens_col,
        )
        self._exporter.export_waterfall(
            rows_e,
            wl_e,
            csv_path,
            dark_intensity=dark_e,
            white_intensity=white_e,
            sensitivity_intensity=sens_e,
            metadata=meta,
        )
        time_display = time.strftime(self.config.export.time_format)
        self._display.state.save_message = f"Last Save: {time_display}"
        print(f"Saved: {csv_path}")
        waterfall_img = self._display.get_waterfall_image()
        if waterfall_img is not None:
            cv2.imwrite(str(csv_path.with_suffix(".png")), waterfall_img)

    def _trim_export_bundle(
        self,
        data: SpectrumData,
        measured_pre: np.ndarray | None,
        dark: np.ndarray | None,
        white: np.ndarray | None,
        reference: np.ndarray | None,
    ) -> tuple[SpectrumData, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Apply ``min_wavelength`` trim consistently with CSV export."""
        measured_e = (
            measured_pre if measured_pre is not None else self._ctx.last_intensity_pre_sensitivity
        )
        data_e = data
        dark_e = dark
        white_e = white
        ref_e = reference
        min_wl = self._ctx.min_wavelength
        if min_wl > 0:
            data_e = trim_spectrum_data_for_export_min_wavelength(data, min_wl)
            n = min(len(data.wavelengths), len(data.intensity))
            mask = export_wavelength_mask(data.wavelengths, n, min_wl)
            if not np.all(mask):
                measured_e = trim_optional_intensity_row(measured_e, n, mask)
                dark_e = trim_optional_intensity_row(dark, n, mask)
                white_e = trim_optional_intensity_row(white, n, mask)
                ref_e = trim_optional_intensity_row(reference, n, mask)
        return data_e, measured_e, dark_e, white_e, ref_e

    def _export_vector_graph(self) -> None:
        """Measurement mode: export current graph viewport to SVG or EPS."""
        if self.mode != "measurement" or self._measurement_mode is None:
            print("[Export] Vector graph is only available in measurement mode")
            return
        ctx = self._ctx
        if ctx.last_data is None:
            print("[Export] No spectrum data")
            return
        note = prompt_label("Export graph", "Label (optional):")
        default_path = self._exporter.generate_filename("graph", label=note).with_suffix(".svg")
        path_str = prompt_save_file_path(
            "Export graph (SVG or EPS)",
            ".svg",
            [("SVG vector", "*.svg"), ("EPS vector", "*.eps"), ("All files", "*.*")],
            initialdir=str(default_path.parent),
            initialfile=default_path.name,
        )
        if not path_str:
            return
        mm = self._measurement_mode
        data = ctx.last_data
        dark = mm.meas_state.dark_spectrum
        white = mm.meas_state.white_spectrum
        ref = ctx.display.state.reference_spectrum
        data_e, pre_e, _, _, ref_e = self._trim_export_bundle(
            data,
            ctx.last_intensity_pre_sensitivity,
            dark,
            white,
            ref,
        )
        vp = ctx.display.viewport
        wl = data_e.wavelengths
        sens_disp = None
        eng = ctx.sensitivity_engine
        if ctx.sensitivity_correction_enabled and eng is not None:
            curve = eng.get_curve_for_display(np.asarray(wl, dtype=np.float64), 256)
            if curve is not None:
                sens_disp = curve[0]
        ref_name = ctx.display.state.reference_name
        if ref_name in ("None", ""):
            ref_name = "Reference"
        meta = mm.snapshot_metadata(ctx)
        req = ViewExportRequest(
            path=Path(path_str),
            wavelengths=wl,
            intensity=data_e.intensity,
            x_axis_label=getattr(data_e, "x_axis_label", "Wavelength (nm)"),
            x_start=vp.x_start,
            x_end=vp.x_end,
            y_min=vp.y_min,
            y_max=vp.y_max,
            peaks=list(data_e.peaks),
            metadata=meta,
            reference=ref_e,
            reference_name=ref_name,
            sensitivity_display=sens_disp,
        )
        out = export_view_vector(req)
        print(f"[Export] Vector graph saved: {out}")

    def _export_colorscience_vector(self) -> None:
        """Color Science: viewport export with XYZ/L*a*b* and sRGB patch."""
        ctx = self._ctx
        cm = self._colorscience_mode
        if cm is None or ctx.last_data is None:
            print("[Export] No spectrum data")
            return
        note = prompt_label("Export graph", "Label (optional):")
        default_path = self._exporter.generate_filename("graph", label=note).with_suffix(".svg")
        path_str = prompt_save_file_path(
            "Export graph (SVG or EPS)",
            ".svg",
            [("SVG vector", "*.svg"), ("EPS vector", "*.eps"), ("All files", "*.*")],
            initialdir=str(default_path.parent),
            initialfile=default_path.name,
        )
        if not path_str:
            return
        data = ctx.last_data
        dark = cm.color_state.dark_spectrum
        is_illum = cm.color_state.measurement_type == ColorMeasurementType.ILLUMINATION
        white = None if is_illum else cm.color_state.white_spectrum
        data_e, _, _, _, _ = self._trim_export_bundle(
            data,
            ctx.last_intensity_pre_sensitivity,
            dark,
            white,
            None,
        )
        vp = ctx.display.viewport
        wl = data_e.wavelengths
        sens_disp = None
        eng = ctx.sensitivity_engine
        if ctx.sensitivity_correction_enabled and eng is not None:
            curve = eng.get_curve_for_display(np.asarray(wl, dtype=np.float64), 256)
            if curve is not None:
                sens_disp = curve[0]
        meta = cm.snapshot_metadata(ctx)
        xyz_t: tuple[float, float, float] | None = None
        lab_t: tuple[float, float, float] | None = None
        xyz_lab = cm._compute_xyz_lab(data_e.intensity, data_e.wavelengths)
        if xyz_lab is not None:
            (x0, y0, z0), (l0, a0, b0) = xyz_lab
            xyz_t = (x0, y0, z0)
            lab_t = (l0, a0, b0)
        cat_w = cm._display_cat_white_point(data_e.wavelengths)
        req = ViewExportRequest(
            path=Path(path_str),
            wavelengths=wl,
            intensity=data_e.intensity,
            x_axis_label=getattr(data_e, "x_axis_label", "Wavelength (nm)"),
            x_start=vp.x_start,
            x_end=vp.x_end,
            y_min=vp.y_min,
            y_max=vp.y_max,
            peaks=list(data_e.peaks),
            metadata=meta,
            reference=None,
            reference_name="Reference",
            sensitivity_display=sens_disp,
            xyz=xyz_t,
            cie_lab=lab_t,
            illuminant_xyz=cat_w,
        )
        out = export_view_vector(req)
        print(f"[Export] Vector graph saved: {out}")

    def _export_colorscience_pdf(self) -> None:
        """Color Science: metadata + one page per swatch (400–700 nm corrected spectrum)."""
        ctx = self._ctx
        cm = self._colorscience_mode
        if cm is None:
            return
        if not cm.color_state.swatches:
            print("[Export] No swatches — use Add to store colors, then export PDF")
            return
        note = prompt_label("Export PDF", "Label for this color science export (optional):")
        pdf_path = self._exporter.generate_filename("colors", label=note).with_suffix(".pdf")
        wl_s = (
            ctx.last_data.wavelengths
            if ctx.last_data is not None
            else np.asarray(self._calibration.wavelengths, dtype=np.float64)
        )
        sens_col, sens_meta = build_sensitivity_for_export(
            correction_enabled=ctx.sensitivity_correction_enabled,
            engine=ctx.sensitivity_engine,
            wavelengths=wl_s,
            sens_cfg=self.config.sensitivity,
        )
        meta = cm.snapshot_metadata(ctx) if ctx.last_data is not None else {
            "Mode": "Color Science",
            "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta["Note"] = note
        bundle = ColorSciencePdfBundle(
            path=pdf_path,
            metadata=meta,
            sensitivity_meta=dict(sens_meta),
            swatches=list(cm.color_state.swatches),
        )
        out = export_colorscience_pdf(bundle)
        time_display = time.strftime(self.config.export.time_format)
        self._display.state.save_message = f"Last PDF: {time_display}"
        print(f"[Export] PDF report saved: {out}")

    def _export_measurement_pdf(self) -> None:
        """Measurement mode: multi-page PDF (metadata, references, sensitivity, peaks)."""
        if self.mode == "colorscience" and self._colorscience_mode is not None:
            self._export_colorscience_pdf()
            return
        if self.mode != "measurement" or self._measurement_mode is None:
            print("[Export] PDF report is only available in measurement or color science mode")
            return
        ctx = self._ctx
        if ctx.last_data is None:
            print("[Export] No spectrum data")
            return
        note = prompt_label("Export PDF", "Label for this measurement (optional):")
        pdf_path = self._exporter.generate_filename("spectrum", label=note).with_suffix(".pdf")
        mm = self._measurement_mode
        data = ctx.last_data
        dark = mm.meas_state.dark_spectrum
        white = mm.meas_state.white_spectrum
        ref = ctx.display.state.reference_spectrum
        data_e, pre_e, d_e, w_e, ref_e = self._trim_export_bundle(
            data,
            ctx.last_intensity_pre_sensitivity,
            dark,
            white,
            ref,
        )
        if pre_e is None or len(pre_e) == 0:
            print("[Export] No pre-sensitivity spectrum available for PDF")
            return
        sens_col, sens_meta = build_sensitivity_for_export(
            correction_enabled=ctx.sensitivity_correction_enabled,
            engine=ctx.sensitivity_engine,
            wavelengths=data_e.wavelengths,
            sens_cfg=self.config.sensitivity,
        )
        ref_name = ctx.display.state.reference_name
        if ref_name in ("None", ""):
            ref_name = "Reference"
        meta = mm.snapshot_metadata(ctx)
        meta["Note"] = note
        overlay_series = mm.overlay_export_columns(ctx, data_e.wavelengths)
        bundle = MeasurementPdfBundle(
            path=pdf_path,
            wavelengths=data_e.wavelengths,
            measured_pre_sensitivity=np.asarray(pre_e, dtype=np.float64),
            measured_corrected=np.asarray(data_e.intensity, dtype=np.float64),
            dark=d_e,
            white=w_e,
            reference=ref_e,
            reference_name=ref_name,
            sensitivity=sens_col,
            sensitivity_meta=dict(sens_meta),
            peaks=list(data_e.peaks),
            metadata=meta,
            x_axis_label=getattr(data_e, "x_axis_label", "Wavelength (nm)"),
            overlay_series=overlay_series or None,
        )
        out = export_measurement_pdf(bundle)
        time_display = time.strftime(self.config.export.time_format)
        self._display.state.save_message = f"Last PDF: {time_display}"
        print(f"[Export] PDF report saved: {out}")

    def run(self) -> None:
        """Run the main loop."""
        mode_names = {
            "calibration": "Calibration",
            "measurement": "Measurement",
            "raman": "Raman",
            "colorscience": "Color Science",
            "waterfall": "Waterfall",
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

        # Threaded capture so long exposure does not freeze the event loop
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        last_frame_received_at: float | None = None
        last_duration_sec: float = 1.0 / 30.0
        capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        capture_thread.start()

        try:
            while self._ctx.running:
                frame = None
                if self._ctx.auto_gain_enabled or self._ctx.auto_exposure_enabled:
                    # Blocking capture so the frame we react to matches the exposure/gain we set.
                    frame = self._capture_frame()
                else:
                    try:
                        frame = self._frame_queue.get_nowait()
                    except Empty:
                        pass

                if frame is not None and frame.size > 0:
                    self._ctx.last_frame = frame
                    intensity, cropped = self._process_intensity(frame)
                    processed = self._run_pipeline(frame, intensity, cropped)
                    processed = self._mode_instance.transform_spectrum_data(processed)
                    processed = self._expand_peaks_with_regions(processed)
                    self._ctx.last_data = processed
                    if self.mode == "waterfall":
                        buf = getattr(self._ctx, "waterfall_raw_buffer", None)
                        if buf is not None and self._ctx.last_raw_intensity is not None:
                            buf.append(self._ctx.last_raw_intensity.copy())
                        rec = getattr(self._ctx, "waterfall_rec_writer", None)
                        if rec is not None and self._ctx.last_raw_intensity is not None:
                            rec.write_row(self._ctx.last_raw_intensity)
                    # Use raw extraction max for AE/AG so saturated shows as 1.0 (not white-ref ~0.8).
                    ae_data = SpectrumData(
                        intensity=np.array([self._ctx.last_raw_extraction_max], dtype=np.float32),
                        wavelengths=np.array([0.0]),
                        exposure_us=processed.exposure_us,
                        gain=processed.gain,
                    )
                    self._ctx.handle_auto_gain_exposure(ae_data)
                    last_frame_received_at = time.time()
                    last_duration_sec = (
                        (processed.exposure_us / 1e6)
                        if (processed.exposure_us is not None and processed.exposure_us > 0)
                        else (1.0 / 30.0)
                    )

                processed = self._ctx.last_data
                progress = 0.0
                if last_frame_received_at is not None and last_duration_sec > 0:
                    elapsed = time.time() - last_frame_received_at
                    progress = min(1.0, max(0.0, elapsed / last_duration_sec))
                self._display.set_capture_progress(progress, last_duration_sec)

                if processed is not None:
                    self._mode_instance.update_display(
                        self._ctx, processed, self.config.display.graph_height
                    )
                    self._render_frame(processed)  # Redraw every iteration so pie updates

                key = cv2.waitKey(1)
                if key != -1:
                    self._display.handle_key(chr(key & 0xFF))
                self._check_window_closed()
        finally:
            self._camera.stop()
            try:
                capture_thread.join(timeout=2.0)
            except Exception:
                pass
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
