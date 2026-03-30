"""Camera-less spectrometer loop for CSV viewer mode."""

import time
from pathlib import Path

import cv2
import numpy as np

from ..bootstrap import build_reference_file_loader
from ..config import Config
from ..core.calibration import Calibration
from ..core.mode_context import ModeContext
from ..core.spectrum import SpectrumData
from ..display.renderer import DisplayManager
from ..export.csv_exporter import (
    CSVExporter,
    build_sensitivity_for_export,
    export_wavelength_mask,
    trim_optional_intensity_row,
    trim_spectrum_data_for_export_min_wavelength,
)
from ..export.graph_export import MeasurementPdfBundle, export_measurement_pdf
from ..processing.auto_controls import AutoExposureController, AutoGainController
from ..processing.extraction import ExtractionMethod, SpectrumExtractor
from ..processing.filters import SavitzkyGolayFilter
from ..processing.peak_detection import PeakDetector, detect_peaks_in_region
from ..processing.pipeline import ProcessingPipeline
from ..processing.sensitivity_correction import SensitivityCorrection
from ..utils.dialog import prompt_label
from .loader import LoadedCsv, load_csv, make_empty
from .mode import CsvViewerMode
from .stubs import NullCamera


def prompt_csv_path(default_dir: Path | None = None) -> Path | None:
    """Show a file-open dialog and return the chosen CSV path, or None if cancelled."""
    import tkinter as tk
    from tkinter import filedialog

    initial = str(default_dir) if (default_dir and default_dir.exists()) else ""
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes("-topmost", True)
    path_str = filedialog.askopenfilename(
        title="Open spectrum CSV",
        initialdir=initial or None,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        parent=root,
    )
    root.destroy()
    return Path(path_str) if path_str else None


def _interpolate_intensity_to_wavelengths(loaded: LoadedCsv, wl_target: np.ndarray) -> np.ndarray:
    """Resample loaded spectrum intensity onto *wl_target* (nm)."""
    wl_src = np.asarray(loaded.wavelengths, dtype=np.float64).ravel()
    y_src = np.asarray(loaded.intensity, dtype=np.float64).ravel()
    n_t = len(wl_target)
    m = min(len(wl_src), len(y_src))
    if m < 2 or n_t == 0:
        return np.zeros(n_t, dtype=np.float32)
    wl_src = wl_src[:m]
    y_src = y_src[:m]
    order = np.argsort(wl_src)
    wl_s = wl_src[order]
    y_s = y_src[order]
    wl_tgt = np.asarray(wl_target, dtype=np.float64).ravel()
    y_new = np.interp(wl_tgt, wl_s, y_s, left=float(y_s[0]), right=float(y_s[-1]))
    return y_new.astype(np.float32)


class CsvViewerSpectrometer:
    """Run the display loop for a loaded spectrum CSV without a camera.

    If *csv_path* is None a file-open dialog is shown immediately so the
    application can be launched without a command-line argument.
    """

    WINDOW_TITLE = "PySpectrometer 3 - CSV Viewer"

    def __init__(
        self,
        csv_path: Path | None = None,
        config: Config | None = None,
        config_path: Path | None = None,
    ) -> None:
        self.config = config or Config()
        self.config.apply_csv_viewer_preset()
        # Align the window title so _check_window_closed matches the actual OpenCV window.
        self.config.spectrograph_title = self.WINDOW_TITLE
        self._config_path = config_path

        self._csv_path: Path | None = csv_path
        loaded = load_csv(csv_path) if csv_path is not None else make_empty()
        n = len(loaded.wavelengths)

        self._camera = NullCamera(
            frame_width=n,
            frame_height=self.config.camera.frame_height,
        )
        self._calibration = self._build_calibration(loaded)
        self._extractor = SpectrumExtractor(
            frame_width=n,
            frame_height=self.config.camera.frame_height,
            method=ExtractionMethod.MEDIAN,
        )
        self._savgol = SavitzkyGolayFilter(
            window_size=self.config.processing.savgol_window,
            poly_order=self.config.processing.savgol_poly,
        )
        self._savgol.enabled = False
        self._peak_detector = PeakDetector(
            min_distance=self.config.processing.peak_min_distance,
            threshold=self.config.processing.peak_threshold,
            max_count=self.config.processing.peak_max_count,
        )
        self._pipeline = ProcessingPipeline([self._savgol, self._peak_detector])
        output_dir = self.config.export.output_dir
        if str(output_dir) == ".":
            output_dir = Path("output")
        self._exporter = CSVExporter(output_dir=output_dir)
        self._auto_gain = AutoGainController()
        self._auto_exposure = AutoExposureController()
        self._sensitivity = SensitivityCorrection(config=self.config.sensitivity)

        self._mode = CsvViewerMode(loaded)
        self._display = DisplayManager(
            self.config,
            self._calibration,
            mode="measurement",
            mode_instance=self._mode,
        )
        self._ctx = self._build_context()
        self._mode.setup(self._ctx)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Display loop — feeds static data each frame, handles UI events."""
        self._camera.start()
        self._display.setup_windows()
        self._ctx.running = True
        self._mode.on_start(self._ctx)
        if self._csv_path is not None:
            print(f"[CSV Viewer] {self._csv_path.name}")

        try:
            while self._ctx.running:
                loaded = self._mode.loaded or make_empty()
                processed = self._build_spectrum(loaded)
                processed = self._mode.transform_spectrum_data(processed)
                processed = self._expand_peaks(processed)
                self._ctx.last_data = processed
                self._mode.update_display(self._ctx, processed, self.config.display.graph_height)
                self._render(processed)

                if getattr(self._ctx, "pending_csv_reload", False):
                    self._ctx.pending_csv_reload = False  # type: ignore[attr-defined]
                    self._prompt_reload()

                if getattr(self._ctx, "pending_csv_overlay", False):
                    self._ctx.pending_csv_overlay = False  # type: ignore[attr-defined]
                    self._prompt_overlay_load()

                key = cv2.waitKey(30)
                if key != -1:
                    self._display.handle_key(chr(key & 0xFF))
                self._check_window_closed()
        finally:
            self._camera.stop()
            self._display.destroy()
            print("[CSV Viewer] stopped.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_calibration(self, loaded: LoadedCsv) -> Calibration:
        cal = Calibration(
            width=len(loaded.wavelengths),
            config=self.config,
            config_path=self._config_path,
            height=self.config.camera.frame_height,
        )
        cal.seed_wavelengths(loaded.wavelengths)
        return cal

    def _build_context(self) -> ModeContext:
        ref_loader = build_reference_file_loader(self.config)
        ctx = ModeContext(
            camera=self._camera,
            calibration=self._calibration,
            display=self._display,
            exporter=self._exporter,
            extractor=self._extractor,
            auto_gain=self._auto_gain,
            auto_exposure=self._auto_exposure,
            reference_file_loader=ref_loader,
        )
        ctx.quit_app = lambda: setattr(ctx, "running", False)
        ctx.save_snapshot = self._save_snapshot
        ctx.export_pdf_report = self._export_csv_viewer_pdf
        ctx.sensitivity_engine = self._sensitivity
        ctx.sensitivity_correction_enabled = True
        ctx.pending_csv_reload = False  # type: ignore[attr-defined]
        ctx.pending_csv_overlay = False  # type: ignore[attr-defined]
        return ctx

    def _build_spectrum(self, loaded: LoadedCsv) -> SpectrumData:
        wl = self._calibration.wavelengths
        n = min(len(wl), len(loaded.intensity))
        i_refs = self._mode.process_spectrum(
            np.asarray(loaded.intensity[:n], dtype=np.float32), wl[:n]
        )
        i_disp = np.asarray(i_refs, dtype=np.float32).copy()
        if self._ctx.sensitivity_correction_enabled:
            i_disp = self._sensitivity.apply(i_disp, wl[:n])
        data = SpectrumData(
            intensity=i_disp[:n],
            wavelengths=wl[:n],
            y_axis_intensity=np.asarray(i_refs[:n], dtype=np.float32),
        )
        return self._pipeline.run(data)

    def _expand_peaks(self, processed: SpectrumData) -> SpectrumData:
        region = self._display.state.peak_include_region
        if region is None:
            return processed
        intensity = processed.intensity.astype(np.float64)
        threshold = self._peak_detector.threshold / 100.0
        min_dist = self._peak_detector.min_distance
        seen = {p.index for p in processed.peaks}
        merged = list(processed.peaks)
        center, half_width = region
        region_peaks = detect_peaks_in_region(
            intensity, processed.wavelengths, center, half_width,
            threshold=threshold, min_dist=min_dist,
        )
        for p in region_peaks:
            if p.index not in seen:
                seen.add(p.index)
                merged.append(p)
        return processed.with_peaks(merged)

    def _render(self, processed: SpectrumData) -> None:
        self._display.render(
            processed,
            camera_gain=self._camera.gain,
            savgol_poly=self._savgol.poly_order,
            peak_min_dist=self._peak_detector.min_distance,
            peak_threshold=self._peak_detector.threshold,
            extraction_method=str(self._extractor.method),
            spectrum_y_center=self._extractor.spectrum_y_center,
            frozen_spectrum=False,
        )

    def _check_window_closed(self) -> None:
        try:
            visible = cv2.getWindowProperty(self.config.spectrograph_title, cv2.WND_PROP_VISIBLE)
            if visible < 1:
                self._ctx.running = False
        except Exception:
            self._ctx.running = False

    def _prompt_reload(self) -> None:
        """Ask user for a new CSV path and swap the loaded data."""
        output_dir = self.config.export.output_dir
        default_dir = (
            self._csv_path.parent
            if self._csv_path is not None
            else (output_dir if str(output_dir) != "." else Path("output"))
        )
        path = prompt_csv_path(default_dir)
        if path is None:
            return
        try:
            loaded = load_csv(path)
        except Exception as exc:
            print(f"[CSV Viewer] Failed to load {path}: {exc}")
            return
        self._csv_path = path
        self._calibration.seed_wavelengths(loaded.wavelengths)
        self._mode.reload(loaded, self._ctx)
        print(f"[CSV Viewer] Loaded {path.name}")

    def _prompt_overlay_load(self) -> None:
        """Add a spectrum from disk as a Load+ overlay (same processing as primary trace)."""
        output_dir = self.config.export.output_dir
        default_dir = (
            self._csv_path.parent
            if self._csv_path is not None
            else (output_dir if str(output_dir) != "." else Path("output"))
        )
        path = prompt_csv_path(default_dir)
        if path is None:
            return
        try:
            loaded = load_csv(path)
        except Exception as exc:
            print(f"[CSV Viewer] Failed to load overlay {path}: {exc}")
            return
        self._mode.append_overlay_from_loaded(loaded, path.stem)
        print(f"[CSV Viewer] Overlay+ {path.name}")

    def _save_snapshot(
        self,
        data: SpectrumData,
        *,
        dark_intensity: np.ndarray | None = None,
        white_intensity: np.ndarray | None = None,
        metadata: dict | None = None,
        **_kwargs: object,
    ) -> None:
        meta = dict(metadata) if metadata else {}
        if not meta.get("Note"):
            meta["Note"] = prompt_label("Save spectrum", "Label for this measurement (optional):")
        csv_path = self._exporter.generate_filename("spectrum", label=meta.get("Note", ""))
        min_wl = self._ctx.min_wavelength
        data_e = data
        dark_e = dark_intensity
        white_e = white_intensity
        if min_wl > 0:
            data_e = trim_spectrum_data_for_export_min_wavelength(data, min_wl)
            n = min(len(data.wavelengths), len(data.intensity))
            mask = export_wavelength_mask(data.wavelengths, n, min_wl)
            if not np.all(mask):
                dark_e = trim_optional_intensity_row(dark_intensity, n, mask)
                white_e = trim_optional_intensity_row(white_intensity, n, mask)
        sens_col, sens_meta = build_sensitivity_for_export(
            correction_enabled=self._ctx.sensitivity_correction_enabled,
            engine=self._sensitivity,
            wavelengths=data_e.wavelengths,
            sens_cfg=self.config.sensitivity,
        )
        meta.update(sens_meta)
        extra_cols = self._mode.overlay_export_columns(self._ctx, data_e.wavelengths)
        self._exporter.export(
            data_e,
            csv_path,
            dark_intensity=dark_e,
            white_intensity=white_e,
            metadata=meta,
            sensitivity_intensity=sens_col,
            extra_columns=extra_cols or None,
        )
        time_display = time.strftime(self.config.export.time_format)
        self._display.state.save_message = f"Last Save: {time_display}"
        print(f"[CSV Viewer] Saved: {csv_path}")

    def _export_csv_viewer_pdf(self) -> None:
        """Write multi-page PDF beside CSV exports (same dated folder, ``.pdf`` suffix)."""
        ctx = self._ctx
        if ctx.last_data is None:
            print("[CSV Viewer] No data for PDF export")
            return
        note = prompt_label("Export PDF", "Label for this measurement (optional):")
        pdf_path = self._exporter.generate_filename("spectrum", label=note).with_suffix(".pdf")
        data = ctx.last_data
        min_wl = ctx.min_wavelength
        data_e = data
        dark_e = self._mode._view.dark
        white_e = self._mode._view.white
        if min_wl > 0:
            data_e = trim_spectrum_data_for_export_min_wavelength(data, min_wl)
            n_full = min(len(data.wavelengths), len(data.intensity))
            mask = export_wavelength_mask(data.wavelengths, n_full, min_wl)
            if not np.all(mask):
                dark_e = trim_optional_intensity_row(dark_e, n_full, mask)
                white_e = trim_optional_intensity_row(white_e, n_full, mask)

        loaded = self._mode.loaded or make_empty()
        n = min(len(data_e.wavelengths), len(loaded.intensity))
        wl = np.asarray(data_e.wavelengths[:n], dtype=np.float32)
        measured_pre = self._mode.process_spectrum(
            np.asarray(loaded.intensity[:n], dtype=np.float32), wl
        )
        measured_pre = np.asarray(measured_pre, dtype=np.float64)
        measured_corrected = np.asarray(data_e.intensity[:n], dtype=np.float64)

        sens_col, sens_meta = build_sensitivity_for_export(
            correction_enabled=ctx.sensitivity_correction_enabled,
            engine=self._sensitivity,
            wavelengths=data_e.wavelengths,
            sens_cfg=self.config.sensitivity,
        )
        meta = self._mode.metadata_for_export(ctx)
        meta["Note"] = note
        overlay_series = self._mode.overlay_export_columns(ctx, data_e.wavelengths)
        bundle = MeasurementPdfBundle(
            path=pdf_path,
            wavelengths=data_e.wavelengths,
            measured_pre_sensitivity=measured_pre,
            measured_corrected=measured_corrected,
            dark=dark_e,
            white=white_e,
            reference=None,
            reference_name="Reference",
            sensitivity=sens_col,
            sensitivity_meta=dict(sens_meta),
            peaks=list(data_e.peaks),
            metadata=meta,
            x_axis_label=getattr(data_e, "x_axis_label", "Wavelength (nm)"),
            overlay_series=overlay_series or None,
        )
        export_measurement_pdf(bundle)
        time_display = time.strftime(self.config.export.time_format)
        self._display.state.save_message = f"Last PDF: {time_display}"
        print(f"[CSV Viewer] PDF saved: {pdf_path}")
