"""Camera-less spectrometer loop for CSV viewer mode."""

import time
from pathlib import Path

import cv2
import numpy as np

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
from ..processing.auto_controls import AutoExposureController, AutoGainController
from ..processing.extraction import ExtractionMethod, SpectrumExtractor
from ..processing.filters import SavitzkyGolayFilter
from ..processing.peak_detection import PeakDetector, detect_peaks_in_region
from ..processing.pipeline import ProcessingPipeline
from ..utils.dialog import prompt_label
from .loader import LoadedCsv, load_csv
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


class CsvViewerSpectrometer:
    """Run the display loop for a loaded spectrum CSV without a camera.

    If *csv_path* is None a file-open dialog is shown immediately so the
    application can be launched without a command-line argument.
    """

    WINDOW_TITLE = "PySpectrometer 3 - CSV Viewer"

    def __init__(self, csv_path: Path | None = None, config: Config | None = None) -> None:
        self.config = config or Config()
        self.config.apply_csv_viewer_preset()
        # Align the window title so _check_window_closed matches the actual OpenCV window.
        self.config.spectrograph_title = self.WINDOW_TITLE

        if csv_path is None:
            output_dir = self.config.export.output_dir
            if str(output_dir) == ".":
                output_dir = Path("output")
            csv_path = prompt_csv_path(output_dir)
            if csv_path is None:
                raise SystemExit(0)

        self._csv_path = csv_path

        loaded = load_csv(csv_path)
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
        print(f"[CSV Viewer] {self._csv_path.name}")
        self._camera.start()
        self._display.setup_windows()
        self._ctx.running = True
        self._mode.on_start(self._ctx)

        try:
            while self._ctx.running:
                loaded = self._mode.loaded
                if loaded is None:
                    break
                processed = self._build_spectrum(loaded)
                processed = self._mode.transform_spectrum_data(processed)
                processed = self._expand_peaks(processed)
                self._ctx.last_data = processed
                self._mode.update_display(self._ctx, processed, self.config.display.graph_height)
                self._render(processed)

                if getattr(self._ctx, "pending_csv_reload", False):
                    self._ctx.pending_csv_reload = False  # type: ignore[attr-defined]
                    self._prompt_reload()

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
            height=self.config.camera.frame_height,
        )
        cal.seed_wavelengths(loaded.wavelengths)
        return cal

    def _build_context(self) -> ModeContext:
        ctx = ModeContext(
            camera=self._camera,
            calibration=self._calibration,
            display=self._display,
            exporter=self._exporter,
            extractor=self._extractor,
            auto_gain=self._auto_gain,
            auto_exposure=self._auto_exposure,
        )
        ctx.quit_app = lambda: setattr(ctx, "running", False)
        ctx.save_snapshot = self._save_snapshot
        ctx.sensitivity_correction_enabled = False
        ctx.pending_csv_reload = False  # type: ignore[attr-defined]
        return ctx

    def _build_spectrum(self, loaded: LoadedCsv) -> SpectrumData:
        wl = self._calibration.wavelengths
        n = min(len(wl), len(loaded.intensity))
        intensity = self._mode.process_spectrum(
            np.asarray(loaded.intensity[:n], dtype=np.float32), wl[:n]
        )
        data = SpectrumData(intensity=intensity[:n], wavelengths=wl[:n])
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
        default_dir = self._csv_path.parent if self._csv_path else None
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
            correction_enabled=False,
            engine=None,
            wavelengths=data_e.wavelengths,
            sens_cfg=self.config.sensitivity,
        )
        meta.update(sens_meta)
        self._exporter.export(
            data_e,
            csv_path,
            dark_intensity=dark_e,
            white_intensity=white_e,
            metadata=meta,
            sensitivity_intensity=sens_col,
        )
        time_display = time.strftime(self.config.export.time_format)
        self._display.state.save_message = f"Last Save: {time_display}"
        print(f"[CSV Viewer] Saved: {csv_path}")
