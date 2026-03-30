"""Composition helpers: wire subsystems from :class:`Config` (single entry points)."""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .data.reference_loader import set_reference_dirs
from .processing.auto_controls import AutoExposureController, AutoGainController
from .processing.extractor_params import ExtractorBuildParams, build_spectrum_extractor
from .processing.extraction import SpectrumExtractor
from .processing.filters import SavitzkyGolayFilter
from .processing.peak_detection import PeakDetector
from .processing.pipeline import ProcessingPipeline


def configure_reference_search(config: Config) -> None:
    """Apply ``config.export.reference_dirs`` for calibration reference CSV lookup."""
    set_reference_dirs(config.export.reference_dirs)


def build_extractor_from_config(config: Config) -> SpectrumExtractor:
    """Build the main-loop spectrum extractor from application config."""
    return build_spectrum_extractor(ExtractorBuildParams.from_config(config))


def build_auto_controllers(
    config: Config,
) -> tuple[AutoGainController, AutoExposureController]:
    """Auto gain / exposure controllers from ``config.auto``."""
    smoothing = config.auto.peak_smoothing_period_sec
    rate_hz = config.auto.max_adjust_rate_hz
    return (
        AutoGainController(peak_smoothing_period_sec=smoothing, max_adjust_rate_hz=rate_hz),
        AutoExposureController(peak_smoothing_period_sec=smoothing, max_adjust_rate_hz=rate_hz),
    )


def build_processing_stack(
    config: Config,
) -> tuple[ProcessingPipeline, SavitzkyGolayFilter, PeakDetector]:
    """Savitzky–Golay + peak pipeline from ``config.processing``."""
    savgol = SavitzkyGolayFilter(
        window_size=config.processing.savgol_window,
        poly_order=config.processing.savgol_poly,
        poly_order_min=config.processing.savgol_poly_min,
        poly_order_max=config.processing.savgol_poly_max,
    )
    savgol.enabled = False
    peak = PeakDetector(
        min_distance=config.processing.peak_min_distance,
        threshold=config.processing.peak_threshold,
        max_count=config.processing.peak_max_count,
        min_distance_min=config.processing.peak_min_distance_min,
        min_distance_max=config.processing.peak_min_distance_max,
        threshold_min=config.processing.peak_threshold_min,
        threshold_max=config.processing.peak_threshold_max,
        max_count_min=config.processing.peak_max_count_min,
        max_count_max=config.processing.peak_max_count_max,
    )
    pipeline = ProcessingPipeline([savgol, peak])
    return pipeline, savgol, peak


def normalized_export_output_dir(config: Config) -> Path:
    """Export directory; treat ``.`` as ``output`` for CSV exporter."""
    output_dir = config.export.output_dir
    if str(output_dir) == ".":
        return Path("output")
    return output_dir
