"""Build :class:`SpectrumExtractor` from :class:`Config` — single mapping site."""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from .extraction import ExtractionMethod, SpectrumExtractor

_METHOD_MAP: dict[str, ExtractionMethod] = {
    "median": ExtractionMethod.MEDIAN,
    "weighted_sum": ExtractionMethod.WEIGHTED_SUM,
    "gaussian": ExtractionMethod.GAUSSIAN,
    "max": ExtractionMethod.MAX,
}


@dataclass(frozen=True)
class ExtractorBuildParams:
    """Immutable inputs for constructing a :class:`SpectrumExtractor`.

    Calibration may override rotation, strip width, and y-center at runtime; see
    :meth:`pyspectrometer.spectrometer.Spectrometer` sync after load.
    """

    frame_width: int
    frame_height: int
    method: ExtractionMethod
    rotation_angle: float
    perpendicular_width: int
    spectrum_y_center: int
    crop_height: int
    background_percentile: float
    frame_black_strip_height: int

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        method: ExtractionMethod | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
    ) -> ExtractorBuildParams:
        """Map ``config`` (and optional overrides) to extractor parameters."""
        fw = config.camera.frame_width if frame_width is None else frame_width
        fh = config.camera.frame_height if frame_height is None else frame_height
        resolved_method = (
            method
            if method is not None
            else _METHOD_MAP.get(config.extraction.method, ExtractionMethod.WEIGHTED_SUM)
        )
        spectrum_y = config.extraction.spectrum_y_center or fh // 2
        return cls(
            frame_width=fw,
            frame_height=fh,
            method=resolved_method,
            rotation_angle=config.extraction.rotation_angle,
            perpendicular_width=config.extraction.perpendicular_width,
            spectrum_y_center=spectrum_y,
            crop_height=config.display.preview_height,
            background_percentile=config.extraction.background_percentile,
            frame_black_strip_height=config.extraction.frame_black_strip_height,
        )


def build_spectrum_extractor(params: ExtractorBuildParams) -> SpectrumExtractor:
    """Construct a :class:`SpectrumExtractor` from build parameters."""
    return SpectrumExtractor(
        frame_width=params.frame_width,
        frame_height=params.frame_height,
        method=params.method,
        rotation_angle=params.rotation_angle,
        perpendicular_width=params.perpendicular_width,
        spectrum_y_center=params.spectrum_y_center,
        crop_height=params.crop_height,
        background_percentile=params.background_percentile,
        frame_black_strip_height=params.frame_black_strip_height,
    )
