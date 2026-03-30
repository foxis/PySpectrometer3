"""Tests for :mod:`pyspectrometer.processing.extractor_params`."""

from pyspectrometer.config import Config
from pyspectrometer.processing.extraction import ExtractionMethod
from pyspectrometer.processing.extractor_params import (
    ExtractorBuildParams,
    build_spectrum_extractor,
)


def test_from_config_maps_extraction_method_string() -> None:
    c = Config()
    c.extraction.method = "gaussian"
    p = ExtractorBuildParams.from_config(c)
    assert p.method == ExtractionMethod.GAUSSIAN


def test_from_config_unknown_method_defaults_to_weighted_sum() -> None:
    c = Config()
    c.extraction.method = "not_a_real_method"
    p = ExtractorBuildParams.from_config(c)
    assert p.method == ExtractionMethod.WEIGHTED_SUM


def test_from_config_frame_overrides() -> None:
    c = Config()
    p = ExtractorBuildParams.from_config(c, frame_width=800, frame_height=600)
    assert p.frame_width == 800
    assert p.frame_height == 600


def test_build_spectrum_extractor_roundtrip() -> None:
    c = Config()
    ext = build_spectrum_extractor(ExtractorBuildParams.from_config(c))
    assert ext.frame_width == c.camera.frame_width
    assert ext.method == ExtractionMethod.WEIGHTED_SUM
