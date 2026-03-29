"""Tests for frame black-level subtraction in spectrum extraction."""

import numpy as np

from pyspectrometer.processing.extraction import ExtractionMethod, SpectrumExtractor


def test_frame_black_uniform_frame_near_zero() -> None:
    """Uniform flat field: black level equals signal; subtracted spectrum ~0 before norm."""
    h, w = 64, 128
    frame = np.full((h, w), 100, dtype=np.float32)
    ex = SpectrumExtractor(
        frame_width=w,
        frame_height=h,
        method=ExtractionMethod.MEDIAN,
        rotation_angle=0.0,
        perpendicular_width=10,
        spectrum_y_center=h // 2,
        frame_black_strip_height=8,
    )
    out = ex.extract(frame, max_val=255.0)
    np.testing.assert_allclose(out.intensity, 0.0, atol=1e-5)


def test_frame_black_always_applied_uniform() -> None:
    """Default extractor always subtracts edge black; uniform field → ~0 normalized."""
    h, w = 48, 96
    frame = np.full((h, w), 50, dtype=np.float32)
    ex = SpectrumExtractor(
        frame_width=w,
        frame_height=h,
        method=ExtractionMethod.MEDIAN,
        spectrum_y_center=h // 2,
        frame_black_strip_height=8,
    )
    on = ex.extract(frame, max_val=255.0).intensity
    np.testing.assert_allclose(on, 0.0, atol=1e-5)


def test_frame_black_offset_removes_bias() -> None:
    """Rows 0–7 and bottom 8 rows at 20; spectrum band at center at 120 → ~100 after subtract."""
    h, w = 64, 32
    frame = np.full((h, w), 20.0, dtype=np.float32)
    frame[28:36, :] = 120.0
    ex = SpectrumExtractor(
        frame_width=w,
        frame_height=h,
        method=ExtractionMethod.MEDIAN,
        perpendicular_width=8,
        spectrum_y_center=32,
        frame_black_strip_height=8,
    )
    out = ex.extract(frame, max_val=255.0)
    # Median in strip is 120; bg = (20+20)/2 = 20 → raw line 100 → 100/255
    np.testing.assert_allclose(out.intensity, 100.0 / 255.0, rtol=1e-5, atol=1e-4)


def test_black_strip_height_clamped_small_frame() -> None:
    ex = SpectrumExtractor(
        frame_width=16,
        frame_height=12,
        method=ExtractionMethod.MEDIAN,
        spectrum_y_center=6,
        frame_black_strip_height=8,
    )
    # 2*8 >= 12 → strip height becomes max(1, 12//4) = 3
    assert ex._black_strip_height(12) == 3


def test_apply_config_frame_black_strip_height() -> None:
    from pyspectrometer.config import Config, _apply_config

    c = Config()
    _apply_config(
        c,
        {"extraction": {"frame_black_strip_height": 12}},
    )
    assert c.extraction.frame_black_strip_height == 12


def test_config_default_frame_black_strip_height() -> None:
    from pyspectrometer.config import Config

    assert Config().extraction.frame_black_strip_height == 8
