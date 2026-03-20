"""Unit tests for csv_viewer/loader.py."""

import io
import textwrap
from pathlib import Path

import numpy as np
import pytest

from pyspectrometer.csv_viewer.loader import CsvType, LoadedCsv, load_csv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tmp(tmp_path: Path, content: str, name: str = "test.csv") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Type detection via # Mode: comment
# ---------------------------------------------------------------------------


def test_detect_spectrum_from_mode_comment(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Mode: Measurement
        # Date: 2024-01-01 12:00:00
        Pixel, Wavelength, Intensity
        0, 400.0, 0.1
        1, 401.0, 0.2
        """,
    )
    result = load_csv(p)
    assert result.csv_type == CsvType.SPECTRUM


def test_detect_calibration_from_mode_comment(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Mode: Calibration
        pixel, intensity, reference_wavelength, reference_intensity, calibrated_wavelength, calibrated_intensity
        0, 0.01, 400.0, 0.9, 400.5, 0.01
        1, 0.02, 401.0, 0.8, 401.5, 0.02
        """,
    )
    result = load_csv(p)
    assert result.csv_type == CsvType.CALIBRATION


def test_detect_waterfall_from_mode_comment(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Mode: Waterfall
        # Rows: 2
        Row, 400.0, 401.0, 402.0
        0, 0.1, 0.2, 0.3
        1, 0.15, 0.25, 0.35
        """,
    )
    result = load_csv(p)
    assert result.csv_type == CsvType.WATERFALL


def test_detect_waterfall_rec_from_mode_comment(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Mode: Waterfall-Rec
        Row, 400.0, 401.0
        0, 0.1, 0.2
        """,
    )
    result = load_csv(p)
    assert result.csv_type == CsvType.WATERFALL


def test_detect_colorscience_from_mode_comment(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Mode: Color Science
        Pixel, Wavelength, Measured
        0, 400.0, 0.1
        """,
    )
    result = load_csv(p)
    assert result.csv_type == CsvType.COLORSCIENCE


# ---------------------------------------------------------------------------
# Fallback detection via column headers
# ---------------------------------------------------------------------------


def test_detect_waterfall_by_headers(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        Row, 400.0, 401.0, 402.0
        0, 0.1, 0.2, 0.3
        """,
    )
    result = load_csv(p)
    assert result.csv_type == CsvType.WATERFALL


def test_detect_calibration_by_headers(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        pixel, intensity, reference_wavelength, reference_intensity, calibrated_wavelength, calibrated_intensity
        0, 0.1, 400.0, 0.9, 400.5, 0.1
        """,
    )
    result = load_csv(p)
    assert result.csv_type == CsvType.CALIBRATION


def test_detect_colorscience_by_swatch_comment(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Swatch_1: S1 R L*=55.0 a*=8.0 b*=30.0
        Pixel, Wavelength, Measured
        0, 400.0, 0.1
        """,
    )
    result = load_csv(p)
    assert result.csv_type == CsvType.COLORSCIENCE


def test_fallback_to_spectrum_without_mode(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        Pixel, Wavelength, Intensity
        0, 400.0, 0.1
        1, 401.0, 0.2
        """,
    )
    result = load_csv(p)
    assert result.csv_type == CsvType.SPECTRUM


# ---------------------------------------------------------------------------
# SPECTRUM loading
# ---------------------------------------------------------------------------


def test_spectrum_basic_arrays(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Mode: Measurement
        Pixel, Wavelength, Intensity
        0, 400.0, 0.10
        1, 401.5, 0.20
        2, 403.0, 0.15
        """,
    )
    result = load_csv(p)
    assert len(result.wavelengths) == 3
    assert len(result.intensity) == 3
    np.testing.assert_allclose(result.wavelengths, [400.0, 401.5, 403.0])
    np.testing.assert_allclose(result.intensity, [0.10, 0.20, 0.15], atol=1e-6)
    assert result.dark is None
    assert result.white is None
    assert result.sensitivity is None


def test_spectrum_with_dark_and_white(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Mode: Measurement
        Pixel, Wavelength, Measured, Dark, White
        0, 400.0, 0.50, 0.01, 0.95
        1, 401.0, 0.60, 0.02, 0.90
        """,
    )
    result = load_csv(p)
    assert result.dark is not None
    assert result.white is not None
    np.testing.assert_allclose(result.dark, [0.01, 0.02], atol=1e-6)
    np.testing.assert_allclose(result.white, [0.95, 0.90], atol=1e-6)
    np.testing.assert_allclose(result.intensity, [0.50, 0.60], atol=1e-6)


def test_spectrum_with_sensitivity_column(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Mode: Measurement
        Pixel, Wavelength, Intensity, Sensitivity
        0, 400.0, 0.10, 0.80
        1, 401.0, 0.20, 0.85
        """,
    )
    result = load_csv(p)
    assert result.sensitivity is not None
    np.testing.assert_allclose(result.sensitivity, [0.80, 0.85], atol=1e-6)


def test_spectrum_extra_columns(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Mode: Color Science
        # Swatch_1: S1 R L*=55.0 a*=8.0 b*=30.0
        Pixel, Wavelength, Measured, Dark, White, S1
        0, 400.0, 0.50, 0.01, 0.95, 0.30
        1, 401.0, 0.60, 0.02, 0.90, 0.35
        """,
    )
    result = load_csv(p)
    assert "S1" in result.extra_columns
    np.testing.assert_allclose(result.extra_columns["S1"], [0.30, 0.35], atol=1e-6)


def test_spectrum_metadata_parsed(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Mode: Measurement
        # Date: 2024-06-15 10:30:00
        # Gain: 12.5
        Pixel, Wavelength, Intensity
        0, 400.0, 0.10
        """,
    )
    result = load_csv(p)
    assert result.metadata["Mode"] == "Measurement"
    assert result.metadata["Date"] == "2024-06-15 10:30:00"
    assert result.metadata["Gain"] == "12.5"


# ---------------------------------------------------------------------------
# WATERFALL loading
# ---------------------------------------------------------------------------


def test_waterfall_wavelengths_from_headers(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Mode: Waterfall
        Row, 400.0, 401.0, 402.0
        0, 0.1, 0.2, 0.3
        1, 0.15, 0.25, 0.35
        """,
    )
    result = load_csv(p)
    np.testing.assert_allclose(result.wavelengths, [400.0, 401.0, 402.0])


def test_waterfall_intensity_is_last_row(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Mode: Waterfall
        Row, 400.0, 401.0, 402.0
        0, 0.10, 0.20, 0.30
        1, 0.15, 0.25, 0.35
        2, 0.20, 0.30, 0.40
        """,
    )
    result = load_csv(p)
    np.testing.assert_allclose(result.intensity, [0.20, 0.30, 0.40], atol=1e-6)


def test_waterfall_dark_white_from_comments(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Mode: Waterfall
        # Dark: 0.01,0.02,0.03
        # White: 0.90,0.91,0.92
        Row, 400.0, 401.0, 402.0
        0, 0.5, 0.6, 0.7
        """,
    )
    result = load_csv(p)
    assert result.dark is not None
    assert result.white is not None
    np.testing.assert_allclose(result.dark, [0.01, 0.02, 0.03], atol=1e-6)
    np.testing.assert_allclose(result.white, [0.90, 0.91, 0.92], atol=1e-6)


def test_waterfall_empty_rows(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Mode: Waterfall
        Row, 400.0, 401.0
        """,
    )
    result = load_csv(p)
    assert len(result.intensity) == 2
    np.testing.assert_allclose(result.intensity, [0.0, 0.0])


# ---------------------------------------------------------------------------
# CALIBRATION loading
# ---------------------------------------------------------------------------


def test_calibration_uses_calibrated_columns(tmp_path):
    p = _write_tmp(
        tmp_path,
        """\
        # Mode: Calibration
        pixel, intensity, reference_wavelength, reference_intensity, calibrated_wavelength, calibrated_intensity
        0, 0.10, 400.0, 0.90, 400.50, 0.10
        1, 0.20, 401.0, 0.85, 401.50, 0.20
        """,
    )
    result = load_csv(p)
    np.testing.assert_allclose(result.wavelengths, [400.50, 401.50])
    np.testing.assert_allclose(result.intensity, [0.10, 0.20], atol=1e-6)
