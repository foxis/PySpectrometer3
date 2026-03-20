"""Measurement / Waterfall CSV export: omit wavelengths below configured floor."""

import numpy as np

from pyspectrometer.core.spectrum import SpectrumData
from pyspectrometer.export.csv_exporter import (
    export_wavelength_mask,
    trim_spectrum_data_for_export_min_wavelength,
    trim_waterfall_export_arrays,
)


def test_export_wavelength_mask():
    wl = np.array([250.0, 290.0, 300.0, 350.0, 400.0])
    m = export_wavelength_mask(wl, 5, 300.0)
    assert m.tolist() == [False, False, True, True, True]
    assert np.all(export_wavelength_mask(wl, 5, 0.0))


def test_trim_spectrum_data_drops_uv():
    wl = np.linspace(200, 800, 601)
    iy = np.ones(601, dtype=np.float32)
    data = SpectrumData(intensity=iy, wavelengths=wl)
    out = trim_spectrum_data_for_export_min_wavelength(data, 300.0)
    assert float(np.min(out.wavelengths)) >= 300.0 - 1.0
    assert len(out.intensity) == len(out.wavelengths)


def test_trim_waterfall_rows_match_header():
    wl = np.array([280.0, 310.0, 400.0])
    rows = [np.ones(3), np.ones(3) * 2]
    r2, wl2, _, _, _, _ = trim_waterfall_export_arrays(
        rows, wl, 300.0, None, None, None
    )
    assert len(wl2) == 2
    assert len(r2[0]) == 2
