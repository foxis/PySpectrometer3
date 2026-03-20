"""Viewport nm-window → data-x mapping (default horizontal span only)."""

import numpy as np

from pyspectrometer.display.viewport import x_span_for_wavelength_window


def test_x_span_increasing_wavelengths():
    wl = np.linspace(300, 900, 601)
    span = x_span_for_wavelength_window(wl, 350.0, 800.0, data_width=601)
    assert span is not None
    x0, x1 = span
    assert 0 <= x0 < x1 <= 600
    assert abs(wl[int(round(x0))] - 350.0) < 2.0
    assert abs(wl[int(round(x1))] - 800.0) < 2.0


def test_x_span_invalid_window_returns_none():
    wl = np.linspace(400, 700, 100)
    assert x_span_for_wavelength_window(wl, 500.0, 500.0, data_width=100) is None
