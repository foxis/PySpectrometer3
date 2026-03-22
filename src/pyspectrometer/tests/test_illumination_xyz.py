"""Illumination mode XYZ uses direct SPD integration with optional reference scaling."""

import numpy as np
import pytest

try:
    from ..colorscience.xyz import calculate_XYZ
except ImportError:
    pytest.skip("colour-science not installed", allow_module_level=True)


def test_illumination_with_reference_y100():
    """When sample equals reference SPD, Y=100 and L*=100."""
    wl = np.linspace(400, 700, 31)
    spd = np.ones(31, dtype=np.float64)  # flat reference
    X, Y, Z = calculate_XYZ(
        spd, wl, "illumination",
        illuminant_spectrum=spd.copy(),
        illuminant_wavelengths=wl,
    )
    assert abs(Y - 100.0) < 0.5
    assert abs(X - 100.0) < 2 and abs(Z - 100.0) < 2


def test_illumination_with_reference_dimmer_sample():
    """When sample is half the reference, Y≈50."""
    wl = np.linspace(400, 700, 31)
    ref = np.ones(31, dtype=np.float64)
    sample = 0.5 * ref
    X, Y, Z = calculate_XYZ(
        sample, wl, "illumination",
        illuminant_spectrum=ref,
        illuminant_wavelengths=wl,
    )
    assert 45 < Y < 55
