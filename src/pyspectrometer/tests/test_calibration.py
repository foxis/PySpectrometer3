"""Calibration integration tests.

Verifies that the correlation-based calibration recovers the correct
wavelength mapping from synthetic "camera" data.
"""

import numpy as np
import pytest

from ..data.reference_spectra import ReferenceSource, get_reference_spectrum
from ..modes.calibration import CalibrationMode


def _camera_sensitivity(wavelength_nm: np.ndarray) -> np.ndarray:
    """Approximate silicon sensor spectral sensitivity (black-body-like).

    Peak around 550nm, typical for CCD/CMOS.
    """
    # Simplified: Gaussian centered at 550nm, FWHM ~200nm
    center = 550.0
    sigma = 85.0  # ~200nm FWHM
    return np.exp(-((wavelength_nm - center) ** 2) / (2 * sigma**2))


def _cmos_nonlinear_response(linear_intensity: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Simulate CMOS sensor non-linear response (attenuation).

    Typical CMOS sensors have a gamma-like response that compresses highlights
    and attenuates low intensities.
    """
    # Ensure non-negative and clip to [0, 1] before gamma
    x = np.clip(linear_intensity.astype(np.float64), 0.0, 1.0)
    return np.power(np.maximum(x, 1e-10), 1.0 / gamma)


def _ground_truth_calibration(n_pixels: int, rng: np.random.Generator) -> np.ndarray:
    """Ground truth pixel->wavelength mapping: somewhat linear with small nonlinearity."""
    p = np.arange(n_pixels, dtype=np.float64)
    c0 = 380.0 + rng.uniform(-5, 5)
    c1 = (750.0 - c0) / max(n_pixels - 1, 1) * rng.uniform(0.95, 1.05)
    c2 = rng.uniform(-5, 5)
    c3 = rng.uniform(-2, 2)
    wl = c0 + c1 * p + c2 * (p**2) / (n_pixels**2) + c3 * (p**3) / (n_pixels**3)
    return wl


def _generate_synthetic_measured(
    n_pixels: int,
    wl_true: np.ndarray,
    source: ReferenceSource,
    noise_std: float = 0.02,
    rng: np.random.Generator | None = None,
    apply_nonlinear: bool = False,
    gamma: float = 2.2,
) -> np.ndarray:
    """Generate synthetic measured spectrum: Hg ref * camera response + noise.

    Args:
        apply_nonlinear: If True, apply CMOS gamma-like non-linear response
        gamma: Gamma for non-linear response (typical CMOS ~2.2)
    """
    rng = rng or np.random.default_rng(42)
    ref = get_reference_spectrum(source, wl_true)
    camera = _camera_sensitivity(wl_true)
    measured = ref * camera
    if measured.max() > 0:
        measured = measured / measured.max()
    if apply_nonlinear:
        measured = _cmos_nonlinear_response(measured, gamma)
    measured = np.clip(measured + rng.normal(0, noise_std, n_pixels), 0, 1.5)
    return measured.astype(np.float32)


def test_calibration_recovers_hg_wavelength_mapping():
    """Calibration should recover correct wavelength mapping from synthetic Hg data."""
    rng = np.random.default_rng(42)
    n_pixels = 640

    # 1. Ground truth calibration (somewhat linear)
    wl_true = _ground_truth_calibration(n_pixels, rng)

    # 2. Synthetic measured spectrum: Hg * camera sensitivity + noise
    measured = _generate_synthetic_measured(n_pixels, wl_true, ReferenceSource.HG, 0.02, rng)

    # 3. Run calibration (correlation-based)
    cal_mode = CalibrationMode()
    cal_mode.select_source(ReferenceSource.HG)
    cal_points = cal_mode.auto_calibrate(
        measured_intensity=measured,
        wavelengths=wl_true,  # dummy; correlation method ignores
        peak_indices=[],
    )

    assert len(cal_points) >= 4, "Calibration must return at least 4 points"

    # 4. Fit polynomial through calibration points
    pixels = np.array([p for p, _ in cal_points], dtype=np.float64)
    wavelengths = np.array([w for _, w in cal_points], dtype=np.float64)
    coeffs = np.polyfit(pixels, wavelengths, 3)
    poly = np.poly1d(coeffs)

    # 5. Compare estimated wavelengths to ground truth
    p_all = np.arange(n_pixels, dtype=np.float64)
    wl_estimated = poly(p_all)
    errors_nm = np.abs(wl_estimated - wl_true)

    max_error = float(np.max(errors_nm))
    mean_error = float(np.mean(errors_nm))

    assert max_error < 8.0, (
        f"Max wavelength error {max_error:.2f} nm exceeds 8 nm tolerance. "
        f"Mean error: {mean_error:.2f} nm"
    )
    assert mean_error < 3.0, (
        f"Mean wavelength error {mean_error:.2f} nm exceeds 3 nm. "
        f"Max error: {max_error:.2f} nm"
    )


def test_calibration_recovers_hg_with_nonlinear_cmos_response():
    """Calibration should recover wavelength mapping when measured spectrum
    is attenuated by non-linear CMOS sensor spectral sensitivity (gamma).
    """
    rng = np.random.default_rng(42)
    n_pixels = 640

    wl_true = _ground_truth_calibration(n_pixels, rng)

    measured = _generate_synthetic_measured(
        n_pixels,
        wl_true,
        ReferenceSource.HG,
        noise_std=0.02,
        rng=rng,
        apply_nonlinear=True,
        gamma=2.2,
    )

    cal_mode = CalibrationMode()
    cal_mode.select_source(ReferenceSource.HG)
    cal_points = cal_mode.auto_calibrate(
        measured_intensity=measured,
        wavelengths=wl_true,
        peak_indices=[],
    )

    assert len(cal_points) >= 4, "Calibration must return at least 4 points"

    pixels = np.array([p for p, _ in cal_points], dtype=np.float64)
    wavelengths = np.array([w for _, w in cal_points], dtype=np.float64)
    coeffs = np.polyfit(pixels, wavelengths, 3)
    poly = np.poly1d(coeffs)

    p_all = np.arange(n_pixels, dtype=np.float64)
    wl_estimated = poly(p_all)
    errors_nm = np.abs(wl_estimated - wl_true)

    max_error = float(np.max(errors_nm))
    mean_error = float(np.mean(errors_nm))

    # Relaxed tolerance for non-linear response (gamma changes peak ratios)
    assert max_error < 12.0, (
        f"Max wavelength error {max_error:.2f} nm exceeds 12 nm with non-linear CMOS. "
        f"Mean error: {mean_error:.2f} nm"
    )
    assert mean_error < 5.0, (
        f"Mean wavelength error {mean_error:.2f} nm exceeds 5 nm with non-linear CMOS. "
        f"Max error: {max_error:.2f} nm"
    )
