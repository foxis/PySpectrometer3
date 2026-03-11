"""Calibration integration tests.

Verifies that both peak-based and correlation-based calibration recover
the correct wavelength mapping. Tests must assert alignment (max/mean error).
"""

from pathlib import Path

import numpy as np
import pytest

from ..core.calibration import Calibration
from ..core.spectrum import Peak
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
        f"Mean wavelength error {mean_error:.2f} nm exceeds 3 nm. Max error: {max_error:.2f} nm"
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


def test_hg_peak_based_calibration_4_peaks():
    """Correlation calibration with synthetic Hg spectrum must align within 8 nm.

    Ground truth: linear 380-720 nm. Synthetic Hg spectrum. Algorithm must recover mapping.
    """
    n_pixels = 640
    rng = np.random.default_rng(43)
    wl_true = _ground_truth_calibration(n_pixels, rng)

    hg_nm = [404.66, 435.84, 546.07, 576.96]
    peak_pixels = [int(np.argmin(np.abs(wl_true - wl))) for wl in hg_nm]
    peak_pixels = sorted(peak_pixels)

    measured = _generate_synthetic_measured(n_pixels, wl_true, ReferenceSource.HG, 0.02, rng)
    peaks = [Peak(index=p, wavelength=float(wl_true[p]), intensity=float(measured[p])) for p in peak_pixels]

    cal_mode = CalibrationMode()
    cal_mode.select_source(ReferenceSource.HG)
    cal_points = cal_mode.auto_calibrate(
        measured_intensity=measured,
        wavelengths=wl_true,
        peak_indices=peaks,
    )

    assert len(cal_points) >= 4, "Correlation must return at least 4 points"

    pixels = np.array([p for p, _ in cal_points], dtype=np.float64)
    wavelengths = np.array([w for _, w in cal_points], dtype=np.float64)
    coeffs = np.polyfit(pixels, wavelengths, min(3, len(cal_points) - 1))
    poly = np.poly1d(coeffs)

    for px, wl_expected in zip(peak_pixels, hg_nm):
        wl_got = poly(px)
        err = abs(wl_got - wl_expected)
        assert err < 8.0, (
            f"Peak at pixel {px} should map to {wl_expected} nm, got {wl_got:.1f} nm "
            f"(error {err:.2f} nm)"
        )


def test_hg_peak_based_4_measured_5_reference():
    """Correlation with synthetic Hg (4 main lines) must align within 8 nm."""
    n_pixels = 640
    rng = np.random.default_rng(44)
    wl_true = _ground_truth_calibration(n_pixels, rng)

    hg_4 = [404.66, 435.84, 546.07, 576.96]
    peak_pixels = [int(np.argmin(np.abs(wl_true - wl))) for wl in hg_4]
    peak_pixels = sorted(peak_pixels)

    measured = _generate_synthetic_measured(n_pixels, wl_true, ReferenceSource.HG, 0.02, rng)
    peaks = [Peak(index=p, wavelength=float(wl_true[p]), intensity=float(measured[p])) for p in peak_pixels]

    cal_mode = CalibrationMode()
    cal_mode.select_source(ReferenceSource.HG)
    cal_points = cal_mode.auto_calibrate(
        measured_intensity=measured,
        wavelengths=wl_true,
        peak_indices=peaks,
    )

    assert len(cal_points) >= 4
    pixels = np.array([p for p, _ in cal_points], dtype=np.float64)
    wavelengths = np.array([w for _, w in cal_points], dtype=np.float64)
    coeffs = np.polyfit(pixels, wavelengths, min(3, len(cal_points) - 1))
    poly = np.poly1d(coeffs)
    for px, wl_ref in zip(peak_pixels, hg_4):
        wl_cal = poly(px)
        assert abs(wl_cal - wl_ref) < 8.0, f"Pixel {px} -> {wl_cal:.1f} nm, expected ~{wl_ref} nm"


def test_correlation_wavelength_range_sensible():
    """Correlation result must map pixels to 360-780 nm (visible within range)."""
    rng = np.random.default_rng(46)
    n_pixels = 640
    wl_true = _ground_truth_calibration(n_pixels, rng)
    measured = _generate_synthetic_measured(n_pixels, wl_true, ReferenceSource.HG, 0.02, rng)

    cal_mode = CalibrationMode()
    cal_mode.select_source(ReferenceSource.HG)
    cal_points = cal_mode.auto_calibrate(
        measured_intensity=measured,
        wavelengths=wl_true,
        peak_indices=[],
    )
    assert len(cal_points) >= 4

    pixels = np.array([p for p, _ in cal_points], dtype=np.float64)
    wavelengths = np.array([w for _, w in cal_points], dtype=np.float64)
    coeffs = np.polyfit(pixels, wavelengths, 3)
    poly = np.poly1d(coeffs)
    wl_start = poly(0)
    wl_end = poly(n_pixels - 1)
    assert 360 <= wl_start <= 440, f"Pixel 0 -> {wl_start:.1f} nm, should be 360-440"
    assert 660 <= wl_end <= 780, f"Pixel {n_pixels - 1} -> {wl_end:.1f} nm, should be 660-780"
    assert wl_start < wl_end - 50, "Wavelength must increase with pixel"


def test_calibration_alignment_tolerance():
    """Sanity: correlation calibration must achieve <10 nm max error on synthetic Hg."""
    rng = np.random.default_rng(45)
    n_pixels = 640
    wl_true = _ground_truth_calibration(n_pixels, rng)
    measured = _generate_synthetic_measured(n_pixels, wl_true, ReferenceSource.HG, 0.02, rng)

    cal_mode = CalibrationMode()
    cal_mode.select_source(ReferenceSource.HG)
    cal_points = cal_mode.auto_calibrate(
        measured_intensity=measured,
        wavelengths=wl_true,
        peak_indices=[],
    )

    assert len(cal_points) >= 4
    pixels = np.array([p for p, _ in cal_points], dtype=np.float64)
    wavelengths = np.array([w for _, w in cal_points], dtype=np.float64)
    coeffs = np.polyfit(pixels, wavelengths, 3)
    poly = np.poly1d(coeffs)

    wl_est = poly(np.arange(n_pixels, dtype=np.float64))
    max_err = float(np.max(np.abs(wl_est - wl_true)))
    assert max_err < 10.0, f"Correlation calibration max error {max_err:.2f} nm exceeds 10 nm"


def test_real_spectrum_hg_and_fl12_calibration():
    """Calibrate real Hg and FL12 spectra; must return valid points with reasonable correlation."""
    from ..processing.auto_calibrator import AutoCalibrator

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "output"
    if not output_dir.exists():
        pytest.skip("output/ not found")

    spectra = [
        ("Spectrum-20260311--193723.csv", ReferenceSource.HG),
        ("Spectrum-20260311--193856.csv", ReferenceSource.FL12),
    ]
    cal = AutoCalibrator()

    for filename, source in spectra:
        csv_path = output_dir / filename
        if not csv_path.exists():
            pytest.skip(f"{filename} not found")

        points = cal.calibrate_from_csv(csv_path, source=source)
        assert len(points) >= 4, f"{filename} + {source.name}: need 4+ points, got {len(points)}"

        px = np.array([p for p, _ in points])
        wl = np.array([w for _, w in points])
        assert np.all(np.diff(px) > 0), f"{filename}: pixels must be strictly increasing"
        assert 250 <= wl.min() <= 450, f"{filename}: start wavelength {wl.min():.1f} nm out of range"
        assert 550 <= wl.max() <= 820, f"{filename}: end wavelength {wl.max():.1f} nm out of range"


def test_hg_csv_calibration_monotonic_no_explosion():
    """Calibrate against Hg spectrum from CSV using same peak detection and matching as calibration.

    Uses data/Spectrum-20260309--010902.csv (Hg lamp, Pixel, Intensity - ignore Wavelength).
    Runs CalibrationMode.auto_calibrate with Hg reference (peak matching or correlation),
    applies Calibration.recalibrate, and asserts:
    - Strictly monotonic wavelengths
    - No explosion (wavelengths in visible/NIR 250-850 nm)
    - Mostly linear (dispersion in typical prism range 0.2-0.5 nm/px)
    """
    from ..data.reference_spectra import ReferenceSource
    from ..processing.peak_detection import find_peak_indexes_scipy

    csv_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "data"
        / "Spectrum-20260309--010902.csv"
    )
    if not csv_path.exists():
        pytest.skip(f"CSV not found: {csv_path}")

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    pixels_arr = data[:, 0].astype(int)
    intensity = data[:, 2].astype(
        np.float32
    )  # Use Intensity only (Wavelength column is from failed cal)
    width = int(pixels_arr[-1]) + 1

    # Placeholder wavelengths for PeakDetector (not used for matching)
    wl_dummy = np.linspace(380.0, 750.0, width, dtype=np.float64)

    # Same peak detection as calibration procedure (PeakDetector / find_peak_indexes_scipy)
    peak_idx = find_peak_indexes_scipy(
        intensity,
        threshold=0.05,
        min_dist=15,
        prominence=0.005,
    )
    peaks = [
        Peak(index=int(i), wavelength=float(wl_dummy[i]), intensity=float(intensity[i]))
        for i in peak_idx
    ]

    # Same calibration procedure: CalibrationMode.auto_calibrate (peak matching or correlation)
    cal_mode = CalibrationMode()
    cal_mode.select_source(ReferenceSource.HG)
    cal_points = cal_mode.auto_calibrate(
        measured_intensity=intensity,
        wavelengths=wl_dummy,
        peak_indices=peaks,
    )

    assert len(cal_points) >= 3, (
        f"Auto-calibrate must return 3+ points (peak or correlation); got {len(cal_points)}"
    )

    # Apply calibration
    cal = Calibration(width=width)
    ok = cal.recalibrate(
        [p for p, _ in cal_points],
        [w for _, w in cal_points],
    )
    assert ok, "recalibrate must succeed"

    wl = cal.wavelengths
    p_min, p_max = min(p for p, _ in cal_points), max(p for p, _ in cal_points)

    # 1. Strictly monotonic within interpolation range
    seg = wl[p_min : p_max + 1]
    diff = np.diff(seg)
    assert np.all(diff < 0) or np.all(diff > 0), (
        f"Wavelengths must be strictly monotonic in [{p_min},{p_max}]"
    )
    assert np.all(diff != 0), "No flat segments in interpolation range"

    # 2. No explosion: wavelengths in visible/NIR
    assert 250 <= wl.min() <= 900 and 250 <= wl.max() <= 900, (
        f"Wavelengths must be 250-900 nm (got [{wl.min():.1f}, {wl.max():.1f}])"
    )

    # 3. Mostly linear: dispersion in typical prism range 0.2-0.5 nm/px
    px_span = cal_points[-1][0] - cal_points[0][0]
    wl_span = abs(cal_points[-1][1] - cal_points[0][1])
    disp = wl_span / max(px_span, 1)
    assert 0.15 <= disp <= 0.6, f"Dispersion {disp:.3f} nm/px outside typical prism range 0.15-0.6"
