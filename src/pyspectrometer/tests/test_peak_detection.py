"""Peak detection tests, including sharp Hg-like emission lines."""

import numpy as np
import pytest

from ..core.spectrum import SpectrumData
from ..processing.peak_detection import (
    _SCIPY_AVAILABLE,
    PeakDetector,
    find_peak_indexes_scipy,
)


def _gaussian_line(center: int, sigma: float, length: int, height: float = 1.0) -> np.ndarray:
    """Sharp 1-2 pixel emission line (Gaussian with small sigma)."""
    x = np.arange(length, dtype=np.float64)
    return height * np.exp(-((x - center) ** 2) / (2 * sigma**2))


def _hg_synthetic_spectrum(
    n_pixels: int,
    wl: np.ndarray,
    pixel_at_wl: np.ndarray,
    sigma: float = 0.8,
) -> np.ndarray:
    """Build synthetic spectrum with sharp Hg-like peaks at given pixel positions."""
    y = np.zeros(n_pixels, dtype=np.float64)
    intensities = [0.6, 0.9, 0.8, 0.7, 0.65]  # approximate Hg relative strengths
    for i, px in enumerate(pixel_at_wl):
        if 0 <= px < n_pixels:
            line = _gaussian_line(int(px), sigma, n_pixels, intensities[i % len(intensities)])
            y += line
    # Add low baseline and noise
    y += 0.05
    y += np.random.default_rng(42).uniform(0, 0.01, n_pixels)
    return y


@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy required")
def test_scipy_finds_sharp_hg_peaks():
    """scipy find_peaks must detect all 4-5 sharp Hg-like peaks."""
    n_pixels = 1000
    wl = np.linspace(380.0, 720.0, n_pixels)
    # Hg lines ~404, 436, 546, 577, 579 nm -> pixel indices
    hg_nm = np.array([404.66, 435.84, 546.07, 576.96, 579.07])
    pixel_at = [int(np.argmin(np.abs(wl - x))) for x in hg_nm]

    y = _hg_synthetic_spectrum(n_pixels, wl, pixel_at, sigma=0.8)

    idx = find_peak_indexes_scipy(y, threshold=0.1, min_dist=5, prominence=0.02)
    assert len(idx) >= 4, f"Expected at least 4 peaks, got {len(idx)}"
    # All Hg pixel positions should be near a detected peak (within 3 px)
    for px in pixel_at:
        near = np.any(np.abs(idx - px) <= 3)
        assert near, f"No peak detected near pixel {px} (Hg line)"


def test_peak_detector_detects_sharp_peaks():
    """PeakDetector must find sharp Hg-like peaks in SpectrumData."""
    n_pixels = 1000
    wl = np.linspace(380.0, 720.0, n_pixels)
    hg_nm = np.array([404.66, 435.84, 546.07, 576.96])
    pixel_at = [int(np.argmin(np.abs(wl - x))) for x in hg_nm]
    y = _hg_synthetic_spectrum(n_pixels, wl, pixel_at, sigma=0.8)

    data = SpectrumData(wavelengths=wl, intensity=y)
    detector = PeakDetector(min_distance=15, threshold=10)
    out = detector.process(data)

    assert len(out.peaks) >= 3, f"Expected at least 3 peaks, got {len(out.peaks)}"
    for px in pixel_at:
        found = any(abs(p.index - px) <= 5 for p in out.peaks)
        assert found, f"No peak near pixel {px}"
