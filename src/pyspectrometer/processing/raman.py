"""Raman shift (wavenumber) conversion utilities."""

import numpy as np


def wavelength_to_wavenumber(wavelength_nm: float, laser_nm: float = 785.0) -> float:
    """Convert wavelength to Raman shift in cm⁻¹.

    Raman shift = (1/λ_laser - 1/λ) × 10⁷
    Stokes (λ > laser): positive shift; anti-Stokes (λ < laser): negative.

    Args:
        wavelength_nm: Detected wavelength in nm
        laser_nm: Excitation laser wavelength in nm

    Returns:
        Raman shift in cm⁻¹
    """
    if wavelength_nm <= 0:
        return np.nan
    return (1.0 / laser_nm - 1.0 / wavelength_nm) * 1e7


def wavelengths_to_wavenumbers(
    wavelengths_nm: np.ndarray,
    laser_nm: float = 785.0,
) -> np.ndarray:
    """Convert wavelength array to Raman shift array in cm⁻¹."""
    return (1.0 / laser_nm - 1.0 / np.maximum(wavelengths_nm, 1e-6)) * 1e7


