"""Reference spectra for calibration.

Contains spectral line data for common calibration sources:
- FL: Compact Fluorescent Lamp (CFL) - Mercury + rare earth phosphors
- Hg: Low-pressure Mercury lamp
- Sun: Solar spectrum (Fraunhofer absorption lines)
- LED: Typical white phosphor-converted LED
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np


class ReferenceSource(Enum):
    """Available reference light sources for calibration."""
    
    FL = auto()      # Compact Fluorescent Lamp
    HG = auto()      # Mercury low-pressure lamp
    SUN = auto()     # Solar spectrum
    LED = auto()     # White phosphor-converted LED


@dataclass
class SpectralLine:
    """A spectral emission or absorption line."""
    
    wavelength: float  # nm
    intensity: float   # relative intensity (0-1)
    label: str = ""    # optional label (e.g., "Hg", "Na-D")


@dataclass
class ReferenceSpectrum:
    """Reference spectrum with peaks and continuous data."""
    
    name: str
    source: ReferenceSource
    peaks: list[SpectralLine]
    description: str = ""
    
    # Optional continuous spectrum (wavelength, intensity arrays)
    wavelengths: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None


# Mercury low-pressure lamp emission lines
# https://physics.nist.gov/PhysRefData/Handbook/Tables/mercurytable2.htm
HG_LINES = [
    SpectralLine(404.66, 0.6, "Hg"),    # Violet
    SpectralLine(435.84, 0.9, "Hg"),    # Blue (strong)
    SpectralLine(546.07, 1.0, "Hg"),    # Green (strongest)
    SpectralLine(576.96, 0.5, "Hg"),    # Yellow doublet
    SpectralLine(579.07, 0.5, "Hg"),    # Yellow doublet
]

# Compact Fluorescent Lamp (CFL) - Mercury + rare earth phosphors
# Combines Hg lines with phosphor emission peaks
FL_LINES = [
    # Mercury lines (from excitation)
    SpectralLine(404.66, 0.4, "Hg"),    # Violet
    SpectralLine(435.84, 0.7, "Hg"),    # Blue
    SpectralLine(546.07, 0.9, "Hg"),    # Green
    SpectralLine(578.0, 0.4, "Hg"),     # Yellow (blended doublet)
    # Rare earth phosphor peaks (Europium, Terbium)
    SpectralLine(487.0, 0.5, "Tb"),     # Terbium blue-green
    SpectralLine(543.0, 0.8, "Tb"),     # Terbium green
    SpectralLine(584.0, 0.6, "Eu"),     # Europium yellow
    SpectralLine(611.0, 1.0, "Eu"),     # Europium red (strongest)
    SpectralLine(629.0, 0.5, "Eu"),     # Europium red
]

# Solar spectrum Fraunhofer absorption lines
# These are ABSORPTION lines (dips in the spectrum)
# https://en.wikipedia.org/wiki/Fraunhofer_lines
SUN_LINES = [
    SpectralLine(393.37, 0.8, "K"),     # Calcium K
    SpectralLine(396.85, 0.7, "H"),     # Calcium H  
    SpectralLine(430.79, 0.5, "G"),     # Iron/Calcium blend
    SpectralLine(486.13, 0.6, "F"),     # Hydrogen beta
    SpectralLine(516.73, 0.4, "b"),     # Magnesium triplet
    SpectralLine(518.36, 0.4, "b"),     # Magnesium triplet
    SpectralLine(527.04, 0.3, "E"),     # Iron
    SpectralLine(589.00, 0.9, "D1"),    # Sodium D1
    SpectralLine(589.59, 0.9, "D2"),    # Sodium D2
    SpectralLine(656.28, 1.0, "C"),     # Hydrogen alpha
    SpectralLine(686.72, 0.5, "B"),     # Oxygen (atmospheric)
    SpectralLine(759.37, 0.6, "A"),     # Oxygen (atmospheric)
]

# White phosphor-converted LED
# Blue LED chip (~450nm) + YAG:Ce phosphor (broad ~550nm peak)
LED_LINES = [
    SpectralLine(450.0, 1.0, "Blue"),   # Blue LED chip peak
    SpectralLine(550.0, 0.9, "Phos"),   # Phosphor emission peak (broad)
    # Additional markers for high-CRI LEDs with red phosphor
    SpectralLine(630.0, 0.4, "Red"),    # Optional red phosphor
]


def _generate_gaussian_spectrum(
    wavelengths: np.ndarray,
    peaks: list[SpectralLine],
    fwhm: float = 15.0,
) -> np.ndarray:
    """Generate continuous spectrum from peaks using Gaussian profiles.
    
    Args:
        wavelengths: Wavelength array to generate spectrum for
        peaks: List of spectral lines
        fwhm: Full width at half maximum for peaks (nm)
        
    Returns:
        Intensity array
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    intensity = np.zeros_like(wavelengths, dtype=np.float64)
    
    for peak in peaks:
        gaussian = peak.intensity * np.exp(
            -((wavelengths - peak.wavelength) ** 2) / (2 * sigma ** 2)
        )
        intensity += gaussian
    
    # Normalize to 0-1 range
    if intensity.max() > 0:
        intensity /= intensity.max()
    
    return intensity


def _generate_led_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """Generate realistic white LED spectrum.
    
    White LEDs have:
    - Narrow blue peak (~450nm, FWHM ~20nm)
    - Broad phosphor emission (~480-700nm, centered ~550nm)
    """
    intensity = np.zeros_like(wavelengths, dtype=np.float64)
    
    # Blue LED peak (narrow)
    blue_center = 450.0
    blue_fwhm = 20.0
    blue_sigma = blue_fwhm / (2 * np.sqrt(2 * np.log(2)))
    intensity += np.exp(-((wavelengths - blue_center) ** 2) / (2 * blue_sigma ** 2))
    
    # Phosphor emission (broad, asymmetric)
    # Modeled as sum of Gaussians for YAG:Ce phosphor
    phos_center = 555.0
    phos_fwhm = 100.0
    phos_sigma = phos_fwhm / (2 * np.sqrt(2 * np.log(2)))
    phosphor = 0.85 * np.exp(-((wavelengths - phos_center) ** 2) / (2 * phos_sigma ** 2))
    
    # Add slight red tail for more realistic shape
    red_tail = 0.2 * np.exp(-((wavelengths - 600) ** 2) / (2 * 40 ** 2))
    
    intensity += phosphor + red_tail
    
    # Normalize
    if intensity.max() > 0:
        intensity /= intensity.max()
    
    return intensity


def _generate_solar_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """Generate approximate solar spectrum with Fraunhofer lines.
    
    Solar spectrum is approximately a 5778K blackbody with absorption lines.
    """
    # Approximate blackbody continuum (simplified)
    # Peak around 500nm for 5778K
    continuum = np.exp(-((wavelengths - 500) ** 2) / (2 * 150 ** 2))
    
    # Add Fraunhofer absorption lines (dips)
    for line in SUN_LINES:
        sigma = 2.0  # Narrow absorption
        dip = line.intensity * 0.3 * np.exp(
            -((wavelengths - line.wavelength) ** 2) / (2 * sigma ** 2)
        )
        continuum = continuum - dip
    
    continuum = np.maximum(continuum, 0)
    
    if continuum.max() > 0:
        continuum /= continuum.max()
    
    return continuum


# Pre-built reference spectra
REFERENCE_SPECTRA: dict[ReferenceSource, ReferenceSpectrum] = {
    ReferenceSource.HG: ReferenceSpectrum(
        name="Mercury (Hg)",
        source=ReferenceSource.HG,
        peaks=HG_LINES,
        description="Low-pressure mercury vapor lamp emission lines",
    ),
    ReferenceSource.FL: ReferenceSpectrum(
        name="Fluorescent (FL)",
        source=ReferenceSource.FL,
        peaks=FL_LINES,
        description="Compact fluorescent lamp (CFL) with Hg + phosphors",
    ),
    ReferenceSource.SUN: ReferenceSpectrum(
        name="Solar (Sun)",
        source=ReferenceSource.SUN,
        peaks=SUN_LINES,
        description="Solar spectrum with Fraunhofer absorption lines",
    ),
    ReferenceSource.LED: ReferenceSpectrum(
        name="White LED",
        source=ReferenceSource.LED,
        peaks=LED_LINES,
        description="Phosphor-converted white LED (blue chip + YAG phosphor)",
    ),
}


def get_reference_spectrum(
    source: ReferenceSource,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """Get reference spectrum intensity for given wavelengths.
    
    Args:
        source: Reference source type
        wavelengths: Wavelength array to generate spectrum for
        
    Returns:
        Intensity array (normalized 0-1)
    """
    ref = REFERENCE_SPECTRA.get(source)
    if ref is None:
        return np.zeros_like(wavelengths)
    
    match source:
        case ReferenceSource.LED:
            return _generate_led_spectrum(wavelengths)
        case ReferenceSource.SUN:
            return _generate_solar_spectrum(wavelengths)
        case _:
            # Use Gaussian peaks for line spectra
            fwhm = 8.0 if source == ReferenceSource.HG else 12.0
            return _generate_gaussian_spectrum(wavelengths, ref.peaks, fwhm)


def get_reference_peaks(source: ReferenceSource) -> list[SpectralLine]:
    """Get calibration peaks for a reference source.
    
    Args:
        source: Reference source type
        
    Returns:
        List of spectral lines with wavelengths for calibration
    """
    ref = REFERENCE_SPECTRA.get(source)
    if ref is None:
        return []
    return ref.peaks.copy()


def get_reference_name(source: ReferenceSource) -> str:
    """Get human-readable name for reference source."""
    ref = REFERENCE_SPECTRA.get(source)
    return ref.name if ref else str(source.name)


def get_all_reference_names() -> list[tuple[ReferenceSource, str]]:
    """Get list of all available reference sources with names."""
    return [(src, get_reference_name(src)) for src in ReferenceSource]
