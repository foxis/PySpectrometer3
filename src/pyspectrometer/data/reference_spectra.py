"""Reference spectra for calibration.

Uses colour-science for actual CIE reference spectra:
- D65: CIE Illuminant D65 (daylight, 6504K) – the standard daylight spectrum
- A: CIE Standard Illuminant A (incandescent, 2856K)
- FL1–FL5: CIE fluorescent illuminants (actual spectra)
- LED: CIE LED-B1 (phosphor white LED)
- Hg: Low-pressure mercury emission lines (for wavelength calibration)

All spectra from colour.SDS_ILLUMINANTS are CIE-standard, not approximations.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import numpy as np

# colour-science provides CIE-standard spectral distributions
try:
    import colour
    _COLOUR_AVAILABLE = True
except ImportError:
    _COLOUR_AVAILABLE = False


class ReferenceSource(Enum):
    """Available reference light sources for calibration."""

    FL = auto()      # CIE FL1 - Daylight fluorescent
    FL2 = auto()     # CIE FL2 - Cool white fluorescent
    FL3 = auto()     # CIE FL3 - White fluorescent
    FL4 = auto()     # CIE FL4 - Warm white fluorescent
    FL5 = auto()     # CIE FL5 - Daylight fluorescent (alternative)
    A = auto()       # CIE Standard Illuminant A (incandescent)
    D65 = auto()     # CIE Illuminant D65 (daylight 6504K) - actual spectrum
    HG = auto()      # Mercury low-pressure lamp
    LED = auto()     # CIE LED-B1 (phosphor white LED)


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
    colour_key: Optional[str] = None  # key in colour.SDS_ILLUMINANTS


# colour-science SDS keys for each ReferenceSource
_COLOUR_SDS_MAP = {
    ReferenceSource.FL: "FL1",
    ReferenceSource.FL2: "FL2",
    ReferenceSource.FL3: "FL3",
    ReferenceSource.FL4: "FL4",
    ReferenceSource.FL5: "FL5",
    ReferenceSource.A: "A",
    ReferenceSource.D65: "D65",
    ReferenceSource.LED: "LED-B1",
}

# Mercury low-pressure lamp emission lines (for peak-matching calibration)
# https://physics.nist.gov/PhysRefData/Handbook/Tables/mercurytable2.htm
HG_LINES = [
    SpectralLine(404.66, 0.6, "Hg"),
    SpectralLine(435.84, 0.9, "Hg"),
    SpectralLine(546.07, 1.0, "Hg"),
    SpectralLine(576.96, 0.5, "Hg"),
    SpectralLine(579.07, 0.5, "Hg"),
]

# CFL / fluorescent phosphor peaks (for peak-matching when using FL sources)
FL_LINES = [
    SpectralLine(404.66, 0.4, "Hg"),
    SpectralLine(435.84, 0.7, "Hg"),
    SpectralLine(487.0, 0.5, "Tb"),
    SpectralLine(543.0, 0.8, "Tb"),
    SpectralLine(546.07, 0.9, "Hg"),
    SpectralLine(578.0, 0.4, "Hg"),
    SpectralLine(584.0, 0.6, "Eu"),
    SpectralLine(611.0, 1.0, "Eu"),
    SpectralLine(629.0, 0.5, "Eu"),
]

# D65 daylight reference lines (for peak-matching if needed)
D65_LINES = [
    SpectralLine(393.37, 0.8, "K"),
    SpectralLine(486.13, 0.6, "F"),
    SpectralLine(589.00, 0.9, "D1"),
    SpectralLine(589.59, 0.9, "D2"),
    SpectralLine(656.28, 1.0, "C"),
]

# White phosphor LED
LED_LINES = [
    SpectralLine(450.0, 1.0, "Blue"),
    SpectralLine(550.0, 0.9, "Phos"),
    SpectralLine(630.0, 0.4, "Red"),
]

REFERENCE_SPECTRA: dict[ReferenceSource, ReferenceSpectrum] = {
    ReferenceSource.HG: ReferenceSpectrum(
        name="Mercury (Hg)",
        source=ReferenceSource.HG,
        peaks=HG_LINES,
        description="Low-pressure mercury vapor lamp emission lines",
        colour_key=None,
    ),
    ReferenceSource.FL: ReferenceSpectrum(
        name="FL1",
        source=ReferenceSource.FL,
        peaks=FL_LINES,
        description="CIE fluorescent illuminant FL1, Daylight fluorescent",
        colour_key="FL1",
    ),
    ReferenceSource.FL2: ReferenceSpectrum(
        name="FL2",
        source=ReferenceSource.FL2,
        peaks=FL_LINES,
        description="CIE fluorescent illuminant FL2, Cool white fluorescent",
        colour_key="FL2",
    ),
    ReferenceSource.FL3: ReferenceSpectrum(
        name="FL3",
        source=ReferenceSource.FL3,
        peaks=FL_LINES,
        description="CIE fluorescent illuminant FL3, White fluorescent",
        colour_key="FL3",
    ),
    ReferenceSource.FL4: ReferenceSpectrum(
        name="FL4",
        source=ReferenceSource.FL4,
        peaks=FL_LINES,
        description="CIE fluorescent illuminant FL4, Warm white fluorescent",
        colour_key="FL4",
    ),
    ReferenceSource.FL5: ReferenceSpectrum(
        name="FL5",
        source=ReferenceSource.FL5,
        peaks=FL_LINES,
        description="CIE fluorescent illuminant FL5, Daylight fluorescent",
        colour_key="FL5",
    ),
    ReferenceSource.A: ReferenceSpectrum(
        name="Illum A",
        source=ReferenceSource.A,
        peaks=[],
        description="CIE Standard Illuminant A (incandescent)",
        colour_key="A",
    ),
    ReferenceSource.D65: ReferenceSpectrum(
        name="D65",
        source=ReferenceSource.D65,
        peaks=D65_LINES,
        description="CIE Illuminant D65 (daylight 6504K)",
        colour_key="D65",
    ),
    ReferenceSource.LED: ReferenceSpectrum(
        name="White LED",
        source=ReferenceSource.LED,
        peaks=LED_LINES,
        description="CIE LED-B1 (phosphor white LED)",
        colour_key="LED-B1",
    ),
}


def _from_colour_sds(wavelengths: np.ndarray, colour_key: str) -> np.ndarray:
    """Get actual CIE spectrum from colour-science, interpolated to wavelengths."""
    if not _COLOUR_AVAILABLE:
        return np.zeros_like(wavelengths, dtype=np.float64)
    try:
        sd = colour.SDS_ILLUMINANTS[colour_key]
        values = np.array([float(sd[w]) for w in wavelengths], dtype=np.float64)
        if values.max() > 0:
            values = values / values.max()
        return values
    except (KeyError, Exception):
        return np.zeros_like(wavelengths, dtype=np.float64)


def _generate_hg_spectrum(wavelengths: np.ndarray) -> np.ndarray:
    """Generate Hg low-pressure spectrum from known emission lines.

    colour-science has HP1–HP5 (high-pressure Hg), not low-pressure.
    Low-pressure Hg is a line source – sharp lines (~1.5 nm FWHM).
    """
    fwhm = 1.5
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    intensity = np.zeros_like(wavelengths, dtype=np.float64)
    for peak in HG_LINES:
        intensity += peak.intensity * np.exp(
            -((wavelengths - peak.wavelength) ** 2) / (2 * sigma ** 2)
        )
    if intensity.max() > 0:
        intensity /= intensity.max()
    return intensity


def get_reference_spectrum(source: ReferenceSource, wavelengths: np.ndarray) -> np.ndarray:
    """Get reference spectrum intensity for given wavelengths.

    Uses colour-science for actual CIE spectra (D65, A, FL1–FL5, LED-B1).
    Hg uses known emission lines (colour has HP1–HP5, not low-pressure Hg).

    Args:
        source: Reference source type
        wavelengths: Wavelength array (nm)

    Returns:
        Intensity array, normalized 0–1
    """
    colour_key = _COLOUR_SDS_MAP.get(source)
    if colour_key is not None and _COLOUR_AVAILABLE:
        return _from_colour_sds(wavelengths, colour_key)

    if source == ReferenceSource.HG:
        return _generate_hg_spectrum(wavelengths)

    return np.zeros_like(wavelengths, dtype=np.float64)


def get_reference_peaks(source: ReferenceSource) -> list[SpectralLine]:
    """Get calibration peaks for a reference source."""
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


# XYZ conversion via colour-science (for spectral -> XYZ when needed)
def spectrum_to_xyz_1931(
    wavelengths: np.ndarray,
    values: np.ndarray,
) -> tuple[float, float, float]:
    """Convert spectrum to CIE XYZ (1931 2° observer) using colour-science.

    Args:
        wavelengths: Wavelengths (nm)
        values: Spectral power distribution

    Returns:
        (X, Y, Z) tuple
    """
    if not _COLOUR_AVAILABLE:
        return (0.0, 0.0, 0.0)
    try:
        sd = colour.SpectralDistribution(
            dict(zip(wavelengths.astype(float), values.astype(float)))
        )
        xyz = colour.sd_to_XYZ(sd)
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    except Exception:
        return (0.0, 0.0, 0.0)
