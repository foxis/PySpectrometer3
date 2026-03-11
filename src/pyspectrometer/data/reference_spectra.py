"""Reference spectra for calibration.

Loads from data/reference/*.csv first, then falls back to colour-science.
- Hg: Low-pressure mercury emission lines (built-in)
- D65, FL1, FL2, FL3, FL12: from CIE_*.csv in data/reference/
- LED1–LED3: from colour-science (no CSV in reference yet)
"""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

# colour-science provides CIE-standard spectral distributions
try:
    import colour

    _COLOUR_AVAILABLE = True
except ImportError:
    _COLOUR_AVAILABLE = False


class ReferenceSource(Enum):
    """Available reference light sources for calibration."""

    HG = auto()  # Mercury low-pressure lamp
    D65 = auto()  # CIE Illuminant D65 (daylight 6504K)
    FL1 = auto()  # CIE FL1 - Daylight fluorescent
    FL2 = auto()  # CIE FL2 - Cool white fluorescent
    FL3 = auto()  # CIE FL3 - White fluorescent
    FL12 = auto()  # CIE FL12 - Daylight fluorescent (narrow band)
    LED = auto()  # CIE LED-B1 (phosphor white LED)
    LED2 = auto()  # CIE LED-B2 (phosphor white LED)
    LED3 = auto()  # CIE LED-B3 (phosphor white LED)


@dataclass
class SpectralLine:
    """A spectral emission or absorption line."""

    wavelength: float  # nm
    intensity: float  # relative intensity (0-1)
    label: str = ""  # optional label (e.g., "Hg", "Na-D")


@dataclass
class ReferenceSpectrum:
    """Reference spectrum with peaks and continuous data."""

    name: str
    source: ReferenceSource
    peaks: list[SpectralLine]
    description: str = ""
    colour_key: str | None = None  # key in colour.SDS_ILLUMINANTS


# colour-science SDS keys for each ReferenceSource
_COLOUR_SDS_MAP = {
    ReferenceSource.FL1: "FL1",
    ReferenceSource.FL2: "FL2",
    ReferenceSource.FL3: "FL3",
    ReferenceSource.FL12: "FL12",
    ReferenceSource.D65: "D65",
    ReferenceSource.LED: "LED-B1",
    ReferenceSource.LED2: "LED-B2",
    ReferenceSource.LED3: "LED-B3",
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
        name="Hg",
        source=ReferenceSource.HG,
        peaks=HG_LINES,
        description="Low-pressure mercury vapor lamp emission lines",
        colour_key=None,
    ),
    ReferenceSource.D65: ReferenceSpectrum(
        name="D65",
        source=ReferenceSource.D65,
        peaks=D65_LINES,
        description="CIE Illuminant D65 (daylight 6504K)",
        colour_key="D65",
    ),
    ReferenceSource.FL1: ReferenceSpectrum(
        name="FL1",
        source=ReferenceSource.FL1,
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
    ReferenceSource.FL12: ReferenceSpectrum(
        name="FL12",
        source=ReferenceSource.FL12,
        peaks=FL_LINES,
        description="CIE fluorescent illuminant FL12, Daylight fluorescent (narrow band)",
        colour_key="FL12",
    ),
    ReferenceSource.LED: ReferenceSpectrum(
        name="LED1",
        source=ReferenceSource.LED,
        peaks=LED_LINES,
        description="CIE LED-B1 (phosphor white LED)",
        colour_key="LED-B1",
    ),
    ReferenceSource.LED2: ReferenceSpectrum(
        name="LED2",
        source=ReferenceSource.LED2,
        peaks=LED_LINES,
        description="CIE LED-B2 (phosphor white LED)",
        colour_key="LED-B2",
    ),
    ReferenceSource.LED3: ReferenceSpectrum(
        name="LED3",
        source=ReferenceSource.LED3,
        peaks=LED_LINES,
        description="CIE LED-B3 (phosphor white LED)",
        colour_key="LED-B3",
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
            -((wavelengths - peak.wavelength) ** 2) / (2 * sigma**2)
        )
    if intensity.max() > 0:
        intensity /= intensity.max()
    return intensity


def get_reference_spectrum(source: ReferenceSource, wavelengths: np.ndarray) -> np.ndarray:
    """Get reference spectrum intensity for given wavelengths.

    Tries data/reference/*.csv first, then colour-science. Hg uses built-in lines.

    Args:
        source: Reference source type
        wavelengths: Wavelength array (nm)

    Returns:
        Intensity array, normalized 0–1
    """
    from .reference_loader import get_reference_spectrum_from_files

    from_file = get_reference_spectrum_from_files(source, wavelengths)
    if from_file is not None:
        return from_file

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


def list_reference_files() -> list[tuple[str, str]]:
    """List CSV files in data/reference. For UI 'load more' support."""
    from .reference_loader import list_available_reference_files

    return list_available_reference_files()


def reload_reference_files() -> None:
    """Clear loaded spectrum cache (call after adding new reference files)."""
    from .reference_loader import clear_spectrum_cache

    clear_spectrum_cache()


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
        sd = colour.SpectralDistribution(dict(zip(wavelengths.astype(float), values.astype(float))))
        xyz = colour.sd_to_XYZ(sd)
        return (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    except Exception:
        return (0.0, 0.0, 0.0)
