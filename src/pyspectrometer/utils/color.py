"""Color utilities for spectrum visualization."""

from typing import NamedTuple


class RGB(NamedTuple):
    """RGB color tuple."""
    r: int
    g: int
    b: int


# Default gray color for wavelengths outside visible range
DEFAULT_GRAY = RGB(155, 155, 155)


def wavelength_to_rgb(nm: float, gamma: float = 0.8) -> RGB:
    """Convert a wavelength in nanometers to an RGB color.
    
    This function converts visible light wavelengths (380-780nm) to
    approximate RGB values for display purposes.
    
    Based on code by Chris Webb:
    https://www.codedrome.com/exploring-the-visible-spectrum-in-python/
    
    Args:
        nm: Wavelength in nanometers
        gamma: Gamma correction factor (default 0.8)
        
    Returns:
        RGB tuple with values 0-255
    """
    max_intensity = 255
    factor = 0.0
    r = 0.0
    g = 0.0
    b = 0.0
    
    if 380 <= nm <= 439:
        r = -(nm - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif 440 <= nm <= 489:
        r = 0.0
        g = (nm - 440) / (490 - 440)
        b = 1.0
    elif 490 <= nm <= 509:
        r = 0.0
        g = 1.0
        b = -(nm - 510) / (510 - 490)
    elif 510 <= nm <= 579:
        r = (nm - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif 580 <= nm <= 644:
        r = 1.0
        g = -(nm - 645) / (645 - 580)
        b = 0.0
    elif 645 <= nm <= 780:
        r = 1.0
        g = 0.0
        b = 0.0
    
    if 380 <= nm <= 419:
        factor = 0.3 + 0.7 * (nm - 380) / (420 - 380)
    elif 420 <= nm <= 700:
        factor = 1.0
    elif 701 <= nm <= 780:
        factor = 0.3 + 0.7 * (780 - nm) / (780 - 700)
    
    r_out = int(max_intensity * ((r * factor) ** gamma)) if r > 0 else 0
    g_out = int(max_intensity * ((g * factor) ** gamma)) if g > 0 else 0
    b_out = int(max_intensity * ((b * factor) ** gamma)) if b > 0 else 0
    
    if r_out + g_out + b_out == 0:
        return DEFAULT_GRAY
    
    return RGB(r_out, g_out, b_out)


def rgb_to_bgr(rgb: RGB) -> tuple[int, int, int]:
    """Convert RGB to BGR for OpenCV."""
    return (rgb.b, rgb.g, rgb.r)


def apply_luminosity(rgb: RGB, luminosity: float) -> RGB:
    """Apply luminosity scaling to an RGB color.
    
    Args:
        rgb: Original RGB color
        luminosity: Luminosity factor (0.0 to 1.0)
        
    Returns:
        Scaled RGB color
    """
    return RGB(
        int(round(rgb.r * luminosity)),
        int(round(rgb.g * luminosity)),
        int(round(rgb.b * luminosity)),
    )
