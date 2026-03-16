"""Waterfall display for time-varying spectrum visualization."""

import cv2
import numpy as np

from ..core.spectrum import SpectrumData
from ..utils.color import apply_luminosity, rgb_to_bgr, wavelength_to_rgb


class WaterfallDisplay:
    """Displays spectrum changes over time in a waterfall format.

    The waterfall display shows successive spectrum measurements as
    horizontal lines, with the most recent at the top. Each pixel's
    color represents the wavelength, and brightness represents intensity.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 320,
        contrast: float = 2.5,
        brightness: int = 10,
    ):
        """Initialize waterfall display.

        Args:
            width: Width of the waterfall display in pixels
            height: Height of the waterfall display (number of history lines)
            contrast: Contrast adjustment factor
            brightness: Brightness offset
        """
        self.width = width
        self.height = height
        self.contrast = contrast
        self.brightness = brightness

        self._buffer = np.zeros([height, width, 3], dtype=np.uint8)

    def set_dimensions(self, width: int, height: int) -> None:
        """Set display height (buffer height). Buffer width follows data length in update()."""
        if height == self.height:
            return
        self.width = width
        self.height = height
        self._buffer = np.zeros([height, 1, 3], dtype=np.uint8)

    def update(self, data: SpectrumData) -> None:
        """Add a new spectrum line to the waterfall.

        Buffer has one column per spectrum point so viewport cropping aligns with
        the spectrum preview strip and graph.
        """
        data_len = len(data.intensity)
        if data_len == 0:
            return
        if self._buffer.shape[1] != data_len or self._buffer.shape[0] != self.height:
            self._buffer = np.zeros([self.height, data_len, 3], dtype=np.uint8)

        line = np.zeros([1, data_len, 3], dtype=np.uint8)
        for i, intensity in enumerate(data.intensity):
            wavelength = round(data.wavelengths[i])
            rgb = wavelength_to_rgb(wavelength)
            luminosity = float(intensity)
            scaled = apply_luminosity(rgb, luminosity)
            line[0, i] = rgb_to_bgr(scaled)

        line = cv2.addWeighted(
            line,
            self.contrast,
            line,
            0,
            self.brightness,
        )

        self._buffer = np.insert(self._buffer, 0, line, axis=0)
        self._buffer = self._buffer[: self.height].copy()

    def render(self) -> np.ndarray:
        """Get the current waterfall display image.

        Returns:
            Waterfall image as numpy array (BGR format)
        """
        return self._buffer.copy()

    def clear(self) -> None:
        """Clear the waterfall display."""
        self._buffer.fill(0)

    @property
    def buffer(self) -> np.ndarray:
        """Direct access to the waterfall buffer (read-only copy)."""
        return self._buffer.copy()
