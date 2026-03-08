"""Signal filtering processors for spectrum data."""

from math import factorial
import numpy as np

from ..core.spectrum import SpectrumData
from .base import ProcessorInterface


def savitzky_golay(
    y: np.ndarray,
    window_size: int,
    order: int,
    deriv: int = 0,
    rate: int = 1,
) -> np.ndarray:
    """Apply Savitzky-Golay smoothing filter to data.
    
    This implementation is based on the SciPy cookbook recipe:
    https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
    
    Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
    All rights reserved. BSD 3-Clause License.
    
    Args:
        y: Input signal array
        window_size: Size of the smoothing window (must be odd)
        order: Polynomial order for fitting
        deriv: Order of derivative to compute (0 = smoothing only)
        rate: Rate parameter for derivative computation
        
    Returns:
        Smoothed signal array
        
    Raises:
        ValueError: If window_size or order are invalid
        TypeError: If window_size is not odd or too small
    """
    try:
        window_size = np.abs(np.int32(window_size))
        order = np.abs(np.int32(order))
    except ValueError as e:
        raise ValueError("window_size and order must be integers") from e
    
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomial order")
    
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    
    b = np.asmatrix([
        [k ** i for i in order_range]
        for k in range(-half_window, half_window + 1)
    ])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y_padded = np.concatenate((firstvals, y, lastvals))
    
    return np.convolve(m[::-1], y_padded, mode="valid")


class SavitzkyGolayFilter(ProcessorInterface):
    """Savitzky-Golay smoothing filter processor.
    
    This processor applies a Savitzky-Golay polynomial smoothing filter
    to the spectrum intensity data, reducing noise while preserving
    peak shapes.
    """
    
    def __init__(
        self,
        window_size: int = 17,
        poly_order: int = 7,
        poly_order_min: int = 0,
        poly_order_max: int = 15,
    ):
        """Initialize Savitzky-Golay filter.
        
        Args:
            window_size: Size of the smoothing window (must be odd)
            poly_order: Polynomial order for fitting
            poly_order_min: Minimum allowed polynomial order
            poly_order_max: Maximum allowed polynomial order
        """
        self._window_size = window_size
        self._poly_order = poly_order
        self._poly_order_min = poly_order_min
        self._poly_order_max = poly_order_max
        self._enabled = True
    
    @property
    def name(self) -> str:
        return "Savitzky-Golay Filter"
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
    
    @property
    def window_size(self) -> int:
        return self._window_size
    
    @window_size.setter
    def window_size(self, value: int) -> None:
        if value % 2 == 0:
            value += 1
        self._window_size = max(3, value)
    
    @property
    def poly_order(self) -> int:
        return self._poly_order
    
    @poly_order.setter
    def poly_order(self, value: int) -> None:
        self._poly_order = max(
            self._poly_order_min,
            min(self._poly_order_max, value)
        )
    
    def increase_poly_order(self) -> int:
        """Increase polynomial order by 1."""
        self.poly_order = self._poly_order + 1
        return self._poly_order
    
    def decrease_poly_order(self) -> int:
        """Decrease polynomial order by 1."""
        self.poly_order = self._poly_order - 1
        return self._poly_order
    
    def process(self, data: SpectrumData) -> SpectrumData:
        """Apply Savitzky-Golay filter to spectrum intensity.

        Preserves bit depth: clips to 0-255 for 8-bit, 0-1023 for 10-bit,
        0-65535 for 16-bit based on input data range.
        
        Args:
            data: Input spectrum data
            
        Returns:
            Spectrum data with smoothed intensity values
        """
        if not self._enabled:
            return data

        intensity = data.intensity.astype(np.float64)
        max_val = float(np.max(intensity)) if intensity.size > 0 else 255.0
        # Preserve bit depth: 8-bit (255), 10-bit (1023), 16-bit (65535)
        clip_max = 255.0 if max_val <= 255 else (65535.0 if max_val > 1023 else 1023.0)
        
        smoothed = savitzky_golay(
            intensity,
            self._window_size,
            self._poly_order,
        )
        
        smoothed = np.clip(smoothed, 0.0, clip_max).astype(np.float32)
        
        return data.with_intensity(smoothed)
