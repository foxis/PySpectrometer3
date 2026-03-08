"""Peak detection processor for spectrum data."""

import numpy as np

from ..core.spectrum import SpectrumData, Peak
from .base import ProcessorInterface


def find_peak_indexes(
    y: np.ndarray,
    threshold: float = 0.3,
    min_dist: int = 1,
    threshold_abs: bool = False,
) -> np.ndarray:
    """Find peak indexes in a signal.
    
    This implementation is based on peakutils:
    https://bitbucket.org/lucashnegri/peakutils
    
    Copyright (c) 2014-2022 Lucas Hermann Negri. MIT License.
    
    Args:
        y: Input signal array
        threshold: Threshold for peak detection (relative to signal range)
        min_dist: Minimum distance between peaks in samples
        threshold_abs: If True, threshold is absolute; if False, relative
        
    Returns:
        Array of peak indexes
        
    Raises:
        ValueError: If y contains unsigned integers
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")
    
    if not threshold_abs:
        threshold = threshold * (np.max(y) - np.min(y)) + np.min(y)
    
    min_dist = int(min_dist)
    
    dy = np.diff(y)
    
    zeros = np.where(dy == 0)[0]
    
    if len(zeros) == len(y) - 1:
        return np.array([])
    
    if len(zeros):
        zeros_diff = np.diff(zeros)
        zeros_diff_not_one = np.add(np.where(zeros_diff != 1), 1)[0]
        zero_plateaus = np.split(zeros, zeros_diff_not_one)
        
        if zero_plateaus[0][0] == 0:
            dy[zero_plateaus[0]] = dy[zero_plateaus[0][-1] + 1]
            zero_plateaus.pop(0)
        
        if len(zero_plateaus) and zero_plateaus[-1][-1] == len(dy) - 1:
            dy[zero_plateaus[-1]] = dy[zero_plateaus[-1][0] - 1]
            zero_plateaus.pop(-1)
        
        for plateau in zero_plateaus:
            median = np.median(plateau)
            dy[plateau[plateau < median]] = dy[plateau[0] - 1]
            dy[plateau[plateau >= median]] = dy[plateau[-1] + 1]
    
    peaks = np.where(
        (np.hstack([dy, 0.0]) < 0.0)
        & (np.hstack([0.0, dy]) > 0.0)
        & (np.greater(y, threshold))
    )[0]
    
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False
        
        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False
        
        peaks = np.arange(y.size)[~rem]
    
    return peaks


class PeakDetector(ProcessorInterface):
    """Peak detection processor for spectrum data.
    
    This processor identifies peaks in the spectrum intensity data
    and adds them to the SpectrumData object.
    """
    
    def __init__(
        self,
        min_distance: int = 50,
        threshold: int = 20,
        min_distance_min: int = 0,
        min_distance_max: int = 100,
        threshold_min: int = 0,
        threshold_max: int = 100,
    ):
        """Initialize peak detector.
        
        Args:
            min_distance: Minimum distance between peaks in pixels
            threshold: Detection threshold (0-100)
            min_distance_min: Minimum allowed min_distance value
            min_distance_max: Maximum allowed min_distance value
            threshold_min: Minimum allowed threshold value
            threshold_max: Maximum allowed threshold value
        """
        self._min_distance = min_distance
        self._threshold = threshold
        self._min_distance_min = min_distance_min
        self._min_distance_max = min_distance_max
        self._threshold_min = threshold_min
        self._threshold_max = threshold_max
        self._enabled = True
    
    @property
    def name(self) -> str:
        return "Peak Detector"
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
    
    @property
    def min_distance(self) -> int:
        return self._min_distance
    
    @min_distance.setter
    def min_distance(self, value: int) -> None:
        self._min_distance = max(
            self._min_distance_min,
            min(self._min_distance_max, value)
        )
    
    @property
    def threshold(self) -> int:
        return self._threshold
    
    @threshold.setter
    def threshold(self, value: int) -> None:
        self._threshold = max(
            self._threshold_min,
            min(self._threshold_max, value)
        )
    
    def increase_min_distance(self) -> int:
        """Increase minimum distance by 1."""
        self.min_distance = self._min_distance + 1
        return self._min_distance
    
    def decrease_min_distance(self) -> int:
        """Decrease minimum distance by 1."""
        self.min_distance = self._min_distance - 1
        return self._min_distance
    
    def increase_threshold(self) -> int:
        """Increase threshold by 1."""
        self.threshold = self._threshold + 1
        return self._threshold
    
    def decrease_threshold(self) -> int:
        """Decrease threshold by 1."""
        self.threshold = self._threshold - 1
        return self._threshold
    
    def process(self, data: SpectrumData) -> SpectrumData:
        """Detect peaks in spectrum data.
        
        Args:
            data: Input spectrum data
            
        Returns:
            Spectrum data with detected peaks
        """
        if not self._enabled:
            return data
        
        intensity = data.intensity.astype(np.int32)
        max_val = max(intensity)
        
        if max_val == 0:
            return data.with_peaks([])
        
        threshold_normalized = self._threshold / max_val
        
        indexes = find_peak_indexes(
            intensity,
            threshold=threshold_normalized,
            min_dist=self._min_distance,
        )
        
        peaks = [
            Peak(
                index=int(idx),
                wavelength=round(data.wavelengths[idx], 1),
                intensity=int(intensity[idx]),
            )
            for idx in indexes
        ]
        
        return data.with_peaks(peaks)
