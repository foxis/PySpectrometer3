"""Peak detection processor for spectrum data."""

import numpy as np

from ..core.spectrum import Peak, SpectrumData
from .base import ProcessorInterface

try:
    from scipy.signal import find_peaks as scipy_find_peaks

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


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
        (np.hstack([dy, 0.0]) < 0.0) & (np.hstack([0.0, dy]) > 0.0) & (np.greater(y, threshold))
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


def find_peak_indexes_scipy(
    y: np.ndarray,
    *,
    threshold: float = 0.2,
    min_dist: int = 15,
    prominence: float = 0.03,
) -> np.ndarray:
    """Find peak indexes using scipy.signal.find_peaks, tuned for sharp emission lines.

    Sharp Hg/fluorescent lines (1-5 pixels wide) are detected reliably with low
    prominence and distance. Prominence filters noise while allowing weak sharp peaks.

    Args:
        y: Input signal (will be cast to float64)
        threshold: Minimum height as fraction of range (0-1)
        min_dist: Minimum distance between peaks in samples
        prominence: Minimum prominence as fraction of range (0-1). Lower = more sensitive.

    Returns:
        Array of peak indexes, sorted by position.
    """
    if not _SCIPY_AVAILABLE:
        return np.array([])

    arr = np.asarray(y, dtype=np.float64)
    if arr.size < 3:
        return np.array([])

    rng = float(np.max(arr) - np.min(arr))
    if rng <= 0:
        return np.array([])

    height = float(np.min(arr) + threshold * rng)
    prom = float(prominence * rng)

    idx, _ = scipy_find_peaks(
        arr,
        height=height,
        prominence=prom,
        distance=max(1, int(min_dist)),
    )
    return np.asarray(idx, dtype=np.intp)


def find_peaks(
    intensity: np.ndarray,
    wavelengths: np.ndarray,
    *,
    threshold: float = 0.1,
    min_dist: int = 15,
    prominence: float = 0.01,
) -> list[Peak]:
    """Find all peaks above threshold. No display coupling (no max_count).

    Core peak detection used by PeakDetector and any consumer needing peaks.
    """
    if not _SCIPY_AVAILABLE or intensity.size < 3:
        return []

    idx = find_peak_indexes_scipy(
        intensity,
        threshold=threshold,
        min_dist=min_dist,
        prominence=prominence,
    )
    arr = np.asarray(intensity, dtype=np.float64)
    return [
        Peak(
            index=int(i),
            wavelength=float(wavelengths[min(i, len(wavelengths) - 1)]),
            intensity=float(arr[i]),
        )
        for i in idx
    ]


def detect_peaks_in_region(
    intensity: np.ndarray,
    wavelengths: np.ndarray,
    center: int,
    half_width: int,
    *,
    threshold: float = 0.2,
    min_dist: int = 15,
    prominence: float = 0.01,
) -> list[Peak]:
    """Find peaks in a region [center - half_width, center + half_width].

    Uses same algorithm as PeakDetector. Returns list of Peak with global indices.

    Args:
        intensity: Full intensity array
        wavelengths: Full wavelength array
        center: Center index of region
        half_width: Half-width of region in pixels
        threshold: Min height as fraction of range (0-1)
        min_dist: Min distance between peaks
        prominence: Min prominence as fraction of range

    Returns:
        List of Peak objects with global indices
    """
    if not _SCIPY_AVAILABLE or intensity.size < 3:
        return []

    lo = max(0, center - half_width)
    hi = min(len(intensity), center + half_width + 1)
    if hi <= lo + 2:
        return []

    slice_intensity = intensity[lo:hi].astype(np.float64)
    rng = float(np.max(slice_intensity) - np.min(slice_intensity))
    if rng <= 0:
        return []

    height = float(np.min(slice_intensity) + threshold * rng)
    prom = float(prominence * rng)

    idx_local, _ = scipy_find_peaks(
        slice_intensity,
        height=height,
        prominence=prom,
        distance=max(1, int(min_dist)),
    )
    peaks = [
        Peak(
            index=int(lo + i),
            wavelength=round(wavelengths[lo + i], 1),
            intensity=float(intensity[lo + i]),
        )
        for i in idx_local
    ]
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
        max_count: int = 10,
        min_distance_min: int = 0,
        min_distance_max: int = 100,
        threshold_min: int = 0,
        threshold_max: int = 100,
        max_count_min: int = 1,
        max_count_max: int = 50,
    ):
        """Initialize peak detector.

        Args:
            min_distance: Minimum distance between peaks in pixels
            threshold: Detection threshold (0-100), min height as % of range
            max_count: Max peaks to display (all above threshold, sorted by size, top N)
            min_distance_min: Minimum allowed min_distance value
            min_distance_max: Maximum allowed min_distance value
            threshold_min: Minimum allowed threshold value
            threshold_max: Maximum allowed threshold value
            max_count_min: Minimum allowed max_count value
            max_count_max: Maximum allowed max_count value
        """
        self._min_distance = min_distance
        self._threshold = threshold
        self._max_count = max_count
        self._min_distance_min = min_distance_min
        self._min_distance_max = min_distance_max
        self._threshold_min = threshold_min
        self._threshold_max = threshold_max
        self._max_count_min = max_count_min
        self._max_count_max = max_count_max
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
        self._min_distance = max(self._min_distance_min, min(self._min_distance_max, value))

    @property
    def threshold(self) -> int:
        return self._threshold

    @threshold.setter
    def threshold(self, value: int) -> None:
        self._threshold = max(self._threshold_min, min(self._threshold_max, value))

    @property
    def max_count(self) -> int:
        return self._max_count

    @max_count.setter
    def max_count(self, value: int) -> None:
        self._max_count = max(self._max_count_min, min(self._max_count_max, value))

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

        Uses find_peaks (core detection), then limits to top N for display.
        """
        if not self._enabled:
            return data

        intensity = data.intensity.astype(np.float64)
        if np.max(intensity) <= 0:
            return data.with_peaks([])

        threshold_norm = self._threshold / 100.0
        if _SCIPY_AVAILABLE:
            peaks = find_peaks(
                intensity,
                data.wavelengths,
                threshold=threshold_norm,
                min_dist=self._min_distance,
                prominence=0.01,
            )
        else:
            indexes = find_peak_indexes(
                intensity,
                threshold=threshold_norm,
                min_dist=self._min_distance,
            )
            peaks = [
                Peak(
                    index=int(i),
                    wavelength=round(data.wavelengths[i], 1),
                    intensity=float(intensity[i]),
                )
                for i in indexes
            ]

        # Display limit: top N by intensity
        peaks = sorted(peaks, key=lambda p: p.intensity, reverse=True)[: self._max_count]
        return data.with_peaks(peaks)
