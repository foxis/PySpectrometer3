"""Peak detection processor for spectrum data."""

import numpy as np

from ..core.spectrum import Extremum, Peak, SpectrumData

from .base import ProcessorInterface

REL_HEIGHT_WIDTH = 1.0 / np.e
DEFAULT_MAX_EXTREMUMS = 20

try:
    from scipy.signal import find_peaks as scipy_find_peaks
    from scipy.signal import peak_widths as scipy_peak_widths

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


def _find_raw_peaks_and_dips(
    arr: np.ndarray,
    *,
    peak_prominence: float = 0.005,
    peak_min_dist: int = 8,
    dip_prominence: float = 0.025,
    dip_min_dist: int = 15,
) -> list[tuple[int, float, bool]]:
    """Returns (index, height_or_depth, is_dip)."""
    if not _SCIPY_AVAILABLE:
        return []
    rng = float(np.max(arr) - np.min(arr))
    if rng <= 0:
        return []
    prom_peak = max(peak_prominence * rng, 1e-12)
    prom_dip = max(dip_prominence * rng, 1e-12)
    result: list[tuple[int, float, bool]] = []
    pk_idx, _ = scipy_find_peaks(arr, prominence=prom_peak, distance=max(1, peak_min_dist))
    for i in pk_idx:
        result.append((int(i), float(arr[i]), False))
    inv = -arr
    dip_idx, _ = scipy_find_peaks(inv, prominence=prom_dip, distance=max(1, dip_min_dist))
    for i in dip_idx:
        depth = float(np.max(arr) - arr[i])
        result.append((int(i), depth, True))
    result.sort(key=lambda x: x[0])
    return result


def _height_and_width_at_rel_height(
    arr: np.ndarray,
    positions: np.ndarray,
    center_idx: int,
    height_raw: float,
    is_dip: bool,
    *,
    rel_height: float = REL_HEIGHT_WIDTH,
) -> tuple[float, float]:
    """Height = max(left, right). Width = 2 * min(left_half, right_half) in nm."""
    n = len(arr)
    if n < 2 or center_idx < 0 or center_idx >= n:
        return 0.0, 0.0
    disp = (positions[-1] - positions[0]) / max(n - 1, 1)
    if is_dip:
        thresh = np.max(arr) - height_raw * (1.0 - rel_height)
    else:
        thresh = height_raw * rel_height
    if thresh <= 0 or (is_dip and thresh >= np.max(arr)):
        return 0.0, 0.0
    left_ips = float(center_idx)
    for i in range(center_idx, 0, -1):
        if i > 0 and (
            (not is_dip and arr[i - 1] < thresh <= arr[i])
            or (is_dip and arr[i - 1] > thresh >= arr[i])
        ):
            t = (thresh - arr[i]) / (arr[i - 1] - arr[i]) if arr[i - 1] != arr[i] else 0.5
            left_ips = i - t
            break
    right_ips = float(center_idx)
    for i in range(center_idx, n - 1):
        if i + 1 < n and (
            (not is_dip and arr[i] >= thresh > arr[i + 1])
            or (is_dip and arr[i] <= thresh < arr[i + 1])
        ):
            t = (thresh - arr[i]) / (arr[i + 1] - arr[i]) if arr[i + 1] != arr[i] else 0.5
            right_ips = i + t
            break

    lo = max(0, int(np.floor(left_ips)))
    hi = min(n, int(np.ceil(right_ips)) + 1)
    apex = float(arr[center_idx])
    if is_dip:
        baseline_left = float(np.max(arr[lo : center_idx + 1])) if lo <= center_idx else apex
        baseline_right = float(np.max(arr[center_idx:hi])) if center_idx < hi else apex
        height = max(baseline_left - apex, baseline_right - apex)
    else:
        baseline_left = float(np.min(arr[lo : center_idx + 1])) if lo <= center_idx else apex
        baseline_right = float(np.min(arr[center_idx:hi])) if center_idx < hi else apex
        height = max(apex - baseline_left, apex - baseline_right)

    left_half = center_idx - left_ips
    right_half = right_ips - center_idx
    width = 2.0 * min(left_half, right_half) * disp
    return height, max(0.0, width)


def extract_extremums(
    intensity: np.ndarray,
    positions: np.ndarray,
    *,
    position_px: np.ndarray | None = None,
    max_count: int = DEFAULT_MAX_EXTREMUMS,
) -> list[Extremum]:
    """Extract peaks and dips. Take N strongest by abs(height). Sorted by position.

    Find all peaks (height positive), all dips (height negative), concatenate,
    sort by abs(height) descending, take max_count. No other filtering.
    """
    raw = _find_raw_peaks_and_dips(intensity)
    if not raw:
        return []
    peaks = [(i, h, False) for i, h, d in raw if not d]
    dips = [(i, h, True) for i, h, d in raw if d]
    combined = peaks + dips
    items: list[tuple[int, float, bool, float, float]] = []
    for idx, height_raw, is_dip in combined:
        h, w = _height_and_width_at_rel_height(
            intensity, positions, idx, height_raw, is_dip, rel_height=REL_HEIGHT_WIDTH
        )
        items.append((idx, height_raw, is_dip, h, w))
    items = sorted(items, key=lambda x: -abs(x[3]))[:max_count]
    items = sorted(items, key=lambda x: x[0])
    max_h = max(x[3] for x in items) or 1.0
    result: list[Extremum] = []
    for k, (idx, _, is_dip, h_computed, w_computed) in enumerate(items):
        pos = float(positions[idx])
        h_norm = h_computed / max_h if max_h > 0 else 0.5
        h_signed = h_norm if not is_dip else -h_norm
        px = int(position_px[idx]) if position_px is not None else None
        result.append(
            Extremum(
                index=k,
                position=pos,
                position_px=px,
                height=h_signed,
                width=w_computed,
                is_dip=is_dip,
            )
        )
    return result


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


def peak_widths_nm(
    intensity: np.ndarray,
    wavelengths: np.ndarray,
    peak_indices: list[int],
    rel_height: float = 0.5,
) -> list[float]:
    """Compute peak width in nm for each peak.

    Uses scipy.signal.peak_widths at rel_height:
    - 0.5 = FWHM (full width at half maximum)
    - 1/e ≈ 0.368 = width at 1/e of peak height

    Args:
        intensity: Signal array
        wavelengths: Wavelength per index (nm)
        peak_indices: Peak indices from find_peaks
        rel_height: Height fraction for width (0.5 = FWHM)

    Returns:
        Width in nm per peak, or 0.0 if unavailable
    """
    if not _SCIPY_AVAILABLE or not peak_indices:
        return [0.0] * len(peak_indices)

    arr = np.asarray(intensity, dtype=np.float64)
    idx = np.array(peak_indices, dtype=np.intp)
    try:
        widths_px, _, left_ips, right_ips = scipy_peak_widths(
            arr, idx, rel_height=rel_height
        )
    except (ValueError, IndexError):
        return [0.0] * len(peak_indices)

    n = len(wavelengths)
    if n < 2:
        return [0.0] * len(peak_indices)

    wl_span = float(wavelengths[-1] - wavelengths[0])
    disp_nm_per_px = wl_span / max(n - 1, 1)

    result = []
    for i in range(len(peak_indices)):
        w_px = float(widths_px[i]) if i < len(widths_px) else 0.0
        result.append(max(0.0, w_px * disp_nm_per_px))
    return result


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

        Uses extract_extremums (canonical), filters to peaks only, top N for display.
        """
        if not self._enabled:
            return data

        intensity = data.intensity.astype(np.float64)
        if np.max(intensity) <= 0:
            return data.with_peaks([])

        extremums = extract_extremums(
            intensity,
            data.wavelengths,
            position_px=np.arange(len(intensity), dtype=np.intp),
            max_count=self._max_count,
        )
        peaks = [
            Peak(
                index=e.index,
                wavelength=round(e.position, 1),
                intensity=float(abs(e.height)),
            )
            for e in extremums
            if not e.is_dip
        ]
        return data.with_peaks(peaks)
