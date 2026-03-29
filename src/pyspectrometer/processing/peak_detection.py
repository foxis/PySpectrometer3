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


def nearest_among_peak_indices(
    candidate_idx: int,
    peak_indices: np.ndarray,
    *,
    n: int,
) -> int:
    """Pick the peak index closest to *candidate_idx* in sample space (same as marker snap).

    Used by calibration assist and any UI that snaps a data index to a precomputed
    peak list (e.g. from :func:`find_peaks` or :func:`find_peak_indexes_scipy`).
    """
    ci = int(np.clip(int(candidate_idx), 0, max(0, n - 1)))
    pk = np.asarray(peak_indices, dtype=np.intp).ravel()
    if pk.size == 0:
        return ci
    return int(pk[int(np.argmin(np.abs(pk - ci)))])


def snap_to_nearest_peak_index(
    y: np.ndarray,
    idx: int,
    *,
    wavelengths: np.ndarray | None = None,
    window: int = 48,
) -> int:
    """Snap *idx* to the nearest peak on *y*; if none detected, use argmax in a local window.

    When *wavelengths* is provided and matches *y* in length, uses :func:`find_peaks`
    (same defaults as the processing pipeline). Otherwise uses
    :func:`find_peak_indexes_scipy`.
    """
    arr = np.asarray(y, dtype=np.float64)
    n = arr.size
    if n < 1:
        return 0
    idx = int(np.clip(idx, 0, n - 1))
    wl = wavelengths
    if wl is not None and len(wl) == n:
        peaks_list = find_peaks(arr, wl)
        peak_ix = np.array([p.index for p in peaks_list], dtype=np.intp)
    else:
        peak_ix = find_peak_indexes_scipy(arr)
    if peak_ix.size > 0:
        return nearest_among_peak_indices(idx, peak_ix, n=n)
    half = max(1, window // 2)
    lo = max(0, idx - half)
    hi = min(n, idx + half + 1)
    seg = arr[lo:hi]
    if seg.size == 0:
        return idx
    return int(lo + int(np.argmax(seg)))


def _widths_nm(
    arr: np.ndarray,
    positions: np.ndarray,
    indices: np.ndarray,
    is_dip: bool,
    rel_height: float = REL_HEIGHT_WIDTH,
) -> list[float]:
    """Width in nm from scipy peak_widths at rel_height."""
    if not _SCIPY_AVAILABLE or indices.size == 0:
        return []
    n = len(positions)
    disp = (positions[-1] - positions[0]) / max(n - 1, 1)
    signal = np.asarray(-arr if is_dip else arr, dtype=np.float64)
    try:
        widths_px, _, _, _ = scipy_peak_widths(
            signal, indices.astype(np.intp), rel_height=rel_height
        )
    except (ValueError, IndexError):
        return [disp] * len(indices)
    return [float(w) * disp if w > 0 else disp for w in widths_px]


def extract_peaks(
    intensity: np.ndarray,
    positions: np.ndarray,
    *,
    prominence: float = 0.005,
    min_dist: int = 8,
    rel_height: float = REL_HEIGHT_WIDTH,
) -> list[tuple[int, float, float, float]]:
    """Extract peaks. Returns (idx, position, height, width) per peak.

    Height from scipy prominence. Width from scipy peak_widths at rel_height.
    """
    if not _SCIPY_AVAILABLE or intensity.size < 3:
        return []
    arr = np.asarray(intensity, dtype=np.float64)
    rng = float(np.max(arr) - np.min(arr))
    if rng <= 0:
        return []
    prom = max(prominence * rng, 1e-12)
    idx, props = scipy_find_peaks(
        arr,
        prominence=prom,
        distance=max(1, min_dist),
    )
    if idx.size == 0:
        return []
    heights = props["prominences"]
    widths = _widths_nm(arr, positions, idx, is_dip=False, rel_height=rel_height)
    n = len(positions)
    result = []
    for i in range(idx.size):
        px = int(idx[i])
        pos = float(positions[px])
        h = float(heights[i])
        w = widths[i] if i < len(widths) else 0.0
        result.append((px, pos, h, w))
    return result


def extract_dips(
    intensity: np.ndarray,
    positions: np.ndarray,
    *,
    prominence: float = 0.08,
    min_dist: int = 15,
    rel_height: float = REL_HEIGHT_WIDTH,
    valley_only: bool = True,
) -> list[tuple[int, float, float, float]]:
    """Extract dips (valleys). Returns (idx, position, height, width) per dip.

    Only valleys between two peaks when valley_only=True.
    Height = depth (scipy prominence on -arr). Width from scipy peak_widths.
    """
    if not _SCIPY_AVAILABLE or intensity.size < 3:
        return []
    arr = np.asarray(intensity, dtype=np.float64)
    rng = float(np.max(arr) - np.min(arr))
    if rng <= 0:
        return []
    prom = max(prominence * rng, 1e-12)
    inv = -arr
    pk_idx, _ = scipy_find_peaks(arr, prominence=prom * 0.5, distance=1)
    peak_set = set(int(p) for p in pk_idx)
    dip_idx, props = scipy_find_peaks(
        inv,
        prominence=prom,
        distance=max(1, min_dist),
    )
    if dip_idx.size == 0:
        return []
    heights = props["prominences"]
    widths = _widths_nm(arr, positions, dip_idx, is_dip=True, rel_height=rel_height)
    result = []
    for i in range(dip_idx.size):
        px = int(dip_idx[i])
        if valley_only and peak_set:
            left_p = max((p for p in peak_set if p < px), default=None)
            right_p = min((p for p in peak_set if p > px), default=None)
            if left_p is None or right_p is None:
                continue
        pos = float(positions[px])
        h = float(heights[i])
        w = widths[i] if i < len(widths) else 0.0
        result.append((px, pos, h, w))
    return result


def debug_raw_peaks_and_dips(
    intensity: np.ndarray,
    positions: np.ndarray,
    *,
    peak_prominence: float = 0.003,
    dip_prominence: float = 0.02,
    no_valley_filter: bool = False,
) -> tuple[list[tuple[int, float, float, float]], list[tuple[int, float, float, float]]]:
    """Return ALL raw peaks and dips with (index, position, height, width).

    For debugging: lower prominence, optional valley filter bypass. Returns (peaks, dips).
    Each item: (idx, pos_nm, height, width).
    """
    peaks_raw = extract_peaks(
        intensity,
        positions,
        prominence=peak_prominence,
        min_dist=1,
    )
    dips_raw = extract_dips(
        intensity,
        positions,
        prominence=dip_prominence,
        min_dist=1,
        valley_only=not no_valley_filter,
    )
    peaks_out = [(px, pos, h, w) for px, pos, h, w in peaks_raw]
    dips_out = [(px, pos, h, w) for px, pos, h, w in dips_raw]
    return peaks_out, dips_out


def extract_extremums(
    intensity: np.ndarray,
    positions: np.ndarray,
    *,
    position_px: np.ndarray | None = None,
    max_count: int = DEFAULT_MAX_EXTREMUMS,
) -> list[Extremum]:
    """Extract peaks and dips ranked by prominence. Works for emission and transmission spectra.

    Combines peaks and dips, sorts by prominence (height/depth) desc, width asc,
    takes top max_count. No preference for peaks over dips or vice versa.
    """
    peaks_raw = extract_peaks(intensity, positions)
    dips_raw = extract_dips(intensity, positions)

    combined: list[tuple[int, float, float, float, bool]] = []
    for px, pos, h, w in peaks_raw:
        combined.append((px, pos, h, w, False))
    for px, pos, h, w in dips_raw:
        combined.append((px, pos, h, w, True))
    if not combined:
        return []

    # Rank by prominence/width (sharpness): narrow prominent features outrank wide shallow ones
    def _rank(x: tuple) -> float:
        _, _, h, w, _ = x
        return h / max(w, 1e-9)

    combined.sort(key=lambda x: -_rank(x))
    top = combined[:max_count]
    # Sort by position for display
    top.sort(key=lambda x: x[1])
    max_h = max(x[2] for x in top) or 1.0
    result: list[Extremum] = []
    for idx, pos, h_computed, w_computed, is_dip in top:
        h_norm = h_computed / max_h if max_h > 0 else 0.5
        h_signed = h_norm if not is_dip else -h_norm
        px = int(position_px[idx]) if position_px is not None else None
        result.append(
            Extremum(
                index=idx,
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
