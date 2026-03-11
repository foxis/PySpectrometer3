"""Triplet-based extremum matching: match (A, center, B) triplets in relative space.

Descriptor: [height, width, A-height/height, B-height/height, A-width/width, B-width/width, rel_pos]
- height: central peak height (positive peak, negative dip)
- width: central width
- A-height/height, B-height/height: neighbor heights relative to center
- A-width/width, B-width/width: neighbor widths relative to center
- rel_pos: (pos - A_pos) / (B_pos - A_pos)

Neighbor selection: for center 2 with peaks 1,2,3,4,5: (1,2,3), (1,2,4), (1,2,5).
Left = immediate left, right = each peak to the right.

Weight vector: all 1s initially (element-wise before scoring).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REL_HEIGHT_WIDTH = 1.0 / np.e
MAX_EXTREMUMS = 15

# Start with all 1s; adjust later
TRIPLET_WEIGHTS = np.ones(7, dtype=np.float64)


@dataclass
class PeakInfo:
    """Single peak or dip with signed height (peak +, dip -)."""

    index: int
    position: float
    position_px: int | None
    height: float  # positive peak, negative dip
    width: float
    is_dip: bool


@dataclass
class Triplet:
    """Triplet (A, center, B) with relative descriptor."""

    center_idx: int
    left_idx: int
    right_idx: int
    descriptor: np.ndarray


def _find_peaks_and_dips(
    arr: np.ndarray,
    *,
    peak_prominence: float = 0.005,
    peak_min_dist: int = 8,
    dip_prominence: float = 0.025,
    dip_min_dist: int = 15,
) -> list[tuple[int, float, bool]]:
    """Returns (index, height_or_depth, is_dip) for peaks and dips.

    Dips use stricter prominence and distance to avoid noise and tiny baseline
    fluctuations being marked as dips.
    """
    try:
        from scipy.signal import find_peaks as scipy_find_peaks
    except ImportError:
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


def _width_at_rel_height(
    arr: np.ndarray,
    positions: np.ndarray,
    center_idx: int,
    height: float,
    is_dip: bool,
    *,
    rel_height: float = REL_HEIGHT_WIDTH,
) -> float:
    """Width in nm at rel_height of extremum height."""
    n = len(arr)
    if n < 2 or center_idx < 0 or center_idx >= n:
        return 0.0
    disp = (positions[-1] - positions[0]) / max(n - 1, 1)

    if is_dip:
        thresh = np.max(arr) - height * (1.0 - rel_height)
    else:
        thresh = height * rel_height
    if thresh <= 0 or (is_dip and thresh >= np.max(arr)):
        return 0.0

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
    return float(abs(right_ips - left_ips) * disp)


def extract_peaks(
    intensity: np.ndarray,
    positions: np.ndarray,
    *,
    position_px: np.ndarray | None = None,
    max_count: int = MAX_EXTREMUMS,
) -> list[PeakInfo]:
    """Extract peaks and dips. Height: positive peak, negative dip. Sorted by position."""
    raw = _find_peaks_and_dips(intensity)
    if not raw:
        return []

    rng = float(np.max(intensity) - np.min(intensity))
    max_val = max(h for _, h, d in raw if not d) or 1.0
    min_peak = 0.08 * max_val
    min_dip_depth = 0.12 * rng if rng > 0 else 0.05
    raw = [
        (i, h, d)
        for i, h, d in raw
        if (not d and h >= min_peak) or (d and h >= min_dip_depth)
    ]

    peaks_only = [(i, h, d) for i, h, d in raw if not d]
    dips_only = [(i, h, d) for i, h, d in raw if d]
    peaks_only = sorted(peaks_only, key=lambda x: -x[1])[:max_count]
    dips_only = sorted(dips_only, key=lambda x: -x[1])[:max_count]
    raw = sorted(peaks_only + dips_only, key=lambda x: x[0])

    widths: list[float] = []
    for idx, height, is_dip in raw:
        w = _width_at_rel_height(
            intensity, positions, idx, height, is_dip, rel_height=REL_HEIGHT_WIDTH
        )
        widths.append(max(0.0, w))

    max_h = max_val
    result: list[PeakInfo] = []
    for k, (idx, height, is_dip) in enumerate(raw):
        pos = float(positions[idx])
        h_norm = height / max_h if max_h > 0 else 0.5
        h_signed = h_norm if not is_dip else -h_norm
        px = int(position_px[idx]) if position_px is not None else None
        result.append(
            PeakInfo(
                index=k,
                position=pos,
                position_px=px,
                height=h_signed,
                width=widths[k],
                is_dip=is_dip,
            )
        )
    return result


def build_triplets(peaks: list[PeakInfo]) -> list[Triplet]:
    """For each center, build triplets (left, center, right): left=immediate, right=each to the right."""
    triplets: list[Triplet] = []
    n = len(peaks)
    eps = 1e-9

    def add(left_idx: int, c: int, right_idx: int) -> None:
        left = peaks[left_idx]
        center = peaks[c]
        right = peaks[right_idx]
        pos_a, pos_b, pos_c = left.position, right.position, center.position
        span = pos_b - pos_a
        rel_pos = (pos_c - pos_a) / span if span > eps else 0.5
        h_abs = abs(center.height) + eps
        a_h_ratio = left.height / h_abs
        b_h_ratio = right.height / h_abs
        w = center.width + eps
        a_w_ratio = left.width / w
        b_w_ratio = right.width / w
        desc = np.array(
            [center.height, center.width, a_h_ratio, b_h_ratio, a_w_ratio, b_w_ratio, rel_pos],
            dtype=np.float64,
        )
        triplets.append(Triplet(center_idx=c, left_idx=left_idx, right_idx=right_idx, descriptor=desc))

    if n >= 2:
        add(0, 0, 1)
    for c in range(1, n - 1):
        left_idx = c - 1
        for right_idx in range(c + 1, n):
            add(left_idx, c, right_idx)
    if n >= 2:
        add(n - 2, n - 1, n - 1)
    return triplets


def _triplet_score(
    ta: Triplet,
    tb: Triplet,
    peaks_a: list[PeakInfo],
    peaks_b: list[PeakInfo],
    weights: np.ndarray,
    metric: str,
) -> float:
    """Score triplet pair. Reject cross-type (peak-dip)."""
    ca = peaks_a[ta.center_idx]
    cb = peaks_b[tb.center_idx]
    if ca.is_dip != cb.is_dip:
        return -1e9

    va = weights * ta.descriptor
    vb = weights * tb.descriptor

    match metric:
        case "euclidean":
            diff = va - vb
            d = float(np.sqrt(np.sum(diff * diff)))
            return 1.0 / (1.0 + d)
        case "cosine":
            va = va + 1e-9
            vb = vb + 1e-9
            cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
            return (cos + 1.0) / 2.0
        case _:
            return -1e9


def _best_triplet_score(
    center_a: int,
    center_b: int,
    triplets_a: list[Triplet],
    triplets_b: list[Triplet],
    peaks_a: list[PeakInfo],
    peaks_b: list[PeakInfo],
    weights: np.ndarray,
    metric: str,
) -> float:
    """Best score over all triplet pairs for (center_a, center_b)."""
    ta_list = [t for t in triplets_a if t.center_idx == center_a]
    tb_list = [t for t in triplets_b if t.center_idx == center_b]
    if not ta_list or not tb_list:
        return -1e9
    best = -1e9
    for ta in ta_list:
        for tb in tb_list:
            s = _triplet_score(ta, tb, peaks_a, peaks_b, weights, metric)
            if s > best:
                best = s
    return best


def _generate_sequential_matches(
    peaks_meas: list[PeakInfo],
    peaks_ref: list[PeakInfo],
    *,
    min_matches: int = 2,
    max_hypotheses: int = 500,
) -> list[list[tuple[int, int]]]:
    """Generate sequential center-to-center match hypotheses.

    No wavelength proximity: we match pixels to nm, so pixels have no wavelength
    until calibrated. Candidates are filtered by type (peak/dip) and sequential
    order only; triplet scoring disambiguates.
    """
    n_meas = len(peaks_meas)
    n_ref = len(peaks_ref)
    if n_meas < min_matches or n_ref < min_matches:
        return []

    hypotheses: list[list[tuple[int, int]]] = []

    def recurse(m_idx: int, r_idx: int, path: list[tuple[int, int]]) -> None:
        if len(hypotheses) >= max_hypotheses:
            return
        if m_idx >= n_meas:
            if len(path) >= min_matches:
                hypotheses.append(path.copy())
            return
        if r_idx >= n_ref:
            return

        m = peaks_meas[m_idx]
        candidates = [
            j for j in range(r_idx, n_ref)
            if peaks_ref[j].is_dip == m.is_dip
        ]

        for j in candidates:
            path.append((m_idx, j))
            recurse(m_idx + 1, j + 1, path)
            path.pop()

        recurse(m_idx + 1, r_idx, path)
        if r_idx + 1 < n_ref:
            recurse(m_idx, r_idx + 1, path)

    recurse(0, 0, [])

    seen: set[tuple[tuple[int, int], ...]] = set()
    unique: list[list[tuple[int, int]]] = []
    for h in hypotheses:
        key = tuple(h)
        if key not in seen:
            seen.add(key)
            unique.append(h)
    return unique


def _path_has_crossing(
    path: list[tuple[int, int]],
    peaks_meas: list[PeakInfo],
    peaks_ref: list[PeakInfo],
) -> bool:
    """True if path produces crossing (measured px increases but ref wl decreases)."""
    if len(path) < 2:
        return False
    pts = [
        (peaks_meas[i_m].position_px, peaks_ref[i_r].position)
        for i_m, i_r in path
        if peaks_meas[i_m].position_px is not None
    ]
    pts.sort(key=lambda x: x[0])
    prev_ref = pts[0][1]
    for _, ref_wl in pts[1:]:
        if ref_wl <= prev_ref:
            return True
        prev_ref = ref_wl
    return False


def _run_matching(
    peaks_meas: list[PeakInfo],
    peaks_ref: list[PeakInfo],
    triplets_meas: list[Triplet],
    triplets_ref: list[Triplet],
    metric: str,
    *,
    max_hypotheses: int = 2000,
) -> list[tuple[int, int]]:
    """Run sequential matching using triplet scores. Reject paths that would cross."""
    hypotheses = _generate_sequential_matches(
        peaks_meas,
        peaks_ref,
        min_matches=2,
        max_hypotheses=max_hypotheses,
    )
    best_score = -1e9
    best_path: list[tuple[int, int]] = []
    for path in hypotheses:
        if _path_has_crossing(path, peaks_meas, peaks_ref):
            continue
        pair_scores = [
            _best_triplet_score(
                i_m,
                i_r,
                triplets_meas,
                triplets_ref,
                peaks_meas,
                peaks_ref,
                TRIPLET_WEIGHTS,
                metric,
            )
            for i_m, i_r in path
        ]
        if any(s < -1e8 for s in pair_scores):
            continue
        score = sum(pair_scores) + 0.05 * len(path)
        if score > best_score:
            best_score = score
            best_path = path
    return best_path


def _filter_non_crossing(cal_points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Enforce no crossing: when sorted by pixel, ref_wl must be strictly increasing."""
    if len(cal_points) < 2:
        return cal_points
    sorted_pts = sorted(cal_points, key=lambda x: x[0])
    result: list[tuple[int, float]] = [sorted_pts[0]]
    for px, ref_wl in sorted_pts[1:]:
        if ref_wl > result[-1][1]:
            result.append((px, ref_wl))
    return result


def _reference_peaks_from_known(
    wavelengths: list[float],
    intensities: list[float],
) -> list[PeakInfo]:
    """Build PeakInfo list from known spectral lines."""
    if not wavelengths or not intensities:
        return []
    max_int = max(intensities) if intensities else 1.0
    result: list[PeakInfo] = []
    for i, (wl, inten) in enumerate(zip(wavelengths, intensities)):
        h_norm = inten / max_int if max_int > 0 else 0.5
        result.append(
            PeakInfo(
                index=i,
                position=float(wl),
                position_px=None,
                height=h_norm,
                width=5.0,
                is_dip=False,
            )
        )
    return result


@dataclass
class MatchOutcome:
    """Result of one matching metric (Euclidean or cosine)."""

    metric: str
    cal_points: list[tuple[int, float]]
    score: float


@dataclass
class AllPeaksInfo:
    """All detected peaks for visualization (matched and unmatched)."""

    pixels: list[int]
    positions: list[float]
    heights: list[float]
    is_dips: list[bool]


def get_selected_path(
    peaks_meas: list[PeakInfo],
    peaks_ref: list[PeakInfo],
    triplets_meas: list[Triplet],
    triplets_ref: list[Triplet],
    *,
    max_hypotheses: int = 2000,
) -> tuple[list[tuple[int, int]], str]:
    """Run matching, return (path, metric). For debugging."""
    path_euc = _run_matching(
        peaks_meas, peaks_ref, triplets_meas, triplets_ref,
        "euclidean", max_hypotheses=max_hypotheses,
    )
    path_cos = _run_matching(
        peaks_meas, peaks_ref, triplets_meas, triplets_ref,
        "cosine", max_hypotheses=max_hypotheses,
    )

    def total_score(path: list[tuple[int, int]], metric: str) -> float:
        if not path:
            return -1e9
        return sum(
            _best_triplet_score(
                i_m, i_r, triplets_meas, triplets_ref,
                peaks_meas, peaks_ref, TRIPLET_WEIGHTS, metric,
            )
            for i_m, i_r in path
        ) + 0.05 * len(path)

    s_euc = total_score(path_euc, "euclidean")
    s_cos = total_score(path_cos, "cosine")
    if s_euc >= s_cos:
        return path_euc, "euclidean"
    return path_cos, "cosine"


def match_extremums(
    measured_intensity: np.ndarray,
    reference_intensity: np.ndarray,
    reference_wavelengths: np.ndarray,
    *,
    initial_wavelengths: np.ndarray | None = None,
    reference_peak_wavelengths: list[float] | None = None,
    reference_peak_intensities: list[float] | None = None,
) -> tuple[list[tuple[int, float]], list[MatchOutcome], AllPeaksInfo]:
    """Match triplets. Returns (best_cal_points, outcomes, all_peaks_for_plot)."""
    n = len(measured_intensity)
    if initial_wavelengths is None or len(initial_wavelengths) != n:
        initial_wavelengths = np.linspace(380, 750, n)

    measured_positions = np.asarray(initial_wavelengths, dtype=np.float64)
    position_px = np.arange(n, dtype=np.intp)
    peaks_meas = extract_peaks(
        measured_intensity,
        measured_positions,
        position_px=position_px,
        max_count=MAX_EXTREMUMS,
    )

    peaks_only = [p for p in peaks_meas if not p.is_dip]
    if len(peaks_only) >= 4:
        peaks_meas = peaks_only
    else:
        peaks_meas = [p for p in peaks_meas if abs(p.height) >= 0.15]

    all_peaks = AllPeaksInfo(
        pixels=[p.position_px for p in peaks_meas if p.position_px is not None],
        positions=[p.position for p in peaks_meas],
        heights=[p.height for p in peaks_meas],
        is_dips=[p.is_dip for p in peaks_meas],
    )

    if (
        reference_peak_wavelengths is not None
        and reference_peak_intensities is not None
        and len(reference_peak_wavelengths) >= 4
    ):
        peaks_ref = _reference_peaks_from_known(
            reference_peak_wavelengths, reference_peak_intensities
        )
    else:
        ref_positions = np.asarray(reference_wavelengths, dtype=np.float64)
        peaks_ref = extract_peaks(
            reference_intensity,
            ref_positions,
            position_px=None,
            max_count=MAX_EXTREMUMS,
        )
        ref_peaks = [p for p in peaks_ref if not p.is_dip]
        if len(ref_peaks) >= 4:
            peaks_ref = ref_peaks
        peaks_ref = [p for p in peaks_ref if abs(p.height) >= 0.15]

    triplets_meas = build_triplets(peaks_meas)
    triplets_ref = build_triplets(peaks_ref)

    path_euc = _run_matching(
        peaks_meas, peaks_ref, triplets_meas, triplets_ref, "euclidean"
    )
    path_cos = _run_matching(
        peaks_meas, peaks_ref, triplets_meas, triplets_ref, "cosine"
    )

    def total_score(path: list[tuple[int, int]], metric: str) -> float:
        if not path:
            return -1e9
        return sum(
            _best_triplet_score(
                i_m,
                i_r,
                triplets_meas,
                triplets_ref,
                peaks_meas,
                peaks_ref,
                TRIPLET_WEIGHTS,
                metric,
            )
            for i_m, i_r in path
        ) + 0.05 * len(path)

    outcomes: list[MatchOutcome] = []
    if path_euc:
        cal_euc = [
            (peaks_meas[i_m].position_px, peaks_ref[i_r].position)
            for i_m, i_r in path_euc
            if peaks_meas[i_m].position_px is not None
        ]
        cal_euc = _filter_non_crossing(cal_euc)
        outcomes.append(
            MatchOutcome(
                metric="euclidean",
                cal_points=cal_euc,
                score=total_score(path_euc, "euclidean"),
            )
        )
    if path_cos:
        cal_cos = [
            (peaks_meas[i_m].position_px, peaks_ref[i_r].position)
            for i_m, i_r in path_cos
            if peaks_meas[i_m].position_px is not None
        ]
        cal_cos = _filter_non_crossing(cal_cos)
        outcomes.append(
            MatchOutcome(
                metric="cosine",
                cal_points=cal_cos,
                score=total_score(path_cos, "cosine"),
            )
        )

    best = max(outcomes, key=lambda o: o.score) if outcomes else None
    best_points = best.cal_points if best else []
    return best_points, outcomes, all_peaks
