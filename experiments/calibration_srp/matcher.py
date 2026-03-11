"""Bootstrap triplet matching: find first match, extend left/right. No criss-cross."""

from __future__ import annotations

import numpy as np

from .descriptor import Triplet, best_pair_score, DEFAULT_WEIGHTS
from .detect import Extremum


def _would_cross(
    path: list[tuple[int, int]],
    new_m: int,
    new_r: int,
    meas: list[Extremum],
    ref: list[Extremum],
) -> bool:
    """True if adding (new_m, new_r) would create crossing."""
    if not path:
        return False
    pts = [
        (meas[i_m].position_px, ref[i_r].position)
        for i_m, i_r in path
        if meas[i_m].position_px is not None
    ]
    pts.append((meas[new_m].position_px, ref[new_r].position))
    pts.sort(key=lambda x: x[0])
    prev_wl = pts[0][1]
    for _, wl in pts[1:]:
        if wl <= prev_wl:
            return True
        prev_wl = wl
    return False


def _extend_left(
    path: list[tuple[int, int]],
    meas: list[Extremum],
    ref: list[Extremum],
    triplets_meas: list[Triplet],
    triplets_ref: list[Triplet],
    weights: np.ndarray,
    metric: str,
) -> list[tuple[int, int]]:
    """Extend path left (smaller pixels, smaller wavelengths)."""
    if not path:
        return path
    m_min = min(meas[i_m].position_px for i_m, _ in path if meas[i_m].position_px is not None)
    r_min_wl = min(ref[i_r].position for _, i_r in path)
    best_m, best_r, best_s = -1, -1, -1e9
    for i_m, e in enumerate(meas):
        if e.position_px is None or e.position_px >= m_min:
            continue
        for i_r, r in enumerate(ref):
            if r.position >= r_min_wl:
                continue
            if _would_cross(path, i_m, i_r, meas, ref):
                continue
            s = best_pair_score(
                i_m, i_r, triplets_meas, triplets_ref, meas, ref, weights, metric
            )
            if s > best_s and s > -1e8:
                best_s, best_m, best_r = s, i_m, i_r
    if best_m >= 0:
        return [(best_m, best_r)] + path
    return path


def _extend_right(
    path: list[tuple[int, int]],
    meas: list[Extremum],
    ref: list[Extremum],
    triplets_meas: list[Triplet],
    triplets_ref: list[Triplet],
    weights: np.ndarray,
    metric: str,
) -> list[tuple[int, int]]:
    """Extend path right (larger pixels, larger wavelengths)."""
    if not path:
        return path
    m_max = max(meas[i_m].position_px for i_m, _ in path if meas[i_m].position_px is not None)
    r_max_wl = max(ref[i_r].position for _, i_r in path)
    best_m, best_r, best_s = -1, -1, -1e9
    for i_m, e in enumerate(meas):
        if e.position_px is None or e.position_px <= m_max:
            continue
        for i_r, r in enumerate(ref):
            if r.position <= r_max_wl:
                continue
            if _would_cross(path, i_m, i_r, meas, ref):
                continue
            s = best_pair_score(
                i_m, i_r, triplets_meas, triplets_ref, meas, ref, weights, metric
            )
            if s > best_s and s > -1e8:
                best_s, best_m, best_r = s, i_m, i_r
    if best_m >= 0:
        return path + [(best_m, best_r)]
    return path


def _bootstrap_from_seed(
    seed: tuple[int, int],
    meas: list[Extremum],
    ref: list[Extremum],
    triplets_meas: list[Triplet],
    triplets_ref: list[Triplet],
    weights: np.ndarray,
    metric: str,
) -> list[tuple[int, int]]:
    """Extend path from single seed."""
    path: list[tuple[int, int]] = [seed]
    while True:
        prev_len = len(path)
        path = _extend_left(path, meas, ref, triplets_meas, triplets_ref, weights, metric)
        path = _extend_right(path, meas, ref, triplets_meas, triplets_ref, weights, metric)
        if len(path) == prev_len:
            break
    return path


def match(
    meas: list[Extremum],
    ref: list[Extremum],
    triplets_meas: list[Triplet],
    triplets_ref: list[Triplet],
    *,
    weights: np.ndarray | None = None,
    metric: str = "euclidean",
    n_seeds: int = 5,
) -> list[tuple[int, int]]:
    """Bootstrap: try top N seeds, extend each, pick best path (most points, then highest score)."""
    if weights is None:
        weights = DEFAULT_WEIGHTS

    if len(meas) < 2 or len(ref) < 2:
        return []

    candidates: list[tuple[float, int, int]] = []
    for i_m in range(len(meas)):
        if meas[i_m].position_px is None:
            continue
        for i_r in range(len(ref)):
            if meas[i_m].is_dip != ref[i_r].is_dip:
                continue
            s = best_pair_score(
                i_m, i_r, triplets_meas, triplets_ref, meas, ref, weights, metric
            )
            if s > -1e8:
                candidates.append((s, i_m, i_r))
    candidates.sort(key=lambda x: -x[0])
    seeds = [(m, r) for _, m, r in candidates[:n_seeds]]

    best_path: list[tuple[int, int]] = []
    best_score = -1e9
    for seed in seeds:
        path = _bootstrap_from_seed(seed, meas, ref, triplets_meas, triplets_ref, weights, metric)
        if len(path) < 2:
            continue
        total = sum(
            best_pair_score(i_m, i_r, triplets_meas, triplets_ref, meas, ref, weights, metric)
            for i_m, i_r in path
        ) + 0.05 * len(path)
        if len(path) > len(best_path) or (len(path) == len(best_path) and total > best_score):
            best_path = path
            best_score = total

    return best_path
