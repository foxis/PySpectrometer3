"""Triplet descriptors: relative position, strength, width for matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detect import Extremum

# [height, width, A-h/h, B-h/h, A-w/w, B-w/w, rel_pos]
DESC_LABELS = ["height", "width", "A-h/h", "B-h/h", "A-w/w", "B-w/w", "rel_pos"]
# Exaggerate rel_pos so matching favors similar relative position of center within span.
DEFAULT_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5], dtype=np.float64)


@dataclass
class Triplet:
    """(left, center, right) with descriptor vector."""

    center_idx: int
    left_idx: int
    right_idx: int
    descriptor: np.ndarray


def build(extremums: list[Extremum]) -> list[Triplet]:
    """Build triplets: left=immediate left, right=each to the right."""
    triplets: list[Triplet] = []
    n = len(extremums)
    eps = 1e-9

    def add(left_idx: int, c: int, right_idx: int) -> None:
        left = extremums[left_idx]
        center = extremums[c]
        right = extremums[right_idx]
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


def score(
    ta: Triplet,
    tb: Triplet,
    a: list[Extremum],
    b: list[Extremum],
    weights: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """Score triplet pair. Reject cross-type (peak-dip)."""
    ca = a[ta.center_idx]
    cb = b[tb.center_idx]
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


def best_pair_score(
    center_a: int,
    center_b: int,
    triplets_a: list[Triplet],
    triplets_b: list[Triplet],
    a: list[Extremum],
    b: list[Extremum],
    weights: np.ndarray,
    metric: str = "euclidean",
) -> float:
    """Best score over all triplet pairs for (center_a, center_b)."""
    ta_list = [t for t in triplets_a if t.center_idx == center_a]
    tb_list = [t for t in triplets_b if t.center_idx == center_b]
    if not ta_list or not tb_list:
        return -1e9
    best = -1e9
    for ta in ta_list:
        for tb in tb_list:
            s = score(ta, tb, a, b, weights, metric)
            if s > best:
                best = s
    return best
