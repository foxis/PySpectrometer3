"""Orchestrate SRP calibration: detect → descriptor → match → fit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .descriptor import DEFAULT_WEIGHTS, build as build_triplets
from .detect import DEFAULT_MAX_COUNT, extract, from_known_lines
from .fit import fit_cal_points
from .matcher import match


@dataclass
class CalibrationResult:
    """Result of triplet-based calibration."""

    cal_points: list[tuple[int, float]]
    wavelengths: np.ndarray
    metric: str


def _valid_positions(wavelengths: np.ndarray, n: int) -> np.ndarray:
    """Valid monotonic positions or linear 380–750."""
    if wavelengths is None or len(wavelengths) < 2:
        return np.linspace(380, 750, n)
    wl = np.asarray(wavelengths)
    if wl.size != n:
        return np.linspace(380, 750, n)
    valid = np.all(np.diff(wl) > 0) and 300 < wl.min() < wl.max() < 900
    return wl if valid else np.linspace(380, 750, n)


def calibrate(
    intensity: np.ndarray,
    positions: np.ndarray,
    reference_wavelengths: list[float],
    reference_intensities: list[float],
    *,
    max_extremums: int = 15,
    weights: np.ndarray | None = None,
    metric: str = "euclidean",
) -> CalibrationResult | None:
    """Detect → build descriptors → bootstrap match → inverse Cauchy fit."""
    n = len(intensity)
    if n < 10 or len(reference_wavelengths) < 4:
        return None

    pos = _valid_positions(positions, n)
    position_px = np.arange(n, dtype=np.intp)

    meas = extract(
        intensity,
        pos,
        position_px=position_px,
        max_count=max_extremums,
    )

    ref = from_known_lines(reference_wavelengths, reference_intensities)
    if len(meas) < 2 or len(ref) < 2:
        return None

    triplets_meas = build_triplets(meas)
    triplets_ref = build_triplets(ref)

    w = weights if weights is not None else DEFAULT_WEIGHTS
    path_euc = match(meas, ref, triplets_meas, triplets_ref, weights=w, metric="euclidean")
    path_cos = match(meas, ref, triplets_meas, triplets_ref, weights=w, metric="cosine")
    path = path_euc if len(path_euc) >= len(path_cos) else path_cos
    metric = "euclidean" if path == path_euc else "cosine"

    cal_points = [
        (meas[i_m].position_px, ref[i_r].position)
        for i_m, i_r in path
        if meas[i_m].position_px is not None
    ]
    cal_points = _filter_non_crossing(cal_points)

    if len(cal_points) < 3:
        return None

    wavelengths = fit_cal_points(cal_points, n)
    return CalibrationResult(cal_points=cal_points, wavelengths=wavelengths, metric=metric)


def _filter_non_crossing(cal_points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Enforce: sorted by pixel, ref_wl strictly increasing."""
    if len(cal_points) < 2:
        return cal_points
    sorted_pts = sorted(cal_points, key=lambda x: x[0])
    result: list[tuple[int, float]] = [sorted_pts[0]]
    for px, ref_wl in sorted_pts[1:]:
        if ref_wl > result[-1][1]:
            result.append((px, ref_wl))
    return result


@dataclass
class Outcome:
    metric: str
    cal_points: list[tuple[int, float]]
    score: float


@dataclass
class AllPeaksInfo:
    pixels: list[int]
    positions: list[float]
    heights: list[float]
    is_dips: list[bool]


def calibrate_extremums(
    intensity: np.ndarray,
    wavelengths: np.ndarray,
    source,
    *,
    debug: bool = False,
    return_outcomes: bool = False,
    max_extremums: int = DEFAULT_MAX_COUNT,
):
    """Feature-based extremum matching. Returns cal_points or (cal_points, outcomes, all_peaks)."""
    from .detect_peaks import get_reference_peaks

    ref_peaks = get_reference_peaks(source)
    ref_wl = [p.wavelength for p in ref_peaks]
    ref_int = [p.intensity for p in ref_peaks]

    n = len(intensity)
    pos = _valid_positions(wavelengths, n)
    position_px = np.arange(n, dtype=np.intp)
    meas = extract(intensity, pos, position_px=position_px, max_count=max_extremums)

    result = calibrate(
        intensity,
        wavelengths,
        ref_wl,
        ref_int,
        max_extremums=max_extremums,
    )

    if result is None:
        fallback = calibrate_peaks(intensity, wavelengths, source, debug=debug)
        ap = AllPeaksInfo(
            pixels=[e.position_px for e in meas if e.position_px is not None],
            positions=[e.position for e in meas],
            heights=[abs(e.height) for e in meas],
            is_dips=[e.is_dip for e in meas],
        )
        return (fallback, [], ap) if return_outcomes else fallback

    cal_points = [(int(p[0]), float(p[1])) for p in result.cal_points]
    if len(cal_points) < 4:
        fallback = calibrate_peaks(intensity, wavelengths, source, debug=debug)
        ap = AllPeaksInfo(
            pixels=[e.position_px for e in meas if e.position_px is not None],
            positions=[e.position for e in meas],
            heights=[abs(e.height) for e in meas],
            is_dips=[e.is_dip for e in meas],
        )
        return (fallback, [], ap) if return_outcomes else fallback

    if debug:
        print(f"  Extremum ({result.metric}): {len(cal_points)} pts")

    score_val = float(len(cal_points)) + 0.1
    outcomes = [Outcome(metric=result.metric, cal_points=cal_points, score=score_val)]
    all_peaks = AllPeaksInfo(
        pixels=[e.position_px for e in meas if e.position_px is not None],
        positions=[e.position for e in meas],
        heights=[abs(e.height) for e in meas],
        is_dips=[e.is_dip for e in meas],
    )
    return (cal_points, outcomes, all_peaks) if return_outcomes else cal_points


def calibrate_peaks(
    intensity: np.ndarray,
    wavelengths: np.ndarray,
    source,
    *,
    debug: bool = False,
) -> list[tuple[int, float]]:
    """Legacy peak-only calibration (hypothesis-based). Fallback when extremum fails."""
    from .detect_peaks import detect_peaks_measured, get_reference_peaks
    from .hypotheses import Hypothesis, generate_sequential_hypotheses
    from .scorer import score_hypothesis

    n = len(intensity)
    measured = detect_peaks_measured(
        intensity,
        wavelengths,
        include_dips=False,
        threshold=0.05,
        prominence=0.005,
        min_dist=10,
    )
    reference = get_reference_peaks(source)
    max_meas = max(p.intensity for p in measured) if measured else 1.0
    measured = [p for p in measured if p.intensity >= 0.05 * max_meas]
    measured.sort(key=lambda x: x.pixel)

    if len(measured) < 4 or len(reference) < 4:
        return []

    initial_wl = _valid_positions(wavelengths, n)
    hypotheses = generate_sequential_hypotheses(
        measured,
        reference,
        min_matches=4,
        tolerance_nm=25.0,
        max_hypotheses=500,
        initial_wavelengths=initial_wl,
    )
    if not hypotheses:
        return []

    best: tuple[Hypothesis, float] | None = None
    for h in hypotheses:
        s = score_hypothesis(h, measured, reference)
        if best is None or s > best[1]:
            best = (h, s)
    return best[0].to_cal_points(measured, reference) if best else []
