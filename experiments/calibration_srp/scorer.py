"""Score hypotheses: polynomial fit, peak alignment, width/strength penalty.

Single responsibility: assign a score to each hypothesis.
"""

from __future__ import annotations

import numpy as np

from .detect_peaks import MeasuredPeak, ReferencePeak
from .hypotheses import Hypothesis


def _linearity_score(points: list[tuple[int, float]]) -> float:
    """R² of linear fit pixel→wavelength. Higher = more linear (Snell's law approx)."""
    if len(points) < 2:
        return 1.0
    px = np.array([p[0] for p in points], dtype=np.float64)
    wl = np.array([p[1] for p in points], dtype=np.float64)
    ss_tot = np.sum((wl - wl.mean()) ** 2)
    if ss_tot < 1e-12:
        return 1.0
    slope, intercept = np.polyfit(px, wl, 1)
    y_pred = slope * px + intercept
    ss_res = np.sum((wl - y_pred) ** 2)
    return 1.0 - ss_res / ss_tot


def _strength_penalty(norm_meas: float, norm_ref: float) -> float:
    """Penalty for strong-weak mismatch. 0=no penalty, 1=full penalty."""
    return abs(norm_meas - norm_ref)


def _strength_bonus(norm_meas: float, norm_ref: float) -> float:
    """Bonus for matching strong peaks. Product: strong-strong=1, strong-weak≈0."""
    return norm_meas * norm_ref


def _width_penalty(w_meas: float, w_ref: float) -> float:
    """Penalty for width mismatch. 0=similar, 1=very different. Neutral when unknown."""
    if w_meas <= 0 or w_ref <= 0:
        return 0.0  # neutral when width unknown
    ratio = min(w_meas, w_ref) / max(w_meas, w_ref)
    return 1.0 - ratio


def _alignment_score(
    cal_points: list[tuple[int, float]],
    measured: list[MeasuredPeak],
    reference: list[ReferencePeak],
    matches: list,
    tolerance_nm: float = 5.0,
) -> float:
    """Fraction of matched peaks that align after polynomial fit.

    Fit poly through cal_points, then check how many (pixel, wl) pairs
    have |poly(pixel) - ref_wl| <= tolerance.
    """
    if len(cal_points) < 3:
        return 0.0
    px = np.array([p[0] for p in cal_points], dtype=np.float64)
    wl = np.array([p[1] for p in cal_points], dtype=np.float64)
    deg = min(3, len(cal_points) - 1)
    coeffs = np.polyfit(px, wl, deg)
    poly = np.poly1d(coeffs)

    aligned = 0
    for m in matches:
        pixel = measured[m.idx_measured].pixel
        ref_wl = reference[m.idx_reference].wavelength
        pred_wl = poly(pixel)
        if abs(pred_wl - ref_wl) <= tolerance_nm:
            aligned += 1
    return aligned / len(matches) if matches else 0.0


def score_hypothesis(
    hypothesis: Hypothesis,
    measured: list[MeasuredPeak],
    reference: list[ReferencePeak],
    *,
    w_linearity: float = 0.2,
    w_alignment: float = 0.2,
    w_strength_bonus: float = 0.5,
    w_strength_pen: float = 0.1,
    w_width: float = 0.0,
) -> float:
    """Score a hypothesis. Higher = better. Prefers strong-strong matches."""
    cal_points = hypothesis.to_cal_points(measured, reference)
    if len(cal_points) < 4:
        return -1e9

    linearity = _linearity_score(cal_points)
    alignment = _alignment_score(cal_points, measured, reference, hypothesis.matches)

    max_meas = max(p.intensity for p in measured) if measured else 1.0
    max_ref = max(p.intensity for p in reference) if reference else 1.0
    strength_pen = 0.0
    strength_bonus = 0.0
    width_pen = 0.0
    weak_match_penalty = 0.0
    dip_to_peak_penalty = 0.0
    strong_match_count = 0
    for m in hypothesis.matches:
        pm = measured[m.idx_measured]
        pr = reference[m.idx_reference]
        if pm.is_dip and not pr.is_dip:
            dip_to_peak_penalty += 1.0
        nm = pm.intensity / max_meas if max_meas > 1e-9 else 0
        nr = pr.intensity / max_ref if max_ref > 1e-9 else 0
        strength_pen += _strength_penalty(nm, nr)
        strength_bonus += _strength_bonus(nm, nr)
        if nm * nr < 0.08:  # Only penalize very weak matches
            weak_match_penalty += 0.3
        if nm >= 0.5 and nr >= 0.5:
            strong_match_count += 1
        wm = pm.width_nm
        wr = pr.width_nm
        width_pen += _width_penalty(wm, wr)
    n = len(hypothesis.matches)
    strength_pen /= n if n else 1
    strength_bonus /= n if n else 1
    width_pen /= n if n else 1

    if strong_match_count < 2:
        return -1e9
    extra_strong_bonus = 0.15 * max(0, strong_match_count - 2)
    match_count_bonus = 0.12 * max(0, n - 4)  # Prefer more matches (5+)
    score = (
        w_linearity * linearity
        + w_alignment * alignment
        + w_strength_bonus * strength_bonus
        - w_strength_pen * strength_pen
        - w_width * width_pen
        - weak_match_penalty
        - 2.0 * dip_to_peak_penalty
        + extra_strong_bonus
        + match_count_bonus
    )
    return score
