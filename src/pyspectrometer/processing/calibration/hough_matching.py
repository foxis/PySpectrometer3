"""Hough-based linear calibration: pixel ↔ wavelength via accumulator voting.

Each (pixel, wavelength) pair implies λ = m·pixel + c. Bin (m, c); dominant bin
is the calibration. Then match by proximity and refine with Cauchy fit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .cauchy_fit import fit_cal_points


@dataclass
class HoughResult:
    """Result of Hough transform calibration."""

    slope: float
    intercept: float
    accumulator: np.ndarray
    m_bins: np.ndarray
    c_bins: np.ndarray
    peak_m_idx: int
    peak_c_idx: int
    cal_points: list[tuple[int, float]]
    wavelengths: np.ndarray


def hough_accumulator(
    pixels: np.ndarray,
    wavelengths: np.ndarray,
    *,
    num_slopes: int = 200,
    num_intercepts: int = 200,
    min_slope: float = 0.2,
    max_slope: float = 0.8,
    min_intercept: float = 300.0,
    max_intercept: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 2D Hough accumulator for λ = m·pixel + c.

    For each (pixel, wavelength) pair, c = λ - m·pixel. Sample m, bin (m, c).

    Args:
        pixels: Measured pixel positions (int or float)
        wavelengths: Reference wavelengths (nm)
        num_slopes: Number of slope bins
        num_intercepts: Number of intercept bins
        min_slope, max_slope: nm per pixel
        min_intercept, max_intercept: wavelength at pixel 0 (nm)

    Returns:
        (accumulator, m_bins, c_bins)
    """
    px = np.asarray(pixels, dtype=np.float64).ravel()
    wl = np.asarray(wavelengths, dtype=np.float64).ravel()

    m_edges = np.linspace(min_slope, max_slope, num_slopes + 1)
    c_edges = np.linspace(min_intercept, max_intercept, num_intercepts + 1)
    acc = np.zeros((num_slopes, num_intercepts), dtype=np.intp)

    for pi in px:
        for lam in wl:
            for mi in range(num_slopes):
                m = (m_edges[mi] + m_edges[mi + 1]) / 2.0
                c = lam - m * pi
                ci = int(np.searchsorted(c_edges, c, side="right")) - 1
                if 0 <= ci < num_intercepts:
                    acc[mi, ci] += 1

    m_bins = (m_edges[:-1] + m_edges[1:]) / 2.0
    c_bins = (c_edges[:-1] + c_edges[1:]) / 2.0
    return acc, m_bins, c_bins


def find_best_linear(
    acc: np.ndarray,
    m_bins: np.ndarray,
    c_bins: np.ndarray,
) -> tuple[float, float, int, int]:
    """Find peak in Hough accumulator. Returns (slope, intercept, m_idx, c_idx)."""
    peak_idx = np.unravel_index(np.argmax(acc), acc.shape)
    m_idx, c_idx = int(peak_idx[0]), int(peak_idx[1])
    return float(m_bins[m_idx]), float(c_bins[c_idx]), m_idx, c_idx


def match_by_proximity(
    pixels: np.ndarray,
    ref_wavelengths: np.ndarray,
    slope: float,
    intercept: float,
    *,
    tolerance_nm: float = 15.0,
) -> list[tuple[int, float]]:
    """Match measured pixels to reference wavelengths by proximity.

    For each pixel: predicted_wl = slope * pixel + intercept.
    Find closest ref wavelength within tolerance.
    """
    ref_wl = np.asarray(ref_wavelengths, dtype=np.float64)
    cal_points: list[tuple[int, float]] = []
    used_ref: set[int] = set()

    for px in np.asarray(pixels, dtype=np.float64):
        pred_wl = slope * px + intercept
        dist = np.abs(ref_wl - pred_wl)
        best = int(np.argmin(dist))
        if dist[best] <= tolerance_nm and best not in used_ref:
            cal_points.append((int(px), float(ref_wl[best])))
            used_ref.add(best)

    return sorted(cal_points, key=lambda x: x[0])


def calibrate_hough(
    measured_pixels: np.ndarray,
    measured_intensities: np.ndarray,
    ref_wavelengths: np.ndarray,
    ref_intensities: np.ndarray,
    n_pixels: int,
    *,
    top_k: int = 12,
    num_slopes: int = 200,
    num_intercepts: int = 200,
    min_slope: float = 0.2,
    max_slope: float = 0.8,
    min_intercept: float = 300.0,
    max_intercept: float = 500.0,
    tolerance_nm: float = 15.0,
) -> HoughResult | None:
    """Full Hough calibration: accumulator → linear → proximity → Cauchy fit.

    Uses top-k by prominence (intensity/height) from measured and reference.
    """
    px = np.asarray(measured_pixels, dtype=np.float64).ravel()
    mi = np.asarray(measured_intensities, dtype=np.float64).ravel()
    rw = np.asarray(ref_wavelengths, dtype=np.float64).ravel()
    ri = np.asarray(ref_intensities, dtype=np.float64).ravel()

    if len(px) < 2 or len(rw) < 2:
        return None

    # Top-k by prominence (intensity)
    meas_rank = np.argsort(-np.abs(mi))[:top_k]
    ref_rank = np.argsort(-np.abs(ri))[:top_k]
    px_top = px[meas_rank]
    rw_top = rw[ref_rank]

    acc, m_bins, c_bins = hough_accumulator(
        px_top,
        rw_top,
        num_slopes=num_slopes,
        num_intercepts=num_intercepts,
        min_slope=min_slope,
        max_slope=max_slope,
        min_intercept=min_intercept,
        max_intercept=max_intercept,
    )

    slope, intercept, m_idx, c_idx = find_best_linear(acc, m_bins, c_bins)
    cal_points = match_by_proximity(
        px_top,
        rw_top,
        slope,
        intercept,
        tolerance_nm=tolerance_nm,
    )

    if len(cal_points) < 3:
        return None

    wavelengths = fit_cal_points(cal_points, n_pixels)
    return HoughResult(
        slope=slope,
        intercept=intercept,
        accumulator=acc,
        m_bins=m_bins,
        c_bins=c_bins,
        peak_m_idx=m_idx,
        peak_c_idx=c_idx,
        cal_points=cal_points,
        wavelengths=wavelengths,
    )


@dataclass
class SpdCorrelationResult:
    """Result of SPD correlation-based calibration."""

    slope: float
    intercept: float
    correlation: float
    score_grid: np.ndarray
    m_bins: np.ndarray
    c_bins: np.ndarray
    cal_points: list[tuple[int, float]]
    wavelengths: np.ndarray


def find_best_linear_spd(
    measured_intensity: np.ndarray,
    ref_wavelengths: np.ndarray,
    ref_intensity: np.ndarray,
    n_pixels: int,
    *,
    wl_min: float = 400.0,
    wl_max: float = 700.0,
    num_slopes: int = 80,
    num_intercepts: int = 80,
    min_slope: float = 0.2,
    max_slope: float = 0.8,
    min_intercept: float = 300.0,
    max_intercept: float = 500.0,
    n_display_points: int = 20,
) -> SpdCorrelationResult | None:
    """Find best linear calibration by correlating full SPD (or 400–700 nm band).

    Grid search over (m, c). For each: resample measured to ref wavelength grid,
    compute Pearson correlation (scale-invariant). Returns best (m, c) and
    N evenly spaced cal points for display and Cauchy fit.
    """
    meas = np.asarray(measured_intensity, dtype=np.float64).ravel()
    rw = np.asarray(ref_wavelengths, dtype=np.float64).ravel()
    ri = np.asarray(ref_intensity, dtype=np.float64).ravel()

    m_edges = np.linspace(min_slope, max_slope, num_slopes + 1)
    c_edges = np.linspace(min_intercept, max_intercept, num_intercepts + 1)
    score_grid = np.zeros((num_slopes, num_intercepts), dtype=np.float64)

    px = np.arange(n_pixels, dtype=np.float64)

    for mi in range(num_slopes):
        m = (m_edges[mi] + m_edges[mi + 1]) / 2.0
        for ci in range(num_intercepts):
            c = (c_edges[ci] + c_edges[ci + 1]) / 2.0
            wl_arr = m * px + c
            score_grid[mi, ci] = _dot_score(meas, wl_arr, rw, ri)

    peak_idx = np.unravel_index(np.argmax(score_grid), score_grid.shape)
    m_idx, c_idx = int(peak_idx[0]), int(peak_idx[1])
    m_bins = (m_edges[:-1] + m_edges[1:]) / 2.0
    c_bins = (c_edges[:-1] + c_edges[1:]) / 2.0
    slope = float(m_bins[m_idx])
    intercept = float(c_bins[c_idx])
    correlation = float(score_grid[m_idx, c_idx])

    # N evenly spaced pixels for display and Cauchy fit
    px_sample = np.linspace(0, n_pixels - 1, n_display_points).astype(np.intp)
    px_sample = np.clip(px_sample, 0, n_pixels - 1)
    cal_points = [(int(p), slope * p + intercept) for p in px_sample]
    cal_points = _dedupe_cal_points(cal_points)

    if len(cal_points) < 3:
        return None

    wavelengths = fit_cal_points(cal_points, n_pixels)
    return SpdCorrelationResult(
        slope=slope,
        intercept=intercept,
        correlation=correlation,
        score_grid=score_grid,
        m_bins=m_bins,
        c_bins=c_bins,
        cal_points=cal_points,
        wavelengths=wavelengths,
    )


def _dedupe_cal_points(cal_points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Remove duplicate pixels, keep first."""
    seen: set[int] = set()
    out: list[tuple[int, float]] = []
    for p, w in cal_points:
        if p not in seen:
            seen.add(p)
            out.append((p, w))
    return sorted(out, key=lambda x: x[0])


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
class RansacResult:
    """Result of RANSAC calibration with edge constraint."""

    slope: float
    intercept: float
    n_inliers: int
    n_iterations: int
    cal_points: list[tuple[int, float]]
    wavelengths: np.ndarray


def _fit_line_two_points(
    p1: float, wl1: float,
    p2: float, wl2: float,
) -> tuple[float, float] | None:
    """Fit λ = m*p + c from two (pixel, wavelength) pairs."""
    if abs(p2 - p1) < 1e-9:
        return None
    m = (wl2 - wl1) / (p2 - p1)
    c = wl1 - m * p1
    return (m, c)


def _edge_constraint_ok(
    slope: float,
    intercept: float,
    pixel_min: float,
    pixel_max: float,
    ref_wl_min: float,
    ref_wl_max: float,
    *,
    edge_margin_nm: float = 100.0,
) -> bool:
    """Check that fit maps pixel edges near ref wavelength edges.

    Spectrum must cover most of the range: not too stretched or shrunk.
    """
    pred_left = slope * pixel_min + intercept
    pred_right = slope * pixel_max + intercept
    left_ok = ref_wl_min - edge_margin_nm <= pred_left <= ref_wl_min + edge_margin_nm
    right_ok = ref_wl_max - edge_margin_nm <= pred_right <= ref_wl_max + edge_margin_nm
    return left_ok and right_ok


def calibrate_ransac(
    measured_pixels: np.ndarray,
    measured_intensities: np.ndarray,
    ref_wavelengths: np.ndarray,
    ref_intensities: np.ndarray,
    n_pixels: int,
    *,
    top_k: int = 20,
    max_iterations: int = 500,
    inlier_tolerance_nm: float = 12.0,
    edge_margin_nm: float = 100.0,
    min_inliers: int = 4,
) -> RansacResult | None:
    """RANSAC on (pixel, wavelength) pairs with edge constraint.

    Top-k features by prominence. Sample 2 pairs, fit line, reject if edges
    don't map near ref range. Keep fit with most inliers, then match by proximity.
    """
    px = np.asarray(measured_pixels, dtype=np.float64).ravel()
    mi = np.asarray(measured_intensities, dtype=np.float64).ravel()
    rw = np.asarray(ref_wavelengths, dtype=np.float64).ravel()
    ri = np.asarray(ref_intensities, dtype=np.float64).ravel()

    if len(px) < 2 or len(rw) < 2:
        return None

    meas_rank = np.argsort(-np.abs(mi))[:top_k]
    ref_rank = np.argsort(-np.abs(ri))[:top_k]
    px_top = px[meas_rank]
    rw_top = rw[ref_rank]

    pixel_min = 0.0
    pixel_max = float(n_pixels - 1)
    ref_wl_min = max(380.0, float(rw_top.min()) - 30.0)
    ref_wl_max = min(750.0, float(rw_top.max()) + 30.0)

    pairs: list[tuple[float, float]] = [
        (float(p), float(w)) for p in px_top for w in rw_top
    ]
    if len(pairs) < 4:
        return None

    rng = np.random.default_rng()
    best_inliers: list[tuple[int, float]] = []
    best_m, best_c = 0.0, 400.0

    for _ in range(max_iterations):
        idx = rng.choice(len(pairs), size=2, replace=False)
        (p1, wl1), (p2, wl2) = pairs[idx[0]], pairs[idx[1]]
        fit = _fit_line_two_points(p1, wl1, p2, wl2)
        if fit is None:
            continue
        m, c = fit
        if not _edge_constraint_ok(
            m, c, pixel_min, pixel_max, ref_wl_min, ref_wl_max,
            edge_margin_nm=edge_margin_nm,
        ):
            continue

        inliers: list[tuple[int, float]] = []
        for p, wl in pairs:
            pred = m * p + c
            if abs(pred - wl) <= inlier_tolerance_nm:
                inliers.append((int(p), float(wl)))

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_m, best_c = m, c

    if len(best_inliers) < min_inliers:
        return None

    cal_points = _dedupe_cal_points(best_inliers)
    cal_points = _filter_non_crossing(cal_points)
    if len(cal_points) < 3:
        return None

    wavelengths = fit_cal_points(cal_points, n_pixels)
    return RansacResult(
        slope=best_m,
        intercept=best_c,
        n_inliers=len(best_inliers),
        n_iterations=max_iterations,
        cal_points=cal_points,
        wavelengths=wavelengths,
    )


@dataclass
class PeakDipResult:
    """Result of peak/dip type-aware alignment."""

    slope: float
    intercept: float
    score: float
    n_peak_match: int
    n_dip_match: int
    n_mismatch: int
    score_grid: np.ndarray
    m_bins: np.ndarray
    c_bins: np.ndarray
    cal_points: list[tuple[int, float]]
    wavelengths: np.ndarray


def _score_peak_dip_alignment(
    meas_px: np.ndarray,
    meas_is_dip: np.ndarray,
    meas_int: np.ndarray,
    ref_wl: np.ndarray,
    ref_is_dip: np.ndarray,
    ref_int: np.ndarray,
    slope: float,
    intercept: float,
    *,
    mismatch_penalty: float = 10.0,
    delta_scale_nm: float = 15.0,
    intensity_penalty_scale: float = 0.2,
    delta_exponent: float = 2.0,
) -> tuple[float, list[tuple[int, float]], list[float]]:
    """Score by x-axis alignment: peak↔peak/dip↔dip +1-(δx/σ)^p, peak↔dip -10.
    Returns (score, cal_points, residuals).
    """
    cal_points: list[tuple[int, float]] = []
    residuals: list[float] = []
    total = 0.0

    for i in range(len(meas_px)):
        px = float(meas_px[i])
        pred_wl = slope * px + intercept
        same_mask = ref_is_dip == meas_is_dip[i]
        if not np.any(same_mask):
            continue
        dist_same = np.abs(ref_wl[same_mask] - pred_wl)
        j = int(np.argmin(dist_same))
        delta_x = float(dist_same[j])
        matched_wl = float(ref_wl[same_mask][j])
        contrib = 1.0 - (delta_x / delta_scale_nm) ** delta_exponent
        int_diff = abs(meas_int[i] - ref_int[same_mask][j])
        contrib -= intensity_penalty_scale * int_diff
        total += contrib
        cal_points.append((int(px), matched_wl))
        residuals.append(delta_x)

    return total, cal_points, residuals


def alignment_score_from_wavelengths(
    meas_px: np.ndarray,
    meas_is_dip: np.ndarray,
    meas_int: np.ndarray,
    ref_wl: np.ndarray,
    ref_is_dip: np.ndarray,
    ref_int: np.ndarray,
    wavelengths: np.ndarray,
    *,
    mismatch_penalty: float = 10.0,
    delta_scale_nm: float = 10.0,
    intensity_penalty_scale: float = 0.2,
    delta_exponent: float = 3.0,
) -> float:
    """Compute alignment score: only same-type matches count (peak↔peak, dip↔dip)."""
    total = 0.0
    n = len(wavelengths)
    for i in range(len(meas_px)):
        px = int(meas_px[i])
        if px < 0 or px >= n:
            continue
        pred_wl = float(wavelengths[px])
        same_mask = ref_is_dip == meas_is_dip[i]
        if not np.any(same_mask):
            continue
        dist_same = np.abs(ref_wl[same_mask] - pred_wl)
        j_same = int(np.argmin(dist_same))
        delta_x = float(dist_same[j_same])
        contrib = 1.0 - (delta_x / delta_scale_nm) ** delta_exponent
        int_diff = abs(meas_int[i] - ref_int[same_mask][j_same])
        contrib -= intensity_penalty_scale * int_diff
        total += contrib
    return total


def _dot_score(
    measured: np.ndarray,
    wl_arr: np.ndarray,
    ref_wl: np.ndarray,
    ref_spd: np.ndarray,
) -> float:
    """Continuous spectrum match in the overlap region only.

    Evaluates only where the measured wavelength range intersects the reference
    grid (no zero-padding artifacts at the edges).  Uses Pearson-style
    mean-subtracted dot product weighted by coverage:
      - peak × peak  →  positive reward
      - peak × low   →  near-zero
      - peak × dip (below mean) → negative penalty (proportional to heights)
    No thresholds; works for any reference spectrum.
    """
    # Only compare inside the measured wavelength window
    in_range = (ref_wl >= wl_arr[0]) & (ref_wl <= wl_arr[-1])
    if not in_range.any():
        return -1e9

    ref_valid = ref_spd[in_range]
    meas_at_ref = np.interp(ref_wl[in_range], wl_arr, measured)

    meas_max = float(meas_at_ref.max())
    ref_max = float(ref_valid.max())
    if meas_max <= 0.0 or ref_max <= 0.0:
        return -1e9

    meas_n = meas_at_ref / meas_max
    ref_n = ref_valid / ref_max

    # Mean subtraction (Pearson-style) avoids baseline sensitivity
    meas_c = meas_n - meas_n.mean()
    ref_c = ref_n - ref_n.mean()

    denom = float(np.sqrt(np.dot(meas_c, meas_c) * np.dot(ref_c, ref_c)))
    if denom <= 0.0:
        return -1e9

    pearson = float(np.dot(meas_c, ref_c) / denom)

    # Penalise solutions where the measured window dwarfs the reference range.
    # meas_coverage = fraction of the measured wavelength span that falls inside ref.
    # A solution mapping 268-909 nm only uses 58% of its span for the ref; correct
    # solutions that almost fully overlap the ref get near 1.0.
    overlap_nm = float(min(wl_arr[-1], ref_wl[-1]) - max(wl_arr[0], ref_wl[0]))
    meas_span_nm = float(wl_arr[-1] - wl_arr[0])
    meas_coverage = max(0.0, overlap_nm / meas_span_nm) if meas_span_nm > 0 else 0.0

    if meas_coverage < 0.25:
        return -1e9

    return pearson * (meas_coverage ** 2)


def _feature_pair_score(
    meas_px: np.ndarray,
    meas_is_dip: np.ndarray,
    meas_h: np.ndarray,
    ref_feat_wl: np.ndarray,
    ref_feat_is_dip: np.ndarray,
    ref_feat_h: np.ndarray,
    wl_arr: np.ndarray,
    sigma_nm: float,
    sigma_int: float,
) -> float:
    """All-pairs 2-D Gaussian score: distance in nm × distance in intensity.

    Same-type pairs add a reward; cross-type pairs subtract a penalty.
    Both are weighted by the product of the two feature heights, so
    shallow features contribute little while prominent ones dominate.
    No hard thresholds; every pair contributes continuously.
    """
    n = len(wl_arr)
    valid = (meas_px >= 0) & (meas_px < n)
    px = meas_px[valid].astype(np.intp)
    is_dip_m = meas_is_dip[valid]
    h_m = meas_h[valid]

    wl_m = wl_arr[px]                               # (n_meas,)
    d_nm = wl_m[:, None] - ref_feat_wl[None, :]     # (n_meas, n_ref)
    d_int = h_m[:, None] - ref_feat_h[None, :]      # (n_meas, n_ref)
    g_nm = np.exp(-d_nm * d_nm / (sigma_nm * sigma_nm))
    g_int = np.exp(-d_int * d_int / (sigma_int * sigma_int))
    weight = h_m[:, None] * ref_feat_h[None, :]
    sign = np.where(is_dip_m[:, None] == ref_feat_is_dip[None, :], 1.0, -1.0)
    return float((weight * g_nm * g_int * sign).sum())


def _best_match_score(
    meas_px: np.ndarray,
    meas_is_dip: np.ndarray,
    meas_h: np.ndarray,
    ref_feat_wl: np.ndarray,
    ref_feat_is_dip: np.ndarray,
    ref_feat_h: np.ndarray,
    wl_arr: np.ndarray,
    sigma_nm: float,
    sigma_int: float,  # kept for API compatibility, not used
) -> float:
    """Coverage-aware best-match score.

    For each reference feature j, compute:
        per_j = h_ref_j × (best_match_j + best_coverage_j − 1)

    where:
        best_match_j    = max_i (h_meas_i × gauss_nm(d_ij) × sign_ij)
        best_coverage_j = max_i of same-type gauss_nm(d_ij)   (0–1)

    This rewards a close same-type match (positive per_j) and *penalises*
    reference features that have no nearby measured counterpart (per_j → −h_ref_j
    when nothing is close).  The penalty prevents false calibrations that perfectly
    align only a subset of reference features from outscoring a correct calibration
    that aligns more features at moderate distances.
    """
    n = len(wl_arr)
    valid = (meas_px >= 0) & (meas_px < n)
    px = meas_px[valid].astype(np.intp)
    is_dip_m = meas_is_dip[valid]
    h_m = meas_h[valid]

    wl_m = wl_arr[px]                                    # (n_meas,)
    d_nm = wl_m[:, None] - ref_feat_wl[None, :]          # (n_meas, n_ref)
    g_nm = np.exp(-d_nm * d_nm / (sigma_nm * sigma_nm))  # (n_meas, n_ref)
    # Intensity similarity in [0, 1]: favour peaks of comparable prominence.
    d_int = h_m[:, None] - ref_feat_h[None, :]
    sigma_i = max(1e-6, sigma_int)
    g_int = np.exp(-d_int * d_int / (sigma_i * sigma_i))
    sign = np.where(is_dip_m[:, None] == ref_feat_is_dip[None, :], 1.0, -1.0)
    same = (sign > 0).astype(np.float64)                 # 1 = same type, 0 = cross

    # Best weighted same-or-cross-type match per reference feature
    best_match = (h_m[:, None] * g_nm * g_int * sign).max(axis=0)     # (n_ref,)
    # Best same-type wavelength coverage per reference feature
    best_coverage = (g_nm * same).max(axis=0)                  # (n_ref,) ∈ [0, 1]

    # Local structure term: compare left–center–right geometry in wavelength
    # between reference and measured, using the best same-type match as center.
    n_ref = ref_feat_wl.shape[0]
    structure = np.ones(n_ref, dtype=np.float64)
    if n_ref >= 3 and h_m.size > 1:
        idx = np.arange(n_ref)
        left_r = np.clip(idx - 1, 0, n_ref - 1)
        right_r = np.clip(idx + 1, 0, n_ref - 1)
        has_triplet = (idx > 0) & (idx < n_ref - 1)

        # Best same-type measured index per reference feature
        weighted_same = h_m[:, None] * g_nm * g_int * same  # (n_meas, n_ref)
        best_idx = np.argmax(weighted_same, axis=0)  # (n_ref,)
        left_m = np.clip(best_idx - 1, 0, h_m.size - 1)
        right_m = np.clip(best_idx + 1, 0, h_m.size - 1)

        sigma_struct = 0.08
        sign_ref = np.where(ref_feat_is_dip, -1.0, 1.0)
        sign_meas = np.where(is_dip_m, -1.0, 1.0)

        t_idx = np.where(has_triplet)[0]
        if t_idx.size > 0:
            lamL_ref = ref_feat_wl[left_r[t_idx]]
            lamC_ref = ref_feat_wl[t_idx]
            lamR_ref = ref_feat_wl[right_r[t_idx]]
            span_ref = np.maximum(lamR_ref - lamL_ref, 1e-6)
            rel_ref = (lamC_ref - lamL_ref) / span_ref

            lamL_meas = wl_m[left_m[t_idx]]
            lamC_meas = wl_m[best_idx[t_idx]]
            lamR_meas = wl_m[right_m[t_idx]]
            span_meas = np.maximum(lamR_meas - lamL_meas, 1e-6)
            rel_meas = (lamC_meas - lamL_meas) / span_meas

            # Neighbour type pattern match (peak/dip of left/right neighbours)
            left_ref_sign = sign_ref[left_r[t_idx]]
            right_ref_sign = sign_ref[right_r[t_idx]]
            left_meas_sign = sign_meas[left_m[t_idx]]
            right_meas_sign = sign_meas[right_m[t_idx]]
            pattern_ok = (left_ref_sign == left_meas_sign) & (right_ref_sign == right_meas_sign)
            pattern_penalty = np.where(pattern_ok, 1.0, 0.3)

            diff_rel = rel_meas - rel_ref
            s_struct = np.exp(-diff_rel * diff_rel / (2.0 * sigma_struct * sigma_struct))
            s_struct *= pattern_penalty

            structure[t_idx] = s_struct

    alpha = 0.7
    per_ref = ref_feat_h * (best_match + best_coverage + alpha * structure - (1.0 + alpha))
    return float(per_ref.sum())


def calibrate_spectrum_anchors(
    n_pixels: int,
    *,
    meas_pixels: np.ndarray,
    meas_is_dip: np.ndarray,
    meas_intensities: np.ndarray,
    ref_feat_wl: np.ndarray,
    ref_feat_is_dip: np.ndarray,
    ref_feat_intensities: np.ndarray,
    min_slope: float = 0.15,
    max_slope: float = 0.70,
    min_intercept: float = 200.0,
    max_intercept: float = 500.0,
    sigma_nm: float = 10.0,
    sigma_int: float = 0.3,
    top_k: int = 18,
) -> PeakDipResult | None:
    """2-anchor RANSAC: fit on a pair of features, score on ALL features.

    For every combination of two measured features (i, j) paired with two
    reference features (a, b) that have the same type and consistent ordering,
    derive the implied linear mapping λ = m·px + c.  Score that mapping using
    the continuous _feature_pair_score on *all* detected features (peaks and
    dips), with no hard threshold anywhere.

    Only the *top_k* most prominent features from each spectrum are used as
    candidate anchors to keep the search fast.  The final scoring uses the
    complete feature sets, so alignment quality is measured against every
    detected line.

    No reference peak wavelength database is required — everything is derived
    from the shape of the two spectra.
    """
    px = np.asarray(meas_pixels, dtype=np.float64).ravel()
    md = np.asarray(meas_is_dip, dtype=bool).ravel()
    mi = np.asarray(meas_intensities, dtype=np.float64).ravel()
    rw = np.asarray(ref_feat_wl, dtype=np.float64).ravel()
    rd = np.asarray(ref_feat_is_dip, dtype=bool).ravel()
    ri = np.asarray(ref_feat_intensities, dtype=np.float64).ravel()

    if len(px) < 2 or len(rw) < 2:
        return None

    mi_n = mi / float(mi.max()) if mi.max() > 0 else mi.copy()
    ri_n = ri / float(ri.max()) if ri.max() > 0 else ri.copy()

    # Top-K by prominence for anchor candidates
    k = min(top_k, len(px), len(rw))
    m_top = np.argsort(mi)[-k:]
    r_top = np.argsort(ri)[-k:]

    px_t = px[m_top]
    md_t = md[m_top]
    rw_t = rw[r_top]
    rd_t = rd[r_top]

    nt, nr = int(k), int(k)
    px_arr = np.arange(n_pixels, dtype=np.float64)

    best_score = -1e9
    best_m, best_c = 0.3, 380.0
    candidates: list[tuple[float, float]] = []

    for i in range(nt):
        for j in range(i + 1, nt):
            dp = float(px_t[j] - px_t[i])
            if abs(dp) < 5:
                continue
            for a in range(nr):
                if md_t[i] != rd_t[a]:
                    continue
                for b in range(nr):
                    if b == a or md_t[j] != rd_t[b]:
                        continue
                    dw = float(rw_t[b] - rw_t[a])
                    m = dw / dp
                    if not (min_slope <= m <= max_slope):
                        continue
                    c = float(rw_t[a]) - m * float(px_t[i])
                    if not (min_intercept <= c <= max_intercept):
                        continue
                    candidates.append((m, c))

    if not candidates:
        return None

    # Evaluate all candidates with a medium sigma, keep top-20 for refinement
    scored: list[tuple[float, float, float]] = []
    for m, c in candidates:
        wl_a = m * px_arr + c
        s = _best_match_score(px, md, mi_n, rw, rd, ri_n, wl_a, sigma_nm, sigma_int)
        scored.append((s, m, c))
        if s > best_score:
            best_score, best_m, best_c = s, m, c

    # Refine the top-20 candidates with progressively tighter sigma
    top20 = sorted(scored, key=lambda x: -x[0])[:20]

    def _neg_best(x: np.ndarray, sigma: float) -> float:
        wl_a = float(x[0]) * px_arr + float(x[1])
        return -_best_match_score(px, md, mi_n, rw, rd, ri_n, wl_a, sigma, sigma_int)

    refined: list[tuple[float, float, float]] = []
    for _, m0, c0 in top20:
        b = [
            (max(min_slope, m0 * 0.97), min(max_slope, m0 * 1.03)),
            (max(min_intercept, c0 - 15.0), min(max_intercept, c0 + 15.0)),
        ]
        r = minimize(lambda x: _neg_best(x, 4.0), x0=[m0, c0], method="L-BFGS-B",
                     bounds=b, options={"maxiter": 400, "ftol": 1e-12})
        rm = float(np.clip(r.x[0], min_slope, max_slope))
        rc = float(np.clip(r.x[1], min_intercept, max_intercept))
        wl_r = rm * px_arr + rc
        s = _best_match_score(px, md, mi_n, rw, rd, ri_n, wl_r, sigma_nm=4.0, sigma_int=sigma_int)
        refined.append((s, rm, rc))
        if s > best_score:
            best_score, best_m, best_c = s, rm, rc

    # Final tight-sigma re-rank: pick the single best refined candidate
    top5 = sorted(refined, key=lambda x: -x[0])[:5]
    for _, m0, c0 in top5:
        b = [
            (max(min_slope, m0 * 0.98), min(max_slope, m0 * 1.02)),
            (max(min_intercept, c0 - 6.0), min(max_intercept, c0 + 6.0)),
        ]
        r = minimize(lambda x: _neg_best(x, 2.5), x0=[m0, c0], method="L-BFGS-B",
                     bounds=b, options={"maxiter": 400, "ftol": 1e-14})
        rm = float(np.clip(r.x[0], min_slope, max_slope))
        rc = float(np.clip(r.x[1], min_intercept, max_intercept))
        wl_r = rm * px_arr + rc
        s = _best_match_score(px, md, mi_n, rw, rd, ri_n, wl_r, sigma_nm=2.5, sigma_int=sigma_int)
        if s > best_score:
            best_score, best_m, best_c = s, rm, rc

    wl_linear = best_m * px_arr + best_c

    # Collect cal_points: all measured features that land within 3% of spectrum range
    soft_tol = (wl_linear[-1] - wl_linear[0]) * 0.03
    cal_points: list[tuple[int, float]] = [
        (0, float(wl_linear[0])),
        (n_pixels - 1, float(wl_linear[-1])),
    ]
    for i in range(len(px)):
        pxi = int(px[i])
        if 0 <= pxi < n_pixels:
            pred = wl_linear[pxi]
            same = rd == md[i]
            if not np.any(same):
                continue
            j = int(np.argmin(np.abs(rw[same] - pred)))
            if float(np.abs(rw[same][j] - pred)) <= soft_tol:
                cal_points.append((pxi, float(rw[same][j])))

    cal_points = _filter_non_crossing(_dedupe_cal_points(sorted(cal_points, key=lambda x: x[0])))
    if len(cal_points) < 3:
        cal_points = [(0, float(wl_linear[0])),
                      (n_pixels // 2, float(wl_linear[n_pixels // 2])),
                      (n_pixels - 1, float(wl_linear[-1]))]

    px_fit = np.array([p for p, _ in cal_points], dtype=np.float64)
    wl_fit = np.array([w for _, w in cal_points], dtype=np.float64)
    coeffs = np.polyfit(px_fit, wl_fit, 1)
    m_fit, c_fit = float(coeffs[0]), float(coeffs[1])
    wl_lin2 = m_fit * px_arr + c_fit
    wl_cauchy = fit_cal_points(cal_points, n_pixels)

    sc_lin = _best_match_score(px, md, mi_n, rw, rd, ri_n, wl_lin2, sigma_nm=4.0, sigma_int=sigma_int)
    sc_cau = _best_match_score(px, md, mi_n, rw, rd, ri_n, wl_cauchy, sigma_nm=4.0, sigma_int=sigma_int)
    wavelengths = wl_cauchy if sc_cau >= sc_lin else wl_lin2
    m_final = m_fit if sc_cau < sc_lin else (float(wl_cauchy[-1]) - float(wl_cauchy[0])) / (n_pixels - 1)
    c_final = c_fit if sc_cau < sc_lin else float(wl_cauchy[0])

    n_peak_match = n_dip_match = n_no_match = 0
    for i in range(len(px)):
        pxi = int(px[i])
        pred = float(wavelengths[pxi]) if 0 <= pxi < n_pixels else m_final * float(px[i]) + c_final
        same = rd == md[i]
        if not np.any(same):
            n_no_match += 1
            continue
        d = float(np.min(np.abs(rw[same] - pred)))
        if d < soft_tol:
            n_dip_match += int(md[i])
            n_peak_match += int(not md[i])
        else:
            n_no_match += 1

    # Build a mock score_grid so downstream display code works
    score_grid = np.array([[best_score]])
    m_bins_arr = np.array([m_final])
    c_bins_arr = np.array([c_final])

    return PeakDipResult(
        slope=m_final,
        intercept=c_final,
        score=best_score,
        n_peak_match=n_peak_match,
        n_dip_match=n_dip_match,
        n_mismatch=n_no_match,
        score_grid=score_grid,
        m_bins=m_bins_arr,
        c_bins=c_bins_arr,
        cal_points=cal_points,
        wavelengths=wavelengths,
    )


def calibrate_peak_dip_grid(
    measured: np.ndarray,
    ref_wavelengths: np.ndarray,
    ref_spd: np.ndarray,
    n_pixels: int,
    *,
    meas_pixels: np.ndarray,
    meas_is_dip: np.ndarray,
    meas_intensities: np.ndarray,
    ref_feat_wl: np.ndarray,
    ref_feat_is_dip: np.ndarray,
    ref_feat_intensities: np.ndarray,
    num_slopes: int = 150,
    num_intercepts: int = 150,
    min_slope: float = 0.22,
    max_slope: float = 0.32,
    min_intercept: float = 370.0,
    max_intercept: float = 420.0,
) -> PeakDipResult | None:
    """Find best linear λ = m·pixel + c by matching two raw spectra.

    Score = continuous dot-product of baseline-subtracted spectra (primary,
    works for any reference, no peak-wavelength knowledge required) PLUS a
    2-D Gaussian all-pairs feature score (secondary, handles fine alignment).

    No hard thresholds anywhere: every feature pair contributes proportionally
    to its prominence and proximity.  Cross-type alignment (peak↔dip) subtracts
    a penalty proportional to both feature heights.
    """
    px = np.asarray(meas_pixels, dtype=np.float64).ravel()
    md = np.asarray(meas_is_dip, dtype=bool).ravel()
    mi = np.asarray(meas_intensities, dtype=np.float64).ravel()
    rfw = np.asarray(ref_feat_wl, dtype=np.float64).ravel()
    rfd = np.asarray(ref_feat_is_dip, dtype=bool).ravel()
    rfi = np.asarray(ref_feat_intensities, dtype=np.float64).ravel()
    ref_wl = np.asarray(ref_wavelengths, dtype=np.float64).ravel()
    ref_s = np.asarray(ref_spd, dtype=np.float64).ravel()
    meas = np.asarray(measured, dtype=np.float64).ravel()

    if len(px) < 2 or len(rfw) < 2:
        return None

    # Normalise intensity heights to [0,1] for consistent Gaussian scaling
    mi_max = float(mi.max()) if mi.max() > 0 else 1.0
    rfi_max = float(rfi.max()) if rfi.max() > 0 else 1.0
    mi_n = mi / mi_max
    rfi_n = rfi / rfi_max

    m_bins_arr = np.linspace(min_slope, max_slope, num_slopes + 1)
    c_bins_arr = np.linspace(min_intercept, max_intercept, num_intercepts + 1)
    m_centers = (m_bins_arr[:-1] + m_bins_arr[1:]) / 2.0
    c_centers = (c_bins_arr[:-1] + c_bins_arr[1:]) / 2.0

    px_arr = np.arange(n_pixels, dtype=np.float64)
    best_score = -1e9
    best_m, best_c = float(m_centers[len(m_centers) // 2]), float(c_centers[len(c_centers) // 2])
    score_grid = np.full((num_slopes, num_intercepts), -1e9, dtype=np.float64)

    def _feat_score(m: float, c: float, sigma_nm: float) -> float:
        wl_a = m * px_arr + c
        return _feature_pair_score(px, md, mi_n, rfw, rfd, rfi_n, wl_a, sigma_nm=sigma_nm, sigma_int=0.3)

    # Coarse search: dot product as the primary discriminator (smooth, no false peaks from
    # discrete feature spacing, penalises over-ranging via meas_coverage²).
    for mi_idx, m in enumerate(m_centers):
        for ci_idx, c in enumerate(c_centers):
            s = _dot_score(meas, m * px_arr + c, ref_wl, ref_s)
            score_grid[mi_idx, ci_idx] = s
            if s > best_score:
                best_score = s
                best_m, best_c = m, c

    # Fine pass: switch to discrete feature-pair score for sub-nm precision,
    # within a tight window around the dot-product best, hard-clamped to original bounds.
    m_step = (max_slope - min_slope) / num_slopes
    c_step = (max_intercept - min_intercept) / num_intercepts
    dm = m_step * 4
    dc = c_step * 4
    m2 = np.linspace(max(min_slope, best_m - dm), min(max_slope, best_m + dm), 61)
    c2 = np.linspace(max(min_intercept, best_c - dc), min(max_intercept, best_c + dc), 61)
    best_feat = -1e9
    for mv in (m2[:-1] + m2[1:]) / 2.0:
        for cv in (c2[:-1] + c2[1:]) / 2.0:
            s = _feat_score(mv, cv, sigma_nm=8.0)
            if s > best_feat:
                best_feat = s
                best_m, best_c = mv, cv

    # Scipy refinement using feature-pair score only
    scipy_bounds = [
        (max(min_slope, best_m - m_step), min(max_slope, best_m + m_step)),
        (max(min_intercept, best_c - c_step), min(max_intercept, best_c + c_step)),
    ]

    def _neg_feat(x: np.ndarray) -> float:
        return -_feat_score(float(x[0]), float(x[1]), sigma_nm=5.0)

    res = minimize(
        _neg_feat,
        x0=np.array([best_m, best_c]),
        method="L-BFGS-B",
        bounds=scipy_bounds,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    best_m = float(np.clip(res.x[0], min_slope, max_slope))
    best_c = float(np.clip(res.x[1], min_intercept, max_intercept))

    wl_linear = best_m * px_arr + best_c

    # Collect cal points: for each measured feature, nearest same-type ref within tolerance
    # Use a relative tolerance based on spectrum width (no hard threshold)
    spec_width_nm = float(wl_linear[-1] - wl_linear[0])
    soft_tol = spec_width_nm * 0.03  # 3% of full range ≈ 10 nm for 350 nm range
    cal_points: list[tuple[int, float]] = [
        (0, float(wl_linear[0])),
        (n_pixels - 1, float(wl_linear[-1])),
    ]
    for i in range(len(px)):
        pxi = int(px[i])
        if 0 <= pxi < n_pixels:
            pred = wl_linear[pxi]
            same_mask = rfd == md[i]
            if not np.any(same_mask):
                continue
            j = int(np.argmin(np.abs(rfw[same_mask] - pred)))
            d = float(np.abs(rfw[same_mask][j] - pred))
            if d <= soft_tol:
                cal_points.append((pxi, float(rfw[same_mask][j])))

    cal_points = _filter_non_crossing(_dedupe_cal_points(sorted(cal_points, key=lambda x: x[0])))
    if len(cal_points) < 3:
        cal_points = [(0, float(wl_linear[0])), (n_pixels // 2, float(wl_linear[n_pixels // 2])),
                      (n_pixels - 1, float(wl_linear[-1]))]

    # Refit: linear and Cauchy from cal_points; pick by feature score
    px_fit = np.array([p for p, _ in cal_points], dtype=np.float64)
    wl_fit = np.array([w for _, w in cal_points], dtype=np.float64)
    coeffs = np.polyfit(px_fit, wl_fit, 1)
    m_fit, c_fit = float(coeffs[0]), float(coeffs[1])
    wl_lin2 = m_fit * px_arr + c_fit
    wl_cauchy = fit_cal_points(cal_points, n_pixels)

    sc_lin = _feature_pair_score(px, md, mi_n, rfw, rfd, rfi_n, wl_lin2, sigma_nm=8.0, sigma_int=0.3)
    sc_cau = _feature_pair_score(px, md, mi_n, rfw, rfd, rfi_n, wl_cauchy, sigma_nm=8.0, sigma_int=0.3)
    wavelengths = wl_cauchy if sc_cau >= sc_lin else wl_lin2
    m_final = m_fit if sc_cau < sc_lin else (wl_cauchy[-1] - wl_cauchy[0]) / (n_pixels - 1)
    c_final = c_fit if sc_cau < sc_lin else float(wl_cauchy[0])

    n_peak_match = n_dip_match = n_no_match = 0
    for i in range(len(px)):
        pxi = int(px[i])
        pred = float(wavelengths[pxi]) if 0 <= pxi < n_pixels else m_final * pxi + c_final
        same_mask = rfd == md[i]
        if not np.any(same_mask):
            n_no_match += 1
            continue
        d = float(np.min(np.abs(rfw[same_mask] - pred)))
        match_tol = soft_tol
        if d < match_tol:
            n_dip_match += 1 if md[i] else 0
            n_peak_match += 1 if not md[i] else 0
        else:
            n_no_match += 1
    n_mismatch = n_no_match

    return PeakDipResult(
        slope=m_final,
        intercept=c_final,
        score=best_score,
        n_peak_match=n_peak_match,
        n_dip_match=n_dip_match,
        n_mismatch=n_mismatch,
        score_grid=score_grid,
        m_bins=m_centers,
        c_bins=c_centers,
        cal_points=cal_points,
        wavelengths=wavelengths,
    )


def _alignment_cost_all_features(
    meas_px: np.ndarray,
    meas_is_dip: np.ndarray,
    ref_wl: np.ndarray,
    ref_is_dip: np.ndarray,
    wavelengths: np.ndarray,
    *,
    sigma: float = 12.0,
) -> float:
    """Negative Gaussian inlier score: reward same-type closeness, ignore cross-type.

    score = sum_i exp(-d_i^2 / sigma^2) where d_i = dist to closest same-type ref.
    Returns -score so lower is better (minimisation convention).
    """
    n = len(wavelengths)
    score = 0.0
    inv_s2 = 1.0 / (sigma * sigma)
    for i in range(len(meas_px)):
        px = int(meas_px[i])
        if px < 0 or px >= n:
            continue
        pred_wl = float(wavelengths[px])
        same_mask = ref_is_dip == meas_is_dip[i]
        if not np.any(same_mask):
            continue
        d = float(np.min(np.abs(ref_wl[same_mask] - pred_wl)))
        score += np.exp(-d * d * inv_s2)
    return -score


def calibrate_peak_dip_ransac(
    meas_pixels: np.ndarray,
    meas_is_dip: np.ndarray,
    meas_intensities: np.ndarray,
    ref_wavelengths: np.ndarray,
    ref_is_dip: np.ndarray,
    ref_intensities: np.ndarray,
    n_pixels: int,
    *,
    min_subset: int = 3,
    max_iterations: int = 500,
    use_cauchy: bool = True,
) -> PeakDipResult | None:
    """RANSAC-style: fit on subset, score on ALL features (distance to closest).

    Sample subsets of min_subset pairs (same type, no crossing). Fit linear/Cauchy.
    Cost = sum over all measured of distance to closest ref of same type.
    Pick subset with minimum cost.
    """
    px = np.asarray(meas_pixels, dtype=np.float64).ravel()
    md = np.asarray(meas_is_dip, dtype=bool).ravel()
    rw = np.asarray(ref_wavelengths, dtype=np.float64).ravel()
    rd = np.asarray(ref_is_dip, dtype=bool).ravel()

    if len(px) < min_subset or len(rw) < min_subset:
        return None

    rng = np.random.default_rng()
    meas_peaks = np.where(~md)[0]
    meas_dips = np.where(md)[0]
    ref_peaks = np.where(~rd)[0]
    ref_dips = np.where(rd)[0]

    def _sample_subset(
        m_idx: np.ndarray,
        r_idx: np.ndarray,
        k: int,
    ) -> list[tuple[int, float]] | None:
        """Sample k ordered pairs (px, ref_wl) from same-type features."""
        if len(m_idx) < k or len(r_idx) < k:
            return None
        m_sorted = np.sort(m_idx)
        r_sorted = np.sort(r_idx)
        m_sel = rng.choice(len(m_sorted), size=k, replace=False)
        r_sel = rng.choice(len(r_sorted), size=k, replace=False)
        m_sel = np.sort(m_sel)
        r_sel = np.sort(r_sel)
        pairs = [
            (int(px[m_sorted[mi]]), float(rw[r_sorted[ri]]))
            for mi, ri in zip(m_sel, r_sel)
        ]
        pairs.sort(key=lambda x: x[0])
        if not all(pairs[i][1] < pairs[i + 1][1] for i in range(len(pairs) - 1)):
            return None
        return pairs

    best_cost = 1e9
    best_cal: list[tuple[int, float]] = []
    best_wl: np.ndarray | None = None

    for _ in range(max_iterations):
        cal: list[tuple[int, float]] | None = None
        if len(meas_peaks) >= min_subset and len(ref_peaks) >= min_subset:
            cal = _sample_subset(meas_peaks, ref_peaks, min_subset)
        if cal is None and len(meas_dips) >= min_subset and len(ref_dips) >= min_subset:
            cal = _sample_subset(meas_dips, ref_dips, min_subset)
        if cal is None:
            continue
        px_fit = np.array([p for p, _ in cal], dtype=np.float64)
        wl_fit = np.array([w for _, w in cal], dtype=np.float64)
        coeffs = np.polyfit(px_fit, wl_fit, 1)
        m_fit, c_fit = float(coeffs[0]), float(coeffs[1])
        pred_lo = m_fit * 0 + c_fit
        pred_hi = m_fit * (n_pixels - 1) + c_fit
        if pred_lo < 350 or pred_hi > 800 or pred_lo > pred_hi:
            continue

        if use_cauchy:
            wl_arr = fit_cal_points(cal, n_pixels)
        else:
            wl_arr = m_fit * np.arange(n_pixels, dtype=np.float64) + c_fit

        cost = _alignment_cost_all_features(px, md, rw, rd, wl_arr)
        if cost < best_cost:
            best_cost = cost
            best_cal = cal
            best_wl = wl_arr

    if best_wl is None or len(best_cal) < min_subset:
        return None

    # Expand: add inliers (all features within threshold of best fit) and refit
    inlier_threshold_nm = 15.0
    expanded: list[tuple[int, float]] = list(best_cal)
    for i in range(len(px)):
        pxi = int(px[i])
        if 0 <= pxi < n_pixels:
            pred = best_wl[pxi]
            same = rd == md[i]
            if np.any(same):
                j = int(np.argmin(np.abs(rw - pred)))
                if md[i] == rd[j] and abs(rw[j] - pred) <= inlier_threshold_nm:
                    cal_points.append((pxi, float(rw[j])))
    best_cal = _filter_non_crossing(_dedupe_cal_points(sorted(expanded, key=lambda x: x[0])))
    if len(best_cal) >= min_subset:
        best_wl = fit_cal_points(best_cal, n_pixels)

    px_fit = np.array([p for p, _ in best_cal], dtype=np.float64)
    wl_fit = np.array([w for _, w in best_cal], dtype=np.float64)
    coeffs = np.polyfit(px_fit, wl_fit, 1)
    m_fit, c_fit = float(coeffs[0]), float(coeffs[1])

    n_peak_match = n_dip_match = n_mismatch = 0
    for i in range(len(px)):
        pred = best_wl[int(px[i])] if 0 <= px[i] < n_pixels else m_fit * px[i] + c_fit
        j = int(np.argmin(np.abs(rw - pred)))
        if np.abs(rw[j] - pred) < 15.0:
            if md[i] == rd[j]:
                n_dip_match += 1 if md[i] else 0
                n_peak_match += 1 if not md[i] else 0
            else:
                n_mismatch += 1

    m_bins = np.array([m_fit - 0.05, m_fit + 0.05])
    c_bins = np.array([c_fit - 20, c_fit + 20])
    score_grid = np.array([[1.0 - best_cost / 1000.0]])

    return PeakDipResult(
        slope=m_fit,
        intercept=c_fit,
        score=-best_cost,
        n_peak_match=n_peak_match,
        n_dip_match=n_dip_match,
        n_mismatch=n_mismatch,
        score_grid=score_grid,
        m_bins=m_bins,
        c_bins=c_bins,
        cal_points=best_cal,
        wavelengths=best_wl,
    )


def _remove_outliers(
    cal_points: list[tuple[int, float]],
    slope: float,
    intercept: float,
    *,
    max_residual_nm: float = 15.0,
) -> list[tuple[int, float]]:
    """Remove cal points with residual > max_residual_nm."""
    out: list[tuple[int, float]] = []
    for px, ref_wl in cal_points:
        pred = slope * px + intercept
        if abs(ref_wl - pred) <= max_residual_nm:
            out.append((px, ref_wl))
    return out


def calibrate_peak_dip(
    meas_pixels: np.ndarray,
    meas_is_dip: np.ndarray,
    meas_intensities: np.ndarray,
    ref_wavelengths: np.ndarray,
    ref_is_dip: np.ndarray,
    ref_intensities: np.ndarray,
    n_pixels: int,
    *,
    num_slopes: int = 100,
    num_intercepts: int = 100,
    min_slope: float = 0.2,
    max_slope: float = 0.8,
    min_intercept: float = 300.0,
    max_intercept: float = 500.0,
    mismatch_penalty: float = 10.0,
    delta_scale_nm: float = 15.0,
    intensity_penalty_scale: float = 0.2,
    delta_exponent: float = 2.0,
    outlier_max_residual_nm: float = 15.0,
) -> PeakDipResult | None:
    """Find linear transform: max aligned peak↔peak/dip↔dip, penalize peak↔dip.

    Score: same-type +1-(δx)² -|Δint|, mismatch -10. Remove outliers by residual.
    """
    px = np.asarray(meas_pixels, dtype=np.float64).ravel()
    mi = np.asarray(meas_intensities, dtype=np.float64).ravel()
    md = np.asarray(meas_is_dip, dtype=bool).ravel()
    rw = np.asarray(ref_wavelengths, dtype=np.float64).ravel()
    rd = np.asarray(ref_is_dip, dtype=bool).ravel()
    ri = np.asarray(ref_intensities, dtype=np.float64).ravel()

    if len(px) < 2 or len(rw) < 2:
        return None

    m_edges = np.linspace(min_slope, max_slope, num_slopes + 1)
    c_edges = np.linspace(min_intercept, max_intercept, num_intercepts + 1)
    score_grid = np.full((num_slopes, num_intercepts), -1e9, dtype=np.float64)

    best_score = -1e9
    best_m, best_c = 0.0, 400.0
    best_cal: list[tuple[int, float]] = []

    for mi_idx in range(num_slopes):
        m = (m_edges[mi_idx] + m_edges[mi_idx + 1]) / 2.0
        for ci_idx in range(num_intercepts):
            c = (c_edges[ci_idx] + c_edges[ci_idx + 1]) / 2.0
            score, cal, _ = _score_peak_dip_alignment(
                px, md, mi, rw, rd, ri,
                m, c,
                mismatch_penalty=mismatch_penalty,
                delta_scale_nm=delta_scale_nm,
                intensity_penalty_scale=intensity_penalty_scale,
                delta_exponent=delta_exponent,
            )
            score_grid[mi_idx, ci_idx] = score
            if score > best_score:
                best_score = score
                best_m, best_c = m, c
                best_cal = cal

    if len(best_cal) < 3:
        return None

    cal_points = _remove_outliers(
        best_cal, best_m, best_c, max_residual_nm=outlier_max_residual_nm
    )
    cal_points = _filter_non_crossing(_dedupe_cal_points(cal_points))
    if len(cal_points) < 3:
        return None

    # Refit slope/intercept via least-squares through cal_points (grid center is coarse)
    px_fit = np.array([p for p, _ in cal_points], dtype=np.float64)
    wl_fit = np.array([w for _, w in cal_points], dtype=np.float64)
    coeffs = np.polyfit(px_fit, wl_fit, 1)
    best_m, best_c = float(coeffs[0]), float(coeffs[1])

    wavelengths = fit_cal_points(cal_points, n_pixels)

    n_peak_match = n_dip_match = n_mismatch = 0
    for i in range(len(px)):
        pred = best_m * px[i] + best_c
        j = int(np.argmin(np.abs(rw - pred)))
        if np.abs(rw[j] - pred) < 10.0:
            if md[i] == rd[j]:
                if md[i]:
                    n_dip_match += 1
                else:
                    n_peak_match += 1
            else:
                n_mismatch += 1

    m_bins = (m_edges[:-1] + m_edges[1:]) / 2.0
    c_bins = (c_edges[:-1] + c_edges[1:]) / 2.0

    return PeakDipResult(
        slope=best_m,
        intercept=best_c,
        score=best_score,
        n_peak_match=n_peak_match,
        n_dip_match=n_dip_match,
        n_mismatch=n_mismatch,
        score_grid=score_grid,
        m_bins=m_bins,
        c_bins=c_bins,
        cal_points=cal_points,
        wavelengths=wavelengths,
    )


def count_aligned_features(
    measured_pixels: np.ndarray,
    ref_wavelengths: np.ndarray,
    wavelengths: np.ndarray,
    *,
    tolerance_nm: float = 5.0,
) -> int:
    """Count how many measured features align with reference after calibration.

    For each measured pixel, predicted_wl = wavelengths[pixel]. Find closest ref.
    """
    n = len(wavelengths)
    count = 0
    for px in np.asarray(measured_pixels, dtype=np.intp):
        if px < 0 or px >= n:
            continue
        pred_wl = float(wavelengths[px])
        dist = np.abs(np.asarray(ref_wavelengths) - pred_wl)
        if np.min(dist) <= tolerance_nm:
            count += 1
    return count
