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

    mask = (rw >= wl_min) & (rw <= wl_max)
    if mask.sum() < 10:
        return None
    wl_ref = rw[mask]
    ref_spd = ri[mask]
    if ref_spd.max() > 0:
        ref_spd = ref_spd / ref_spd.max()

    m_edges = np.linspace(min_slope, max_slope, num_slopes + 1)
    c_edges = np.linspace(min_intercept, max_intercept, num_intercepts + 1)
    score_grid = np.zeros((num_slopes, num_intercepts), dtype=np.float64)

    px = np.arange(n_pixels, dtype=np.float64)
    meas_norm = meas.copy()
    if meas_norm.max() > 0:
        meas_norm = meas_norm / meas_norm.max()

    for mi in range(num_slopes):
        m = (m_edges[mi] + m_edges[mi + 1]) / 2.0
        for ci in range(num_intercepts):
            c = (c_edges[ci] + c_edges[ci + 1]) / 2.0
            wl_meas = m * px + c
            valid = (wl_meas >= wl_min) & (wl_meas <= wl_max)
            if valid.sum() < 10:
                score_grid[mi, ci] = -2.0
                continue
            meas_at_ref = np.interp(wl_ref, wl_meas[valid], meas_norm[valid])
            if np.std(meas_at_ref) < 1e-12 or np.std(ref_spd) < 1e-12:
                score_grid[mi, ci] = -2.0
                continue
            corr = np.corrcoef(meas_at_ref, ref_spd)[0, 1]
            score_grid[mi, ci] = float(corr) if not np.isnan(corr) else -2.0

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


def calibrate_peak_dip_grid(
    meas_pixels: np.ndarray,
    meas_is_dip: np.ndarray,
    meas_intensities: np.ndarray,
    ref_wavelengths: np.ndarray,
    ref_is_dip: np.ndarray,
    ref_intensities: np.ndarray,
    n_pixels: int,
    *,
    num_slopes: int = 120,
    num_intercepts: int = 120,
    min_slope: float = 0.22,
    max_slope: float = 0.32,
    min_intercept: float = 370.0,
    max_intercept: float = 420.0,
) -> PeakDipResult | None:
    """Grid search: find (m,c) that maximizes alignment score over ALL features.

    Score = alignment_score_from_wavelengths (rewards same-type, penalizes mismatch).
    Deterministic. Then collect inliers, refit Cauchy.
    """
    px = np.asarray(meas_pixels, dtype=np.float64).ravel()
    md = np.asarray(meas_is_dip, dtype=bool).ravel()
    mi = np.asarray(meas_intensities, dtype=np.float64).ravel()
    rw = np.asarray(ref_wavelengths, dtype=np.float64).ravel()
    rd = np.asarray(ref_is_dip, dtype=bool).ravel()
    ri = np.asarray(ref_intensities, dtype=np.float64).ravel()

    if len(px) < 2 or len(rw) < 2:
        return None

    m_edges = np.linspace(min_slope, max_slope, num_slopes + 1)
    c_edges = np.linspace(min_intercept, max_intercept, num_intercepts + 1)

    best_cost = 1e9
    best_m, best_c = 0.27, 395.0
    score_grid = np.full((num_slopes, num_intercepts), -1e9, dtype=np.float64)

    px_arr = np.arange(n_pixels, dtype=np.float64)

    for mi_idx in range(num_slopes):
        m = (m_edges[mi_idx] + m_edges[mi_idx + 1]) / 2.0
        for ci_idx in range(num_intercepts):
            c = (c_edges[ci_idx] + c_edges[ci_idx + 1]) / 2.0
            wl_arr = m * px_arr + c
            if wl_arr[0] < 350 or wl_arr[-1] > 800:
                continue
            cost = _alignment_cost_all_features(px, md, rw, rd, wl_arr, sigma=15.0)
            score_grid[mi_idx, ci_idx] = -cost
            if cost < best_cost:
                best_cost = cost
                best_m, best_c = m, c

    # Second-pass: finer grid in a ±2 bin window around the best
    dm = (max_slope - min_slope) / num_slopes * 2
    dc = (max_intercept - min_intercept) / num_intercepts * 2
    m2_edges = np.linspace(best_m - dm, best_m + dm, 81)
    c2_edges = np.linspace(best_c - dc, best_c + dc, 81)
    for m2 in (m2_edges[:-1] + m2_edges[1:]) / 2.0:
        for c2 in (c2_edges[:-1] + c2_edges[1:]) / 2.0:
            wl_arr = m2 * px_arr + c2
            if wl_arr[0] < 350 or wl_arr[-1] > 800:
                continue
            cost = _alignment_cost_all_features(px, md, rw, rd, wl_arr, sigma=8.0)
            if cost < best_cost:
                best_cost = cost
                best_m, best_c = m2, c2

    m_step = (max_slope - min_slope) / num_slopes
    c_step = (max_intercept - min_intercept) / num_intercepts

    def _cost_fine(x: np.ndarray) -> float:
        m, c = float(x[0]), float(x[1])
        wl_arr = m * px_arr + c
        if wl_arr[0] < 350 or wl_arr[-1] > 800:
            return 1e9
        return _alignment_cost_all_features(px, md, rw, rd, wl_arr, sigma=5.0)

    res = minimize(
        _cost_fine,
        x0=np.array([best_m, best_c]),
        method="L-BFGS-B",
        bounds=[
            (best_m - 2 * m_step, best_m + 2 * m_step),
            (best_c - 2 * c_step, best_c + 2 * c_step),
        ],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if res.fun < best_cost:
        best_m, best_c = float(res.x[0]), float(res.x[1])
        best_cost = res.fun

    wl_linear = best_m * px_arr + best_c

    # Collect inliers: closest same-type ref within threshold for each measured feature
    inlier_threshold_nm = 20.0
    cal_points: list[tuple[int, float]] = [
        (0, best_c),
        (n_pixels - 1, best_m * (n_pixels - 1) + best_c),
    ]
    for i in range(len(px)):
        pxi = int(px[i])
        if 0 <= pxi < n_pixels:
            pred = wl_linear[pxi]
            same_mask = rd == md[i]
            if not np.any(same_mask):
                continue
            j_same = int(np.argmin(np.abs(rw[same_mask] - pred)))
            d = abs(rw[same_mask][j_same] - pred)
            if d <= inlier_threshold_nm:
                cal_points.append((pxi, float(rw[same_mask][j_same])))

    cal_points = _filter_non_crossing(_dedupe_cal_points(sorted(cal_points, key=lambda x: x[0])))
    if len(cal_points) < 3:
        return None

    # Refit: linear from cal_points; Cauchy for wavelengths
    px_fit = np.array([p for p, _ in cal_points], dtype=np.float64)
    wl_fit = np.array([w for _, w in cal_points], dtype=np.float64)
    coeffs = np.polyfit(px_fit, wl_fit, 1)
    m_fit, c_fit = float(coeffs[0]), float(coeffs[1])
    wl_linear = m_fit * np.arange(n_pixels, dtype=np.float64) + c_fit
    wl_cauchy = fit_cal_points(cal_points, n_pixels)
    # Use whichever gives better alignment
    sc_lin = alignment_score_from_wavelengths(px, md, mi, rw, rd, ri, wl_linear, delta_scale_nm=15.0, delta_exponent=2.0)
    sc_cau = alignment_score_from_wavelengths(px, md, mi, rw, rd, ri, wl_cauchy, delta_scale_nm=15.0, delta_exponent=2.0)
    wavelengths = wl_cauchy if sc_cau >= sc_lin else wl_linear

    n_peak_match = n_dip_match = n_no_match = 0
    for i in range(len(px)):
        pxi = int(px[i])
        pred = wavelengths[pxi] if 0 <= pxi < n_pixels else m_fit * pxi + c_fit
        same_mask = rd == md[i]
        if not np.any(same_mask):
            n_no_match += 1
            continue
        d = float(np.min(np.abs(rw[same_mask] - pred)))
        if d < 15.0:
            n_dip_match += 1 if md[i] else 0
            n_peak_match += 1 if not md[i] else 0
        else:
            n_no_match += 1
    n_mismatch = n_no_match

    m_bins = (m_edges[:-1] + m_edges[1:]) / 2.0
    c_bins = (c_edges[:-1] + c_edges[1:]) / 2.0

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
