"""SPD-based peak/dip calibration: 2-anchor RANSAC, no emission line database.

calibrate_spectrum_anchors derives λ = m·pixel + c from same-type feature pairs,
scores on all features via coverage-aware _best_match_score, refines with Cauchy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .cauchy_fit import fit_cal_points


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

    # Collect cal_points from reference features via inverse linear mapping.
    # This avoids local kinks caused by a single mis-snapped measured feature.
    cal_points: list[tuple[int, float]] = []
    for w_ref in rw:
        px_f = (float(w_ref) - best_c) / best_m if best_m != 0.0 else 0.0
        if 0.0 <= px_f <= float(n_pixels - 1):
            cal_points.append((int(round(px_f)), float(w_ref)))

    # Always anchor the full span.
    cal_points.append((0, float(wl_linear[0])))
    cal_points.append((n_pixels - 1, float(wl_linear[-1])))

    cal_points = _filter_non_crossing(_dedupe_cal_points(sorted(cal_points, key=lambda x: x[0])))
    if len(cal_points) < 3:
        cal_points = [
            (0, float(wl_linear[0])),
            (n_pixels // 2, float(wl_linear[n_pixels // 2])),
            (n_pixels - 1, float(wl_linear[-1])),
        ]

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

    soft_tol = (float(wavelengths[-1]) - float(wavelengths[0])) * 0.03

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
