"""Exhaustive 2-peak-pair calibration - no known emission wavelengths required.

For each pair of measured peaks (pixels) x each pair of reference peaks (nm)
the implied linear mapping lambda=m*pixel+c is unique.  All candidates are
scored against ALL detected features with a vectorised 2-D Gaussian.
Same-type pairs reward; cross-type subtract proportionally to heights.
No hard thresholds anywhere.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .cauchy_fit import fit_cal_points
from .extremum import extract as _extract
from .hough_matching import (PeakDipResult, _dedupe_cal_points,
                              _feature_pair_score, _filter_non_crossing)


def _pair_candidates(meas_px_top, ref_wl_top, min_slope, max_slope,
                     min_intercept, max_intercept):
    """All (m, c) implied by matching any measured-peak pair to any ref-peak pair."""
    cands = []
    for i in range(len(meas_px_top)):
        for j in range(i + 1, len(meas_px_top)):
            dpx = float(meas_px_top[j] - meas_px_top[i])
            if abs(dpx) < 20.0:
                continue
            for a in range(len(ref_wl_top)):
                for b in range(a + 1, len(ref_wl_top)):
                    dnm = float(ref_wl_top[b] - ref_wl_top[a])
                    if abs(dnm) < 5.0:
                        continue
                    for sign in (1.0, -1.0):
                        m = sign * dnm / dpx
                        if not (min_slope <= m <= max_slope):
                            continue
                        c = float(ref_wl_top[a]) - m * float(meas_px_top[i])
                        if min_intercept <= c <= max_intercept:
                            cands.append((m, c))
    return np.array(cands, dtype=np.float64) if cands else np.zeros((0, 2), dtype=np.float64)


def _score_batch(candidates, meas_px, meas_is_dip, meas_h, ref_wl, ref_is_dip,
                 ref_h, sigma_nm, sigma_int):
    """Vectorised 2-D Gaussian score for all candidates simultaneously."""
    if len(candidates) == 0:
        return np.zeros(0, dtype=np.float64)
    wl_m = candidates[:, 0:1] * meas_px[None, :] + candidates[:, 1:2]
    d_nm = wl_m[:, :, None] - ref_wl[None, None, :]
    g_nm = np.exp(-d_nm * d_nm / (sigma_nm * sigma_nm))
    d_int = meas_h[:, None] - ref_h[None, :]
    g_int = np.exp(-d_int * d_int / (sigma_int * sigma_int))
    sign = np.where(meas_is_dip[:, None] == ref_is_dip[None, :], 1.0, -1.0)
    ws = meas_h[:, None] * ref_h[None, :] * g_int * sign
    return (g_nm * ws[None, :, :]).sum(axis=(1, 2))


def calibrate_exhaustive_pairs(measured, ref_wavelengths, ref_spd, n_pixels, *,
                                n_top_meas=12, n_top_ref=15,
                                n_all_meas=35, n_all_ref=35,
                                min_slope=0.15, max_slope=0.70,
                                min_intercept=200.0, max_intercept=500.0,
                                sigma_nm=10.0, sigma_int=0.3,
                                n_refine=8):
    """Find best linear lambda=m*pixel+c matching two raw spectra.

    No known emission wavelengths required - works with any reference SPD.
    Fits on a subset (top-K peak pairs), validates on ALL detected peaks/dips.
    Goal: maximise number of aligned spectral lines.
    """
    meas = np.asarray(measured, dtype=np.float64).ravel()
    rw_grid = np.asarray(ref_wavelengths, dtype=np.float64).ravel()
    rs = np.asarray(ref_spd, dtype=np.float64).ravel()
    px_arr = np.arange(n_pixels, dtype=np.float64)

    feats_mt = _extract(meas, px_arr, position_px=None, max_count=n_top_meas)
    feats_rt = _extract(rs, rw_grid, position_px=None, max_count=n_top_ref)
    if len(feats_mt) < 2 or len(feats_rt) < 2:
        return None

    feats_ma = _extract(meas, px_arr, position_px=None, max_count=n_all_meas)
    feats_ra = _extract(rs, rw_grid, position_px=None, max_count=n_all_ref)

    px_top = np.array([e.position for e in feats_mt], dtype=np.float64)
    wl_top = np.array([e.position for e in feats_rt], dtype=np.float64)
    px_all = np.array([e.position for e in feats_ma], dtype=np.float64)
    dip_m  = np.array([e.is_dip   for e in feats_ma], dtype=bool)
    h_m    = np.array([abs(e.height) for e in feats_ma], dtype=np.float64)
    wl_all = np.array([e.position for e in feats_ra], dtype=np.float64)
    dip_r  = np.array([e.is_dip   for e in feats_ra], dtype=bool)
    h_r    = np.array([abs(e.height) for e in feats_ra], dtype=np.float64)

    mh = float(h_m.max()) if h_m.size and h_m.max() > 0 else 1.0
    rh = float(h_r.max()) if h_r.size and h_r.max() > 0 else 1.0
    h_mn = h_m / mh
    h_rn = h_r / rh

    cands = _pair_candidates(px_top, wl_top, min_slope, max_slope,
                              min_intercept, max_intercept)
    if len(cands) == 0:
        return None

    scores = _score_batch(cands, px_all, dip_m, h_mn, wl_all, dip_r, h_rn,
                           sigma_nm=sigma_nm, sigma_int=sigma_int)

    sigma_f = sigma_nm * 0.5
    top_idx = np.argsort(scores)[::-1][:n_refine]
    best_s = float(scores[top_idx[0]])
    best_m = float(cands[top_idx[0], 0])
    best_c = float(cands[top_idx[0], 1])

    def _neg(x):
        return -_feature_pair_score(px_all, dip_m, h_mn, wl_all, dip_r, h_rn,
                                    float(x[0]) * px_arr + float(x[1]),
                                    sigma_nm=sigma_f, sigma_int=sigma_int)

    for idx in top_idx:
        m0, c0 = float(cands[idx, 0]), float(cands[idx, 1])
        dm = max(m0 * 0.06, 0.01)
        dc = max(abs(c0) * 0.03, 8.0)
        res = minimize(_neg, x0=np.array([m0, c0]), method="L-BFGS-B",
                       bounds=[(max(min_slope, m0 - dm), min(max_slope, m0 + dm)),
                               (max(min_intercept, c0 - dc), min(max_intercept, c0 + dc))],
                       options={"maxiter": 400, "ftol": 1e-12})
        s = -float(res.fun)
        if s > best_s:
            best_s = s
            best_m = float(np.clip(res.x[0], min_slope, max_slope))
            best_c = float(np.clip(res.x[1], min_intercept, max_intercept))

    wl_lin = best_m * px_arr + best_c
    tol = float(wl_lin[-1] - wl_lin[0]) * 0.025

    cal = [(0, float(wl_lin[0])), (n_pixels - 1, float(wl_lin[-1]))]
    for i in range(len(px_all)):
        pxi = int(px_all[i])
        if 0 <= pxi < n_pixels:
            pred = wl_lin[pxi]
            same = dip_r == dip_m[i]
            if not np.any(same):
                continue
            j = int(np.argmin(np.abs(wl_all[same] - pred)))
            if float(np.abs(wl_all[same][j] - pred)) <= tol:
                cal.append((pxi, float(wl_all[same][j])))

    cal = _filter_non_crossing(_dedupe_cal_points(sorted(cal, key=lambda x: x[0])))
    if len(cal) < 3:
        cal = [(0, float(wl_lin[0])),
               (n_pixels // 2, float(wl_lin[n_pixels // 2])),
               (n_pixels - 1, float(wl_lin[-1]))]

    px_fit = np.array([p for p, _ in cal], dtype=np.float64)
    wl_fit = np.array([w for _, w in cal], dtype=np.float64)
    cf = np.polyfit(px_fit, wl_fit, 1)
    wl_l2  = float(cf[0]) * px_arr + float(cf[1])
    wl_cau = fit_cal_points(cal, n_pixels)

    def _fs(wa):
        return _feature_pair_score(px_all, dip_m, h_mn, wl_all, dip_r, h_rn,
                                   wa, sigma_nm=sigma_f, sigma_int=sigma_int)

    use_c = _fs(wl_cau) >= _fs(wl_l2)
    wls = wl_cau if use_c else wl_l2
    m_f = (wl_cau[-1] - wl_cau[0]) / (n_pixels - 1) if use_c else float(cf[0])
    c_f = float(wl_cau[0]) if use_c else float(cf[1])

    np_m = nd_m = nn_m = 0
    for i in range(len(px_all)):
        pxi  = int(px_all[i])
        pred = float(wls[pxi]) if 0 <= pxi < n_pixels else m_f * pxi + c_f
        same = dip_r == dip_m[i]
        if not np.any(same):
            nn_m += 1
            continue
        d = float(np.min(np.abs(wl_all[same] - pred)))
        if d < tol:
            nd_m += 1 if dip_m[i] else 0
            np_m += 1 if not dip_m[i] else 0
        else:
            nn_m += 1

    return PeakDipResult(slope=m_f, intercept=c_f, score=best_s,
                         n_peak_match=np_m, n_dip_match=nd_m, n_mismatch=nn_m,
                         score_grid=np.array([[best_s]]),
                         m_bins=np.array([best_m]), c_bins=np.array([best_c]),
                         cal_points=cal, wavelengths=wls)
