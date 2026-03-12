"""Benchmark: compare our algorithm output to ground-truth calibration from CSVs."""
import sys, numpy as np
sys.path.insert(0, 'src')
from pyspectrometer.processing.calibration.hough_matching import (
    _alignment_cost_all_features, calibrate_peak_dip_grid, find_best_linear_spd,
    alignment_score_from_wavelengths,
)
from pyspectrometer.data.reference_spectra import ReferenceSource, get_reference_spectrum
from pyspectrometer.processing.calibration.extremum import extract
from pyspectrometer.processing.calibration.cauchy_fit import fit_cal_points

def load(f):
    ints, wls = [], []
    with open(f) as fp:
        for line in fp:
            s = line.strip()
            if s.startswith('#') or s.lower().startswith('pixel'):
                continue
            p = s.split(',')
            if len(p) >= 5:
                try:
                    ints.append(float(p[1]))
                    wls.append(float(p[4]))
                except Exception:
                    pass
    return np.array(ints), np.array(wls)

def gaussian_score(meas_px, meas_dip, ref_wl, ref_is_dip, wl_arr, sigma=8.0):
    return -_alignment_cost_all_features(meas_px, meas_dip, ref_wl, ref_is_dip, wl_arr, sigma=sigma)

def exact_matches(meas_px, meas_dip, ref_wl, ref_is_dip, wl_arr, threshold=5.0):
    return sum(
        1 for i, px in enumerate(meas_px)
        if np.any(ref_is_dip == meas_dip[i]) and
        np.min(np.abs(ref_wl[ref_is_dip == meas_dip[i]] - float(wl_arr[int(px)]))) < threshold
    )

for csv_path, src_name in [
    ('output/Spectrum-20260311--214345.csv', 'FL12'),
    ('output/Spectrum-20260311--213319.csv', 'FL12'),
    ('output/Spectrum-20260311--193856.csv', 'FL12'),
]:
    intensity, wl_gt = load(csv_path)
    n = len(intensity)
    src = ReferenceSource[src_name]

    wl_ref_grid = np.linspace(380, 750, 500)
    ref_spd = get_reference_spectrum(src, wl_ref_grid)
    ref_exts = extract(ref_spd, wl_ref_grid, position_px=None, max_count=25)
    ref_wl = np.array([e.position for e in ref_exts])
    ref_is_dip = np.array([e.is_dip for e in ref_exts])
    ref_int = np.array([abs(e.height) for e in ref_exts])

    px_arr = np.arange(n, dtype=np.intp)
    meas_exts = extract(intensity, wl_gt, position_px=px_arr, max_count=25)
    meas_px = np.array([e.position_px for e in meas_exts if e.position_px is not None], dtype=float)
    meas_dip = np.array([e.is_dip for e in meas_exts if e.position_px is not None])
    meas_int = np.array([abs(e.height) for e in meas_exts if e.position_px is not None])

    # GT scores
    g8_gt = gaussian_score(meas_px, meas_dip, ref_wl, ref_is_dip, wl_gt, sigma=8.0)
    em5_gt = exact_matches(meas_px, meas_dip, ref_wl, ref_is_dip, wl_gt, threshold=5.0)
    em10_gt = exact_matches(meas_px, meas_dip, ref_wl, ref_is_dip, wl_gt, threshold=10.0)
    slope_gt = (wl_gt[-1] - wl_gt[0]) / (n - 1)

    # Run our algorithm (using bounds from GT if GT is within FL12 range)
    min_slope = max(0.22, slope_gt - 0.04)
    max_slope_b = min(0.55, slope_gt + 0.04)
    min_int = max(250.0, float(wl_gt[0]) - 15.0)
    max_int_b = min(420.0, float(wl_gt[0]) + 15.0)

    result = calibrate_peak_dip_grid(
        meas_px, meas_dip, meas_int, ref_wl, ref_is_dip, ref_int, n,
        num_slopes=150, num_intercepts=150,
        min_slope=min_slope, max_slope=max_slope_b,
        min_intercept=min_int, max_intercept=max_int_b,
    )
    if result is not None:
        g8_algo = gaussian_score(meas_px, meas_dip, ref_wl, ref_is_dip, result.wavelengths, sigma=8.0)
        em5_algo = exact_matches(meas_px, meas_dip, ref_wl, ref_is_dip, result.wavelengths, threshold=5.0)
        em10_algo = exact_matches(meas_px, meas_dip, ref_wl, ref_is_dip, result.wavelengths, threshold=10.0)
        dm = result.slope - slope_gt
        dc = result.intercept - float(wl_gt[0])
    else:
        g8_algo = em5_algo = em10_algo = dm = dc = float('nan')

    print(f"\n{'='*65}")
    print(f"  {csv_path.split('/')[-1]}  src={src_name}")
    print(f"  GT:   slope={slope_gt:.4f}  c={wl_gt[0]:.2f}  range={wl_gt[0]:.0f}-{wl_gt[-1]:.0f} nm")
    print(f"  GT score:   gaussian(8)={g8_gt:.2f}  match<5nm={em5_gt}/{len(meas_px)}  match<10nm={em10_gt}/{len(meas_px)}")
    if result is not None:
        print(f"  Algo: slope={result.slope:.4f} (d={dm:+.4f})  c={result.intercept:.2f} (d={dc:+.2f})")
        print(f"  Algo score: gaussian(8)={g8_algo:.2f}  match<5nm={em5_algo}/{len(meas_px)}  match<10nm={em10_algo}/{len(meas_px)}")
