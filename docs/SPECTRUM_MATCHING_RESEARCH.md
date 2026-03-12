# Spectrum Matching for Wavelength Calibration — Research Summary

## Problem

Wavelength calibration maps **pixel → wavelength**. We need to identify which detected peaks correspond to which reference lines, then fit a dispersion model (polynomial or inverse Cauchy).

## Established Approaches

### 1. Hough Transform (RASCAL, IEEE)

**Idea:** Each (pixel, wavelength) pair defines a line in slope–intercept space: `λ = m·pixel + c`. Correct calibration = many pairs agreeing on (m, c).

**Process:**
1. Detect peaks → pixel positions
2. Reference atlas → known wavelengths
3. For each (pixel, ref_wavelength) pair, compute supporting (m, c) via `c = λ - m·pixel`
4. Sample m over plausible range (dispersion), solve for c
5. Bin (m, c) in 2D accumulator; peak = best calibration hypothesis
6. Use top candidates for polynomial fit

**Pros:** Robust to outliers, no explicit matching step, works with many candidate pairs.  
**Refs:** RASCAL (rascal.readthedocs.io), IEEE 8393138 (Hough + RANSAC).

---

### 2. RANSAC (Robust Fitting)

**Idea:** Sample minimal subsets (e.g. 2–3 points), fit model, count inliers. Repeat; keep model with most inliers.

**Process:**
1. Get candidate (pixel, wavelength) pairs (from Hough or brute-force)
2. Sample k points (k = polynomial degree + 1)
3. Fit polynomial
4. Count inliers (residual < tolerance)
5. Iterate; refine with all inliers

**Pros:** Handles outliers, standard in astronomy (PypeIt, specreduce).  
**Refs:** RASCAL, specreduce.

---

### 3. Cross-Correlation

**Idea:** Shift reference spectrum in wavelength; find shift that maximizes correlation with measured spectrum.

**Process:**
1. Interpolate measured spectrum to reference wavelength grid
2. Compute cross-correlation at many shifts
3. Peak of correlation → wavelength zero-point shift
4. Often used for relative calibration (align spectra) or when dispersion is known

**Pros:** No peak matching, works with blended lines, robust to SNR.  
**Refs:** STScI HST notebooks, IUE wavestudy.

---

### 4. Standard Peak-Matching Workflow (Astropy, PypeIt, specreduce)

**Process:**
1. Detect peaks in arc/calibration spectrum
2. Load reference line list (NIST, Hg, Ne, etc.)
3. **Match:** Try hypotheses (e.g. “first peak = 404.66 nm”) and fit; or use Hough to get candidates
4. Fit polynomial (or linear) to matched (pixel, wavelength) pairs
5. Validate with residuals; reject bad matches

**Key:** Matching is the hard part. Hough and RANSAC automate it.

---

## What We Do vs. Literature

| Aspect | Our approach | Literature |
|--------|--------------|------------|
| Matching | Triplet descriptors (height, width, ratios, rel_pos) | Hough (pixel–wavelength pairs) or RANSAC |
| Model | Inverse Cauchy | Polynomial (2nd–4th order) common |
| Robustness | Bootstrap + score ranking | RANSAC, Hough accumulator |
| Reference | Known lines (FL12, Hg) | NIST, arc lamp atlases |

**Issues with triplet matching:**
- Descriptors are sensitive to noise, baseline, and line shape
- No direct use of pixel–wavelength geometry
- Complex scoring; many tunable weights
- Hough/RANSAC use simple geometry and are well validated

---

## Recommendations

1. **Add Hough-based matching** as an alternative path:
   - Input: peak pixels + reference wavelengths
   - Generate (pixel, λ) pairs for all combinations (or a pruned set)
   - Hough: bin (slope, intercept), take top bins
   - Fit polynomial to inlier pairs

2. **Use RANSAC** on candidate (pixel, wavelength) pairs from Hough or from a coarse linear guess.

3. **Keep correlation** as a fallback when few lines are available.

4. **Simplify or phase out triplet descriptors** in favor of geometry-based methods for calibration.

---

## References

- **RASCAL:** https://github.com/jveitchmichaelis/rascal — Hough + RANSAC
- **RASCAL Hough:** https://rascal.readthedocs.io/en/latest/background/houghtransform.html
- **IEEE 8393138:** Hough + RANSAC for spectrometer calibration
- **PypeIt:** https://pypeit.readthedocs.io — Arc lamp calibration, NIST lines
- **specreduce:** WavelengthCalibration1D, polynomial fit to line positions
- **STScI cross-correlation:** Correcting missing wavecals
