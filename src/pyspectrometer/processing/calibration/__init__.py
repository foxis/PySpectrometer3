"""Public calibration API: SPD-based peak/dip matching → Cauchy fit.

This module exposes only the current “winning” strategy:

  - feature extraction (`extract`)
  - SPD-based 2-anchor calibration (`calibrate_spectrum_anchors`)
  - score helpers for analysis/visualisation
"""

from .cauchy_fit import fit_cal_points
from .detect_peaks import get_reference_peaks
from .extremum import extract, from_known_lines
from .hough_matching import (
    PeakDipResult,
    alignment_score_from_wavelengths,
    calibrate_spectrum_anchors,
    compute_score_grid,
    count_aligned_features,
    dot_score,
)

__all__ = [
    "PeakDipResult",
    "calibrate_spectrum_anchors",
    "compute_score_grid",
    "alignment_score_from_wavelengths",
    "count_aligned_features",
    "dot_score",
    "extract",
    "from_known_lines",
    "get_reference_peaks",
    "fit_cal_points",
]

