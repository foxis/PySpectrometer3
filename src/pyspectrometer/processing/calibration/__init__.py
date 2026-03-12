"""Wavelength calibration: triplet/Hough matching → Cauchy fit."""

from .calibrate import (
    AllPeaksInfo,
    CalibrationResult,
    Outcome,
    calibrate,
    calibrate_extremums,
    calibrate_peaks,
)
from .detect_peaks import get_reference_peaks
from .extremum import extract, from_known_lines
from .hough_matching import (
    HoughResult,
    PeakDipResult,
    RansacResult,
    SpdCorrelationResult,
    alignment_score_from_wavelengths,
    calibrate_hough,
    calibrate_peak_dip,
    calibrate_ransac,
    count_aligned_features,
    find_best_linear_spd,
)
from .triplet import DEFAULT_WEIGHTS, Triplet, best_pair_score, build as build_triplets, score as triplet_score

__all__ = [
    "HoughResult",
    "PeakDipResult",
    "RansacResult",
    "SpdCorrelationResult",
    "AllPeaksInfo",
    "CalibrationResult",
    "Outcome",
    "DEFAULT_WEIGHTS",
    "Triplet",
    "best_pair_score",
    "build_triplets",
    "calibrate",
    "calibrate_extremums",
    "calibrate_hough",
    "calibrate_peaks",
    "calibrate_peak_dip",
    "calibrate_ransac",
    "count_aligned_features",
    "find_best_linear_spd",
    "get_reference_peaks",
    "extract",
    "from_known_lines",
    "triplet_score",
]
