"""Triplet-based wavelength calibration: detect → descriptor → match → fit."""

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
from .triplet import DEFAULT_WEIGHTS, Triplet, best_pair_score, build as build_triplets, score as triplet_score

__all__ = [
    "AllPeaksInfo",
    "CalibrationResult",
    "Outcome",
    "DEFAULT_WEIGHTS",
    "Triplet",
    "best_pair_score",
    "build_triplets",
    "calibrate",
    "calibrate_extremums",
    "calibrate_peaks",
    "get_reference_peaks",
    "extract",
    "from_known_lines",
    "triplet_score",
]
