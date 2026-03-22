"""Stacked spectrum overlays (Load+): store raw SPD columns; display follows UI sensitivity toggle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from typing import Any

from ..csv_viewer.loader import LoadedCsv
from ..processing.reference_correction import apply_dark_white_correction


@dataclass
class SpectrumOverlay:
    """Why: preserve each overlay's measured SPD and its own dark/white/sensitivity columns.

    Display applies dark/white correction first, then sensitivity only when the UI
    toggle is on — using the embedded sensitivity column when present, else the
    active correction engine.
    """

    label: str
    source_wavelengths: np.ndarray
    measured: np.ndarray
    dark: np.ndarray | None = None
    white: np.ndarray | None = None
    sensitivity: np.ndarray | None = None


def overlay_from_loaded_csv(loaded: LoadedCsv, label: str) -> SpectrumOverlay:
    """Build overlay storage from a parsed CSV (same columns as export)."""
    return SpectrumOverlay(
        label=label,
        source_wavelengths=np.asarray(loaded.wavelengths, dtype=np.float64).copy(),
        measured=np.asarray(loaded.intensity, dtype=np.float64).copy(),
        dark=np.asarray(loaded.dark, dtype=np.float64).copy()
        if loaded.dark is not None
        else None,
        white=np.asarray(loaded.white, dtype=np.float64).copy()
        if loaded.white is not None
        else None,
        sensitivity=np.asarray(loaded.sensitivity, dtype=np.float64).copy()
        if loaded.sensitivity is not None
        else None,
    )


def _interp_to_target(
    src_wl: np.ndarray,
    values: np.ndarray,
    target_wl: np.ndarray,
) -> np.ndarray:
    """Linear interpolation onto target_wl (sorted src assumed)."""
    if len(src_wl) < 2 or len(values) == 0:
        return np.zeros(len(target_wl), dtype=np.float64)
    m = min(len(src_wl), len(values))
    sw = np.asarray(src_wl[:m], dtype=np.float64)
    yv = np.asarray(values[:m], dtype=np.float64)
    order = np.argsort(sw)
    sw = sw[order]
    yv = yv[order]
    tw = np.asarray(target_wl, dtype=np.float64).ravel()
    return np.interp(tw, sw, yv, left=float(yv[0]), right=float(yv[-1]))


def _apply_sensitivity_engine(
    intensity: np.ndarray,
    wavelengths: np.ndarray,
    loaded_sens: np.ndarray | None,
    src_wl: np.ndarray,
    ctx: Any,
) -> np.ndarray:
    """Match measurement-mode Load+ sensitivity: CSV column divides intensity; else engine."""
    if loaded_sens is not None:
        sens = np.interp(
            wavelengths.astype(np.float64),
            src_wl.astype(np.float64),
            loaded_sens.astype(np.float64),
        )
        peak = float(np.max(sens))
        if peak > 1e-6:
            sens = sens / peak
        corrected = np.asarray(intensity, dtype=np.float64).copy()
        mask = sens > 1e-6
        corrected[mask] = intensity[mask] / sens[mask]
        peak_c = float(np.max(corrected))
        if peak_c > 1e-6:
            corrected /= peak_c
        return np.clip(corrected, 0, 1).astype(np.float32)

    engine = getattr(ctx, "sensitivity_engine", None)
    if engine is not None:
        return engine.apply(np.asarray(intensity, dtype=np.float32), wavelengths)
    return np.asarray(intensity, dtype=np.float32)


def overlay_dark_white_corrected(
    overlay: SpectrumOverlay,
    target_wl: np.ndarray,
    *,
    dark_session: np.ndarray | None,
    white_session: np.ndarray | None,
) -> np.ndarray:
    """Dark/white corrected trace on target_wl; prefers overlay columns, else session refs."""
    sw = np.asarray(overlay.source_wavelengths, dtype=np.float64).ravel()
    meas = _interp_to_target(sw, overlay.measured, target_wl)
    dark_o = (
        _interp_to_target(sw, overlay.dark, target_wl)
        if overlay.dark is not None
        else None
    )
    white_o = (
        _interp_to_target(sw, overlay.white, target_wl)
        if overlay.white is not None
        else None
    )
    dark = dark_o
    white = white_o
    if dark is None and dark_session is not None:
        dark = np.asarray(dark_session, dtype=np.float64)[: len(target_wl)]
    if white is None and white_session is not None:
        white = np.asarray(white_session, dtype=np.float64)[: len(target_wl)]
    return apply_dark_white_correction(meas, dark, white)


def overlay_display_intensity(
    overlay: SpectrumOverlay,
    target_wl: np.ndarray,
    ctx: Any,
    *,
    sensitivity_enabled: bool,
    dark_session: np.ndarray | None,
    white_session: np.ndarray | None,
) -> np.ndarray:
    """Intensity values for graph/export: same units as main trace for current S toggle."""
    base = overlay_dark_white_corrected(
        overlay,
        target_wl,
        dark_session=dark_session,
        white_session=white_session,
    )
    base = np.clip(np.asarray(base, dtype=np.float64), 0, 1)
    if not sensitivity_enabled:
        return base.astype(np.float32)
    sw = np.asarray(overlay.source_wavelengths, dtype=np.float64).ravel()
    return _apply_sensitivity_engine(
        base.astype(np.float32),
        target_wl,
        overlay.sensitivity,
        sw,
        ctx,
    )
