"""Reference spectrum correction (dark/white)."""

from typing import Optional
import numpy as np


def apply_dark_white_correction(
    intensity: np.ndarray,
    dark: Optional[np.ndarray],
    white: Optional[np.ndarray],
) -> np.ndarray:
    """Apply dark subtraction and white reference normalization.

    Output is normalized 0-1 for pipeline consistency.

    Args:
        intensity: Raw spectrum intensity (any scale)
        dark: Dark/black reference spectrum (same length as intensity)
        white: White reference spectrum (same length as intensity)

    Returns:
        Corrected intensity in 0-1 range, float32
    """
    result = intensity.astype(np.float64)

    if dark is not None:
        result = result - np.asarray(dark, dtype=np.float64)
        result = np.maximum(result, 0)

    if white is not None:
        white_arr = np.asarray(white, dtype=np.float64)
        if dark is not None:
            white_arr = white_arr - np.asarray(dark, dtype=np.float64)
        white_arr = np.maximum(white_arr, 1)
        result = result / white_arr

    return np.clip(result, 0, 1).astype(np.float32)
