"""Display-related utilities (scaling, etc). Pure functions, no UI dependencies."""

from typing import Optional
import numpy as np


def scale_to_uint8(arr: np.ndarray, max_val: Optional[float] = None) -> np.ndarray:
    """Scale array to 8-bit (0-255) for display.

    Handles uint8 (passthrough copy), uint16 (scale by max_val).
    Caller provides max_val; use max(arr.max(), 1) for dynamic range,
    or (1 << bit_depth) - 1 for bit-depth-based scaling.

    Args:
        arr: Input array (uint8 or uint16).
        max_val: Scale denominator. If None, uses max(arr.max(), 1).

    Returns:
        uint8 array in 0-255 range.
    """
    if arr.dtype == np.uint8:
        return arr.copy()
    if max_val is None or max_val <= 0:
        max_val = max(float(np.max(arr)), 1.0)
    return (arr.astype(np.float32) * 255.0 / max_val).astype(np.uint8)
