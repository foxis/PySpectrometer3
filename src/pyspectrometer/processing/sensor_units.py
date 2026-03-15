"""Sensor units and conversion helpers (debug / pre-pipeline).

Convert normalized intensity (0–1) to raw counts and to a flux proxy for checking
linearity of sensor response vs exposure×gain. Not part of the main pipeline yet.
"""

import numpy as np


def peak_to_counts(peak: float, bit_depth: int = 10) -> float:
    """Convert normalized peak (0–1) to raw ADC counts for the given bit depth.

    For 10-bit sensor, full scale is 1023; saturated peak 1.0 → 1023 counts.
    """
    max_val = (1 << bit_depth) - 1
    return float(np.clip(peak, 0.0, 1.0) * max_val)


def counts_to_flux_proxy(
    counts: float, exposure_us: int | None, gain: float | None
) -> float | None:
    """Counts per (exposure_sec × gain): proxy for scene intensity (stable across E×G for same light).

    Only valid when the frame was captured with the same exposure and gain used here.
    With pipeline lag (e.g. Picamera2), the frame may have been shot with previous E×G
    while we pass current E×G, so flux_proxy will be wrong (typically understated after
    increasing E). Use in steady state or when the camera reports per-frame E×G.

    Returns None if exposure or gain is missing/zero. Unit: counts · s⁻¹ · gain⁻¹.
    """
    if not exposure_us or not gain or gain <= 0:
        return None
    exposure_s = exposure_us / 1e6
    return counts / (exposure_s * gain)
