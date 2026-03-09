"""Graph scaling utilities. Pure functions, no display dependencies."""

from typing import Union
import numpy as np


def scale_intensity_to_graph(
    intensity: Union[np.ndarray, float],
    graph_height: int,
    margin: int = 10,
) -> np.ndarray:
    """Scale intensity (0-1) to graph Y coordinate range (pixels from bottom).

    Args:
        intensity: Intensity value(s) in 0-1 range
        graph_height: Graph height in pixels
        margin: Pixels to leave at top (default 10). Use 1 for full height.

    Returns:
        Scaled value(s) in [0, graph_height - margin], float32. Same shape as intensity.
    """
    effective_height = max(0, graph_height - margin)
    scale = float(effective_height) if effective_height > 0 else 1.0
    result = np.asarray(intensity, dtype=np.float32) * scale
    return np.clip(result, 0, effective_height).astype(np.float32)
