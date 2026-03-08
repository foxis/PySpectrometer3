"""Shared utilities for overlay rendering."""

from typing import Optional
import cv2
import numpy as np


def render_polyline_overlay(
    graph: np.ndarray,
    intensity: np.ndarray,
    color: tuple[int, int, int],
    resample_to_width: Optional[int] = None,
    thickness: int = 2,
) -> None:
    """Render intensity as polyline on graph.

    Intensity values represent vertical position (pixels from bottom).
    Larger values draw higher on the graph.

    Args:
        graph: BGR image to draw on
        intensity: 1D array of values (graph coordinates from bottom)
        color: BGR color tuple
        resample_to_width: If set, resample intensity to match graph width
        thickness: Line thickness
    """
    height, width = graph.shape[0], graph.shape[1]

    if resample_to_width is not None and len(intensity) != resample_to_width:
        x_old = np.linspace(0, width - 1, num=len(intensity), dtype=np.float32)
        x_new = np.arange(width, dtype=np.float32)
        intensity = np.interp(
            x_new,
            x_old,
            np.asarray(intensity, dtype=np.float32),
        )

    n = min(len(intensity), width)
    points = []
    for i in range(n):
        val = intensity[i]
        y = height - int(min(val, height - 1))
        points.append((i, max(0, y)))

    if len(points) > 1:
        pts = np.array(points, dtype=np.int32)
        cv2.polylines(
            graph,
            [pts],
            isClosed=False,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )
