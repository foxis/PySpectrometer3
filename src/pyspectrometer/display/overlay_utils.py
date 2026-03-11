"""Shared utilities for overlay rendering."""

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .viewport import Viewport


def render_polyline_overlay(
    graph: np.ndarray,
    intensity: np.ndarray,
    color: tuple[int, int, int],
    resample_to_width: int | None = None,
    thickness: int = 2,
    viewport: "Viewport | None" = None,
) -> None:
    """Render intensity as polyline on graph.

    Intensity values represent vertical position (pixels from bottom).
    Larger values draw higher on the graph.

    Args:
        graph: BGR image to draw on
        intensity: 1D array of values (graph coordinates from bottom, or 0-1 if viewport)
        color: BGR color tuple
        resample_to_width: If set, resample intensity to match graph width
        thickness: Line thickness
        viewport: Optional viewport for zoom/pan. If set, intensity is 0-1 and mapped via viewport.
    """
    height, width = graph.shape[0], graph.shape[1]

    if viewport is not None:
        _render_with_viewport(graph, intensity, color, viewport, resample_to_width, thickness)
        return

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


def _render_with_viewport(
    graph: np.ndarray,
    intensity: np.ndarray,
    color: tuple[int, int, int],
    viewport: "Viewport",
    resample_to_width: int | None,
    thickness: int,
) -> None:
    """Render polyline with viewport zoom/pan."""
    height, width = graph.shape[0], graph.shape[1]
    n_data = len(intensity)

    if resample_to_width is not None and n_data != resample_to_width:
        x_old = np.linspace(0, n_data - 1, num=n_data, dtype=np.float32)
        x_new = np.linspace(0, n_data - 1, num=resample_to_width, dtype=np.float32)
        intensity = np.interp(x_new, x_old, np.asarray(intensity, dtype=np.float32))
        n_data = len(intensity)

    points = []
    for i in range(n_data):
        data_x = float(i)
        if data_x < viewport.x_start or data_x > viewport.x_end:
            continue
        val = float(intensity[i])
        sx = viewport.data_x_to_screen(data_x, width)
        sy = viewport.intensity_to_screen_y(val, height)
        points.append((sx, sy))

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
