"""Graticule rendering for spectrum display."""

from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..core.calibration import GraticuleData

if TYPE_CHECKING:
    from .viewport import Viewport

# Tick colors: light gray for minor, gray for major; both thin
TICK_LIGHT_GRAY = (200, 200, 200)
TICK_GRAY = (150, 150, 150)


class GraticuleRenderer:
    """Renders calibrated graticule lines on spectrum graphs."""

    def __init__(
        self,
        font: int | None = None,
        font_scale: float = 1.0,
        text_offset: int = 12,
    ):
        """Initialize graticule renderer.

        Args:
            font: OpenCV font type
            font_scale: Font scale factor
            text_offset: Horizontal text offset for labels
        """
        self.font = font if font is not None else cv2.FONT_HERSHEY_PLAIN
        self.font_scale = font_scale
        self.text_offset = text_offset

    def render_lines_on_graph(
        self,
        graph: np.ndarray,
        graticule: GraticuleData,
        viewport: "Viewport | None" = None,
    ) -> None:
        """Render graticule lines only (no labels)."""
        height = graph.shape[0]
        width = graph.shape[1]

        def to_screen(data_pos: int) -> int:
            if viewport is not None:
                return viewport.data_x_to_screen(float(data_pos), width)
            return data_pos

        def in_range(data_pos: int) -> bool:
            if viewport is None:
                return True
            return viewport.x_start <= data_pos <= viewport.x_end

        for position in graticule.tens:
            if not in_range(position):
                continue
            screen_pos = to_screen(position)
            cv2.line(graph, (screen_pos, 15), (screen_pos, height), TICK_LIGHT_GRAY, 1)

        for pos, _ in graticule.fifties:
            if not in_range(pos):
                continue
            screen_pos = to_screen(pos)
            cv2.line(graph, (screen_pos, 15), (screen_pos, height), TICK_GRAY, 1)

    def render_labels_on_graph(
        self,
        graph: np.ndarray,
        graticule: GraticuleData,
        viewport: "Viewport | None" = None,
        *,
        peak_screen_x: list[int] | None = None,
        spectrum_screen_y: np.ndarray | None = None,
    ) -> None:
        """Render graticule labels at fixed positions relative to their tick lines.

        Labels are always placed at the bottom of the graph, to the right of the
        tick line. Position is deterministic (no live-data decisions) so labels
        never jump between frames.

        peak_screen_x and spectrum_screen_y are accepted for API compatibility
        but are intentionally unused here.
        """
        height = graph.shape[0]
        width = graph.shape[1]
        baseline = 12

        def to_screen(data_pos: int) -> int:
            if viewport is not None:
                return viewport.data_x_to_screen(float(data_pos), width)
            return data_pos

        def in_range(data_pos: int) -> bool:
            if viewport is None:
                return True
            return viewport.x_start <= data_pos <= viewport.x_end

        for pos, label in graticule.fifties:
            if not in_range(pos):
                continue
            screen_pos = to_screen(pos)
            text = str(label)
            (text_w, _), _ = cv2.getTextSize(text, self.font, self.font_scale, 1)

            # Place to the right of the tick; clamp so it never leaves the canvas
            x = screen_pos + self.text_offset
            if x + text_w > width:
                x = screen_pos - self.text_offset - text_w
            x = max(0, x)

            cv2.putText(
                graph,
                text,
                (x, baseline),
                self.font,
                self.font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    def render_on_graph(
        self,
        graph: np.ndarray,
        graticule: GraticuleData,
        viewport: "Viewport | None" = None,
    ) -> None:
        """Render graticule lines and labels (legacy: labels use line crossings only)."""
        self.render_lines_on_graph(graph, graticule, viewport)
        self.render_labels_on_graph(graph, graticule, viewport)

    def render_horizontal_lines(
        self,
        graph: np.ndarray,
        spacing: int = 64,
    ) -> None:
        """Render horizontal grid lines on a graph.

        Args:
            graph: Graph image to draw on (modified in place)
            spacing: Spacing between horizontal lines
        """
        height = graph.shape[0]
        width = graph.shape[1]

        for i in range(spacing, height, spacing):
            cv2.line(
                graph,
                (0, i),
                (width, i),
                TICK_GRAY,
                1,
            )

    def render_on_waterfall(
        self,
        waterfall: np.ndarray,
        graticule: GraticuleData,
        y_offset: int = 162,
        viewport: "Viewport | None" = None,
    ) -> None:
        """Render graticule markers on waterfall display.

        Args:
            waterfall: Waterfall image to draw on (modified in place)
            graticule: Graticule data with line positions
            y_offset: Y offset for the waterfall region
            viewport: Optional viewport for horizontal zoom/pan
        """
        height = waterfall.shape[0]
        width = waterfall.shape[1]

        for pos, label in graticule.fifties:
            if viewport is not None:
                if not (viewport.x_start <= pos <= viewport.x_end):
                    continue
                screen_x = viewport.data_x_to_screen(float(pos), width)
            else:
                screen_x = pos

            for i in range(y_offset, height - 5, 20):
                cv2.line(waterfall, (screen_x, i), (screen_x, i + 1), (0, 0, 0), 2)
                cv2.line(waterfall, (screen_x, i), (screen_x, i + 1), (255, 255, 255), 1)

            label_x = max(0, screen_x - self.text_offset)
            cv2.putText(
                waterfall,
                str(label),
                (label_x, height - 5),
                self.font,
                self.font_scale,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                waterfall,
                str(label),
                (label_x, height - 5),
                self.font,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
