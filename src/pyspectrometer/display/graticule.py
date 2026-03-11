"""Graticule rendering for spectrum display."""

from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..core.calibration import GraticuleData

if TYPE_CHECKING:
    from .viewport import Viewport

# Tick colors: light gray for minor, gray for tens, black for labeled lines
TICK_LIGHT_GRAY = (200, 200, 200)
TICK_GRAY = (150, 150, 150)
LABEL_LINE_BLACK = (0, 0, 0)

# Overlap weights: peak lines > spectrum > other graticule lines
WEIGHT_PEAK_LINE = 10
WEIGHT_SPECTRUM = 5
WEIGHT_GRATICULE_LINE = 1


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
            cv2.line(graph, (screen_pos, 15), (screen_pos, height), LABEL_LINE_BLACK, 1)

    def render_labels_on_graph(
        self,
        graph: np.ndarray,
        graticule: GraticuleData,
        viewport: "Viewport | None" = None,
        *,
        peak_screen_x: list[int] | None = None,
        spectrum_screen_y: np.ndarray | None = None,
    ) -> None:
        """Render graticule labels with overlap-aware placement.

        Call after spectrum and peaks are drawn so labels stay visible.
        Chooses left/right to minimize overlap with peak lines, spectrum, other labels.

        Args:
            graph: Graph image to draw on (modified in place)
            graticule: Graticule data with line positions
            viewport: Optional viewport for zoom
            peak_screen_x: Screen x of peak vertical lines (priority 1)
            spectrum_screen_y: Spectrum y per screen x, length = width (priority 2)
        """
        height = graph.shape[0]
        width = graph.shape[1]
        unit = getattr(graticule, "unit", "nm")

        def to_screen(data_pos: int) -> int:
            if viewport is not None:
                return viewport.data_x_to_screen(float(data_pos), width)
            return data_pos

        def in_range(data_pos: int) -> bool:
            if viewport is None:
                return True
            return viewport.x_start <= data_pos <= viewport.x_end

        all_screen_pos: set[int] = set()
        for p in graticule.tens:
            if in_range(p):
                all_screen_pos.add(to_screen(p))
        for p, _ in graticule.fifties:
            if in_range(p):
                all_screen_pos.add(to_screen(p))

        peak_x = peak_screen_x if peak_screen_x is not None else []
        spec_y = spectrum_screen_y if spectrum_screen_y is not None else None

        label_y = 12
        placed_spans: list[tuple[int, int, int, int]] = []

        for pos, label in graticule.fifties:
            if not in_range(pos):
                continue
            screen_pos = to_screen(pos)
            text = f"{str(label)}{unit}"
            (text_w, text_h), _ = cv2.getTextSize(text, self.font, self.font_scale, 1)
            label_h = text_h + 4
            label_y1 = 0
            label_y2 = label_y + label_h

            left_x = max(0, screen_pos - self.text_offset - text_w)
            right_x = min(width - text_w, screen_pos + self.text_offset)

            def overlap_score(x1: int, x2: int) -> float:
                """Lower is better. Priority: peak lines > spectrum > graticule lines."""
                score = 0.0
                for px in peak_x:
                    if x1 <= px <= x2:
                        score += WEIGHT_PEAK_LINE
                if spec_y is not None and len(spec_y) > x1:
                    for sx in range(max(0, x1), min(len(spec_y), x2 + 1)):
                        sy = int(spec_y[sx])
                        if 0 <= sy < height and label_y1 <= sy <= label_y2:
                            score += WEIGHT_SPECTRUM
                            break
                for gx in all_screen_pos:
                    if gx != screen_pos and x1 <= gx <= x2:
                        score += WEIGHT_GRATICULE_LINE
                for px1, px2, py1, py2 in placed_spans:
                    if not (x2 <= px1 or x1 >= px2) and not (label_y2 <= py1 or label_y1 >= py2):
                        score += WEIGHT_GRATICULE_LINE * 2
                return score

            left_score = overlap_score(left_x, left_x + text_w)
            right_score = overlap_score(right_x, right_x + text_w)

            x = left_x if left_score <= right_score else right_x
            x = max(0, min(width - text_w, x))

            cv2.putText(
                graph,
                text,
                (x, label_y),
                self.font,
                self.font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            placed_spans.append((x, x + text_w, label_y1, label_y2))

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
    ) -> None:
        """Render graticule markers on waterfall display.

        Args:
            waterfall: Waterfall image to draw on (modified in place)
            graticule: Graticule data with line positions
            y_offset: Y offset for the waterfall region
        """
        height = waterfall.shape[0]

        for pos, label in graticule.fifties:
            for i in range(y_offset, height - 5, 20):
                cv2.line(
                    waterfall,
                    (pos, i),
                    (pos, i + 1),
                    (0, 0, 0),
                    2,
                )
                cv2.line(
                    waterfall,
                    (pos, i),
                    (pos, i + 1),
                    (255, 255, 255),
                    1,
                )

            unit = getattr(graticule, "unit", "nm")
            cv2.putText(
                waterfall,
                f"{label}{unit}",
                (pos - self.text_offset, height - 5),
                self.font,
                self.font_scale,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                waterfall,
                f"{label}{unit}",
                (pos - self.text_offset, height - 5),
                self.font,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
