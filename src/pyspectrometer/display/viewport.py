"""Viewport for zoom and pan of spectrum graph."""

from dataclasses import dataclass


@dataclass
class Viewport:
    """Viewport state for horizontal and vertical zoom/pan.

    Data space:
    - X: pixel index 0 to n-1 (spectrum width)
    - Y: intensity 0.0 to 1.0

    Default (no zoom): x_start=0, x_end=n-1, y_min=0, y_max=1
    """

    x_start: float = 0.0
    x_end: float = 0.0  # Set from data width on first render
    y_min: float = 0.0
    y_max: float = 1.0

    def reset(self, data_width: int) -> None:
        """Reset to full view."""
        self.x_start = 0.0
        self.x_end = float(max(0, data_width - 1))
        self.y_min = 0.0
        self.y_max = 1.0

    def init_if_needed(self, data_width: int) -> None:
        """Initialize x_end if not yet set."""
        if self.x_end <= self.x_start and data_width > 0:
            self.x_end = float(data_width - 1)

    def data_x_to_screen(self, data_x: float, graph_width: int) -> int:
        """Map data x (index) to screen x pixel."""
        span = self.x_end - self.x_start
        if span <= 0:
            return 0
        ratio = (data_x - self.x_start) / span
        return int(ratio * (graph_width - 1))

    def screen_x_to_data(self, screen_x: int, graph_width: int) -> float:
        """Map screen x pixel to data x (index)."""
        if graph_width <= 1:
            return self.x_start
        ratio = screen_x / (graph_width - 1)
        return self.x_start + ratio * (self.x_end - self.x_start)

    def intensity_to_screen_y(self, intensity: float, graph_height: int, margin: int = 1) -> int:
        """Map intensity (0-1) to screen y pixel (0=top, height=bottom)."""
        span = self.y_max - self.y_min
        if span <= 0:
            return graph_height - 1
        effective = max(0, graph_height - margin)
        normalized = (intensity - self.y_min) / span
        normalized = max(0.0, min(1.0, normalized))
        return graph_height - 1 - int(normalized * effective)

    def screen_y_to_intensity(self, screen_y: int, graph_height: int, margin: int = 1) -> float:
        """Map screen y pixel to intensity (0-1)."""
        effective = max(0, graph_height - margin)
        if effective <= 0:
            return self.y_min
        from_bottom = graph_height - 1 - screen_y
        normalized = from_bottom / effective
        normalized = max(0.0, min(1.0, normalized))
        return self.y_min + normalized * (self.y_max - self.y_min)

    def pan_x(self, delta_data: float) -> None:
        """Pan horizontally by delta in data units."""
        span = self.x_end - self.x_start
        self.x_start += delta_data
        self.x_end += delta_data
        # Clamp to valid range (caller provides data_width for clamping if needed)

    def pan_y(self, delta_intensity: float) -> None:
        """Pan vertically by delta in intensity units. Range can extend beyond 0-1."""
        self.y_min += delta_intensity
        self.y_max += delta_intensity
        if self.y_max <= self.y_min:
            self.y_max = self.y_min + 0.01

    def zoom_x(self, factor: float, center_data_x: float | None = None) -> None:
        """Zoom horizontally. factor>1 zooms in."""
        span = self.x_end - self.x_start
        if span <= 0:
            return
        center = center_data_x if center_data_x is not None else (self.x_start + self.x_end) / 2
        new_span = span / factor
        half = new_span / 2
        x_start = center - half
        x_end = center + half

        # Clamp and push the opposite edge to preserve the full requested span
        if x_start < 0.0:
            x_end -= x_start
            x_start = 0.0
        self.x_start = x_start
        self.x_end = x_end

    def zoom_y(self, factor: float, center_intensity: float | None = None) -> None:
        """Zoom vertically. factor>1 zooms in. Range can extend beyond 0-1."""
        span = self.y_max - self.y_min
        if span <= 0:
            return
        center = (
            center_intensity
            if center_intensity is not None
            else (self.y_min + self.y_max) / 2
        )
        new_span = span / factor
        half = new_span / 2
        y_min = center - half
        y_max = center + half
        if y_max <= y_min:
            y_max = y_min + 0.01
        self.y_min = y_min
        self.y_max = y_max

    def clamp_x(self, data_width: int) -> None:
        """Clamp x range to valid data indices."""
        max_end = float(data_width - 1)
        if self.x_start < 0:
            self.x_end -= self.x_start
            self.x_start = 0
        if self.x_end > max_end:
            self.x_start -= self.x_end - max_end
            self.x_end = max_end
        if self.x_start < 0:
            self.x_start = 0
        if self.x_end - self.x_start < 1:
            self.x_end = self.x_start + 1
