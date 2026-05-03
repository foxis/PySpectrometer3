"""Vertical slider controls for gain, exposure, and LED intensity (PWM)."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class SliderMode(Enum):
    """Slider interaction mode (deprecated: all sliders use knob behavior)."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"  # Knob: click anywhere, drag by distance


@dataclass
class SliderStyle:
    """Visual style for vertical sliders."""

    track_color: tuple[int, int, int] = (60, 60, 60)
    fill_color: tuple[int, int, int] = (80, 150, 80)
    thumb_color: tuple[int, int, int] = (200, 200, 200)
    thumb_hover_color: tuple[int, int, int] = (255, 255, 255)
    border_color: tuple[int, int, int] = (100, 100, 100)
    label_color: tuple[int, int, int] = (200, 200, 200)
    value_color: tuple[int, int, int] = (0, 255, 255)

    track_width: int = 20
    thumb_height: int = 8
    border_width: int = 1
    font_scale: float = 0.35
    font_thickness: int = 1


DEFAULT_SLIDER_STYLE = SliderStyle()



class VerticalSlider:
    """A vertical draggable slider control."""

    def __init__(
        self,
        x: int,
        y: int,
        height: int,
        min_val: float,
        max_val: float,
        value: float,
        label: str = "",
        style: SliderStyle | None = None,
        log_scale: bool = False,
        mode: SliderMode = SliderMode.RELATIVE,
    ):
        """Initialize vertical slider.

        Args:
            x: X position (left edge)
            y: Y position (top edge)
            height: Height of the slider
            min_val: Minimum value
            max_val: Maximum value
            value: Initial value
            label: Label to display above slider
            style: Visual style
            log_scale: Use logarithmic scale
            mode: ABSOLUTE = position maps to value; RELATIVE = first click ref, drag delta
        """
        self.x = x
        self.y = y
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
        self._value = value
        self.label = label
        self.style = style or DEFAULT_SLIDER_STYLE
        self.log_scale = log_scale
        self.mode = mode

        self.visible = False
        self.dragging = False
        self.hovered = False
        self._ref_py: int | None = None
        self._ref_value: float | None = None

        self.on_change: Callable[[float], None] | None = None

    @property
    def value(self) -> float:
        """Get current value."""
        return self._value

    @value.setter
    def value(self, val: float) -> None:
        """Set value (clamped to range)."""
        self._value = max(self.min_val, min(self.max_val, val))

    @property
    def width(self) -> int:
        """Total width including label area."""
        return self.style.track_width + 30

    def _value_to_position(self, val: float) -> int:
        """Convert value to Y position."""
        if self.log_scale and self.min_val > 0:
            import math

            log_min = math.log(self.min_val)
            log_max = math.log(self.max_val)
            log_val = math.log(max(val, self.min_val))
            ratio = (log_val - log_min) / (log_max - log_min) if log_max > log_min else 0
        else:
            ratio = (
                (val - self.min_val) / (self.max_val - self.min_val)
                if self.max_val > self.min_val
                else 0
            )

        # Invert: top = max, bottom = min
        track_height = self.height - self.style.thumb_height
        return int(self.y + track_height * (1 - ratio))

    def _position_to_value(self, py: int) -> float:
        """Convert Y position to value."""
        track_height = self.height - self.style.thumb_height
        ratio = 1 - (py - self.y) / track_height if track_height > 0 else 0
        ratio = max(0, min(1, ratio))

        if self.log_scale and self.min_val > 0:
            import math

            log_min = math.log(self.min_val)
            log_max = math.log(self.max_val)
            log_val = log_min + ratio * (log_max - log_min)
            return math.exp(log_val)

        return self.min_val + ratio * (self.max_val - self.min_val)

    def contains(self, px: int, py: int) -> bool:
        """Check if point is inside slider area."""
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height

    def render(self, image: np.ndarray) -> None:
        """Render slider onto image."""
        if not self.visible:
            return

        style = self.style

        # Track area
        track_x = self.x + (self.width - style.track_width) // 2
        track_y = self.y
        track_h = self.height

        # Draw track background
        cv2.rectangle(
            image,
            (track_x, track_y),
            (track_x + style.track_width, track_y + track_h),
            style.track_color,
            -1,
        )
        cv2.rectangle(
            image,
            (track_x, track_y),
            (track_x + style.track_width, track_y + track_h),
            style.border_color,
            style.border_width,
        )

        # Fill bar (from bottom to current value)
        thumb_y = self._value_to_position(self._value)
        cv2.rectangle(
            image,
            (track_x + 2, thumb_y),
            (track_x + style.track_width - 2, track_y + track_h - 2),
            style.fill_color,
            -1,
        )

        # Thumb
        thumb_color = (
            style.thumb_hover_color if self.hovered or self.dragging else style.thumb_color
        )
        cv2.rectangle(
            image,
            (track_x, thumb_y),
            (track_x + style.track_width, thumb_y + style.thumb_height),
            thumb_color,
            -1,
        )

        # Value text (above slider)
        val_text = f"{self._value:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(
            val_text, font, style.font_scale, style.font_thickness
        )
        text_x = self.x + (self.width - text_w) // 2
        text_y = self.y - 3
        cv2.putText(
            image,
            val_text,
            (text_x, text_y),
            font,
            style.font_scale,
            style.value_color,
            style.font_thickness,
            cv2.LINE_AA,
        )

        # Label (below slider)
        if self.label:
            (label_w, _), _ = cv2.getTextSize(
                self.label, font, style.font_scale, style.font_thickness
            )
            label_x = self.x + (self.width - label_w) // 2
            label_y = self.y + self.height + text_h + 3
            cv2.putText(
                image,
                self.label,
                (label_x, label_y),
                font,
                style.font_scale,
                style.label_color,
                style.font_thickness,
                cv2.LINE_AA,
            )

    def handle_mouse_move(self, px: int, py: int) -> bool:
        """Handle mouse move. Returns True if slider state changed."""
        if not self.visible:
            return False

        was_hovered = self.hovered
        self.hovered = self.contains(px, py)

        if self.dragging and self._ref_py is not None and self._ref_value is not None:
            track_height = self.height - self.style.thumb_height
            if track_height > 0:
                delta_py = self._ref_py - py  # Up = positive
                # Ratio of track moved (0 to 1 for full track)
                ratio = delta_py / track_height

                if self.log_scale and self.min_val > 0:
                    # Log scale: multiply by a factor based on drag distance
                    # Full track drag = multiply by max/min ratio
                    import math
                    log_range = math.log(self.max_val / self.min_val)
                    factor = math.exp(ratio * log_range)
                    new_val = self._ref_value * factor
                else:
                    # Linear scale: add proportional to range
                    value_range = self.max_val - self.min_val
                    delta_val = ratio * value_range
                    new_val = self._ref_value + delta_val

                new_val = max(self.min_val, min(self.max_val, new_val))
                if abs(new_val - self._value) > 1e-9:
                    self._value = new_val
                    if self.on_change:
                        self.on_change(self._value)
                    return True

        return self.hovered != was_hovered

    def handle_mouse_down(self, px: int, py: int) -> bool:
        """Handle mouse down. Returns True if handled. Click anywhere on track to start drag."""
        if not self.visible:
            return False

        if self.contains(px, py):
            self.dragging = True
            self._ref_py = py
            self._ref_value = self._value
            return True

        return False

    def handle_mouse_up(self) -> bool:
        """Handle mouse up. Returns True if was dragging."""
        if self.dragging:
            self.dragging = False
            self._ref_py = None
            self._ref_value = None
            return True
        return False


class SliderPanel:
    """Panel containing gain, exposure, and LED intensity sliders."""

    def __init__(
        self,
        x: int,
        y: int,
        height: int,
        style: SliderStyle | None = None,
    ):
        """Initialize slider panel (gain, exposure, LED intensity).

        Args:
            x: X position (right side of display)
            y: Y position
            height: Height available for sliders
            style: Visual style
        """
        self.x = x
        self.y = y
        self.height = height
        self.style = style or DEFAULT_SLIDER_STYLE

        slider_height = height - 30  # Leave room for labels

        # Gain range: 1x to 16x (typical camera max)
        self.gain_slider = VerticalSlider(
            x=x,
            y=y + 15,  # Leave room for value text
            height=slider_height,
            min_val=1.0,
            max_val=16.0,
            value=1.0,
            label="G",
            style=self.style,
            log_scale=True,
            mode=SliderMode.RELATIVE,
        )

        # Exposure range: 100us to 30 seconds (30,000,000 us)
        # Continuous mode limited to 1s by auto-exposure algorithm
        self.exposure_slider = VerticalSlider(
            x=x + self.gain_slider.width + 5,
            y=y + 15,
            height=slider_height,
            min_val=100,
            max_val=30_000_000,  # 30 seconds max for frozen/long exposure
            value=10000,
            label="E",
            style=self.style,
            log_scale=True,
            mode=SliderMode.RELATIVE,
        )

        # LED intensity: 0–100% software PWM duty cycle (Measurement, Color Science)
        self.led_intensity_slider = VerticalSlider(
            x=x + self.gain_slider.width + 5 + self.exposure_slider.width + 5,
            y=y + 15,
            height=slider_height,
            min_val=0.0,
            max_val=100.0,
            value=100.0,
            label="LED",
            style=self.style,
            log_scale=False,
            mode=SliderMode.RELATIVE,
        )
        self.led_intensity_slider.visible = False  # Shown only in Measurement/Color Science

        self._sliders = [self.gain_slider, self.exposure_slider, self.led_intensity_slider]

    @property
    def width(self) -> int:
        """Total width of panel."""
        return (
            self.gain_slider.width
            + self.exposure_slider.width
            + self.led_intensity_slider.width
            + 10
        )

    def render(self, image: np.ndarray) -> None:
        """Render all visible sliders."""
        for slider in self._sliders:
            slider.render(image)

    def handle_mouse_move(self, px: int, py: int) -> bool:
        """Handle mouse move. Returns True if any slider changed."""
        changed = False
        for slider in self._sliders:
            if slider.handle_mouse_move(px, py):
                changed = True
        return changed

    def handle_mouse_down(self, px: int, py: int) -> bool:
        """Handle mouse down. Returns True if handled."""
        for slider in self._sliders:
            if slider.handle_mouse_down(px, py):
                return True
        return False

    def handle_mouse_up(self) -> bool:
        """Handle mouse up. Returns True if any slider was dragging."""
        handled = False
        for slider in self._sliders:
            if slider.handle_mouse_up():
                handled = True
        return handled

    def any_visible(self) -> bool:
        """Check if any slider is visible."""
        return any(s.visible for s in self._sliders)

    def any_dragging(self) -> bool:
        """Check if any slider is being dragged."""
        return any(s.dragging for s in self._sliders)


class HorizontalSlider:
    """A horizontal draggable slider (e.g. for zoom)."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        min_val: float,
        max_val: float,
        value: float,
        label: str = "",
        style: SliderStyle | None = None,
        mode: SliderMode = SliderMode.RELATIVE,
    ):
        """Initialize horizontal slider.

        Args:
            x: X position (left edge)
            y: Y position (top edge)
            width: Width of the slider
            min_val: Minimum value
            max_val: Maximum value
            value: Initial value
            label: Label to display
            style: Visual style
            mode: ABSOLUTE = position maps to value; RELATIVE = first click ref, drag delta
        """
        self.x = x
        self.y = y
        self.width = width
        self.min_val = min_val
        self.max_val = max_val
        self._value = value
        self.label = label
        self.style = style or DEFAULT_SLIDER_STYLE
        self.mode = mode

        self.visible = False
        self.dragging = False
        self.hovered = False
        self._ref_px: int | None = None
        self._ref_value: float | None = None

        self.on_change: Callable[[float], None] | None = None

    @property
    def value(self) -> float:
        """Get current value."""
        return self._value

    @value.setter
    def value(self, val: float) -> None:
        """Set value (clamped to range)."""
        self._value = max(self.min_val, min(self.max_val, val))

    @property
    def height(self) -> int:
        """Total height including track."""
        return self.style.track_width + 8

    def _value_to_position(self, val: float) -> int:
        """Convert value to X position."""
        ratio = (
            (val - self.min_val) / (self.max_val - self.min_val)
            if self.max_val > self.min_val
            else 0
        )
        ratio = max(0, min(1, ratio))
        track_width = self.width - self.style.thumb_height
        return int(self.x + track_width * ratio)

    def _position_to_value(self, px: int) -> float:
        """Convert X position to value."""
        track_width = self.width - self.style.thumb_height
        if track_width <= 0:
            return self.min_val
        ratio = (px - self.x) / track_width
        ratio = max(0, min(1, ratio))
        return self.min_val + ratio * (self.max_val - self.min_val)

    def contains(self, px: int, py: int) -> bool:
        """Check if point is inside slider area."""
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height

    def render(self, image: np.ndarray) -> None:
        """Render slider onto image."""
        if not self.visible:
            return

        style = self.style
        track_x = self.x
        track_y = self.y + (self.height - style.track_width) // 2
        track_w = self.width

        cv2.rectangle(
            image,
            (track_x, track_y),
            (track_x + track_w, track_y + style.track_width),
            style.track_color,
            -1,
        )
        cv2.rectangle(
            image,
            (track_x, track_y),
            (track_x + track_w, track_y + style.track_width),
            style.border_color,
            style.border_width,
        )

        thumb_x = self._value_to_position(self._value)
        thumb_color = (
            style.thumb_hover_color if self.hovered or self.dragging else style.thumb_color
        )
        cv2.rectangle(
            image,
            (thumb_x, track_y),
            (thumb_x + style.thumb_height, track_y + style.track_width),
            thumb_color,
            -1,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        val_text = f"{self._value:.1f}"
        (text_w, text_h), _ = cv2.getTextSize(
            val_text, font, style.font_scale, style.font_thickness
        )
        text_x = thumb_x + (style.thumb_height - text_w) // 2
        text_y = track_y - 2
        cv2.putText(
            image,
            val_text,
            (text_x, text_y),
            font,
            style.font_scale,
            style.value_color,
            style.font_thickness,
            cv2.LINE_AA,
        )

        if self.label:
            cv2.putText(
                image,
                self.label,
                (self.x, self.y - 2),
                font,
                style.font_scale,
                style.label_color,
                style.font_thickness,
                cv2.LINE_AA,
            )

    def handle_mouse_move(self, px: int, py: int) -> bool:
        """Handle mouse move. Returns True if slider state changed."""
        if not self.visible:
            return False

        was_hovered = self.hovered
        self.hovered = self.contains(px, py)

        if self.dragging and self._ref_px is not None:
            track_width = self.width - self.style.thumb_height
            if track_width > 0:
                delta_px = px - self._ref_px  # Right = positive
                value_range = self.max_val - self.min_val
                # 10px drag on 100px track = 10% of range (e.g. 5 for 0-50, 0.1 for 0-1)
                delta_val = (delta_px / track_width) * value_range
                new_val = max(
                    self.min_val,
                    min(self.max_val, (self._ref_value or self._value) + delta_val),
                )
                if new_val != self._value:
                    self._value = new_val
                    self._ref_px = px
                    self._ref_value = new_val
                    if self.on_change:
                        self.on_change(self._value)
                    return True

        return self.hovered != was_hovered

    def handle_mouse_down(self, px: int, py: int) -> bool:
        """Handle mouse down. Returns True if handled. Click anywhere on track to start drag."""
        if not self.visible:
            return False

        if self.contains(px, py):
            self.dragging = True
            self._ref_px = px
            self._ref_value = self._value
            return True

        return False

    def handle_mouse_up(self) -> bool:
        """Handle mouse up. Returns True if was dragging."""
        if self.dragging:
            self.dragging = False
            self._ref_px = None
            self._ref_value = None
            return True
        return False
