"""Clickable button system for OpenCV-based GUI."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional
import cv2
import numpy as np


class ButtonState(Enum):
    """Visual state of a button."""
    
    NORMAL = auto()
    HOVER = auto()
    ACTIVE = auto()
    DISABLED = auto()


@dataclass
class ButtonStyle:
    """Visual style configuration for buttons."""
    
    bg_normal: tuple[int, int, int] = (60, 60, 60)
    bg_hover: tuple[int, int, int] = (80, 80, 80)
    bg_active: tuple[int, int, int] = (40, 120, 40)
    bg_disabled: tuple[int, int, int] = (40, 40, 40)
    
    fg_normal: tuple[int, int, int] = (200, 200, 200)
    fg_hover: tuple[int, int, int] = (255, 255, 255)
    fg_active: tuple[int, int, int] = (255, 255, 255)
    fg_disabled: tuple[int, int, int] = (100, 100, 100)
    
    border_normal: tuple[int, int, int] = (100, 100, 100)
    border_hover: tuple[int, int, int] = (150, 150, 150)
    border_active: tuple[int, int, int] = (100, 200, 100)
    border_disabled: tuple[int, int, int] = (60, 60, 60)
    
    font_scale: float = 0.4
    font_thickness: int = 1
    padding_x: int = 6
    padding_y: int = 4
    border_width: int = 1


DEFAULT_STYLE = ButtonStyle()


@dataclass
class Button:
    """A clickable button with label and callback."""
    
    label: str
    action_name: str
    callback: Optional[Callable[[], None]] = None
    
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    
    state: ButtonState = ButtonState.NORMAL
    is_toggle: bool = False
    is_active: bool = False
    style: ButtonStyle = field(default_factory=lambda: DEFAULT_STYLE)
    
    # Keyboard shortcut (used by keyboard handler, not displayed in button)
    shortcut: str = ""
    # Icon type: "playback" = red circle when active, gray square when stopped
    icon_type: str = ""
    
    def contains(self, px: int, py: int) -> bool:
        """Check if point is inside button bounds."""
        return (
            self.x <= px < self.x + self.width and
            self.y <= py < self.y + self.height
        )
    
    def render(self, image: np.ndarray) -> None:
        """Render button onto image."""
        style = self.style
        
        effective_state = self.state
        if self.is_toggle and self.is_active and self.state != ButtonState.DISABLED:
            effective_state = ButtonState.ACTIVE
        
        bg_colors = {
            ButtonState.NORMAL: style.bg_normal,
            ButtonState.HOVER: style.bg_hover,
            ButtonState.ACTIVE: style.bg_active,
            ButtonState.DISABLED: style.bg_disabled,
        }
        fg_colors = {
            ButtonState.NORMAL: style.fg_normal,
            ButtonState.HOVER: style.fg_hover,
            ButtonState.ACTIVE: style.fg_active,
            ButtonState.DISABLED: style.fg_disabled,
        }
        border_colors = {
            ButtonState.NORMAL: style.border_normal,
            ButtonState.HOVER: style.border_hover,
            ButtonState.ACTIVE: style.border_active,
            ButtonState.DISABLED: style.border_disabled,
        }
        
        bg = bg_colors[effective_state]
        fg = fg_colors[effective_state]
        border = border_colors[effective_state]
        
        x1, y1 = self.x, self.y
        x2, y2 = self.x + self.width - 1, self.y + self.height - 1
        
        cv2.rectangle(image, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), border, style.border_width)
        
        if self.icon_type == "playback":
            # Red circle when active (recording), gray square when stopped
            cx = x1 + self.width // 2
            cy = y1 + self.height // 2
            r = min(self.width, self.height) // 4
            if self.is_toggle and self.is_active:
                cv2.circle(image, (cx, cy), max(2, r), (0, 0, 255), -1)
            else:
                sz = max(2, r - 1)
                cv2.rectangle(
                    image,
                    (cx - sz, cy - sz),
                    (cx + sz, cy + sz),
                    (120, 120, 120),
                    -1,
                )
        else:
            display_text = self.label
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, text_h), baseline = cv2.getTextSize(
                display_text, font, style.font_scale, style.font_thickness
            )
            text_x = x1 + (self.width - text_w) // 2
            text_y = y1 + (self.height + text_h) // 2
            cv2.putText(
                image,
                display_text,
                (text_x, text_y),
                font,
                style.font_scale,
                fg,
                style.font_thickness,
                cv2.LINE_AA,
            )
    
    def on_click(self) -> bool:
        """Handle click event. Returns True if handled."""
        if self.state == ButtonState.DISABLED:
            return False
        
        if self.is_toggle:
            self.is_active = not self.is_active
        
        print(f"[BUTTON] '{self.label}' clicked -> action: {self.action_name}")
        
        if self.callback is not None:
            self.callback()
            return True
        
        return True
    
    def set_active(self, active: bool) -> None:
        """Set active state for toggle buttons."""
        self.is_active = active


class ButtonBar:
    """A horizontal bar of buttons."""
    
    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        height: int = 24,
        spacing: int = 4,
        style: Optional[ButtonStyle] = None,
    ):
        """Initialize button bar.
        
        Args:
            x: Starting X position
            y: Y position
            height: Height of buttons
            spacing: Spacing between buttons
            style: Button style (uses default if None)
        """
        self.x = x
        self.y = y
        self.height = height
        self.spacing = spacing
        self.style = style or DEFAULT_STYLE
        
        self._buttons: list[Button] = []
        self._next_x = x
    
    def add_button(
        self,
        label: str,
        action_name: str,
        callback: Optional[Callable[[], None]] = None,
        width: Optional[int] = None,
        is_toggle: bool = False,
        shortcut: str = "",
        icon_type: str = "",
    ) -> Button:
        """Add a button to the bar.
        
        Args:
            label: Button text
            action_name: Name for logging/identification
            callback: Function to call on click
            width: Button width (auto-calculated if None)
            is_toggle: Whether button toggles on/off
            shortcut: Keyboard shortcut hint
            
        Returns:
            The created button
        """
        if width is None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_w, _), _ = cv2.getTextSize(
                label, font, self.style.font_scale, self.style.font_thickness
            )
            width = text_w + self.style.padding_x * 2
        
        button = Button(
            label=label,
            action_name=action_name,
            callback=callback,
            x=self._next_x,
            y=self.y,
            width=width,
            height=self.height,
            is_toggle=is_toggle,
            style=self.style,
            shortcut=shortcut,
            icon_type=icon_type,
        )
        
        self._buttons.append(button)
        self._next_x += width + self.spacing
        
        return button
    
    def add_separator(self, width: int = 10) -> None:
        """Add spacing between button groups."""
        self._next_x += width
    
    def render(self, image: np.ndarray) -> None:
        """Render all buttons onto image."""
        for button in self._buttons:
            button.render(image)
    
    def handle_mouse_move(self, px: int, py: int) -> Optional[Button]:
        """Handle mouse move, updating hover states.
        
        Returns:
            Button under cursor, or None
        """
        hovered = None
        for button in self._buttons:
            if button.contains(px, py):
                button.state = ButtonState.HOVER
                hovered = button
            else:
                if button.state == ButtonState.HOVER:
                    button.state = ButtonState.NORMAL
        return hovered
    
    def handle_click(self, px: int, py: int) -> Optional[Button]:
        """Handle mouse click.
        
        Returns:
            Clicked button, or None
        """
        for button in self._buttons:
            if button.contains(px, py):
                button.on_click()
                return button
        return None
    
    def get_button(self, action_name: str) -> Optional[Button]:
        """Get button by action name."""
        for button in self._buttons:
            if button.action_name == action_name:
                return button
        return None
    
    @property
    def buttons(self) -> list[Button]:
        """Get all buttons."""
        return self._buttons.copy()
    
    @property
    def total_width(self) -> int:
        """Get total width of all buttons."""
        if not self._buttons:
            return 0
        last = self._buttons[-1]
        return last.x + last.width - self.x
