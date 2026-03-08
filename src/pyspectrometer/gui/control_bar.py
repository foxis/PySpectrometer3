"""Control bar with two rows of buttons for the spectrum display."""

from dataclasses import dataclass
from typing import Callable, Optional
import cv2
import numpy as np

from .buttons import Button, ButtonBar, ButtonStyle


@dataclass
class ControlBarConfig:
    """Configuration for the control bar."""
    
    height: int = 80
    row_height: int = 24
    row_spacing: int = 4
    button_spacing: int = 4
    margin_x: int = 5
    margin_y: int = 8
    
    bg_color: tuple[int, int, int] = (20, 20, 20)
    
    font_scale: float = 0.35
    font_thickness: int = 1


class ControlBar:
    """Two-row control bar with clickable buttons.
    
    Replaces the static message banner with interactive controls.
    """
    
    def __init__(
        self,
        width: int = 800,
        config: Optional[ControlBarConfig] = None,
    ):
        """Initialize control bar.
        
        Args:
            width: Width of the control bar
            config: Configuration (uses defaults if None)
        """
        self.width = width
        self.config = config or ControlBarConfig()
        
        self._row1: Optional[ButtonBar] = None
        self._row2: Optional[ButtonBar] = None
        self._status_values: dict[str, str] = {}
        
        self._setup_buttons()
    
    def _setup_buttons(self) -> None:
        """Create the button rows."""
        cfg = self.config
        
        style = ButtonStyle(
            font_scale=cfg.font_scale,
            font_thickness=cfg.font_thickness,
            padding_x=5,
            padding_y=3,
        )
        
        row1_y = cfg.margin_y
        row2_y = cfg.margin_y + cfg.row_height + cfg.row_spacing
        
        self._row1 = ButtonBar(
            x=cfg.margin_x,
            y=row1_y,
            height=cfg.row_height,
            spacing=cfg.button_spacing,
            style=style,
        )
        
        self._row2 = ButtonBar(
            x=cfg.margin_x,
            y=row2_y,
            height=cfg.row_height,
            spacing=cfg.button_spacing,
            style=style,
        )
        
        self._row1.add_button("Capture", "capture_current", shortcut="")
        self._row1.add_button("Peak", "capture_peak", shortcut="h", is_toggle=True)
        self._row1.add_button("Save", "save", shortcut="s")
        self._row1.add_button("Load", "load_reference", shortcut="")
        self._row1.add_separator(8)
        self._row1.add_button("Ref", "use_as_reference", shortcut="")
        self._row1.add_button("Dark", "capture_dark", shortcut="", is_toggle=True)
        self._row1.add_separator(8)
        self._row1.add_button("Gain+", "gain_up", shortcut="t")
        self._row1.add_button("Gain-", "gain_down", shortcut="g")
        self._row1.add_button("AutoG", "auto_gain", shortcut="", is_toggle=True)
        
        self._row2.add_button("Light", "light_toggle", shortcut="", is_toggle=True)
        self._row2.add_button("XYZ", "fit_xyz", shortcut="")
        self._row2.add_separator(8)
        self._row2.add_button("PixMode", "toggle_pixel_mode", shortcut="p", is_toggle=True)
        self._row2.add_button("Measure", "toggle_measure", shortcut="m", is_toggle=True)
        self._row2.add_button("Cal", "calibrate", shortcut="c")
        self._row2.add_button("ClearPts", "clear_clicks", shortcut="x")
        self._row2.add_separator(8)
        self._row2.add_button("Ext", "cycle_extraction", shortcut="e")
        self._row2.add_button("AutoAng", "auto_detect_angle", shortcut="a")
        self._row2.add_button("Quit", "quit", shortcut="q")
    
    def register_callback(self, action_name: str, callback: Callable[[], None]) -> bool:
        """Register a callback for a button action.
        
        Args:
            action_name: Name of the action (matches button action_name)
            callback: Function to call when button is clicked
            
        Returns:
            True if button was found and callback registered
        """
        for bar in [self._row1, self._row2]:
            if bar is None:
                continue
            button = bar.get_button(action_name)
            if button is not None:
                button.callback = callback
                return True
        return False
    
    def set_button_active(self, action_name: str, active: bool) -> bool:
        """Set the active state of a toggle button.
        
        Args:
            action_name: Name of the action
            active: Whether button should appear active
            
        Returns:
            True if button was found
        """
        for bar in [self._row1, self._row2]:
            if bar is None:
                continue
            button = bar.get_button(action_name)
            if button is not None:
                button.set_active(active)
                return True
        return False
    
    def set_status(self, key: str, value: str) -> None:
        """Set a status value to display.
        
        Args:
            key: Status key (e.g., "gain", "cal")
            value: Value to display
        """
        self._status_values[key] = value
    
    def render(self) -> np.ndarray:
        """Render the control bar.
        
        Returns:
            Image of the control bar
        """
        cfg = self.config
        
        image = np.zeros([cfg.height, self.width, 3], dtype=np.uint8)
        image[:] = cfg.bg_color
        
        if self._row1 is not None:
            self._row1.render(image)
        if self._row2 is not None:
            self._row2.render(image)
        
        self._render_status(image)
        
        return image
    
    def _render_status(self, image: np.ndarray) -> None:
        """Render status values on the right side."""
        cfg = self.config
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        status_x = self.width - 200
        y = cfg.margin_y + 12
        line_height = 14
        
        color = (0, 200, 200)
        
        for key, value in self._status_values.items():
            text = f"{key}: {value}"
            cv2.putText(
                image,
                text,
                (status_x, y),
                font,
                cfg.font_scale,
                color,
                cfg.font_thickness,
                cv2.LINE_AA,
            )
            y += line_height
    
    def handle_mouse_move(self, px: int, py: int) -> Optional[Button]:
        """Handle mouse move event.
        
        Args:
            px: Mouse X position (relative to control bar)
            py: Mouse Y position (relative to control bar)
            
        Returns:
            Button under cursor, or None
        """
        hovered = None
        
        if self._row1 is not None:
            btn = self._row1.handle_mouse_move(px, py)
            if btn is not None:
                hovered = btn
        
        if self._row2 is not None:
            btn = self._row2.handle_mouse_move(px, py)
            if btn is not None:
                hovered = btn
        
        return hovered
    
    def handle_click(self, px: int, py: int) -> Optional[Button]:
        """Handle mouse click event.
        
        Args:
            px: Mouse X position (relative to control bar)
            py: Mouse Y position (relative to control bar)
            
        Returns:
            Clicked button, or None
        """
        if self._row1 is not None:
            btn = self._row1.handle_click(px, py)
            if btn is not None:
                return btn
        
        if self._row2 is not None:
            btn = self._row2.handle_click(px, py)
            if btn is not None:
                return btn
        
        return None
    
    def get_button(self, action_name: str) -> Optional[Button]:
        """Get a button by action name.
        
        Args:
            action_name: Name of the action
            
        Returns:
            Button, or None if not found
        """
        for bar in [self._row1, self._row2]:
            if bar is None:
                continue
            button = bar.get_button(action_name)
            if button is not None:
                return button
        return None
    
    @property
    def height(self) -> int:
        """Get control bar height."""
        return self.config.height
