"""Control bar with two rows of buttons for the spectrum display."""

from dataclasses import dataclass
from typing import Callable, Optional
import cv2
import numpy as np

from .buttons import Button, ButtonBar, ButtonStyle


def _estimate_button_width(label: str, shortcut: str, style: ButtonStyle) -> int:
    """Estimate button width from label (shortcut not displayed in button)."""
    (text_w, _), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX,
        style.font_scale, style.font_thickness,
    )
    return text_w + style.padding_x * 2


@dataclass
class ControlBarConfig:
    """Configuration for the control bar."""
    
    height: int = 80
    row_height: int = 24
    row_spacing: int = 12  # Vertical separation between button rows
    button_spacing: int = 4
    margin_x: int = 5
    margin_y: int = 8
    
    bg_color: tuple[int, int, int] = (20, 20, 20)
    
    font_scale: float = 0.35
    font_thickness: int = 1
    
    def __post_init__(self) -> None:
        """Adjust row sizes to fit within height."""
        # Calculate what fits: margin + row + spacing + row + margin
        min_needed = self.margin_y * 2 + self.row_height * 2 + self.row_spacing
        if min_needed > self.height:
            # Reduce row height and margins to fit, but keep at least 4px vertical spacing
            available = self.height - 4  # minimal margins
            self.margin_y = 2
            self.row_spacing = max(4, min(self.row_spacing, available - 2 * 20))  # min 4px between rows
            self.row_height = (available - self.row_spacing) // 2


class ControlBar:
    """Two-row control bar with clickable buttons.

    Replaces the static message banner with interactive controls.
    Uses mode.get_buttons() as single source of truth when buttons provided.
    """

    def __init__(
        self,
        width: int = 800,
        config: Optional[ControlBarConfig] = None,
        mode: str = "measurement",
        buttons: Optional[list] = None,
    ):
        """Initialize control bar.

        Args:
            width: Width of the control bar
            config: Configuration (uses defaults if None)
            mode: Operating mode
            buttons: Button definitions from mode.get_buttons() (required)
        """
        if buttons is None:
            raise ValueError("buttons required; use mode.get_buttons()")
        self.width = width
        self.config = config or ControlBarConfig()
        self.mode = mode

        self._row1: Optional[ButtonBar] = None
        self._row2: Optional[ButtonBar] = None
        self._status_values: dict[str, str] = {}
        self._button_style: Optional[ButtonStyle] = None

        self._setup_buttons(buttons)

    def _setup_buttons(self, buttons: list) -> None:
        """Set up buttons from list (ButtonDefinition from mode.get_buttons())."""
        cfg = self.config
        
        self._button_style = ButtonStyle(
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
            style=self._button_style,
        )
        
        self._row2 = ButtonBar(
            x=cfg.margin_x,
            y=row2_y,
            height=cfg.row_height,
            spacing=cfg.button_spacing,
            style=self._button_style,
        )
        
        for btn_def in buttons:
            bar = self._row1 if btn_def.row == 1 else self._row2
            if btn_def.action_name.startswith("__spacer"):
                left_width = bar._next_x - bar.x
                spacer_idx = next(i for i, b in enumerate(buttons) if b is btn_def)
                same_row = btn_def.row
                right_defs = [b for i, b in enumerate(buttons) if i > spacer_idx and b.row == same_row
                              and not b.action_name.startswith("__spacer")]
                right_width = 0
                for rd in right_defs:
                    w = _estimate_button_width(rd.label, rd.shortcut, self._button_style)
                    right_width += w + cfg.button_spacing
                if right_defs:
                    right_width -= cfg.button_spacing
                available = self.width - cfg.margin_x * 2 - left_width - right_width
                spacer_width = max(10, min(available, 300))
                bar.add_separator(width=spacer_width)
                continue
            bar.add_button(
                btn_def.label,
                btn_def.action_name,
                is_toggle=btn_def.is_toggle,
                shortcut=btn_def.shortcut,
                icon_type=getattr(btn_def, "icon_type", ""),
            )
    
    def set_mode(self, mode: str, buttons: list) -> None:
        """Change operating mode (rebuilds buttons).

        Args:
            mode: New operating mode
            buttons: Button definitions from mode.get_buttons() (required)
        """
        self.mode = mode
        self._status_values.clear()
        self._setup_buttons(buttons)
    
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
        """Render the control bar (buttons only, no status).
        
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
        
        return image
    
    def get_status_text(self) -> str:
        """Get formatted status text for display elsewhere.
        
        Returns:
            Compact status string like "Gain:10 Cal:OK Avg:5"
        """
        parts = [f"{k}:{v}" for k, v in self._status_values.items()]
        return "  ".join(parts)
    
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
