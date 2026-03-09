"""Keyboard input handling for PySpectrometer3."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import cv2


class Action(Enum):
    """Available keyboard actions."""

    QUIT = auto()
    TOGGLE_HOLD_PEAKS = auto()
    SAVE = auto()
    CALIBRATE = auto()
    CLEAR_CLICKS = auto()
    TOGGLE_MEASURE = auto()
    TOGGLE_PIXEL_MODE = auto()
    SAVGOL_UP = auto()
    SAVGOL_DOWN = auto()
    PEAK_WIDTH_UP = auto()
    PEAK_WIDTH_DOWN = auto()
    THRESHOLD_UP = auto()
    THRESHOLD_DOWN = auto()
    GAIN_UP = auto()
    GAIN_DOWN = auto()
    CYCLE_EXTRACTION_METHOD = auto()
    AUTO_DETECT_ANGLE = auto()
    PERP_WIDTH_UP = auto()
    PERP_WIDTH_DOWN = auto()
    SAVE_EXTRACTION_PARAMS = auto()
    CYCLE_REFERENCE_SPECTRUM = auto()


@dataclass
class KeyBinding:
    """A key binding mapping a key to an action."""

    key: str
    action: Action
    description: str = ""


DEFAULT_BINDINGS: list[KeyBinding] = [
    KeyBinding("q", Action.QUIT, "Quit application"),
    KeyBinding("h", Action.TOGGLE_HOLD_PEAKS, "Toggle peak hold"),
    KeyBinding("s", Action.SAVE, "Save spectrum"),
    KeyBinding("c", Action.CALIBRATE, "Perform calibration"),
    KeyBinding("x", Action.CLEAR_CLICKS, "Clear click points"),
    KeyBinding("m", Action.TOGGLE_MEASURE, "Toggle measure mode"),
    KeyBinding("p", Action.TOGGLE_PIXEL_MODE, "Toggle pixel mode"),
    KeyBinding("o", Action.SAVGOL_UP, "Increase Savitzky-Golay order"),
    KeyBinding("l", Action.SAVGOL_DOWN, "Decrease Savitzky-Golay order"),
    KeyBinding("i", Action.PEAK_WIDTH_UP, "Increase peak width"),
    KeyBinding("k", Action.PEAK_WIDTH_DOWN, "Decrease peak width"),
    KeyBinding("u", Action.THRESHOLD_UP, "Increase threshold"),
    KeyBinding("j", Action.THRESHOLD_DOWN, "Decrease threshold"),
    KeyBinding("t", Action.GAIN_UP, "Increase camera gain"),
    KeyBinding("g", Action.GAIN_DOWN, "Decrease camera gain"),
    KeyBinding("e", Action.CYCLE_EXTRACTION_METHOD, "Cycle extraction method"),
    KeyBinding("a", Action.AUTO_DETECT_ANGLE, "Auto-detect rotation angle"),
    KeyBinding("]", Action.PERP_WIDTH_UP, "Increase perpendicular width"),
    KeyBinding("[", Action.PERP_WIDTH_DOWN, "Decrease perpendicular width"),
    KeyBinding("w", Action.SAVE_EXTRACTION_PARAMS, "Save extraction parameters"),
    KeyBinding("r", Action.CYCLE_REFERENCE_SPECTRUM, "Cycle reference spectrum"),
]


class KeyboardHandler:
    """Handles keyboard input and dispatches to registered callbacks.

    This class decouples keyboard input from the main loop by using
    a callback-based approach. Actions are registered with callbacks,
    and the handler dispatches to the appropriate callback when a
    key is pressed.
    """

    def __init__(self, bindings: list[KeyBinding] | None = None):
        """Initialize keyboard handler.

        Args:
            bindings: List of key bindings (uses defaults if None)
        """
        self._bindings = bindings or DEFAULT_BINDINGS.copy()
        self._callbacks: dict[Action, Callable[[], None]] = {}
        self._key_to_action: dict[int, Action] = {}

        self._build_key_map()

    def _build_key_map(self) -> None:
        """Build mapping from key codes to actions."""
        self._key_to_action.clear()
        for binding in self._bindings:
            key_code = ord(binding.key)
            self._key_to_action[key_code] = binding.action

    def register(self, action: Action, callback: Callable[[], None]) -> None:
        """Register a callback for an action.

        Args:
            action: Action to register callback for
            callback: Function to call when action is triggered
        """
        self._callbacks[action] = callback

    def unregister(self, action: Action) -> None:
        """Unregister a callback for an action.

        Args:
            action: Action to unregister
        """
        self._callbacks.pop(action, None)

    def poll(self, wait_ms: int = 1) -> Action | None:
        """Poll for keyboard input and dispatch callbacks.

        Args:
            wait_ms: Milliseconds to wait for key press

        Returns:
            The action that was triggered, or None if no key pressed
        """
        key_press_raw = cv2.waitKey(wait_ms)

        if key_press_raw == -1:
            return None

        # Try raw key first
        action = self._key_to_action.get(key_press_raw)

        # Fall back to masked key (for compatibility with some platforms)
        if action is None:
            key_press_masked = key_press_raw & 0xFF
            action = self._key_to_action.get(key_press_masked)

        if action is None:
            return None

        callback = self._callbacks.get(action)
        if callback is not None:
            callback()

        return action

    def get_bindings(self) -> list[KeyBinding]:
        """Get list of all key bindings."""
        return self._bindings.copy()

    def get_help_text(self) -> str:
        """Generate help text for all key bindings."""
        lines = ["Keyboard Shortcuts:", ""]
        for binding in self._bindings:
            desc = binding.description or binding.action.name
            lines.append(f"  {binding.key}: {desc}")
        return "\n".join(lines)
