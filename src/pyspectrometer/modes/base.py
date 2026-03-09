"""Base class for operating modes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional
import numpy as np


class ModeType(Enum):
    """Available operating modes."""
    
    CALIBRATION = auto()
    MEASUREMENT = auto()
    RAMAN = auto()
    COLORSCIENCE = auto()


@dataclass
class ModeState:
    """Common state shared across modes."""
    
    # Spectrum capture
    frozen: bool = False
    averaging_enabled: bool = False
    averaging_count: int = 1
    accumulated_frames: int = 0
    accumulated_intensity: Optional[np.ndarray] = None
    
    # References
    black_reference: Optional[np.ndarray] = None
    white_reference: Optional[np.ndarray] = None
    
    # Auto controls
    auto_gain_enabled: bool = False
    auto_gain_target_min: int = 200
    auto_gain_target_max: int = 240
    
    # GPIO
    lamp_enabled: bool = False


@dataclass
class ButtonDefinition:
    """Definition of a button for the mode's control bar."""

    label: str
    action_name: str
    is_toggle: bool = False
    shortcut: str = ""
    row: int = 1
    icon_type: str = ""  # e.g. "playback" for freeze button


class BaseMode(ABC):
    """Abstract base class for operating modes.
    
    Each mode defines its own:
    - Control bar buttons
    - Processing pipeline modifications
    - Overlay rendering
    - State management
    """
    
    def __init__(self):
        """Initialize base mode."""
        self.state = ModeState()
        self._callbacks: dict[str, Callable[[], None]] = {}
    
    @property
    @abstractmethod
    def mode_type(self) -> ModeType:
        """Get the mode type."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get human-readable mode name."""
        pass
    
    @abstractmethod
    def get_buttons(self) -> list[ButtonDefinition]:
        """Get button definitions for this mode's control bar.
        
        Returns:
            List of button definitions for row 1 and row 2
        """
        pass
    
    @abstractmethod
    def process_spectrum(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Process spectrum data according to mode logic.
        
        Args:
            intensity: Raw intensity array
            wavelengths: Wavelength array
            
        Returns:
            Processed intensity array
        """
        pass
    
    @abstractmethod
    def get_overlay(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> Optional[tuple[np.ndarray, tuple[int, int, int]]]:
        """Get overlay spectrum to render on graph.
        
        Args:
            wavelengths: Current wavelength calibration array
            graph_height: Height of the graph in pixels
            
        Returns:
            Tuple of (intensity array scaled to graph height, BGR color) or None
        """
        pass
    
    def register_callback(self, action_name: str, callback: Callable[[], None]) -> None:
        """Register a callback for a button action."""
        self._callbacks[action_name] = callback
    
    def handle_action(self, action_name: str) -> bool:
        """Handle a button action.
        
        Returns:
            True if action was handled
        """
        callback = self._callbacks.get(action_name)
        if callback is not None:
            callback()
            return True
        return False
    
    def toggle_freeze(self) -> bool:
        """Toggle spectrum freeze state."""
        self.state.frozen = not self.state.frozen
        return self.state.frozen
    
    def toggle_averaging(self) -> bool:
        """Toggle averaging mode."""
        self.state.averaging_enabled = not self.state.averaging_enabled
        if not self.state.averaging_enabled:
            self.state.accumulated_frames = 0
            self.state.accumulated_intensity = None
        return self.state.averaging_enabled
    
    def accumulate_spectrum(self, intensity: np.ndarray) -> np.ndarray:
        """Accumulate spectrum intensity for averaging.
        
        Args:
            intensity: Current spectrum intensity (1D array)
            
        Returns:
            Averaged spectrum intensity
        """
        if not self.state.averaging_enabled:
            return intensity
        
        if self.state.accumulated_intensity is None:
            self.state.accumulated_intensity = intensity.astype(np.float64)
            self.state.accumulated_frames = 1
        else:
            self.state.accumulated_intensity = self.state.accumulated_intensity.astype(np.float32) + np.asarray(intensity, dtype=np.float32)
            self.state.accumulated_frames += 1
        
        return (self.state.accumulated_intensity / self.state.accumulated_frames).astype(np.float32)
    
    def set_black_reference(self, intensity: np.ndarray) -> None:
        """Set black/dark reference spectrum."""
        self.state.black_reference = intensity.copy()
        print(f"[{self.name}] Black reference set")
    
    def set_white_reference(self, intensity: np.ndarray) -> None:
        """Set white reference spectrum."""
        self.state.white_reference = intensity.copy()
        print(f"[{self.name}] White reference set")
    
    def clear_references(self) -> None:
        """Clear all references."""
        self.state.black_reference = None
        self.state.white_reference = None
        print(f"[{self.name}] References cleared")
    
    def apply_references(self, intensity: np.ndarray) -> np.ndarray:
        """Apply black/white reference correction.

        Returns 0-1 normalized intensity for pipeline consistency.

        Args:
            intensity: Raw intensity (0-1)

        Returns:
            Corrected intensity in 0-1 range
        """
        from ..processing.reference_correction import apply_dark_white_correction
        return apply_dark_white_correction(
            intensity,
            self.state.black_reference,
            self.state.white_reference,
        )
    
    def calculate_auto_gain_adjustment(
        self,
        current_max: float,
        current_gain: float,
        step: float = 0.5,
    ) -> float:
        """Calculate gain adjustment for auto-gain.
        
        Args:
            current_max: Current maximum intensity
            current_gain: Current camera gain
            step: Gain adjustment step
            
        Returns:
            New gain value
        """
        if not self.state.auto_gain_enabled:
            return current_gain
        
        target_min = self.state.auto_gain_target_min
        target_max = self.state.auto_gain_target_max
        
        if current_max > target_max:
            return max(0, current_gain - step)
        elif current_max < target_min:
            return min(50, current_gain + step)
        
        return current_gain
