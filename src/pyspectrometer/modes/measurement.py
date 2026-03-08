"""Measurement mode for general spectrum measurement.

Features:
- Save/Load spectrum
- Dark reference subtraction
- White reference normalization
- Spectrum averaging
- Auto gain control
- GPIO lamp control
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
import numpy as np

from .base import BaseMode, ModeType, ButtonDefinition


@dataclass
class MeasurementState:
    """State specific to measurement mode."""
    
    # Reference spectra
    dark_spectrum: Optional[np.ndarray] = None
    white_spectrum: Optional[np.ndarray] = None
    reference_spectrum: Optional[np.ndarray] = None
    
    # Current captured spectrum (for saving)
    captured_spectrum: Optional[np.ndarray] = None
    captured_wavelengths: Optional[np.ndarray] = None
    
    # Display settings
    show_reference: bool = False
    normalize_to_reference: bool = False
    subtract_dark: bool = True
    
    # Loaded spectrum info
    loaded_spectrum_name: str = ""


class MeasurementMode(BaseMode):
    """Measurement mode for general spectrum analysis."""
    
    def __init__(self):
        """Initialize measurement mode."""
        super().__init__()
        self.meas_state = MeasurementState()
        
        # Callbacks to be set by spectrometer
        self._on_save: Optional[Callable[[], None]] = None
        self._on_load: Optional[Callable[[], None]] = None
    
    @property
    def mode_type(self) -> ModeType:
        return ModeType.MEASUREMENT
    
    @property
    def name(self) -> str:
        return "Measurement"
    
    def get_buttons(self) -> list[ButtonDefinition]:
        """Get measurement mode buttons."""
        return [
            # Row 1: Capture and references
            ButtonDefinition("Capture", "capture", row=1),
            ButtonDefinition("Peak", "capture_peak", is_toggle=True, shortcut="h", row=1),
            ButtonDefinition("Avg", "toggle_averaging", is_toggle=True, row=1),
            ButtonDefinition("Dark", "set_dark", row=1),
            ButtonDefinition("White", "set_white", row=1),
            ButtonDefinition("ClrRef", "clear_refs", row=1),
            ButtonDefinition("Save", "save", shortcut="s", row=1),
            ButtonDefinition("Load", "load", row=1),
            
            # Row 2: Display and control
            ButtonDefinition("ShowRef", "show_reference", is_toggle=True, row=2),
            ButtonDefinition("Norm", "normalize", is_toggle=True, row=2),
            ButtonDefinition("AutoG", "auto_gain", is_toggle=True, row=2),
            ButtonDefinition("Gain+", "gain_up", shortcut="t", row=2),
            ButtonDefinition("Gain-", "gain_down", shortcut="g", row=2),
            ButtonDefinition("Lamp", "lamp_toggle", is_toggle=True, row=2),
            ButtonDefinition("Quit", "quit", shortcut="q", row=2),
        ]
    
    def process_spectrum(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Process spectrum with dark/white correction and normalization."""
        result = intensity.astype(np.float64)
        
        # Apply averaging if enabled
        if self.state.averaging_enabled:
            result = self.accumulate_spectrum(result)
        
        # Subtract dark reference
        if self.meas_state.subtract_dark and self.meas_state.dark_spectrum is not None:
            dark = self.meas_state.dark_spectrum.astype(np.float64)
            result = result - dark
            result = np.maximum(result, 0)
        
        # Normalize to white reference
        if self.meas_state.white_spectrum is not None:
            white = self.meas_state.white_spectrum.astype(np.float64)
            if self.meas_state.dark_spectrum is not None:
                white = white - self.meas_state.dark_spectrum.astype(np.float64)
            white = np.maximum(white, 1)  # Avoid division by zero
            result = (result / white) * 255
        
        # Normalize to reference spectrum
        if self.meas_state.normalize_to_reference and self.meas_state.reference_spectrum is not None:
            ref = self.meas_state.reference_spectrum.astype(np.float64)
            ref = np.maximum(ref, 1)
            result = (result / ref) * 255
        
        return np.clip(result, 0, 255)
    
    def get_overlay(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> Optional[tuple[np.ndarray, tuple[int, int, int]]]:
        """Get reference spectrum overlay if enabled."""
        if not self.meas_state.show_reference:
            return None
        
        ref = self.meas_state.reference_spectrum
        if ref is None:
            return None
        
        # Scale to graph height
        max_val = max(ref.max(), 1)
        scaled = (ref / max_val) * (graph_height - 10)
        
        # Light gray color for reference
        return (scaled, (150, 150, 150))
    
    def capture_current(self, intensity: np.ndarray, wavelengths: np.ndarray) -> None:
        """Capture current spectrum for later use."""
        self.meas_state.captured_spectrum = intensity.copy()
        self.meas_state.captured_wavelengths = wavelengths.copy()
        print("[Measurement] Spectrum captured")
    
    def set_dark_reference(self, intensity: np.ndarray) -> None:
        """Set dark/black reference spectrum."""
        self.meas_state.dark_spectrum = intensity.copy()
        self.meas_state.subtract_dark = True
        print("[Measurement] Dark reference set")
    
    def set_white_reference(self, intensity: np.ndarray) -> None:
        """Set white reference spectrum."""
        self.meas_state.white_spectrum = intensity.copy()
        print("[Measurement] White reference set")
    
    def set_reference_spectrum(self, intensity: np.ndarray, name: str = "Current") -> None:
        """Set reference spectrum for normalization/overlay."""
        self.meas_state.reference_spectrum = intensity.copy()
        self.meas_state.loaded_spectrum_name = name
        print(f"[Measurement] Reference spectrum set: {name}")
    
    def clear_references(self) -> None:
        """Clear all reference spectra."""
        self.meas_state.dark_spectrum = None
        self.meas_state.white_spectrum = None
        self.meas_state.reference_spectrum = None
        self.meas_state.subtract_dark = True
        self.meas_state.normalize_to_reference = False
        self.meas_state.show_reference = False
        self.meas_state.loaded_spectrum_name = ""
        print("[Measurement] All references cleared")
    
    def toggle_show_reference(self) -> bool:
        """Toggle reference spectrum overlay."""
        self.meas_state.show_reference = not self.meas_state.show_reference
        print(f"[Measurement] Show reference: {'ON' if self.meas_state.show_reference else 'OFF'}")
        return self.meas_state.show_reference
    
    def toggle_normalize(self) -> bool:
        """Toggle normalization to reference."""
        self.meas_state.normalize_to_reference = not self.meas_state.normalize_to_reference
        print(f"[Measurement] Normalize: {'ON' if self.meas_state.normalize_to_reference else 'OFF'}")
        return self.meas_state.normalize_to_reference
    
    def get_status(self) -> dict[str, str]:
        """Get status values for display."""
        status = {}
        
        if self.meas_state.dark_spectrum is not None:
            status["Dark"] = "SET"
        if self.meas_state.white_spectrum is not None:
            status["White"] = "SET"
        if self.meas_state.reference_spectrum is not None:
            status["Ref"] = self.meas_state.loaded_spectrum_name or "SET"
        
        if self.state.averaging_enabled:
            status["Avg"] = str(self.state.accumulated_frames)
        
        return status
