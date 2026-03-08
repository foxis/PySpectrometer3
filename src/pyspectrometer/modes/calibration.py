"""Calibration mode for wavelength calibration.

Workflow:
1. Select reference source (FL, Hg, Sun, LED)
2. Point spectrometer at light source
3. Use auto-level and auto-shift to fit spectrum in view
4. Freeze spectrum and click calibrate
5. Algorithm matches peaks and computes 4-point polynomial fit
6. Verify overlay aligns with measured spectrum
7. Save calibration manually when satisfied
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np

from .base import BaseMode, ModeType, ButtonDefinition
from ..data.reference_spectra import (
    ReferenceSource,
    get_reference_spectrum,
    get_reference_peaks,
    get_reference_name,
    SpectralLine,
)


@dataclass
class CalibrationState:
    """State specific to calibration mode."""
    
    # Current reference source
    current_source: ReferenceSource = ReferenceSource.FL
    
    # Overlay visibility
    overlay_visible: bool = True
    
    # Detected peaks from measured spectrum
    detected_peaks: list[int] = field(default_factory=list)  # pixel positions
    
    # Matched calibration points (pixel, wavelength)
    calibration_points: list[tuple[int, float]] = field(default_factory=list)
    
    # Auto-level target settings
    target_intensity_min: int = 200
    target_intensity_max: int = 240
    
    # Y-axis shift (for auto-center)
    y_shift: int = 0
    
    # Frozen spectrum for calibration
    frozen_intensity: Optional[np.ndarray] = None


class CalibrationMode(BaseMode):
    """Calibration mode for wavelength calibration."""
    
    # Reference sources in cycle order
    SOURCES = [
        ReferenceSource.FL,
        ReferenceSource.HG,
        ReferenceSource.SUN,
        ReferenceSource.LED,
    ]
    
    def __init__(self):
        """Initialize calibration mode."""
        super().__init__()
        self.cal_state = CalibrationState()
        
        # Enable auto-gain by default
        self.state.auto_gain_enabled = True
        
        # Callbacks to be set by spectrometer
        self._on_save_calibration: Optional[Callable[[], None]] = None
        self._on_load_calibration: Optional[Callable[[], None]] = None
        self._on_auto_calibrate: Optional[Callable[[list[tuple[int, float]]], None]] = None
        self._on_gain_change: Optional[Callable[[float], None]] = None
        self._on_y_shift_change: Optional[Callable[[int], None]] = None
    
    @property
    def mode_type(self) -> ModeType:
        return ModeType.CALIBRATION
    
    @property
    def name(self) -> str:
        return "Calibration"
    
    def get_buttons(self) -> list[ButtonDefinition]:
        """Get calibration mode buttons."""
        return [
            # Row 1: Source selection and calibration actions
            ButtonDefinition("FL", "source_fl", row=1),
            ButtonDefinition("Hg", "source_hg", row=1),
            ButtonDefinition("Sun", "source_sun", row=1),
            ButtonDefinition("LED", "source_led", row=1),
            ButtonDefinition("AutoLvl", "auto_level", is_toggle=True, row=1),
            ButtonDefinition("AutoCal", "auto_calibrate", row=1),
            ButtonDefinition("Save", "save_cal", shortcut="w", row=1),
            ButtonDefinition("Load", "load_cal", row=1),
            
            # Row 2: Display and control
            ButtonDefinition("Freeze", "freeze", is_toggle=True, shortcut="f", row=2),
            ButtonDefinition("Overlay", "toggle_overlay", is_toggle=True, row=2),
            ButtonDefinition("AutoG", "auto_gain", is_toggle=True, row=2),
            ButtonDefinition("Gain+", "gain_up", shortcut="t", row=2),
            ButtonDefinition("Gain-", "gain_down", shortcut="g", row=2),
            ButtonDefinition("Clear", "clear_points", shortcut="x", row=2),
            ButtonDefinition("Quit", "quit", shortcut="q", row=2),
        ]
    
    def process_spectrum(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Process spectrum - apply freeze if enabled."""
        # If frozen, return the frozen spectrum
        if self.state.frozen and self.cal_state.frozen_intensity is not None:
            return self.cal_state.frozen_intensity
        
        # Apply averaging if enabled
        if self.state.averaging_enabled:
            intensity = self.accumulate_frame(intensity)
        
        # Store for freeze
        self.cal_state.frozen_intensity = intensity.copy()
        
        return intensity
    
    def get_overlay(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> Optional[tuple[np.ndarray, tuple[int, int, int]]]:
        """Get reference spectrum overlay."""
        if not self.cal_state.overlay_visible:
            return None
        
        # Generate reference spectrum
        ref_intensity = get_reference_spectrum(
            self.cal_state.current_source,
            wavelengths,
        )
        
        # Scale to graph height
        scaled = ref_intensity * (graph_height - 10)
        
        # Color based on source
        colors = {
            ReferenceSource.FL: (255, 200, 0),    # Cyan
            ReferenceSource.HG: (255, 0, 255),    # Magenta
            ReferenceSource.SUN: (0, 200, 255),   # Orange/yellow
            ReferenceSource.LED: (200, 200, 200), # White/gray
        }
        color = colors.get(self.cal_state.current_source, (200, 200, 200))
        
        return (scaled, color)
    
    def select_source(self, source: ReferenceSource) -> None:
        """Select a reference source."""
        self.cal_state.current_source = source
        print(f"[Calibration] Selected reference: {get_reference_name(source)}")
    
    def cycle_source(self) -> ReferenceSource:
        """Cycle to next reference source."""
        current_idx = self.SOURCES.index(self.cal_state.current_source)
        next_idx = (current_idx + 1) % len(self.SOURCES)
        self.cal_state.current_source = self.SOURCES[next_idx]
        print(f"[Calibration] Reference: {get_reference_name(self.cal_state.current_source)}")
        return self.cal_state.current_source
    
    def toggle_overlay(self) -> bool:
        """Toggle reference overlay visibility."""
        self.cal_state.overlay_visible = not self.cal_state.overlay_visible
        print(f"[Calibration] Overlay: {'ON' if self.cal_state.overlay_visible else 'OFF'}")
        return self.cal_state.overlay_visible
    
    def run_auto_level(self, current_intensity: np.ndarray, current_gain: float) -> float:
        """Run auto-level algorithm once to find optimal gain.
        
        Adjusts gain to bring spectrum peak into target range (200-240).
        
        Args:
            current_intensity: Current spectrum intensity values
            current_gain: Current camera gain setting
            
        Returns:
            New gain value to apply (or same if already optimal)
        """
        current_max = float(np.max(current_intensity))
        target_mid = (self.cal_state.target_intensity_min + self.cal_state.target_intensity_max) / 2
        
        if current_max < 1:
            print("[AutoLevel] No signal detected")
            return current_gain
        
        # Calculate required gain ratio
        ratio = target_mid / current_max
        new_gain = current_gain * ratio
        
        # Clamp to reasonable range
        new_gain = max(1.0, min(50.0, new_gain))
        
        print(f"[AutoLevel] Current max: {current_max:.0f}, Target: {target_mid:.0f}")
        print(f"[AutoLevel] Gain: {current_gain:.1f} -> {new_gain:.1f}")
        
        return new_gain
    
    def auto_calibrate(
        self,
        measured_intensity: np.ndarray,
        wavelengths: np.ndarray,
        peak_indices: list[int],
    ) -> list[tuple[int, float]]:
        """Perform automatic calibration by matching peaks.
        
        Args:
            measured_intensity: Measured spectrum intensity
            wavelengths: Current wavelength calibration
            peak_indices: Detected peak pixel positions
            
        Returns:
            List of (pixel, wavelength) calibration points (minimum 4)
        """
        reference_peaks = get_reference_peaks(self.cal_state.current_source)
        
        if len(peak_indices) < 4:
            print(f"[Calibration] Need at least 4 peaks, found {len(peak_indices)}")
            return []
        
        if len(reference_peaks) < 4:
            print(f"[Calibration] Reference has only {len(reference_peaks)} peaks")
            return []
        
        # Convert peak pixel positions to approximate wavelengths
        peak_wavelengths = [wavelengths[min(idx, len(wavelengths)-1)] for idx in peak_indices]
        
        # Match peaks to reference lines
        matches = self._match_peaks_to_reference(
            peak_indices,
            peak_wavelengths,
            reference_peaks,
        )
        
        if len(matches) < 4:
            print(f"[Calibration] Could only match {len(matches)} peaks, need 4")
            return []
        
        # Take best 4-6 matches
        matches = sorted(matches, key=lambda m: m[2])[:min(6, len(matches))]
        
        # Extract (pixel, wavelength) pairs
        calibration_points = [(m[0], m[1]) for m in matches]
        
        self.cal_state.calibration_points = calibration_points
        
        print(f"[Calibration] Matched {len(calibration_points)} points:")
        for pixel, wl in calibration_points:
            print(f"  Pixel {pixel} -> {wl:.1f} nm")
        
        return calibration_points
    
    def _match_peaks_to_reference(
        self,
        peak_pixels: list[int],
        peak_wavelengths: list[float],
        reference_peaks: list[SpectralLine],
    ) -> list[tuple[int, float, float]]:
        """Match detected peaks to reference spectral lines.
        
        Uses a greedy nearest-neighbor approach with consistency checking.
        
        Args:
            peak_pixels: Detected peak pixel positions
            peak_wavelengths: Approximate wavelengths from current calibration
            reference_peaks: Known reference spectral lines
            
        Returns:
            List of (pixel, reference_wavelength, error) tuples
        """
        matches = []
        used_refs = set()
        
        # Sort reference peaks by wavelength
        ref_sorted = sorted(reference_peaks, key=lambda p: p.wavelength)
        
        # For each detected peak, find nearest reference
        for pixel, approx_wl in zip(peak_pixels, peak_wavelengths):
            best_match = None
            best_error = float('inf')
            
            for ref in ref_sorted:
                if ref.wavelength in used_refs:
                    continue
                
                error = abs(approx_wl - ref.wavelength)
                
                # Allow larger error tolerance (up to 50nm for uncalibrated)
                if error < best_error and error < 50:
                    best_error = error
                    best_match = ref
            
            if best_match is not None:
                matches.append((pixel, best_match.wavelength, best_error))
                used_refs.add(best_match.wavelength)
        
        # Sort by pixel position for consistency
        matches.sort(key=lambda m: m[0])
        
        # Verify monotonicity (wavelengths should increase with pixels)
        valid_matches = []
        last_wl = 0
        for pixel, wl, err in matches:
            if wl > last_wl:
                valid_matches.append((pixel, wl, err))
                last_wl = wl
        
        return valid_matches
    
    def get_calibration_points(self) -> list[tuple[int, float]]:
        """Get current calibration points for saving.
        
        Returns:
            List of (pixel, wavelength) tuples
        """
        return self.cal_state.calibration_points.copy()
    
    def clear_calibration_points(self) -> None:
        """Clear all calibration points."""
        self.cal_state.calibration_points.clear()
        print("[Calibration] Points cleared")
    
    def get_current_source_name(self) -> str:
        """Get name of current reference source."""
        return get_reference_name(self.cal_state.current_source)
