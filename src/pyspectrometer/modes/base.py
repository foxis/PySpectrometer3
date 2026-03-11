"""Base class for operating modes."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from ..core.calibration import GraticuleData
    from ..core.mode_context import ModeContext
    from ..core.spectrum import SpectrumData


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
    integration_mode: Literal["none", "avg", "acc"] = "none"
    accumulated_frames: int = 0
    accumulated_intensity: np.ndarray | None = None

    # References
    black_reference: np.ndarray | None = None
    white_reference: np.ndarray | None = None

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
        self._ctx: ModeContext | None = None

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
    ) -> tuple[np.ndarray, tuple[int, int, int]] | None:
        """Get overlay spectrum to render on graph.

        Args:
            wavelengths: Current wavelength calibration array
            graph_height: Height of the graph in pixels

        Returns:
            Tuple of (intensity array scaled to graph height, BGR color) or None
        """
        pass

    def on_start(self, ctx: "ModeContext") -> None:
        """Called when the main loop starts. Override for mode-specific startup (e.g. default source)."""
        pass

    def transform_spectrum_data(self, data: "SpectrumData") -> "SpectrumData":
        """Transform spectrum data for display/export. Override for Raman (wavelength→wavenumber)."""
        return data

    def get_graticule(self, data: "SpectrumData") -> Optional["GraticuleData"]:
        """Custom graticule for display. Override for Raman (wavenumber axis). Returns None to use calibration default."""
        return None

    def update_display(
        self,
        ctx: "ModeContext",
        processed: "SpectrumData",
        graph_height: int,
    ) -> None:
        """Update overlay and status. Override in subclasses. Default: clear overlays, legacy ref if available."""
        ctx.display.set_mode_overlay(None)
        ctx.display.set_sensitivity_overlay(None)
        if ctx.reference_manager is not None:
            ref_intensity = ctx.reference_manager.get_interpolated(
                processed.wavelengths, processed.intensity
            )
            ctx.display.state.reference_spectrum = ref_intensity

    def setup(self, ctx: "ModeContext") -> None:
        """Register handlers using the given context. Subclasses override to add mode-specific handlers."""
        self._ctx = ctx
        self._register_common_handlers()

    def _register_common_handlers(self) -> None:
        """Register handlers for buttons common to all modes."""
        ctx = self._ctx
        if ctx is None:
            return
        self.register_callback("quit", ctx.quit_app)
        self.register_callback("capture_peak", lambda: self._on_toggle_hold())
        self.register_callback("cycle_preview", lambda: ctx.display.cycle_preview_mode())
        self.register_callback("show_gain_slider", lambda: self._on_toggle_gain_slider())
        self.register_callback("show_exposure_slider", lambda: self._on_toggle_exposure_slider())
        self.register_callback("show_spectrum_bars", lambda: self._on_toggle_spectrum_bars())
        self.register_callback("clear_peak_region", lambda: self._on_clear_peak_region())
        self.register_callback("auto_gain", lambda: self._on_toggle_auto_gain())
        self.register_callback("auto_exposure", lambda: self._on_toggle_auto_exposure())
        ctx.display.register_slider_callbacks(
            gain_cb=lambda v: setattr(ctx.camera, "gain", v),
            exposure_cb=lambda v: setattr(ctx.camera, "exposure", int(v)),
        )
        ctx.display.set_gain_value(ctx.camera.gain)
        ctx.display.set_exposure_value(getattr(ctx.camera, "exposure", 10000))

    def _on_toggle_hold(self) -> None:
        """Toggle peak hold."""
        ctx = self._ctx
        if ctx is None:
            return
        ctx.display.state.hold_peaks = not ctx.display.state.hold_peaks
        ctx.display.set_button_active("capture_peak", ctx.display.state.hold_peaks)
        if not ctx.display.state.hold_peaks:
            ctx.held_intensity = None
        print(f"[HOLD] Peak hold {'ON' if ctx.display.state.hold_peaks else 'OFF'}")

    def _on_toggle_gain_slider(self) -> None:
        """Toggle gain slider visibility."""
        ctx = self._ctx
        if ctx is None:
            return
        visible = ctx.display.toggle_gain_slider()
        ctx.display.set_button_active("show_gain_slider", visible)
        print(f"[GAIN] Gain slider: {'VISIBLE' if visible else 'HIDDEN'}")

    def _on_toggle_exposure_slider(self) -> None:
        """Toggle exposure slider visibility."""
        ctx = self._ctx
        if ctx is None:
            return
        visible = ctx.display.toggle_exposure_slider()
        ctx.display.set_button_active("show_exposure_slider", visible)
        print(f"[EXPOSURE] Exposure slider: {'VISIBLE' if visible else 'HIDDEN'}")

    def _on_toggle_spectrum_bars(self) -> None:
        """Toggle vertical colored bars on spectrum graph."""
        ctx = self._ctx
        if ctx is None:
            return
        visible = ctx.display.toggle_spectrum_bars()
        ctx.display.set_button_active("show_spectrum_bars", visible)
        print(f"[BARS] Spectrum bars: {'ON' if visible else 'OFF'}")

    def _on_clear_peak_region(self) -> None:
        """Clear the click-to-include peak region."""
        ctx = self._ctx
        if ctx is None:
            return
        ctx.display.clear_peak_include_region()
        print("[PEAK] Cleared peak include region")

    def _on_toggle_auto_gain(self) -> None:
        """Toggle auto gain."""
        ctx = self._ctx
        if ctx is None:
            return
        ctx.auto_gain_enabled = not ctx.auto_gain_enabled
        ctx.display.set_button_active("auto_gain", ctx.auto_gain_enabled)
        print(f"[AUTO_GAIN] Auto gain: {'ON' if ctx.auto_gain_enabled else 'OFF'}")

    def _on_toggle_auto_exposure(self) -> None:
        """Toggle auto exposure."""
        ctx = self._ctx
        if ctx is None:
            return
        ctx.auto_exposure_enabled = not ctx.auto_exposure_enabled
        ctx.display.set_button_active("auto_exposure", ctx.auto_exposure_enabled)
        print(f"[AUTO_EXPOSURE] Auto exposure: {'ON' if ctx.auto_exposure_enabled else 'OFF'}")

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
        """Toggle averaging mode (mutually exclusive with accumulation)."""
        if self.state.integration_mode == "avg":
            self.state.integration_mode = "none"
            self.state.accumulated_frames = 0
            self.state.accumulated_intensity = None
            return False
        self.state.integration_mode = "avg"
        self.state.accumulated_frames = 0
        self.state.accumulated_intensity = None
        return True

    def toggle_accumulation(self) -> bool:
        """Toggle accumulation mode (mutually exclusive with averaging)."""
        if self.state.integration_mode == "acc":
            self.state.integration_mode = "none"
            self.state.accumulated_frames = 0
            self.state.accumulated_intensity = None
            return False
        self.state.integration_mode = "acc"
        self.state.accumulated_frames = 0
        self.state.accumulated_intensity = None
        return True

    def accumulate_spectrum(self, intensity: np.ndarray) -> np.ndarray:
        """Accumulate spectrum intensity. Avg: sum/N. Acc: sum normalized by max for display.

        Args:
            intensity: Current spectrum intensity (1D array)

        Returns:
            Processed intensity (0-1 for pipeline)
        """
        if self.state.integration_mode == "none":
            return intensity

        if self.state.accumulated_intensity is None:
            self.state.accumulated_intensity = intensity.astype(np.float64)
            self.state.accumulated_frames = 1
        else:
            self.state.accumulated_intensity = self.state.accumulated_intensity.astype(
                np.float64
            ) + np.asarray(intensity, dtype=np.float64)
            self.state.accumulated_frames += 1

        acc = self.state.accumulated_intensity
        if self.state.integration_mode == "avg":
            return (acc / self.state.accumulated_frames).astype(np.float32)
        # Acc: normalize by max for 0-1 display
        m = float(np.max(acc))
        if m < 1e-9:
            return acc.astype(np.float32)
        return (acc / m).astype(np.float32)

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
