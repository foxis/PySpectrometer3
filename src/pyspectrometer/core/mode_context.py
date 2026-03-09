"""Shared context for modes - dependencies and mutable state.

The orchestrator creates ModeContext, populates it with services and callbacks,
and passes it to modes. Modes use the context to implement their handlers
without depending on the orchestrator.
"""

from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np

from .spectrum import SpectrumData


def _noop() -> None:
    pass


def _noop_save(_: SpectrumData) -> None:
    pass


@dataclass
class ModeContext:
    """Shared context passed to modes for implementing handlers.

    The orchestrator owns these references and updates frame state each loop.
    Modes read/write via context to perform actions (save, set_dark, etc.).
    """

    # Services (read-only for modes)
    camera: object  # CameraInterface
    calibration: object  # Calibration
    display: object  # DisplayManager
    exporter: object  # CSVExporter
    extractor: object  # SpectrumExtractor
    auto_gain: object  # AutoGainController
    auto_exposure: object  # AutoExposureController

    # Quit callback - mode calls to stop the main loop
    quit_app: Callable[[], None] = _noop

    # Save snapshot - mode calls with SpectrumData
    save_snapshot: Callable[[SpectrumData], None] = _noop_save

    # Frame state (orchestrator updates each frame; modes read)
    last_data: Optional[SpectrumData] = None
    last_frame: Optional[np.ndarray] = None
    last_intensity_pre_sensitivity: Optional[np.ndarray] = None

    # Mutable state (modes read/write; orchestrator reads for loop logic)
    running: bool = True
    frozen_spectrum: bool = False
    frozen_intensity: Optional[np.ndarray] = None
    held_intensity: Optional[np.ndarray] = None
    auto_gain_enabled: bool = False
    auto_exposure_enabled: bool = False

    # Auto-level overlay dismiss (calibration mode sets)
    clear_autolevel_overlay: Callable[[], None] = _noop

    # Debounce for auto-calibrate (calibration mode)
    last_auto_calibrate_time: float = 0.0
    auto_calibrate_debounce_sec: float = 1.5

    def handle_auto_gain_exposure(self, data: SpectrumData) -> None:
        """Run auto-gain and auto-exposure adjustment (orchestrator calls each frame)."""
        if not self.auto_gain_enabled and not self.auto_exposure_enabled:
            return
        if self.frozen_spectrum:
            return

        def get_gain() -> float:
            return self.camera.gain

        def set_gain(v: float) -> None:
            self.camera.gain = v

        def get_exposure() -> int:
            return getattr(self.camera, "exposure", 10000)

        def set_exposure(v: int) -> None:
            if hasattr(self.camera, "exposure"):
                self.camera.exposure = v

        if self.auto_exposure_enabled:
            self.auto_exposure.adjust(
                data,
                get_exposure,
                set_exposure,
                self.display.set_exposure_value,
            )
        if self.auto_gain_enabled:
            self.auto_gain.adjust(
                data,
                get_gain,
                set_gain,
                self.display.set_gain_value,
            )
