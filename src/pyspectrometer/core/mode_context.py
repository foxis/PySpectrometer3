"""Shared context for modes - dependencies and mutable state.

The orchestrator creates ModeContext, populates it with services and callbacks,
and passes it to modes. Modes use the context to implement their handlers
without depending on the orchestrator.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from .spectrum import SpectrumData

if TYPE_CHECKING:
    from ..capture.base import CameraInterface
    from ..core.reference_spectrum import ReferenceSpectrumManager
    from ..display.renderer import DisplayManager
    from ..export.csv_exporter import CSVExporter
    from ..processing.auto_controls import AutoExposureController, AutoGainController
    from ..processing.extraction import SpectrumExtractor
    from .calibration import Calibration


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
    camera: "CameraInterface"
    calibration: "Calibration"
    display: "DisplayManager"
    exporter: "CSVExporter"
    extractor: "SpectrumExtractor"
    auto_gain: "AutoGainController"
    auto_exposure: "AutoExposureController"
    reference_manager: Optional["ReferenceSpectrumManager"] = None

    # Quit callback - mode calls to stop the main loop
    quit_app: Callable[[], None] = _noop

    # Save snapshot - mode calls with SpectrumData and optional kwargs (reference_intensity, metadata)
    save_snapshot: Callable[..., None] = _noop_save

    # Frame state (orchestrator updates each frame; modes read)
    last_data: SpectrumData | None = None
    last_frame: np.ndarray | None = None
    last_intensity_pre_sensitivity: np.ndarray | None = None
    last_raw_intensity: np.ndarray | None = None  # Pre-process_spectrum for dark capture
    last_raw_extraction_max: float = 0.0  # Max pixel in extraction ROI (0-1) for AE/AG

    # Mutable state (modes read/write; orchestrator reads for loop logic)
    running: bool = True
    frozen_spectrum: bool = False
    frozen_intensity: np.ndarray | None = None
    held_intensity: np.ndarray | None = None
    auto_gain_enabled: bool = False
    auto_exposure_enabled: bool = False

    # Auto-level overlay dismiss (calibration mode sets)
    clear_autolevel_overlay: Callable[[], None] = _noop

    # Debounce for auto-calibrate (calibration mode)
    last_auto_calibrate_time: float = 0.0
    auto_calibrate_debounce_sec: float = 1.5

    # After an exposure adjustment, skip gain for this many frames so we see the new exposure before changing gain (avoids alternating E-down then G-down and overshoot).
    gain_cooldown_frames_remaining: int = 0

    def handle_auto_gain_exposure(self, data: SpectrumData) -> None:
        """Run auto-gain and auto-exposure adjustment (orchestrator calls each frame)."""
        if not self.auto_gain_enabled and not self.auto_exposure_enabled:
            return
        if self.frozen_spectrum:
            return

        if self.gain_cooldown_frames_remaining > 0:
            self.gain_cooldown_frames_remaining -= 1

        def get_gain() -> float:
            return self.camera.gain

        def set_gain(v: float) -> None:
            self.camera.gain = v

        def get_exposure() -> int:
            return getattr(self.camera, "exposure", 10000)

        def set_exposure(v: int) -> None:
            if hasattr(self.camera, "exposure"):
                self.camera.exposure = v

        # Adjust at most one per frame; after exposure change, cooldown gain so we don't alternate E-down then G-down and overshoot.
        if self.auto_exposure_enabled:
            if self.auto_exposure.adjust(
                data,
                get_exposure,
                set_exposure,
                self.display.set_exposure_value,
            ):
                self.gain_cooldown_frames_remaining = 3
                return
        if self.auto_gain_enabled and self.gain_cooldown_frames_remaining <= 0:
            self.auto_gain.adjust(
                data,
                get_gain,
                set_gain,
                self.display.set_gain_value,
            )
