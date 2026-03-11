"""Configuration management for PySpectrometer3."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


def config_search_paths(explicit: Path | None = None) -> list[Path]:
    """Return paths to try for config, in order."""
    if explicit is not None:
        return [explicit.expanduser()]
    paths = []
    if p := os.environ.get("PYSPECTROMETER_CONFIG"):
        paths.append(Path(p).expanduser())
    paths.append(Path.cwd() / "pyspectrometer.toml")
    paths.append(Path.home() / ".config" / "pyspectrometer" / "config.toml")
    return paths


def default_config_path(explicit: Path | None = None) -> Path:
    """Return primary config path (for display). First in search order."""
    return config_search_paths(explicit)[0]


def load_config(path: Path | None = None) -> tuple["Config", Path | None]:
    """Load config from file, merging with defaults.

    Args:
        path: Explicit config file path. If None, searches default locations.

    Returns:
        (Config, loaded_path). loaded_path is None if no file was loaded.
    """
    config = Config()
    for p in _config_search_paths(path):
        if p.exists():
            with open(p, "rb") as f:
                data = tomllib.load(f)
            _apply_config(config, data)
            return config, p
    return config, None


def _apply_config(config: "Config", data: dict) -> None:

    def apply(obj: object, d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict) and k != "data_file" and hasattr(obj, k):
                apply(getattr(obj, k), v)
            elif hasattr(obj, k):
                setattr(obj, k, v)

    if "camera" in data:
        apply(config.camera, data["camera"])
    if "display" in data:
        apply(config.display, data["display"])
    if "calibration" in data:
        cal = data["calibration"]
        if "data_file" in cal:
            config.calibration.data_file = Path(cal["data_file"])
        if "default_pixels" in cal:
            config.calibration.default_pixels = tuple(cal["default_pixels"])
        if "default_wavelengths" in cal:
            config.calibration.default_wavelengths = tuple(cal["default_wavelengths"])


@dataclass
class CameraConfig:
    """Camera-related configuration.

    Default 1280x720 matches OV9281 and common spectrometer sensors.
    OV9281 modes: 640x400, 1280x720, 1280x800 (no 800x600).
    """

    frame_width: int = 1280
    frame_height: int = 720
    gain: float = 10.0
    gain_min: float = 0.0
    gain_max: float = 50.0
    fps: int = 30

    # Monochrome camera support (default: True for spectrometer use)
    monochrome: bool = True
    bit_depth: int = 10  # 8, 10, or 16 (for monochrome cameras)


@dataclass
class DisplayConfig:
    """Display-related configuration."""

    fullscreen: bool = False
    waterfall_enabled: bool = False
    graph_height: int = 320
    preview_height: int = 80
    message_height: int = 80

    # Window size (canvas will be scaled to fit)
    window_width: int = 1280
    window_height: int = 720

    # UI scaling for small displays
    font_scale: float = 0.4
    text_thickness: int = 1
    status_col1_x: int = 490
    status_col2_x: int = 800

    @property
    def stack_height(self) -> int:
        """Total height of the stacked display."""
        return self.graph_height + self.preview_height + self.message_height


@dataclass
class ProcessingConfig:
    """Signal processing configuration."""

    savgol_window: int = 17
    savgol_poly: int = 7
    savgol_poly_min: int = 0
    savgol_poly_max: int = 15

    peak_min_distance: int = 15  # Lower for sharp Hg lines (577/579 ~7 px apart)
    peak_min_distance_min: int = 1
    peak_min_distance_max: int = 100

    peak_threshold: int = 20
    peak_threshold_min: int = 0
    peak_threshold_max: int = 100

    pixel_rows_to_average: int = 3


@dataclass
class CalibrationConfig:
    """Calibration-related configuration."""

    data_file: Path = field(default_factory=lambda: Path("caldata.txt"))
    default_pixels: tuple[int, ...] = (0, 640, 1280)
    default_wavelengths: tuple[float, ...] = (380.0, 560.0, 750.0)


@dataclass
class ExportConfig:
    """Export-related configuration."""

    output_dir: Path = field(default_factory=lambda: Path("."))
    timestamp_format: str = "%Y%m%d--%H%M%S"
    time_format: str = "%H:%M:%S"


@dataclass
class WaterfallConfig:
    """Waterfall display configuration."""

    contrast: float = 2.5
    brightness: int = 10


@dataclass
class ExtractionConfig:
    """Spectrum extraction configuration."""

    method: str = "weighted_sum"  # "median", "weighted_sum", "gaussian"
    rotation_angle: float = 0.0  # degrees, loaded from calibration
    perpendicular_width: int = 20  # pixels to sample perpendicular to axis
    perpendicular_width_min: int = 5
    perpendicular_width_max: int = 100
    background_percentile: float = 10.0  # for background subtraction
    spectrum_y_center: int = 0  # 0 = auto (frame center)


@dataclass
class Config:
    """Main configuration container for PySpectrometer3.

    This dataclass aggregates all configuration options for the spectrometer
    application, providing sensible defaults that match the original behavior.
    """

    camera: CameraConfig = field(default_factory=CameraConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    waterfall: WaterfallConfig = field(default_factory=WaterfallConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)

    # Window titles
    spectrograph_title: str = "PySpectrometer 3 - Spectrograph"
    waterfall_title: str = "PySpectrometer 3 - Waterfall"

    @classmethod
    def from_args(
        cls,
        base: "Config | None" = None,
        fullscreen: bool = False,
        waterfall: bool = False,
        waveshare: bool = False,
        gain: float | None = None,
        width: int | None = None,
        height: int | None = None,
        monochrome: bool = False,
        bit_depth: int = 10,
    ) -> "Config":
        """Create or merge configuration from command-line arguments."""
        config = base if base is not None else cls()

        config.display.fullscreen = fullscreen
        config.display.waterfall_enabled = waterfall

        if gain is not None:
            config.camera.gain = gain
        if width is not None:
            config.camera.frame_width = width
        if height is not None:
            config.camera.frame_height = height

        config.camera.monochrome = monochrome
        config.camera.bit_depth = bit_depth

        if waveshare:
            config.apply_waveshare_preset()

        return config

    def apply_waveshare_preset(self) -> None:
        """Apply settings optimized for Waveshare display (640x400).

        Layout breakdown for 400px height:
        - message_height: 60 (control bar with 2 rows of buttons)
        - preview_height: 60 (camera preview window)
        - graph_height: 280 (spectrum graph)
        Total: 400px
        """
        self.camera.frame_width = 640
        self.camera.frame_height = 400

        self.display.window_width = 640
        self.display.window_height = 400
        self.display.graph_height = 280
        self.display.preview_height = 60
        self.display.message_height = 60

        self.display.font_scale = 0.35
        self.display.text_thickness = 1
        self.display.status_col1_x = 340
        self.display.status_col2_x = 500

        self.calibration.default_pixels = (0, 320, 640)

        self.processing.peak_min_distance = 15  # Sharp Hg lines

    @classmethod
    def waveshare_35(cls) -> "Config":
        """Create configuration for Waveshare 3.5\" display (640x480)."""
        config = cls()
        config.apply_waveshare_preset()
        return config

    @classmethod
    def standard_800(cls) -> "Config":
        """Create standard 800x480 configuration."""
        return cls()
