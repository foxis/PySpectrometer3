"""Configuration management for PySpectrometer3."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import tomllib
import tomli_w


def config_search_paths(explicit: Path | None = None) -> list[Path]:
    """Return paths to try for config, in order. Primary: user config dir."""
    if explicit is not None:
        return [explicit.expanduser()]
    paths = []
    if p := os.environ.get("PYSPECTROMETER_CONFIG"):
        paths.append(Path(p).expanduser())
    paths.append(Path.home() / ".config" / "pyspectrometer" / "config.toml")
    paths.append(Path.cwd() / "pyspectrometer.toml")
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
    for p in config_search_paths(path):
        if p.exists():
            with open(p, "rb") as f:
                data = tomllib.load(f)
            _apply_config(config, data)
            return config, p
    return config, None


def save_config(config: "Config", config_path: Path | None) -> bool:
    """Write config to file including calibration data.

    Always saves to user config directory (~/.config/pyspectrometer/config.toml)
    so project directory is not polluted. config_path is ignored for write location.

    Args:
        config: Config to save
        config_path: Ignored; kept for API compatibility.

    Returns:
        True if saved successfully
    """
    path = Path.home() / ".config" / "pyspectrometer" / "config.toml"
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = _config_to_dict(config)
    try:
        with open(path, "wb") as f:
            tomli_w.dump(doc, f)
        return True
    except OSError as e:
        print(f"Failed to save config: {e}")
        return False


def _config_to_dict(config: "Config") -> dict:
    """Build TOML-serializable dict from Config."""
    return {
        "camera": {
            "frame_width": config.camera.frame_width,
            "frame_height": config.camera.frame_height,
            "gain": config.camera.gain,
            "fps": config.camera.fps,
            "monochrome": config.camera.monochrome,
            "bit_depth": config.camera.bit_depth,
        },
        "display": {
            "fullscreen": config.display.fullscreen,
            "waterfall_enabled": config.display.waterfall_enabled,
            "graph_height": config.display.graph_height,
            "preview_height": config.display.preview_height,
            "message_height": config.display.message_height,
            "window_width": config.display.window_width,
            "window_height": config.display.window_height,
            "font_scale": config.display.font_scale,
            "text_thickness": config.display.text_thickness,
            "status_col1_x": config.display.status_col1_x,
            "status_col2_x": config.display.status_col2_x,
        },
        "calibration": {
            "cal_pixels": list(config.calibration.cal_pixels),
            "cal_wavelengths": [float(w) for w in config.calibration.cal_wavelengths],
            "rotation_angle": config.calibration.rotation_angle,
            "spectrum_y_center": config.calibration.spectrum_y_center,
            "perpendicular_width": config.calibration.perpendicular_width,
            "default_pixels": list(config.calibration.default_pixels),
            "default_wavelengths": [float(w) for w in config.calibration.default_wavelengths],
        },
        "processing": {
            "savgol_window": config.processing.savgol_window,
            "savgol_poly": config.processing.savgol_poly,
            "peak_min_distance": config.processing.peak_min_distance,
            "peak_threshold": config.processing.peak_threshold,
            "peak_max_count": config.processing.peak_max_count,
            "pixel_rows_to_average": config.processing.pixel_rows_to_average,
        },
        "export": {
            "output_dir": str(config.export.output_dir),
            "timestamp_format": config.export.timestamp_format,
            "time_format": config.export.time_format,
            "reference_dirs": [str(p) for p in config.export.reference_dirs],
        },
        "waterfall": {
            "contrast": config.waterfall.contrast,
            "brightness": config.waterfall.brightness,
        },
        "extraction": {
            "method": config.extraction.method,
            "rotation_angle": config.extraction.rotation_angle,
            "perpendicular_width": config.extraction.perpendicular_width,
            "spectrum_y_center": config.extraction.spectrum_y_center,
            "background_percentile": config.extraction.background_percentile,
        },
    }


def _apply_config(config: "Config", data: dict) -> None:

    def apply(obj: object, d: dict) -> None:
        for k, v in d.items():
            if isinstance(v, dict) and hasattr(obj, k):
                apply(getattr(obj, k), v)
            elif hasattr(obj, k):
                setattr(obj, k, v)

    if "camera" in data:
        apply(config.camera, data["camera"])
    if "display" in data:
        apply(config.display, data["display"])
    if "calibration" in data:
        cal = data["calibration"]
        if "cal_pixels" in cal:
            config.calibration.cal_pixels = list(cal["cal_pixels"])
        if "cal_wavelengths" in cal:
            config.calibration.cal_wavelengths = [float(w) for w in cal["cal_wavelengths"]]
        if "rotation_angle" in cal:
            config.calibration.rotation_angle = float(cal["rotation_angle"])
        if "spectrum_y_center" in cal:
            config.calibration.spectrum_y_center = int(cal["spectrum_y_center"])
        if "perpendicular_width" in cal:
            config.calibration.perpendicular_width = int(cal["perpendicular_width"])
        if "default_pixels" in cal:
            config.calibration.default_pixels = tuple(cal["default_pixels"])
        if "default_wavelengths" in cal:
            config.calibration.default_wavelengths = tuple(cal["default_wavelengths"])
    if "processing" in data:
        apply(config.processing, data["processing"])
    if "export" in data:
        exp = data["export"]
        if "output_dir" in exp:
            config.export.output_dir = Path(exp["output_dir"])
        if "timestamp_format" in exp:
            config.export.timestamp_format = exp["timestamp_format"]
        if "time_format" in exp:
            config.export.time_format = exp["time_format"]
        if "reference_dirs" in exp:
            config.export.reference_dirs = [Path(p) for p in exp["reference_dirs"]]
    if "waterfall" in data:
        apply(config.waterfall, data["waterfall"])
    if "extraction" in data:
        apply(config.extraction, data["extraction"])


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

    # Window size (canvas scaled to fit). Default 640x400 for Waveshare 640 displays.
    window_width: int = 640
    window_height: int = 400

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

    peak_threshold: int = 20  # Min height as % of range (0-100). Peaks below are ignored.
    peak_threshold_min: int = 0
    peak_threshold_max: int = 100

    peak_max_count: int = 10  # Max peaks to display. All above threshold, sorted by size, top N shown.
    peak_max_count_min: int = 1
    peak_max_count_max: int = 50

    pixel_rows_to_average: int = 3


@dataclass
class CalibrationConfig:
    """Calibration-related configuration.

    Calibration data (cal_pixels, cal_wavelengths, rotation_angle, etc.)
    is stored in the config file alongside other settings.
    """

    # Loaded/saved calibration data
    cal_pixels: list[int] = field(default_factory=lambda: [0, 640, 1280])
    cal_wavelengths: list[float] = field(default_factory=lambda: [380.0, 560.0, 750.0])
    rotation_angle: float = 0.0
    spectrum_y_center: int = 0
    perpendicular_width: int = 20

    # Defaults when no calibration in config
    default_pixels: tuple[int, ...] = (0, 640, 1280)
    default_wavelengths: tuple[float, ...] = (380.0, 560.0, 750.0)


@dataclass
class ExportConfig:
    """Export-related configuration."""

    output_dir: Path = field(default_factory=lambda: Path("output"))
    timestamp_format: str = "%Y%m%d--%H%M%S"
    time_format: str = "%H:%M:%S"
    reference_dirs: list[Path] = field(
        default_factory=lambda: [Path("data/reference"), Path("output")]
    )


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
        monochrome: bool = False,
        bit_depth: int = 10,
    ) -> "Config":
        """Create or merge configuration from command-line arguments.

        Height is derived from width: 640->400, 1280->720 (OV9281 modes).
        """
        config = base if base is not None else cls()

        config.display.fullscreen = fullscreen
        config.display.waterfall_enabled = waterfall

        if gain is not None:
            config.camera.gain = gain
        if width is not None:
            config.camera.frame_width = width
            config.camera.frame_height = 400 if width == 640 else 720

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
