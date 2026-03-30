"""Configuration management for PySpectrometer3."""

import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tomli_w
import tomllib


def app_config_dir() -> Path:
    """Base directory for user config (~/.config/itohio/spectral)."""
    return Path.home() / ".config" / "itohio" / "spectral"


def resolve_explicit_config_path(path: str | Path) -> Path:
    """Normalize a user-supplied config path (CLI, env, tools).

    If the string form starts with ``!``, the rest is taken as a path relative to
    :func:`app_config_dir` (the same directory as the default ``config.toml``),
    e.g. ``!garden.toml`` → ``<app_config_dir>/garden.toml``. A lone ``!`` maps
    to :func:`default_main_config_path`. Otherwise behaves like
    ``Path(...).expanduser()``.
    """
    s = str(path).strip()
    if s.startswith("!"):
        rest = s[1:].lstrip("/\\")
        if not rest:
            return default_main_config_path()
        return (app_config_dir() / rest).expanduser()
    return Path(s).expanduser()


def config_search_paths(explicit: Path | None = None) -> list[Path]:
    """Return paths to try for config, in order.

    If *explicit* is set (CLI ``--config`` / ``-c``), only that path is used (after
    :func:`resolve_explicit_config_path`, so ``!name`` is under :func:`app_config_dir`);
    it overrides ``PYSPECTROMETER_CONFIG`` and the usual default search order.
    Otherwise: env var, then ``~/.config/.../config.toml``, then cwd
    ``pyspectrometer.toml``.
    """
    if explicit is not None:
        return [resolve_explicit_config_path(explicit)]
    paths = []
    if p := os.environ.get("PYSPECTROMETER_CONFIG"):
        paths.append(resolve_explicit_config_path(p))
    paths.append(app_config_dir() / "config.toml")
    paths.append(Path.cwd() / "pyspectrometer.toml")
    return paths


def default_config_path(explicit: Path | None = None) -> Path:
    """Return primary config path (for display). First in search order."""
    return config_search_paths(explicit)[0]


def explicit_config_path_from_argv(argv: list[str] | None = None) -> Path | None:
    """First ``--config`` or ``-c`` value in *argv* (default: ``sys.argv`` tail).

    Use for Poetry scripts and tools that do not use :mod:`argparse`, so they
    load/save the same file as ``python -m pyspectrometer --config ...``.
    """
    if argv is None:
        argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] in ("--config", "-c"):
            if i + 1 < len(argv):
                nxt = argv[i + 1]
                if not nxt.startswith("-"):
                    return resolve_explicit_config_path(nxt)
            return None
        i += 1
    return None


def load_config(path: Path | None = None) -> tuple["Config", Path | None]:
    """Load config from file, merging with defaults.

    Args:
        path: Explicit config file path. If None, searches default locations.

    Returns:
        (Config, path_for_save). ``path_for_save`` is the file that was read when one
        exists; if *path* was given but the file is missing, returns that expanded path
        so callers persist to the same file on first save. If implicit search finds no
        file, returns None (``save_config`` then uses the default main config path).
    """
    config = Config()
    for p in config_search_paths(path):
        if p.exists():
            with open(p, "rb") as f:
                data = tomllib.load(f)
            _apply_config(config, data)
            return config, p
    if path is not None:
        return config, resolve_explicit_config_path(path)
    return config, None


def parse_window_geometry(s: str) -> tuple[int, int]:
    """Parse a display size string: ``WxH`` with ``x``, ``X``, ``*``, comma, or Unicode ``×``.

    Raises:
        ValueError: If the string does not match two positive integers.

    Returns:
        ``(width, height)`` in pixels.
    """
    raw = s.strip().lower().replace("*", "x")
    m = re.fullmatch(r"(\d+)\s*[x×,]\s*(\d+)", raw)
    if not m:
        raise ValueError(f"expected WxH (e.g. 1280x720), got {s!r}")
    w, h = int(m.group(1)), int(m.group(2))
    if w < 1 or h < 1:
        raise ValueError(f"width and height must be positive, got {w}x{h}")
    return w, h


def default_main_config_path() -> Path:
    """Canonical path for the main (measurement) config."""
    return (app_config_dir() / "config.toml").resolve()


def csv_viewer_config_path() -> Path:
    """Canonical path for the CSV viewer config (separate from main config)."""
    return (app_config_dir() / "csv_viewer.toml").resolve()


def load_csv_viewer_config() -> tuple["Config", Path]:
    """Load config for the CSV viewer.

    Loads the main config as baseline (calibration, sensitivity, processing),
    then overlays any previously-saved CSV viewer settings, then enforces the
    CSV viewer display preset.  Saves always go to csv_viewer.toml so the main
    config (used by measure/calibrate modes) is never touched.

    Returns:
        (config, save_path) where save_path is csv_viewer_config_path().
    """
    config, _ = load_config(None)
    config.apply_csv_viewer_preset()

    csv_path = csv_viewer_config_path()
    if csv_path.exists():
        with open(csv_path, "rb") as f:
            data = tomllib.load(f)
        _apply_config(config, data)
        config.apply_csv_viewer_preset()

    return config, csv_path


def save_config(config: "Config", config_path: Path | None) -> bool:
    """Write config to file including calibration data.

    Saves to *config_path* when provided, otherwise to the default main config
    location (~/.config/itohio/spectral/config.toml).

    Args:
        config: Config to save.
        config_path: Explicit destination path. None → default main config.

    Returns:
        True if saved successfully.
    """
    path = config_path.resolve() if config_path else default_main_config_path()
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
        # Single source of truth: fields come from CameraConfig (asdict); add fields on the dataclass only.
        "camera": asdict(config.camera),
        "display": {
            "fullscreen": config.display.fullscreen,
            "graph_height": config.display.graph_height,
            "preview_height": config.display.preview_height,
            "message_height": config.display.message_height,
            "window_width": config.display.window_width,
            "font_scale": config.display.font_scale,
            "text_thickness": config.display.text_thickness,
            "graph_label_font": config.display.graph_label_font,
            "graph_label_font_scale": config.display.graph_label_font_scale,
            "status_col1_x": config.display.status_col1_x,
            "status_col2_x": config.display.status_col2_x,
        },
        "measurement": {
            "auto_viewport": config.measurement.auto_viewport,
            "viewport_wl_min": config.measurement.viewport_wl_min,
            "viewport_wl_max": config.measurement.viewport_wl_max,
        },
        "color_science": {
            "auto_viewport": config.color_science.auto_viewport,
            "viewport_wl_min": config.color_science.viewport_wl_min,
            "viewport_wl_max": config.color_science.viewport_wl_max,
        },
        "calibration": {
            "cal_pixels": list(config.calibration.cal_pixels),
            "cal_wavelengths": [float(w) for w in config.calibration.cal_wavelengths],
            "rotation_angle": config.calibration.rotation_angle,
            "spectrum_y_center": config.calibration.spectrum_y_center,
            "perpendicular_width": config.calibration.perpendicular_width,
            "monotonicity_threshold_nm": config.calibration.monotonicity_threshold_nm,
            "default_pixels": list(config.calibration.default_pixels),
            "default_wavelengths": [float(w) for w in config.calibration.default_wavelengths],
        },
        "processing": {
            "savgol_window": config.processing.savgol_window,
            "savgol_poly": config.processing.savgol_poly,
            "peak_min_distance": config.processing.peak_min_distance,
            "peak_threshold": config.processing.peak_threshold,
            "peak_max_count": config.processing.peak_max_count,
            "peak_include_region_half_width": config.processing.peak_include_region_half_width,
            "pixel_rows_to_average": config.processing.pixel_rows_to_average,
        },
        "export": {
            "output_dir": str(config.export.output_dir),
            "timestamp_format": config.export.timestamp_format,
            "time_format": config.export.time_format,
            "reference_dirs": [str(p) for p in config.export.reference_dirs],
            "min_wavelength": config.export.min_wavelength,
        },
        "waterfall": {
            "enabled": config.waterfall.enabled,
            "viewport_wl_min": config.waterfall.viewport_wl_min,
            "viewport_wl_max": config.waterfall.viewport_wl_max,
            "contrast": config.waterfall.contrast,
            "brightness": config.waterfall.brightness,
        },
        "extraction": {
            "method": config.extraction.method,
            "rotation_angle": config.extraction.rotation_angle,
            "perpendicular_width": config.extraction.perpendicular_width,
            "spectrum_y_center": config.extraction.spectrum_y_center,
            "background_percentile": config.extraction.background_percentile,
            "frame_black_strip_height": config.extraction.frame_black_strip_height,
        },
        "auto": {
            "peak_smoothing_period_sec": config.auto.peak_smoothing_period_sec,
            "max_adjust_rate_hz": config.auto.max_adjust_rate_hz,
        },
        "hardware": {
            "led_pin": config.hardware.led_pin,
            "led_pwm_frequency_hz": config.hardware.led_pwm_frequency_hz,
        },
        "sensitivity": {
            "use_custom_curve": config.sensitivity.use_custom_curve,
            "custom_wavelengths": [float(w) for w in config.sensitivity.custom_wavelengths],
            "custom_values": [float(v) for v in config.sensitivity.custom_values],
            "calibration_reference": config.sensitivity.calibration_reference,
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
        if "monotonicity_threshold_nm" in cal:
            config.calibration.monotonicity_threshold_nm = float(cal["monotonicity_threshold_nm"])
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
        if "min_wavelength" in exp:
            config.export.min_wavelength = float(exp["min_wavelength"])
    if "measurement" in data:
        apply(config.measurement, data["measurement"])
    if "color_science" in data:
        apply(config.color_science, data["color_science"])
    if "waterfall" in data:
        apply(config.waterfall, data["waterfall"])
    if "extraction" in data:
        apply(config.extraction, data["extraction"])
    if "auto" in data:
        apply(config.auto, data["auto"])
    if "hardware" in data:
        hw = dict(data["hardware"])
        if "led_pin" not in hw and "led_bcm_pin" in hw:
            hw["led_pin"] = hw["led_bcm_pin"]
        apply(config.hardware, hw)
    if "sensitivity" in data:
        sens = data["sensitivity"]
        if "use_custom_curve" in sens:
            config.sensitivity.use_custom_curve = bool(sens["use_custom_curve"])
        if "custom_wavelengths" in sens:
            config.sensitivity.custom_wavelengths = [float(w) for w in sens["custom_wavelengths"]]
        if "custom_values" in sens:
            config.sensitivity.custom_values = [float(v) for v in sens["custom_values"]]
        if "calibration_reference" in sens:
            config.sensitivity.calibration_reference = str(sens["calibration_reference"] or "")


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

    # Mirror camera image left-right after capture (all backends)
    flip_horizontal: bool = False

    # OpenCV backend only: device index, URL, or v4l:/dev/videoN (ignored by Picamera2)
    opencv_source: int | str = 0


@dataclass
class DisplayConfig:
    """Display-related configuration.

    Canvas is built at (window_width, stack_height) - pixel perfect, no scaling.
    Default 640x400 for Waveshare 640 displays. Layout sums to stack_height.
    """

    fullscreen: bool = False
    graph_height: int = 280
    preview_height: int = 60
    message_height: int = 60

    # Canvas/window dimensions. No scaling - built at this size.
    window_width: int = 640

    # Fonts: tuned for small displays (readable at 640px width)
    font_scale: float = 0.35
    text_thickness: int = 1
    graph_label_font: int = 1  # cv2.FONT_HERSHEY_PLAIN
    graph_label_font_scale: float = 0.9
    status_col1_x: int = 340
    status_col2_x: int = 500

    @property
    def stack_height(self) -> int:
        """Total height of the stacked display (canvas height)."""
        return self.graph_height + self.preview_height + self.message_height


@dataclass
class MeasurementConfig:
    """Default horizontal wavelength window for spectrum modes (not calibration or Raman)."""

    auto_viewport: bool = True
    viewport_wl_min: float = 350.0
    viewport_wl_max: float = 950.0


@dataclass
class ColorScienceConfig:
    """Default horizontal wavelength window for color science mode."""

    auto_viewport: bool = True
    viewport_wl_min: float = 400.0
    viewport_wl_max: float = 700.0


@dataclass
class ProcessingConfig:
    """Signal processing configuration."""

    savgol_window: int = 17
    savgol_poly: int = 7
    savgol_poly_min: int = 0
    savgol_poly_max: int = 15

    peak_min_distance: int = 5  # Min pixels between peaks
    peak_min_distance_min: int = 1
    peak_min_distance_max: int = 100

    peak_threshold: int = 10  # Min height as % of range (0-100). Peaks below are ignored.
    peak_threshold_min: int = 0
    peak_threshold_max: int = 100

    peak_max_count: int = 10  # Max peaks to display. All above threshold, sorted by size, top N shown.
    peak_max_count_min: int = 1
    peak_max_count_max: int = 50

    # Click-to-include: half-width in data pixels for region around click
    peak_include_region_half_width: int = 25

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

    # Monotonicity tolerance (nm): allow diff up to -threshold before rejecting Cauchy fit.
    # Set to 0.1 nm — well below the ~0.29 nm/pixel sensor resolution, but safely above
    # floating-point noise in polynomial evaluation (~0.001–0.01 nm range).
    monotonicity_threshold_nm: float = 0.5

    # Defaults when no calibration in config
    default_pixels: tuple[int, ...] = (0, 640, 1280)
    default_wavelengths: tuple[float, ...] = (380.0, 560.0, 750.0)


@dataclass
class SensitivityConfig:
    """User-fitted spectral sensitivity curve (saved with wavelength calibration)."""

    use_custom_curve: bool = False
    custom_wavelengths: list[float] = field(default_factory=list)
    custom_values: list[float] = field(default_factory=list)
    # Reference illuminant name used for last CRR fit (e.g. CIE Illuminant A); empty if datasheet only
    calibration_reference: str = ""


@dataclass
class ExportConfig:
    """Export-related configuration."""

    output_dir: Path = field(default_factory=lambda: Path("output"))
    timestamp_format: str = "%Y%m%d--%H%M%S"
    time_format: str = "%H:%M:%S"
    reference_dirs: list[Path] = field(
        default_factory=lambda: [Path("data/reference"), Path("output")]
    )
    # CSV: omit points below this wavelength (nm). 0 = full axis. Modes that trim read this via context.
    min_wavelength: float = 300.0


@dataclass
class WaterfallConfig:
    """Waterfall strip and its spectrum plot: on/off, wavelength span, contrast, brightness."""

    enabled: bool = False
    # Default visible wavelength interval (nm) on the spectrum graph tied to this feature.
    viewport_wl_min: float = 350.0
    viewport_wl_max: float = 950.0
    contrast: float = 2.5
    brightness: int = 10


@dataclass
class HardwareConfig:
    """GPIO / board peripherals (spectrometer lamp on Raspberry Pi Zero)."""

    # Broadcom GPIO number (same scheme as gpiozero / `gpio readall`), not the physical header pin index.
    # Default 22 is ordinary GPIO; PWM here is software-timed via gpiozero. GPIO 18 can use SoC hardware
    # PWM when enabled in device tree; this app still drives brightness through PWMLED unless you change drivers.
    led_pin: int = 22
    # PWM frequency requested for gpiozero (Hz); on many pins this is implemented as software PWM.
    led_pwm_frequency_hz: int = 3000


@dataclass
class AutoConfig:
    """Auto gain / auto exposure configuration."""

    # Exponential smoothing of spectrum peak (seconds). Shorter = faster reaction to over/underexposure.
    peak_smoothing_period_sec: float = 0.02
    # Max adjustments per second (rate limit). Higher = faster convergence, more risk of overshoot.
    max_adjust_rate_hz: float = 20.0


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
    # Rows (top and bottom) used for per-column bias estimate; bias removal is always applied in extract.
    frame_black_strip_height: int = 8


@dataclass
class DisplayRuntimeView:
    """References UI-facing slices and titles from :class:`Config` for the display layer.

    Holds the same sub-objects as the aggregate config; mutations apply to persisted settings.
    """

    display: DisplayConfig
    waterfall: WaterfallConfig
    measurement: MeasurementConfig
    color_science: ColorScienceConfig
    processing: ProcessingConfig
    spectrograph_title: str
    waterfall_title: str

    @classmethod
    def from_config(cls, config: "Config") -> "DisplayRuntimeView":
        return cls(
            display=config.display,
            waterfall=config.waterfall,
            measurement=config.measurement,
            color_science=config.color_science,
            processing=config.processing,
            spectrograph_title=config.spectrograph_title,
            waterfall_title=config.waterfall_title,
        )


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
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    color_science: ColorScienceConfig = field(default_factory=ColorScienceConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    auto: AutoConfig = field(default_factory=AutoConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)

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
        config.waterfall.enabled = waterfall

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

        Layout: message_height 60 + preview_height 60 + graph_height 280 = 400px.
        """
        self.camera.frame_width = 640
        self.camera.frame_height = 400

        self.display.window_width = 640
        self.display.graph_height = 280
        self.display.preview_height = 60
        self.display.message_height = 60

        self.display.font_scale = 0.35
        self.display.text_thickness = 1
        self.display.graph_label_font_scale = 0.9
        self.display.status_col1_x = 340
        self.display.status_col2_x = 500

        self.calibration.default_pixels = (0, 320, 640)

        self.processing.peak_min_distance = 5

    def apply_csv_viewer_preset(self) -> None:
        """Apply settings for the camera-less CSV viewer (1024x800).

        No camera strip — full window height goes to the spectrum graph.
        """
        self.display.window_width = 1024
        self.display.graph_height = 740
        self.display.preview_height = 0
        self.display.message_height = 60

    def apply_desktop_preset(self) -> None:
        """Apply settings for desktop/large displays (1280x720)."""
        self.display.window_width = 1280
        self.display.graph_height = 320
        self.display.preview_height = 80
        self.display.message_height = 80

        self.display.font_scale = 0.4
        self.display.text_thickness = 1
        self.display.graph_label_font_scale = 1.0
        self.display.status_col1_x = 490
        self.display.status_col2_x = 800

    def apply_window_geometry(self, width: int, height: int) -> None:
        """Set display canvas to ``width`` × ``height`` with a consistent stacked layout.

        Splits *height* into message bar, preview strip, and graph. Interpolates
        status column X positions and font scales between Waveshare (640px) and
        desktop (1280px); for wider windows uses proportional column placement.
        Does not change camera or processing settings.
        """
        w = max(320, min(width, 7680))
        h = max(180, min(height, 4320))
        self.display.window_width = w

        min_bar = 36
        max_bar = min(120, h // 3)
        bar = max(min_bar, min(max_bar, max(h // 6, min_bar)))
        graph_h = h - 2 * bar
        if graph_h < 100:
            bar = max(36, (h - 100) // 2)
            graph_h = h - 2 * bar

        self.display.message_height = bar
        self.display.preview_height = bar
        self.display.graph_height = graph_h

        self.display.text_thickness = 1
        if w <= 640:
            self.display.font_scale = 0.35
            self.display.graph_label_font_scale = 0.9
            c1, c2 = 340, 500
        elif w < 1280:
            t = (w - 640) / (1280 - 640)
            self.display.font_scale = 0.35 + t * (0.4 - 0.35)
            self.display.graph_label_font_scale = 0.9 + t * (1.0 - 0.9)
            c1 = int(340 + t * (490 - 340))
            c2 = int(500 + t * (800 - 500))
        else:
            self.display.font_scale = min(0.5, 0.4 + (w - 1280) / 8000.0)
            self.display.graph_label_font_scale = min(1.15, 1.0 + (w - 1280) / 8000.0)
            c1 = int(w * 0.383)
            c2 = int(w * 0.625)

        c1 = max(10, min(c1, w - 80))
        c2 = max(c1 + 40, min(c2, w - 10))
        self.display.status_col1_x = c1
        self.display.status_col2_x = c2

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
