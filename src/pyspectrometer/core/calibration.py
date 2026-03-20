"""Wavelength calibration for PySpectrometer3."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import Config, SensitivityConfig

try:
    from scipy.interpolate import PchipInterpolator

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


@dataclass
class CalibrationResult:
    """Result of calibration computation."""

    wavelengths: np.ndarray
    order: int
    coefficients: np.ndarray
    r_squared: float | None = None
    rotation_angle: float = 0.0
    spectrum_y_center: int = 0
    perpendicular_width: int = 20

    @property
    def is_accurate(self) -> bool:
        """Check if calibration is considered accurate (4+ points)."""
        return self.order > 0

    @property
    def status_message(self) -> str:
        """Human-readable calibration status."""
        if self.order == 0:
            return "UNCALIBRATED"
        return "OK"

    @property
    def detail_message(self) -> str:
        """Human-readable calibration detail."""
        if self.order == 0:
            return "Perform Calibration!"
        return "Cauchy (prism)"


@dataclass
class GraticuleData:
    """Data for rendering calibrated graticule lines."""

    tens: list[int] = field(default_factory=list)
    fifties: list[tuple[int, int]] = field(default_factory=list)
    unit: str = "nm"


class Calibration:
    """Handles wavelength calibration for the spectrometer.

    Calibration data is stored in the config file (pyspectrometer.toml).
    """

    def __init__(
        self,
        width: int,
        config: "Config",
        config_path: Path | None = None,
        height: int = 480,
        default_pixels: tuple[int, ...] = (0, 400, 800),
        default_wavelengths: tuple[float, ...] = (380.0, 560.0, 750.0),
    ):
        self.width = width
        self.height = height
        self._config = config
        self._config_path = config_path
        self.default_pixels = default_pixels
        self.default_wavelengths = default_wavelengths

        self._result: CalibrationResult | None = None
        self._graticule: GraticuleData | None = None

        cal = config.calibration
        self._rotation_angle = cal.rotation_angle
        self._spectrum_y_center = cal.spectrum_y_center or height // 2
        self._perpendicular_width = cal.perpendicular_width
        self._cal_pixels = list(cal.cal_pixels) if cal.cal_pixels else list(default_pixels)
        self._cal_wavelengths = (
            list(cal.cal_wavelengths) if cal.cal_wavelengths else list(default_wavelengths)
        )

    @property
    def sensitivity_settings(self) -> "SensitivityConfig":
        """Sensitivity section of app config (curve + CRR reference name)."""
        return self._config.sensitivity

    @property
    def wavelengths(self) -> np.ndarray:
        """Get calibrated wavelength array."""
        if self._result is None:
            self.load()
        return self._result.wavelengths

    @property
    def result(self) -> CalibrationResult:
        """Get calibration result."""
        if self._result is None:
            self.load()
        return self._result

    @property
    def graticule(self) -> GraticuleData:
        """Get graticule data for display."""
        if self._graticule is None:
            self._graticule = self._generate_graticule()
        return self._graticule

    def seed_wavelengths(self, wavelengths: np.ndarray) -> None:
        """Bypass polynomial fitting and use a pre-computed wavelength array directly.

        Used by the CSV viewer to inject the axis from a loaded CSV without touching
        the calibration config or running a polynomial fit.
        """
        self._result = CalibrationResult(
            wavelengths=np.asarray(wavelengths, dtype=np.float64),
            order=1,
            coefficients=np.array([]),
        )
        self._graticule = None

    def load(self) -> CalibrationResult:
        """Load calibration from config and compute wavelengths."""
        cal = self._config.calibration
        self._cal_pixels = list(cal.cal_pixels) if cal.cal_pixels else list(self.default_pixels)
        self._cal_wavelengths = (
            list(cal.cal_wavelengths) if cal.cal_wavelengths else list(self.default_wavelengths)
        )
        self._rotation_angle = cal.rotation_angle
        self._spectrum_y_center = cal.spectrum_y_center or self.height // 2
        self._perpendicular_width = cal.perpendicular_width

        has_errors = len(self._cal_pixels) < 4 or len(self._cal_pixels) != len(self._cal_wavelengths)
        if has_errors:
            self._cal_pixels = list(self.default_pixels)
            self._cal_wavelengths = list(self.default_wavelengths)
            print("Calibration: using default wavelength data (need 4+ points in config)")
        else:
            path = self._config_path or "default"
            print(f"Loading calibration from config ({path})")
            print(f"  rotation={self._rotation_angle:.2f}° y={self._spectrum_y_center} width={self._perpendicular_width}")

        self._result = self._compute_calibration(
            self._cal_pixels, self._cal_wavelengths, has_errors
        )
        self._graticule = None
        return self._result

    def save(
        self,
        pixel_data: list[int],
        wavelength_data: list[float],
        rotation_angle: float | None = None,
        spectrum_y_center: int | None = None,
        perpendicular_width: int | None = None,
    ) -> bool:
        """Save calibration data to config file.

        Args:
            pixel_data: List of pixel positions
            wavelength_data: List of corresponding wavelengths
            rotation_angle: Spectrum rotation angle in degrees
            spectrum_y_center: Y coordinate of spectrum center
            perpendicular_width: Width of perpendicular sampling region

        Returns:
            True if calibration was saved successfully
        """
        if err := _validate_cal_points(pixel_data, wavelength_data):
            print(err)
            return False

        if rotation_angle is not None:
            self._rotation_angle = rotation_angle
        if spectrum_y_center is not None:
            self._spectrum_y_center = spectrum_y_center
        if perpendicular_width is not None:
            self._perpendicular_width = perpendicular_width

        return self._write_config(pixel_data, wavelength_data)

    def save_extraction_params(
        self,
        rotation_angle: float,
        spectrum_y_center: int,
        perpendicular_width: int,
    ) -> bool:
        """Save only extraction parameters (preserves wavelength calibration).

        Args:
            rotation_angle: Spectrum rotation angle in degrees
            spectrum_y_center: Y coordinate of spectrum center
            perpendicular_width: Width of perpendicular sampling region

        Returns:
            True if saved successfully
        """
        self._rotation_angle = rotation_angle
        self._spectrum_y_center = spectrum_y_center
        self._perpendicular_width = perpendicular_width
        return self._write_config(self._cal_pixels, self._cal_wavelengths)

    def _write_config(
        self, pixel_data: list[int], wavelength_data: list[float]
    ) -> bool:
        """Update config with calibration data and save to file."""
        from ..config import save_config

        cal = self._config.calibration
        cal.cal_pixels = list(pixel_data)
        cal.cal_wavelengths = [float(w) for w in wavelength_data]
        cal.rotation_angle = self._rotation_angle
        cal.spectrum_y_center = self._spectrum_y_center
        cal.perpendicular_width = self._perpendicular_width

        if save_config(self._config, self._config_path):
            self._cal_pixels = list(pixel_data)
            self._cal_wavelengths = [float(w) for w in wavelength_data]
            self._result = self._compute_calibration(
                self._cal_pixels, self._cal_wavelengths, has_errors=False
            )
            self._graticule = None
            print(
                f"Calibration saved to config: "
                f"angle={self._rotation_angle:.2f}°, y={self._spectrum_y_center}, width={self._perpendicular_width}"
            )
            return True
        return False

    def save_sensitivity_curve(
        self,
        wavelengths: list[float],
        values: list[float],
        *,
        calibration_reference: str = "",
    ) -> bool:
        """Persist user-fitted sensitivity curve to config."""
        from ..config import save_config

        if len(wavelengths) != len(values) or len(wavelengths) < 4:
            print("[Sensitivity] Need at least 4 wavelength/value pairs")
            return False
        sens = self._config.sensitivity
        sens.use_custom_curve = True
        sens.custom_wavelengths = [float(w) for w in wavelengths]
        sens.custom_values = [float(v) for v in values]
        sens.calibration_reference = (calibration_reference or "").strip()
        if save_config(self._config, self._config_path):
            print(f"[Sensitivity] Saved user curve ({len(wavelengths)} points) to config")
            return True
        return False

    def clear_sensitivity_curve(self) -> bool:
        """Remove user curve from config; app uses datasheet CMOS only after reset."""
        from ..config import save_config

        sens = self._config.sensitivity
        sens.use_custom_curve = False
        sens.custom_wavelengths = []
        sens.custom_values = []
        sens.calibration_reference = ""
        if save_config(self._config, self._config_path):
            print("[Sensitivity] User curve cleared; use CMOS button to reload datasheet in session")
            return True
        return False

    @property
    def rotation_angle(self) -> float:
        """Get spectrum rotation angle in degrees."""
        return self._rotation_angle

    @property
    def spectrum_y_center(self) -> int:
        """Get spectrum Y center position."""
        return self._spectrum_y_center

    @property
    def perpendicular_width(self) -> int:
        """Get perpendicular sampling width."""
        return self._perpendicular_width

    @property
    def cal_pixels(self) -> list[int]:
        """Get current calibration pixel positions."""
        return self._cal_pixels.copy()

    @property
    def cal_wavelengths(self) -> list[float]:
        """Get current calibration wavelengths."""
        return self._cal_wavelengths.copy()

    def recalibrate(
        self,
        pixel_data: list[int],
        wavelength_data: list[float],
    ) -> bool:
        """Apply new calibration without saving to file.

        Useful for previewing calibration before committing.

        Args:
            pixel_data: List of pixel positions
            wavelength_data: List of corresponding wavelengths

        Returns:
            True if calibration was applied successfully
        """
        if err := _validate_cal_points(pixel_data, wavelength_data):
            print(err)
            return False

        self._cal_pixels = list(pixel_data)
        self._cal_wavelengths = list(wavelength_data)

        self._result = self._compute_calibration(pixel_data, wavelength_data, has_errors=False)
        self._graticule = None

        return True

    def _compute_calibration(
        self,
        pixels: list[int],
        wavelengths: list[float],
        has_errors: bool,
    ) -> CalibrationResult:
        """Compute wavelength calibration using prism dispersion model.

        Tries Cauchy-inverse fit first (1/λ² = poly(pixel)), physics-motivated for
        prism spectrometers and strictly monotonic. Falls back to PCHIP or linear
        if Cauchy fails (non-positive fit, non-monotonic, or <3 points).
        """
        # Sort by pixel
        sorted_pairs = sorted(zip(pixels, wavelengths), key=lambda p: p[0])
        px = np.array([p[0] for p in sorted_pairs], dtype=np.float64)
        wl = np.array([p[1] for p in sorted_pairs], dtype=np.float64)

        # Reject only if cal data has plateaus or reverses (allow strictly increasing or decreasing)
        diff_wl = np.diff(wl)
        if len(px) >= 2 and (np.any(diff_wl == 0) or (np.any(diff_wl > 0) and np.any(diff_wl < 0))):
            print("Cal points not strictly monotonic; using linear interpolation.")
            interp_wl = np.interp(np.arange(self.width), px, wl)
            method = "linear"
        else:
            interp_wl, method = self._try_cauchy_fit(px, wl)
            if method != "cauchy":
                interp_wl = self._fallback_interp(px, wl, method)

        wavelength_data = np.round(interp_wl, 6).astype(np.float64)
        order = 0 if has_errors else 3

        pixel_grid = np.arange(self.width, dtype=np.float64)
        predicted = np.interp(px, pixel_grid, wavelength_data)
        corr_matrix = np.corrcoef(wl, predicted)
        r_squared = float(corr_matrix[0, 1] ** 2) if not np.isnan(corr_matrix[0, 1]) else None

        coefficients = np.array([], dtype=np.float64)
        return CalibrationResult(
            wavelengths=wavelength_data,
            order=order,
            coefficients=coefficients,
            r_squared=r_squared,
            rotation_angle=self._rotation_angle,
            spectrum_y_center=self._spectrum_y_center,
            perpendicular_width=self._perpendicular_width,
        )

    def _try_cauchy_fit(self, px: np.ndarray, wl: np.ndarray) -> tuple[np.ndarray, str]:
        """Try Cauchy-inverse fit: 1/λ² = poly(pixel) → λ = 1/√poly.

        Evaluates the polynomial over the full pixel range [0, width-1] so
        wavelength mapping covers the entire sensor.
        """
        if len(px) < 3:
            return np.zeros(self.width), "linear"

        # 1/λ² in µm⁻² for numerical stability (nm→µm)
        inv_wl_sq = 1.0 / ((wl * 1e-3) ** 2)
        deg = min(2, len(px) - 1)
        try:
            coeffs = np.polyfit(px, inv_wl_sq, deg)
            poly = np.poly1d(coeffs)
        except np.linalg.LinAlgError:
            return np.zeros(self.width), "linear"

        # Evaluate polynomial over full pixel range
        pixel_grid = np.arange(self.width, dtype=np.float64)
        pred_inv = poly(pixel_grid)
        eps = 1e-12
        if np.any(pred_inv <= eps):
            print("Cauchy fit produced non-positive values; falling back to linear.")
            return np.zeros(self.width), "pchip" if _SCIPY_AVAILABLE else "linear"

        full_wl = 1.0 / np.sqrt(pred_inv) * 1000.0  # µm⁻¹ → nm
        d = np.diff(full_wl)
        tol = getattr(
            self._config.calibration, "monotonicity_threshold_nm", 0.001
        )
        inc_ok = np.all(d > -tol)
        dec_ok = np.all(d < tol)
        if not (inc_ok or dec_ok):
            # Determine intended direction from majority of steps, then find
            # only the actual violation for that direction.
            if float(np.median(d)) >= 0:
                direction, bad = "increasing", d <= -tol
                limit = f"-{tol}"
            else:
                direction, bad = "decreasing", d >= tol
                limit = f"+{tol}"
            idx = int(np.where(bad)[0][0])  # first actual violation
            wl_a = float(full_wl[idx])
            wl_b = float(full_wl[idx + 1])
            diff_val = float(d[idx])
            print(
                f"Cauchy fit monotonicity failed at pixel {idx}→{idx + 1}: "
                f"wl={wl_a:.3f} nm → {wl_b:.3f} nm, diff={diff_val:.4f} nm "
                f"(limit {limit} nm for {direction}); falling back to linear."
            )
            return np.zeros(self.width), "pchip" if _SCIPY_AVAILABLE else "linear"

        print("Calculating Cauchy-inverse fit (prism dispersion) over full pixel range...")
        return full_wl, "cauchy"

    def _fallback_interp(self, px: np.ndarray, wl: np.ndarray, method: str) -> np.ndarray:
        """Fallback interpolation when Cauchy fails.

        Uses linear interpolation within cal range and linear EXTRAPOLATION
        (extending slope) outside, so wavelengths span full sensor (e.g. 380–750 nm)
        rather than being capped to the source spectrum range.
        """
        if method == "pchip" and _SCIPY_AVAILABLE and len(px) >= 3:
            print("Using PCHIP interpolation (strictly monotonic)...")
            p_min, p_max = float(px.min()), float(px.max())
            interp = PchipInterpolator(px, wl, extrapolate=False)
            out = self._linear_interp_extrap(px, wl)
            pixel_grid = np.arange(self.width, dtype=np.float64)
            mask = (pixel_grid >= p_min) & (pixel_grid <= p_max)
            out[mask] = interp(pixel_grid[mask])
            d = np.diff(out)
            if np.all(d > 0) or np.all(d < 0):
                return out
        print("Using linear interpolation with slope extrapolation (full sensor range)...")
        return self._linear_interp_extrap(px, wl)

    def _linear_interp_extrap(self, px: np.ndarray, wl: np.ndarray) -> np.ndarray:
        """Linear interpolation within range, linear extrapolation (extend slope) outside."""
        pixel_grid = np.arange(self.width, dtype=np.float64)
        out = np.interp(pixel_grid, px, wl)
        p_min, p_max = float(px.min()), float(px.max())
        slope_lo = (wl[1] - wl[0]) / (px[1] - px[0]) if px[1] != px[0] else 0.0
        slope_hi = (wl[-1] - wl[-2]) / (px[-1] - px[-2]) if px[-1] != px[-2] else 0.0
        mask_lo = pixel_grid < p_min
        mask_hi = pixel_grid > p_max
        out[mask_lo] = wl[0] + (pixel_grid[mask_lo] - px[0]) * slope_lo
        out[mask_hi] = wl[-1] + (pixel_grid[mask_hi] - px[-1]) * slope_hi
        return out

    def _generate_graticule(self) -> GraticuleData:
        """Generate graticule data for wavelength markers.

        Uses searchsorted for deterministic tick placement so ticks stay fixed
        when wavelength data has slight float variations between frames.
        """
        wavelength_data = self.wavelengths

        low = int(round(wavelength_data[0])) - 10
        high = int(round(wavelength_data[-1])) + 10

        tens: list[int] = []
        fifties: list[tuple[int, int]] = []

        for target_nm in range(low, high):
            if target_nm % 10 != 0:
                continue

            idx = _closest_index_stable(wavelength_data, target_nm)
            if idx < 0 or abs(wavelength_data[idx] - target_nm) >= 1:
                continue

            if target_nm % 50 == 0:
                label = int(round(wavelength_data[idx]))
                fifties.append((idx, label))
            else:
                tens.append(idx)

        return GraticuleData(tens=tens, fifties=fifties)


_MIN_CAL_POINTS = 4
_MIN_CAL_RANGE_NM = 100.0


def _validate_cal_points(pixels: list[int], wavelengths: list[float]) -> str | None:
    """Return an error message if cal points are invalid, else None."""
    if len(pixels) < _MIN_CAL_POINTS:
        return f"Need at least {_MIN_CAL_POINTS} calibration points!"
    if len(pixels) != len(wavelengths):
        return "Pixel and wavelength arrays must have same length!"
    wl_range = max(wavelengths) - min(wavelengths)
    if wl_range < _MIN_CAL_RANGE_NM:
        return f"Calibration range too narrow: {wl_range:.1f} nm (need {_MIN_CAL_RANGE_NM:.0f}+ nm)."
    return None


def _closest_index_stable(x_values: np.ndarray, target: float) -> int:
    """Index of closest value to target, deterministic for ties.

    Uses searchsorted and prefers lower index when equidistant,
    so ticks stay fixed across frames with slight float variations.
    """
    idx = int(np.searchsorted(x_values, target))
    if idx <= 0:
        return 0
    if idx >= len(x_values):
        return len(x_values) - 1
    # Prefer lower index when equidistant for stable placement
    if abs(x_values[idx - 1] - target) <= abs(x_values[idx] - target):
        return idx - 1
    return idx


def graticule_from_x_axis(
    x_values: np.ndarray,
    step: int = 50,
    unit_suffix: str = "nm",
) -> GraticuleData:
    """Generate graticule from arbitrary x-axis (wavelength or wavenumber).

    Uses searchsorted for deterministic tick placement so ticks stay fixed
    when x-axis has slight float variations between frames.

    Args:
        x_values: X-axis values (same length as spectrum)
        step: Step for major ticks (e.g. 50 for nm, 500 for cm⁻¹)
        unit_suffix: Suffix for labels (e.g. "nm" or " cm⁻¹")

    Returns:
        GraticuleData for display
    """
    low = int(round(float(np.min(x_values)))) - step // 2
    high = int(round(float(np.max(x_values)))) + step // 2
    tens: list[int] = []
    fifties: list[tuple[int, int]] = []
    for target in range(low, high):
        if target % (step // 5) != 0 and target % step != 0:
            continue
        idx = _closest_index_stable(x_values, target)
        if abs(x_values[idx] - target) >= step / 10:
            continue
        if target % step == 0:
            fifties.append((idx, target))
        else:
            tens.append(idx)
    return GraticuleData(tens=tens, fifties=fifties, unit=unit_suffix)
