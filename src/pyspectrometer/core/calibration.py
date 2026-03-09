"""Wavelength calibration for PySpectrometer3."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np

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
    r_squared: Optional[float] = None
    rotation_angle: float = 0.0
    spectrum_y_center: int = 0
    perpendicular_width: int = 20
    
    @property
    def is_accurate(self) -> bool:
        """Check if calibration is considered accurate (3rd order poly)."""
        return self.order >= 3
    
    @property
    def status_message(self) -> str:
        """Human-readable calibration status."""
        if self.order == 0:
            return "UNCALIBRATED!"
        elif self.order == 2:
            return "Calibrated!!"
        else:
            return "Calibrated!!!"
    
    @property
    def detail_message(self) -> str:
        """Human-readable calibration detail."""
        if self.order == 0:
            return "Perform Calibration!"
        elif self.order == 2:
            return "Monotonic (3 points)"
        else:
            return "Cauchy (prism)"


@dataclass
class GraticuleData:
    """Data for rendering calibrated graticule lines."""
    
    tens: list[int] = field(default_factory=list)
    fifties: list[tuple[int, int]] = field(default_factory=list)


class Calibration:
    """Handles wavelength calibration for the spectrometer.
    
    This class manages loading, saving, and computing calibration data
    that maps pixel positions to wavelengths using polynomial fitting.
    """
    
    def __init__(
        self,
        width: int,
        height: int = 480,
        cal_file: Optional[Path] = None,
        default_pixels: tuple[int, ...] = (0, 400, 800),
        default_wavelengths: tuple[float, ...] = (380.0, 560.0, 750.0),
    ):
        self.width = width
        self.height = height
        self.cal_file = cal_file or Path("caldata.txt")
        self.default_pixels = default_pixels
        self.default_wavelengths = default_wavelengths
        
        self._result: Optional[CalibrationResult] = None
        self._graticule: Optional[GraticuleData] = None
        
        self._rotation_angle: float = 0.0
        self._spectrum_y_center: int = height // 2
        self._perpendicular_width: int = 20
        
        # Store current calibration data points
        self._cal_pixels: list[int] = list(default_pixels)
        self._cal_wavelengths: list[float] = list(default_wavelengths)
    
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
    
    def load(self) -> CalibrationResult:
        """Load calibration data from file and compute wavelengths."""
        pixels, wavelengths, has_errors = self._read_cal_file()
        
        if has_errors:
            pixels = list(self.default_pixels)
            wavelengths = list(self.default_wavelengths)
            print("Loading of Calibration data failed!")
            print("Loading placeholder data...")
            print("You MUST perform a Calibration to use this software!\n")
        
        self._result = self._compute_calibration(pixels, wavelengths, has_errors)
        self._graticule = None
        return self._result
    
    def save(
        self,
        pixel_data: list[int],
        wavelength_data: list[float],
        rotation_angle: Optional[float] = None,
        spectrum_y_center: Optional[int] = None,
        perpendicular_width: Optional[int] = None,
    ) -> bool:
        """Save calibration data to file.
        
        Args:
            pixel_data: List of pixel positions
            wavelength_data: List of corresponding wavelengths
            rotation_angle: Spectrum rotation angle in degrees
            spectrum_y_center: Y coordinate of spectrum center
            perpendicular_width: Width of perpendicular sampling region
            
        Returns:
            True if calibration was saved successfully
        """
        if len(pixel_data) < 3:
            print("Need at least 3 calibration points!")
            return False
            
        if len(pixel_data) != len(wavelength_data):
            print("Pixel and wavelength arrays must have same length!")
            return False
        
        if rotation_angle is not None:
            self._rotation_angle = rotation_angle
        if spectrum_y_center is not None:
            self._spectrum_y_center = spectrum_y_center
        if perpendicular_width is not None:
            self._perpendicular_width = perpendicular_width
        
        try:
            pixels_str = ",".join(map(str, pixel_data))
            wavelengths_str = ",".join(map(str, wavelength_data))
            
            with open(self.cal_file, "w") as f:
                f.write(f"{pixels_str}\r\n")
                f.write(f"{wavelengths_str}\r\n")
                f.write(f"{self._rotation_angle}\r\n")
                f.write(f"{self._spectrum_y_center}\r\n")
                f.write(f"{self._perpendicular_width}\r\n")
            
            print("Calibration Data Written!")
            self.load()
            return True
            
        except Exception as e:
            print(f"Failed to save calibration: {e}")
            return False
    
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

        try:
            pixels, wavelengths, has_errors = self._read_pixels_wavelengths_only()
            if has_errors:
                pixels = list(self.default_pixels)
                wavelengths = list(self.default_wavelengths)

            pixels_str = ",".join(map(str, pixels))
            wavelengths_str = ",".join(map(str, wavelengths))

            with open(self.cal_file, "w") as f:
                f.write(f"{pixels_str}\r\n")
                f.write(f"{wavelengths_str}\r\n")
                f.write(f"{rotation_angle}\r\n")
                f.write(f"{spectrum_y_center}\r\n")
                f.write(f"{perpendicular_width}\r\n")

            self._rotation_angle = rotation_angle
            self._spectrum_y_center = spectrum_y_center
            self._perpendicular_width = perpendicular_width

            print(f"Extraction params saved: angle={rotation_angle:.2f}°, y={spectrum_y_center}, width={perpendicular_width}")
            return True
            
        except Exception as e:
            print(f"Failed to save extraction params: {e}")
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
        if len(pixel_data) < 3:
            print("Need at least 3 calibration points!")
            return False
        
        if len(pixel_data) != len(wavelength_data):
            print("Pixel and wavelength arrays must have same length!")
            return False
        
        self._cal_pixels = list(pixel_data)
        self._cal_wavelengths = list(wavelength_data)
        
        self._result = self._compute_calibration(pixel_data, wavelength_data, has_errors=False)
        self._graticule = None
        
        return True
    
    @staticmethod
    def _parse_pixels_wavelengths(lines: list[str]) -> tuple[list[int], list[float], bool]:
        """Parse pixels and wavelengths from first two lines. Returns (pixels, wavelengths, errors)."""
        pixels: list[int] = []
        wavelengths: list[float] = []
        try:
            line0 = lines[0].strip()
            pixels = [int(x) for x in line0.split(",")]
            line1 = lines[1].strip()
            wavelengths = [float(x) for x in line1.split(",")]
            if len(pixels) != len(wavelengths) or len(pixels) < 3:
                return pixels, wavelengths, True
        except (IndexError, ValueError):
            return pixels, wavelengths, True
        return pixels, wavelengths, False

    def _read_pixels_wavelengths_only(self) -> tuple[list[int], list[float], bool]:
        """Read only pixels and wavelengths (lines 0-1). No side effects on extraction params."""
        try:
            with open(self.cal_file, "r") as f:
                lines = f.readlines()
            return self._parse_pixels_wavelengths(lines)
        except OSError:
            return [], [], True

    def _read_cal_file(self) -> tuple[list[int], list[float], bool]:
        """Read calibration data from file.
        Pixels/wavelengths (lines 0-1) and extraction params (lines 2-4)
        are read independently so offset/rotation still load when lines 0-1 fail.
        """
        try:
            print("Loading calibration data...")
            with open(self.cal_file, "r") as f:
                lines = f.readlines()
        except OSError:
            return [], [], True

        pixels, wavelengths, errors = self._parse_pixels_wavelengths(lines)
        if not errors and len(pixels) >= 3:
            self._cal_pixels = pixels
            self._cal_wavelengths = wavelengths

        # Always load extraction params (offset, rotation, perpendicular width)
        # so they apply even when pixel/wavelength lines fail
        try:
            if len(lines) >= 3:
                self._rotation_angle = float(lines[2].strip())
                print(f"Loaded rotation angle: {self._rotation_angle:.2f} deg")
            if len(lines) >= 4:
                self._spectrum_y_center = int(float(lines[3].strip()))
                print(f"Loaded spectrum Y center (offset): {self._spectrum_y_center}")
            if len(lines) >= 5:
                self._perpendicular_width = int(float(lines[4].strip()))
                print(f"Loaded perpendicular width: {self._perpendicular_width}")
        except (IndexError, ValueError) as e:
            print(f"[Calibration] Could not parse extraction params: {e}")

        return pixels, wavelengths, errors
    
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
        order = 0 if has_errors else (3 if len(px) >= 4 else 2)

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
        Uses Cauchy only within [min(px), max(px)]; extrapolates linearly outside
        to avoid explosion at red/blue ends.
        """
        if len(px) < 3:
            return np.zeros(self.width), "linear"

        p_min, p_max = float(px.min()), float(px.max())
        # 1/λ² in µm⁻² for numerical stability (nm→µm)
        inv_wl_sq = 1.0 / ((wl * 1e-3) ** 2)
        deg = min(2, len(px) - 1)
        try:
            coeffs = np.polyfit(px, inv_wl_sq, deg)
            poly = np.poly1d(coeffs)
        except np.linalg.LinAlgError:
            return np.zeros(self.width), "linear"

        # Evaluate only within cal range to avoid polynomial extrapolation explosion
        interp_in = np.arange(max(0, int(p_min)), min(self.width, int(p_max) + 1), dtype=np.float64)
        pred_inv = poly(interp_in)
        eps = 1e-12
        if np.any(pred_inv <= eps):
            print("Cauchy fit produced non-positive values; falling back.")
            return np.zeros(self.width), "pchip" if _SCIPY_AVAILABLE else "linear"

        wl_in = 1.0 / np.sqrt(pred_inv) * 1000.0  # µm⁻¹ → nm
        d = np.diff(wl_in)
        if not (np.all(d > 0) or np.all(d < 0)):
            print("Cauchy fit produced non-monotonic result; falling back.")
            return np.zeros(self.width), "pchip" if _SCIPY_AVAILABLE else "linear"

        # Build full array: linear extrapolation outside cal range
        full_wl = np.interp(np.arange(self.width), px, wl)  # base
        i0, i1 = int(p_min), int(p_max)
        if i0 <= i1:
            full_wl[i0 : i1 + 1] = wl_in
        print("Calculating Cauchy-inverse fit (prism dispersion)...")
        return full_wl, "cauchy"

    def _fallback_interp(self, px: np.ndarray, wl: np.ndarray, method: str) -> np.ndarray:
        """Fallback interpolation when Cauchy fails. Uses linear (no PCHIP extrapolation
        beyond cal range) to avoid red/blue explosion.
        """
        if method == "pchip" and _SCIPY_AVAILABLE and len(px) >= 3:
            print("Using PCHIP interpolation (strictly monotonic)...")
            p_min, p_max = float(px.min()), float(px.max())
            interp = PchipInterpolator(px, wl, extrapolate=False)  # no wild extrapolation
            # Evaluate only inside [p_min, p_max]; use linear for rest
            out = np.interp(np.arange(self.width), px, wl)
            pixel_grid = np.arange(self.width, dtype=np.float64)
            mask = (pixel_grid >= p_min) & (pixel_grid <= p_max)
            out[mask] = interp(pixel_grid[mask])
            d = np.diff(out)
            if np.all(d > 0) or np.all(d < 0):
                return out
        print("Using linear interpolation (strictly monotonic)...")
        return np.interp(np.arange(self.width), px, wl)
    
    def _generate_graticule(self) -> GraticuleData:
        """Generate graticule data for wavelength markers."""
        wavelength_data = self.wavelengths
        
        low = int(round(wavelength_data[0])) - 10
        high = int(round(wavelength_data[-1])) + 10
        
        tens: list[int] = []
        fifties: list[tuple[int, int]] = []
        
        for target_nm in range(low, high):
            if target_nm % 10 != 0:
                continue
                
            position = min(
                enumerate(wavelength_data),
                key=lambda x: abs(target_nm - x[1])
            )
            
            if abs(target_nm - position[1]) >= 1:
                continue
                
            if target_nm % 50 == 0:
                label = int(round(position[1]))
                fifties.append((position[0], label))
            else:
                tens.append(position[0])
        
        return GraticuleData(tens=tens, fifties=fifties)
