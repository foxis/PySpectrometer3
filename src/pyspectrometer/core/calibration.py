"""Wavelength calibration for PySpectrometer3."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np


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
            return "2nd Order Polyfit"
        else:
            return "3rd Order Polyfit"


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
            pixels, wavelengths, has_errors = self._read_cal_file()
            if has_errors:
                pixels = list(self.default_pixels)
                wavelengths = list(self.default_wavelengths)
            
            pixels_str = ",".join(map(str, pixels))
            wavelengths_str = ",".join(map(str, wavelengths))
            
            with open(self.cal_file, "w") as f:
                f.write(f"{pixels_str}\r\n")
                f.write(f"{wavelengths_str}\r\n")
                f.write(f"{self._rotation_angle}\r\n")
                f.write(f"{self._spectrum_y_center}\r\n")
                f.write(f"{self._perpendicular_width}\r\n")
            
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
    
    def _read_cal_file(self) -> tuple[list[int], list[float], bool]:
        """Read calibration data from file."""
        errors = False
        pixels: list[int] = []
        wavelengths: list[float] = []
        
        try:
            print("Loading calibration data...")
            with open(self.cal_file, "r") as f:
                lines = f.readlines()
                
            line0 = lines[0].strip()
            pixels = [int(x) for x in line0.split(",")]
            
            line1 = lines[1].strip()
            wavelengths = [float(x) for x in line1.split(",")]
            
            if len(pixels) != len(wavelengths):
                errors = True
            if len(pixels) < 3 or len(wavelengths) < 3:
                errors = True
            
            if not errors:
                self._cal_pixels = pixels
                self._cal_wavelengths = wavelengths
            
            if len(lines) >= 3:
                line2 = lines[2].strip()
                self._rotation_angle = float(line2)
                print(f"Loaded rotation angle: {self._rotation_angle:.2f} deg")
            
            if len(lines) >= 4:
                line3 = lines[3].strip()
                self._spectrum_y_center = int(float(line3))
                print(f"Loaded spectrum Y center: {self._spectrum_y_center}")
            
            if len(lines) >= 5:
                line4 = lines[4].strip()
                self._perpendicular_width = int(float(line4))
                print(f"Loaded perpendicular width: {self._perpendicular_width}")
                
        except Exception:
            errors = True
        
        return pixels, wavelengths, errors
    
    def _compute_calibration(
        self,
        pixels: list[int],
        wavelengths: list[float],
        has_errors: bool,
    ) -> CalibrationResult:
        """Compute wavelength calibration using polynomial fit."""
        wavelength_data = np.zeros(self.width)
        
        if len(pixels) == 3:
            print("Calculating second order polynomial...")
            coefficients = np.polyfit(pixels, wavelengths, 2)
            poly = np.poly1d(coefficients)
            print(poly)
            
            for pixel in range(self.width):
                wavelength_data[pixel] = round(poly(pixel), 6)
            
            print("Done! Note that calibration with only 3 wavelengths will not be accurate!")
            order = 0 if has_errors else 2
            
            return CalibrationResult(
                wavelengths=wavelength_data,
                order=order,
                coefficients=coefficients,
                rotation_angle=self._rotation_angle,
                spectrum_y_center=self._spectrum_y_center,
                perpendicular_width=self._perpendicular_width,
            )
        
        print("Calculating third order polynomial...")
        coefficients = np.polyfit(pixels, wavelengths, 3)
        poly = np.poly1d(coefficients)
        print(poly)
        
        print("Generating Wavelength Data!\n")
        for pixel in range(self.width):
            wavelength_data[pixel] = round(poly(pixel), 6)
        
        predicted = [poly(px) for px in pixels]
        corr_matrix = np.corrcoef(wavelengths, predicted)
        corr = corr_matrix[0, 1]
        r_squared = corr ** 2
        
        print(f"R-Squared={r_squared}")
        
        return CalibrationResult(
            wavelengths=wavelength_data,
            order=3,
            coefficients=coefficients,
            r_squared=r_squared,
            rotation_angle=self._rotation_angle,
            spectrum_y_center=self._spectrum_y_center,
            perpendicular_width=self._perpendicular_width,
        )
    
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
