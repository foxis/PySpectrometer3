"""File I/O for wavelength calibration data."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CalibrationFileData:
    """Data read from or to be written to calibration file."""

    pixels: list[int]
    wavelengths: list[float]
    has_errors: bool
    rotation_angle: float
    spectrum_y_center: int
    perpendicular_width: int


class CalibrationFileIO:
    """Reads and writes calibration data to/from file.

    File format: 5 lines — pixels, wavelengths, rotation_angle,
    spectrum_y_center, perpendicular_width.
    """

    def __init__(self, cal_file: Path | str) -> None:
        self.cal_file = Path(cal_file)

    def read(self) -> CalibrationFileData:
        """Read full calibration from file."""
        try:
            with open(self.cal_file) as f:
                lines = f.readlines()
        except OSError:
            return CalibrationFileData(
                pixels=[],
                wavelengths=[],
                has_errors=True,
                rotation_angle=0.0,
                spectrum_y_center=0,
                perpendicular_width=20,
            )

        pixels, wavelengths, has_errors = self._parse_pixels_wavelengths(lines)
        rotation_angle = 0.0
        spectrum_y_center = 0
        perpendicular_width = 20

        try:
            if len(lines) >= 3:
                rotation_angle = float(lines[2].strip())
            if len(lines) >= 4:
                spectrum_y_center = int(float(lines[3].strip()))
            if len(lines) >= 5:
                perpendicular_width = int(float(lines[4].strip()))
        except (IndexError, ValueError):
            pass

        return CalibrationFileData(
            pixels=pixels,
            wavelengths=wavelengths,
            has_errors=has_errors,
            rotation_angle=rotation_angle,
            spectrum_y_center=spectrum_y_center,
            perpendicular_width=perpendicular_width,
        )

    def write(
        self,
        pixels: list[int],
        wavelengths: list[float],
        rotation_angle: float,
        spectrum_y_center: int,
        perpendicular_width: int,
    ) -> bool:
        """Write calibration data to file."""
        try:
            pixels_str = ",".join(map(str, pixels))
            wavelengths_str = ",".join(map(str, wavelengths))
            with open(self.cal_file, "w") as f:
                f.write(f"{pixels_str}\r\n")
                f.write(f"{wavelengths_str}\r\n")
                f.write(f"{rotation_angle}\r\n")
                f.write(f"{spectrum_y_center}\r\n")
                f.write(f"{perpendicular_width}\r\n")
            return True
        except OSError as e:
            print(f"Failed to save calibration: {e}")
            return False

    @staticmethod
    def _parse_pixels_wavelengths(lines: list[str]) -> tuple[list[int], list[float], bool]:
        """Parse pixels and wavelengths from first two lines."""
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
