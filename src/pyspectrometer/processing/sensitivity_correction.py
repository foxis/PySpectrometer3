"""CMOS spectral sensitivity correction for calibration."""

from pathlib import Path

import numpy as np

from ..utils.graph_scale import scale_intensity_to_graph

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DEFAULT_CSV = Path("sensor_sensitivity") / "OV9281_spectral_sensitivity_extracted.csv"
_FALLBACK_CSV = Path("silicon_CMOS_spectral_sensitivity.csv")


class SensitivityCorrection:
    """Loads and applies CMOS spectral sensitivity correction.

    Interpolates sensitivity at wavelengths and divides intensity to linearize
    spectrum. Used in calibration mode for reference overlay and correlation.
    """

    def __init__(self) -> None:
        self._wl: np.ndarray | None = None
        self._val: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        """Load CMOS sensitivity curve from CSV. OV9281 by default, silicon_CMOS fallback."""
        candidates = [
            _DATA_DIR / _DEFAULT_CSV,
            Path.cwd() / "data" / _DEFAULT_CSV,
            _DATA_DIR / _FALLBACK_CSV,
            Path.cwd() / "data" / _FALLBACK_CSV,
        ]
        csv_path = None
        for p in candidates:
            if p.exists():
                csv_path = p
                break
        if csv_path is None:
            print(
                f"[Calibration] Sensitivity file not found (tried: {[str(p) for p in candidates]})"
            )
            return

        skiprows = 1 if "OV9281" in str(csv_path) else 2
        try:
            data = np.loadtxt(csv_path, delimiter=",", skiprows=skiprows)
            self._wl = data[:, 0]
            self._val = data[:, 1]
            print(
                f"[Calibration] Loaded CMOS sensitivity: {len(self._wl)} points, "
                f"{self._wl[0]:.0f}-{self._wl[-1]:.0f} nm"
            )
        except Exception as e:
            print(f"[Calibration] Failed to load CMOS sensitivity: {e}")
            self._wl = None
            self._val = None

    def interpolate(self, wavelengths: np.ndarray) -> np.ndarray | None:
        """Interpolate sensitivity at wavelengths. Returns None if not loaded."""
        if self._wl is None or self._val is None:
            return None
        return np.interp(
            wavelengths,
            self._wl,
            self._val,
            left=0.0,
            right=0.0,
        )

    def apply(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Apply sensitivity correction to linearize spectrum."""
        sensitivity = self.interpolate(wavelengths)
        if sensitivity is None:
            return intensity

        n_wl = len(wavelengths)
        n_int = len(intensity)
        n = min(n_wl, n_int)
        intensity = intensity[:n]
        sensitivity = sensitivity[:n]

        intensity_f = np.asarray(intensity, dtype=np.float32)
        corrected = np.zeros(n, dtype=np.float32)
        mask = sensitivity > 1e-6
        corrected[mask] = intensity_f[mask] / sensitivity[mask]
        corrected[~mask] = intensity_f[~mask]

        return corrected

    def get_curve_for_display(
        self,
        wavelengths: np.ndarray,
        graph_height: int,
    ) -> tuple[np.ndarray, tuple[int, int, int]] | None:
        """Get sensitivity curve scaled for graph overlay. Returns None if not loaded."""
        sensitivity = self.interpolate(wavelengths)
        if sensitivity is None:
            return None
        scaled = scale_intensity_to_graph(sensitivity, graph_height)
        return (scaled, (100, 255, 100))
