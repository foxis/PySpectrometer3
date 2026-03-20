"""CMOS spectral sensitivity correction: datasheet curve and optional user calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import SensitivityConfig
from .sensitivity_curve_fit import fit_sensitivity_values
from ..utils.graph_scale import scale_intensity_to_graph

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DEFAULT_CSV = Path("sensor_sensitivity") / "OV9281_spectral_sensitivity_extracted.csv"
_FALLBACK_CSV = Path("silicon_CMOS_spectral_sensitivity.csv")


class SensitivityCorrection:
    """Loads datasheet CMOS curve; optional user-fitted curve from reference illuminant.

    Pipeline divides intensity by the active sensitivity curve to flatten spectral response.
    User calibration (CRR) fits measured/reference ratio, smoothed and merged with CMOS at edges.
    """

    def __init__(self, config: SensitivityConfig | None = None) -> None:
        self._wl_cmos: np.ndarray | None = None
        self._val_cmos: np.ndarray | None = None
        self._wl_active: np.ndarray | None = None
        self._val_active: np.ndarray | None = None
        self._using_custom: bool = False
        self._load_cmos_csv()
        self.apply_config(config)

    def _load_cmos_csv(self) -> None:
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
                f"[Sensitivity] CMOS file not found (tried: {[str(p) for p in candidates]})"
            )
            return

        skiprows = 1 if "OV9281" in str(csv_path) else 2
        try:
            data = np.loadtxt(csv_path, delimiter=",", skiprows=skiprows)
            self._wl_cmos = np.asarray(data[:, 0], dtype=np.float64)
            self._val_cmos = np.asarray(data[:, 1], dtype=np.float64)
            print(
                f"[Sensitivity] Loaded datasheet CMOS: {len(self._wl_cmos)} points, "
                f"{self._wl_cmos[0]:.0f}-{self._wl_cmos[-1]:.0f} nm"
            )
        except Exception as e:
            print(f"[Sensitivity] Failed to load CMOS CSV: {e}")
            self._wl_cmos = None
            self._val_cmos = None

    def apply_config(self, config: SensitivityConfig | None) -> None:
        """Activate saved user curve from config or fall back to datasheet."""
        self.reset_to_datasheet()
        if config is None:
            return
        if not config.use_custom_curve:
            return
        wl = list(config.custom_wavelengths)
        val = list(config.custom_values)
        if len(wl) != len(val) or len(wl) < 4:
            return
        self.set_custom_curve(np.asarray(wl, dtype=np.float64), np.asarray(val, dtype=np.float64))
        print(f"[Sensitivity] Loaded user curve from config ({len(wl)} points)")

    def reset_to_datasheet(self) -> None:
        """Use only the sensor datasheet CMOS curve (discard in-memory custom)."""
        self._using_custom = False
        if self._wl_cmos is not None and self._val_cmos is not None:
            self._wl_active = self._wl_cmos
            self._val_active = self._val_cmos
        else:
            self._wl_active = None
            self._val_active = None

    def set_custom_curve(self, wavelengths: np.ndarray, values: np.ndarray) -> None:
        """Set active curve to user-calibrated samples (must be sorted by wavelength)."""
        self._wl_active = np.asarray(wavelengths, dtype=np.float64).copy()
        self._val_active = np.asarray(values, dtype=np.float64).copy()
        self._using_custom = True

    @property
    def using_custom_curve(self) -> bool:
        return self._using_custom

    def _interpolate_cmos_only(self, wavelengths: np.ndarray) -> np.ndarray | None:
        """Interpolate datasheet CMOS at wavelengths (ignores custom curve)."""
        if self._wl_cmos is None or self._val_cmos is None:
            return None
        v0, v1 = float(self._val_cmos[0]), float(self._val_cmos[-1])
        return np.interp(
            np.asarray(wavelengths, dtype=np.float64),
            self._wl_cmos,
            self._val_cmos,
            left=v0,
            right=v1,
        )

    def interpolate(self, wavelengths: np.ndarray) -> np.ndarray | None:
        """Interpolate active sensitivity at wavelengths."""
        if self._wl_active is None or self._val_active is None:
            return None
        v0, v1 = float(self._val_active[0]), float(self._val_active[-1])
        return np.interp(
            np.asarray(wavelengths, dtype=np.float64),
            self._wl_active,
            self._val_active,
            left=v0,
            right=v1,
        )

    def recalibrate_from_measurement(
        self,
        measured: np.ndarray,
        wavelengths: np.ndarray,
        reference: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Build new sensitivity from measured vs reference SPD. Returns (wl, sens) or None."""
        cmos_at = self._interpolate_cmos_only(wavelengths)
        if cmos_at is None:
            print("[CRR] No datasheet CMOS curve; cannot scale fit")
            return None

        sens = fit_sensitivity_values(
            wavelengths,
            measured,
            reference,
            cmos_at,
        )
        wl_out = np.asarray(wavelengths, dtype=np.float64).copy()
        n = min(len(wl_out), len(sens))
        return (wl_out[:n], sens[:n])

    def apply(
        self,
        intensity: np.ndarray,
        wavelengths: np.ndarray,
    ) -> np.ndarray:
        """Divide intensity by active sensitivity curve."""
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
        """Scale active sensitivity for graph overlay."""
        sensitivity = self.interpolate(wavelengths)
        if sensitivity is None:
            return None
        scaled = scale_intensity_to_graph(sensitivity, graph_height)
        return (scaled, (100, 255, 100))
