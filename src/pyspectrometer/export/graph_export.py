"""Vector (SVG/EPS) and multi-page PDF graph export for spectrum data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.spectrum import Peak


def _configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _apply_spectrum_ticks(ax, *, x_is_wavelength: bool = True) -> None:
    """Major/minor grid for spectrum plots (wavelength on X, intensity on Y)."""
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator

    if x_is_wavelength:
        x0, x1 = ax.get_xlim()
        if x1 > x0:
            span = x1 - x0
            if span > 200:
                major, minor = 50, 10
            elif span > 80:
                major, minor = 20, 5
            elif span > 40:
                major, minor = 10, 2
            else:
                major, minor = 5, 1
            ax.xaxis.set_major_locator(MultipleLocator(major))
            ax.xaxis.set_minor_locator(MultipleLocator(minor))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.15, linestyle=":")


@dataclass
class ViewExportRequest:
    """Parameters for a single vector export matching the on-screen graph viewport."""

    path: Path
    wavelengths: np.ndarray
    intensity: np.ndarray
    x_axis_label: str
    x_start: float
    x_end: float
    y_min: float
    y_max: float
    peaks: list[Peak]
    metadata: dict[str, Any]
    reference: np.ndarray | None = None
    reference_name: str = "Reference"
    sensitivity_display: np.ndarray | None = None
    # Color Science: optional CIE values for annotation + sRGB patch
    xyz: tuple[float, float, float] | None = None
    cie_lab: tuple[float, float, float] | None = None


@dataclass
class ColorSciencePdfBundle:
    """Payload for Color Science PDF: metadata + one page per swatch (400–700 nm spectrum)."""

    path: Path
    metadata: dict[str, Any]
    sensitivity_meta: dict[str, str]
    swatches: list[Any]
    wl_range_nm: tuple[float, float] = (400.0, 700.0)


@dataclass
class MeasurementPdfBundle:
    """Payload for the measurement-mode multi-page PDF report."""

    path: Path
    wavelengths: np.ndarray
    measured_pre_sensitivity: np.ndarray
    measured_corrected: np.ndarray
    dark: np.ndarray | None
    white: np.ndarray | None
    reference: np.ndarray | None
    reference_name: str
    sensitivity: np.ndarray | None
    sensitivity_meta: dict[str, str]
    peaks: list[Peak]
    metadata: dict[str, Any]
    x_axis_label: str = "Wavelength (nm)"
    # Load+ overlays at export time (same processing as on-screen for current S toggle)
    overlay_series: list[tuple[str, np.ndarray]] | None = None


def _wl_window(
    wavelengths: np.ndarray,
    x_start: float,
    x_end: float,
) -> tuple[np.ndarray, slice]:
    """Slice wavelength array by pixel-index viewport (inclusive index range)."""
    n = len(wavelengths)
    if n == 0:
        return np.array([]), slice(0, 0)
    i0 = int(np.clip(round(x_start), 0, n - 1))
    i1 = int(np.clip(round(x_end), 0, n - 1))
    if i1 < i0:
        i0, i1 = i1, i0
    sl = slice(i0, i1 + 1)
    return np.asarray(wavelengths[sl], dtype=np.float64), sl


def _align_series(y: np.ndarray | None, n: int) -> np.ndarray | None:
    if y is None or len(y) == 0:
        return None
    y = np.asarray(y, dtype=np.float64)
    if len(y) >= n:
        return y[:n].copy()
    return np.pad(y, (0, n - len(y)))


def _peak_y_on_curve(p: Peak, wl: np.ndarray, y: np.ndarray, y_min: float, y_max: float) -> float:
    """Y position for peak marker: interpolate onto the plotted curve (not stored peak height)."""
    if len(wl) == 0:
        return float(np.clip(p.intensity, y_min, y_max))
    py = float(np.interp(float(p.wavelength), np.asarray(wl, dtype=np.float64), np.asarray(y, dtype=np.float64)))
    return float(np.clip(py, y_min, y_max))


def _plot_peaks_vector(ax, peaks: list[Peak], wl: np.ndarray, y: np.ndarray, y_min: float, y_max: float) -> None:
    """Peaks on vector export — wavelength-aligned markers on the current trace."""
    for p in peaks:
        py = _peak_y_on_curve(p, wl, y, y_min, y_max)
        ax.axvline(p.wavelength, color="#c44e52", alpha=0.35, linewidth=1.0, zorder=1)
        ax.scatter(
            [p.wavelength],
            [py],
            color="#c44e52",
            s=36,
            zorder=4,
            edgecolors="white",
            linewidths=0.5,
        )


def _plot_peaks_pdf(ax, peaks: list[Peak], wl: np.ndarray, y: np.ndarray, y_min: float, y_max: float) -> None:
    """Peaks on PDF report with nm labels at the curve."""
    for p in peaks:
        if len(wl) == 0:
            continue
        if not (float(wl[0]) <= p.wavelength <= float(wl[-1])):
            continue
        py = _peak_y_on_curve(p, wl, y, y_min, y_max)
        ax.axvline(p.wavelength, color="#c44e52", alpha=0.35, linewidth=1.0, zorder=1)
        ax.scatter(
            [p.wavelength],
            [py],
            color="#c44e52",
            s=36,
            zorder=4,
            edgecolors="white",
            linewidths=0.5,
        )
        ax.annotate(
            f"{p.wavelength:.1f} nm",
            (p.wavelength, py),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=7,
            color="#8b0000",
        )


def export_view_vector(req: ViewExportRequest) -> Path:
    """Export the current graph view as SVG or EPS (path suffix selects format).

    Axes match the on-screen viewport. Peaks are marked; legend lists series and metadata.

    Path simplification is disabled so dense spectra are not collapsed to a straight line.
    """
    import matplotlib as mpl

    plt = _configure_matplotlib()
    path = Path(req.path)
    if path.suffix.lower() not in (".svg", ".eps"):
        path = path.with_suffix(".svg")

    n = min(len(req.wavelengths), len(req.intensity))
    wl = np.asarray(req.wavelengths[:n], dtype=np.float64)
    y_main = np.asarray(req.intensity[:n], dtype=np.float64)
    wl_win, sl = _wl_window(wl, req.x_start, req.x_end)
    y_win = y_main[sl]
    if len(wl_win) > 0:
        wl_lo, wl_hi = float(wl_win[0]), float(wl_win[-1])
        if wl_lo > wl_hi:
            wl_lo, wl_hi = wl_hi, wl_lo
        peaks_view = [p for p in req.peaks if wl_lo <= p.wavelength <= wl_hi]
    else:
        peaks_view = []

    with mpl.rc_context({"path.simplify": False}):
        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        ax.plot(wl_win, y_win, color="#1f77b4", linewidth=1.4, label="Measured (display)")

        ref_a = _align_series(req.reference, n)
        if ref_a is not None:
            ax.plot(
                wl_win,
                ref_a[sl],
                color="#7f7f7f",
                linewidth=1.0,
                linestyle="--",
                label=req.reference_name,
            )

        sens_a = _align_series(req.sensitivity_display, n)
        if sens_a is not None:
            ax.plot(
                wl_win,
                sens_a[sl],
                color="#2ca02c",
                linewidth=1.0,
                linestyle=":",
                label="Sensitivity (peak-norm.)",
            )

        _plot_peaks_vector(ax, peaks_view, wl_win, y_win, req.y_min, req.y_max)

        ax.set_xlim(float(wl_win[0]) if len(wl_win) else 0.0, float(wl_win[-1]) if len(wl_win) else 1.0)
        ax.set_ylim(req.y_min, req.y_max)
        ax.set_xlabel(req.x_axis_label)
        ax.set_ylabel("Intensity (display scale)")
        _apply_spectrum_ticks(ax)

        meta_parts = [f"{k}: {v}" for k, v in req.metadata.items() if v not in (None, "")]
        if meta_parts:
            fig.text(0.5, 0.02, " | ".join(meta_parts[:8]), ha="center", fontsize=7, color="#333333")

        if req.xyz is not None:
            from ..colorscience.swatches import xyz_to_display_bgr

            bgr = xyz_to_display_bgr(*req.xyz)
            rgb = (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)
            ax_patch = ax.inset_axes([0.02, 0.62, 0.12, 0.28])
            ax_patch.imshow(np.full((1, 1, 3), rgb, dtype=float), aspect="auto")
            ax_patch.set_xticks([])
            ax_patch.set_yticks([])
            for s in ax_patch.spines.values():
                s.set_edgecolor("#333333")
                s.set_linewidth(1.0)
            lab_lines = [f"X={req.xyz[0]:.2f}  Y={req.xyz[1]:.2f}  Z={req.xyz[2]:.2f}"]
            if req.cie_lab is not None:
                lab_lines.append(
                    f"L*={req.cie_lab[0]:.2f}  a*={req.cie_lab[1]:.2f}  b*={req.cie_lab[2]:.2f}"
                )
            ax.text(
                0.16,
                0.98,
                "\n".join(lab_lines),
                transform=ax.transAxes,
                va="top",
                fontsize=9,
                family="monospace",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
            )

        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.12)
        fig.savefig(path, format=path.suffix[1:])
        plt.close(fig)
    return path


def _norm_peak(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    m = float(np.max(np.abs(a))) if a.size else 1.0
    return a / m if m > 1e-12 else a


def _pdf_line_page(
    plt: Any,
    pdf: Any,
    title: str,
    x_axis: str,
    wl: np.ndarray,
    series: list[tuple[str, np.ndarray, str]],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, values, ls in series:
        ax.plot(wl, values, label=label, linestyle=ls)
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    _apply_spectrum_ticks(ax)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _pdf_blank(plt: Any, pdf: Any, msg: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
    ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def export_measurement_pdf(bundle: MeasurementPdfBundle) -> Path:
    """Write a multi-page PDF: metadata, sensitivity, dark, white, measured, corrected, peaks, refs."""
    plt = _configure_matplotlib()
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(bundle.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(
        len(bundle.wavelengths),
        len(bundle.measured_pre_sensitivity),
        len(bundle.measured_corrected),
    )
    wl = np.asarray(bundle.wavelengths[:n], dtype=np.float64)
    m_pre = np.asarray(bundle.measured_pre_sensitivity[:n], dtype=np.float64)
    m_cor = np.asarray(bundle.measured_corrected[:n], dtype=np.float64)
    xa = bundle.x_axis_label

    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        lines = ["Experiment", "=" * 40]
        for k, v in bundle.metadata.items():
            if v is None or v == "":
                continue
            lines.append(f"{k}: {v}")
        lines.extend(["", "Sensitivity (export)", "-" * 20])
        lines.extend(f"{k}: {v}" for k, v in bundle.sensitivity_meta.items() if v)
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", family="monospace", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

        if bundle.sensitivity is not None and len(bundle.sensitivity) > 0:
            s = np.asarray(bundle.sensitivity[:n], dtype=np.float64)
            _pdf_line_page(plt, pdf, "Sensitivity curve (exported values)", xa, wl, [(r"Relative sensitivity", s, "-")])
        else:
            _pdf_blank(plt, pdf, "Sensitivity curve not available")

        d = _align_series(bundle.dark, n)
        if d is not None:
            _pdf_line_page(plt, pdf, "Dark reference (raw)", xa, wl, [(r"Dark", d, "-")])
        else:
            _pdf_blank(plt, pdf, "Dark reference — not set")

        wht = _align_series(bundle.white, n)
        if wht is not None:
            _pdf_line_page(plt, pdf, "White reference (raw)", xa, wl, [(r"White", wht, "-")])
        else:
            _pdf_blank(plt, pdf, "White reference — not set")

        _pdf_line_page(plt, pdf, "Measured (pre-sensitivity correction)", xa, wl, [(r"Measured", m_pre, "-")])

        _pdf_line_page(plt, pdf, "Measured (sensitivity-corrected)", xa, wl, [(r"Measured (corrected)", m_cor, "-")])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wl, m_cor, label=r"Measured (corrected)", color="#1f77b4")
        ylo, yhi = float(np.min(m_cor)), float(np.max(m_cor))
        pad = max(0.02 * (yhi - ylo), 0.01)
        y_a, y_b = ylo - pad, yhi + pad
        _plot_peaks_pdf(ax, bundle.peaks, wl, m_cor, y_a, y_b)
        ax.set_ylim(y_a, y_b)
        ax.set_title("Sensitivity-corrected with peaks")
        ax.set_xlabel(xa)
        _apply_spectrum_ticks(ax)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        ovs = bundle.overlay_series
        if ovs:
            series: list[tuple[str, np.ndarray, str]] = [
                (name, np.asarray(y[:n], dtype=np.float64), "-") for name, y in ovs
            ]
            _pdf_line_page(
                plt,
                pdf,
                "Load+ overlays (export-time display scale)",
                xa,
                wl,
                series,
            )
        else:
            _pdf_blank(plt, pdf, "No Load+ overlays")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(wl, _norm_peak(m_cor), label=r"Measured (corrected, peak-norm.)", color="#1f77b4", linewidth=1.5)
        if d is not None:
            ax.plot(wl, _norm_peak(d), label=r"Dark (peak-norm.)", color="#444444", alpha=0.8)
        if wht is not None:
            ax.plot(wl, _norm_peak(wht), label=r"White (peak-norm.)", color="#ff7f0e", alpha=0.8)
        ref = _align_series(bundle.reference, n)
        if ref is not None:
            name = bundle.reference_name or "Reference"
            ax.plot(
                wl,
                _norm_peak(ref),
                label=f"{name} (peak-norm.)",
                color="#7f7f7f",
                linestyle="--",
            )
        ax.set_title("Corrected measurement with references (each trace ÷ its max)")
        ax.set_xlabel(xa)
        ax.set_ylabel("Peak-normalized")
        _apply_spectrum_ticks(ax)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return path


def spectrum_in_wl_window(
    wl: np.ndarray,
    spec: np.ndarray,
    lo: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return wavelength and intensity rows limited to ``[lo, hi]`` nm (inclusive)."""
    wl = np.asarray(wl, dtype=np.float64)
    spec = np.asarray(spec, dtype=np.float64)
    n = min(len(wl), len(spec))
    if n == 0:
        return np.array([]), np.array([])
    wl, spec = wl[:n], spec[:n]
    m = (wl >= lo) & (wl <= hi)
    return wl[m], spec[m]


def export_colorscience_pdf(bundle: ColorSciencePdfBundle) -> Path:
    """Multi-page PDF: experiment metadata, then one page per swatch (patch + XYZ/LAB + 400–700 nm spectrum)."""
    plt = _configure_matplotlib()
    from matplotlib.backends.backend_pdf import PdfPages

    path = Path(bundle.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = bundle.wl_range_nm
    xa = "Wavelength (nm)"

    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        lines = ["Color Science — experiment", "=" * 40]
        for k, v in bundle.metadata.items():
            if v is None or v == "":
                continue
            lines.append(f"{k}: {v}")
        lines.extend(["", "Sensitivity (export)", "-" * 20])
        lines.extend(f"{k}: {v}" for k, v in bundle.sensitivity_meta.items() if v)
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", family="monospace", fontsize=9)
        pdf.savefig(fig)
        plt.close(fig)

        swatches = list(bundle.swatches)
        if not swatches:
            _pdf_blank(plt, pdf, "No swatches — press Add to store measurements, then export PDF again.")

        for sw in swatches:
            wl_w, sp_w = spectrum_in_wl_window(sw.wavelengths, sw.spectrum, lo, hi)
            bgr = sw.bgr
            rgb = (bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0)

            fig = plt.figure(figsize=(10, 7))
            gs = fig.add_gridspec(2, 2, height_ratios=[0.38, 1.0], width_ratios=[0.22, 1.0], hspace=0.35, wspace=0.25)
            ax_patch = fig.add_subplot(gs[0, 0])
            ax_patch.imshow(np.full((1, 1, 3), rgb, dtype=float), aspect="auto")
            ax_patch.set_xticks([])
            ax_patch.set_yticks([])
            for s in ax_patch.spines.values():
                s.set_edgecolor("#222222")
                s.set_linewidth(1.2)

            ax_txt = fig.add_subplot(gs[0, 1])
            ax_txt.axis("off")
            ax_txt.text(
                0.0,
                0.92,
                f"{sw.label}  [{sw.mode}]",
                fontsize=15,
                fontweight="bold",
                va="top",
                transform=ax_txt.transAxes,
            )
            ax_txt.text(
                0.0,
                0.58,
                f"XYZ  X = {sw.X:.4f}    Y = {sw.Y:.4f}    Z = {sw.Z:.4f}",
                fontsize=11,
                va="top",
                family="monospace",
                transform=ax_txt.transAxes,
            )
            ax_txt.text(
                0.0,
                0.32,
                f"L*a*b*   L* = {sw.L:.4f}    a* = {sw.a:.4f}    b* = {sw.b:.4f}",
                fontsize=11,
                va="top",
                family="monospace",
                transform=ax_txt.transAxes,
            )
            ax_txt.text(
                0.0,
                0.06,
                f"Spectrum after corrections, {lo:.0f}–{hi:.0f} nm",
                fontsize=9,
                va="top",
                color="#555555",
                transform=ax_txt.transAxes,
            )

            ax_sp = fig.add_subplot(gs[1, :])
            if len(wl_w) > 0:
                ax_sp.plot(wl_w, sp_w, color="#1f77b4", linewidth=1.5, label="Measured (corrected)")
            ax_sp.set_xlim(lo, hi)
            ax_sp.set_xlabel(xa)
            ax_sp.set_ylabel("Intensity (display scale)")
            ax_sp.set_title(f"Swatch {sw.label}")
            _apply_spectrum_ticks(ax_sp)
            ax_sp.legend(loc="best", fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return path
