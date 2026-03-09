# PySpectrometer3 Code Quality Report

**Generated:** 2026-03-09  
**Last updated:** 2026-03-10  
**Focus:** Duplicate functionality, dead code, SRP violations, long methods  
**Priority:** Duplicate/dead first, then SRP, then long methods

---

## What Was Improved (2026-03-10)

### Duplicate Functionality — DONE
- **1.1** `mode.get_buttons()` single source; ControlBar accepts buttons from mode_instance.
- **1.2** `utils.display.scale_to_uint8()` centralizes uint16→uint8 scaling.
- **1.3** `_interp_sensitivity(wavelengths)` in calibration mode.
- **1.4** `_parse_pixels_wavelengths(lines)` shared in `core/calibration.py`.

### Dead Code Removed
- **2.1–2.6** Spectrometer refactored to thin orchestrator (~306 lines); unregistered callbacks removed.
- **2.7–2.9, 2.11, 2.13** `capture_normalized`, `_draw_sampling_lines`, `_draw_sampling_lines_full`, `cycle_source`, `set_preview_mode` removed.

### SRP / Structure
- **3.1** Spectrometer reduced to thin orchestrator; mode logic delegated to mode classes.
- **3.4** `CalibrationFileIO` extracted for load/save; Calibration keeps compute, graticule, extraction params.
- **4.2** `run()` main loop simplified (~30 lines).
- **4.8** `_read_cal_file()` uses `_parse_pixels_wavelengths()`.

### Tooling & CLI
- Poetry migration; Ruff for lint/format.
- Scripts: `calibrate`, `measure`, `colors`, `raman`, `stream`.
- `poetry run stream` — MJPEG camera stream (Picamera2/OpenCV).

---

## 1. Duplicate Functionality (Priority: HIGH) — DONE

| # | Location | Description | Status |
|---|----------|-------------|--------|
| 1.1 | `gui/control_bar.py` vs `modes/*.py` | Button definitions duplicated. | FIXED |
| 1.2 | renderer, extraction, picamera | uint16→uint8 display scaling repeated. | FIXED |
| 1.3 | `modes/calibration.py` | Sensitivity np.interp duplicated. | FIXED |
| 1.4 | `core/calibration.py` | _read_cal_file and _read_pixels_wavelengths_only share parsing. | FIXED |

---

## 2. Dead Functionality (Priority: HIGH)

| # | Location | Description | Status |
|---|----------|-------------|--------|
| 2.1–2.6 | `spectrometer.py` | Unregistered callbacks. | FIXED (removed with refactor) |
| 2.7 | `capture/picamera.py` | `capture_normalized()` | FIXED (removed) |
| 2.8 | `display/renderer.py` | `_draw_sampling_lines()` | FIXED (removed) |
| 2.9 | `display/renderer.py` | `_draw_sampling_lines_full()` | FIXED (removed) |
| 2.10 | `modes/calibration.py:638` | `calibrate_from_peaks()` — never called. | Remove or expose via UI. |
| 2.11 | `modes/calibration.py` | `cycle_source()` | FIXED (removed) |
| 2.12 | `modes/base.py:174–206` | `set_black_reference`, `set_white_reference`, `apply_references`, `calculate_auto_gain_adjustment` — MeasurementMode uses own API. | Remove or document if kept. |
| 2.13 | `display/renderer.py` | `set_preview_mode()` | FIXED (removed) |

---

## 3. SRP Violations (Priority: MEDIUM)

| # | Location | Description | Status |
|---|----------|-------------|--------|
| 3.1 | `spectrometer.py` | Was ~1000 lines. | IMPROVED: thin orchestrator ~306 lines |
| 3.2 | `display/renderer.py` DisplayManager | Window + mouse + graph/peak/cursor/overlay/status/slider/control bar. | Extract `GraphRenderer`, `PreviewComposer`, `StatusRenderer`. |
| 3.3 | `modes/calibration.py` CalibrationMode | Source + overlay + sensitivity + auto-cal + CMOS load. | Extract `AutoCalibrator`, `SensitivityCorrection`. |
| 3.4 | `core/calibration.py` Calibration | Load, save, compute, graticule, extraction params. | FIXED: `CalibrationFileIO` extracted. |

---

## 4. Long Methods (Priority: LOW)

| # | Location | Approx. Lines | Status |
|---|----------|---------------|--------|
| 4.1 | `display/renderer.py` `render()` | ~155 | Split into `_render_graph()`, `_compose_preview()`, `_compose_window()`. |
| 4.2 | `spectrometer.py` `run()` | — | DONE (simplified) |
| 4.3 | `capture/picamera.py` `_process_raw_frame()` | ~70 | Extract `_unpack_packed_uint8()`, `_handle_uint16()`. |
| 4.4 | `processing/extraction.py` `detect_angle()` | ~105 | Extract `_build_visualization()`. |
| 4.5 | `modes/calibration.py` `_auto_calibrate_peak_ordered()` | ~68 | Extract `dispersion_score`, `_try_shift_mapping()`. |
| 4.6 | `modes/calibration.py` `_auto_calibrate_correlation()` | ~95 | Extract `_build_loss_fn()`, `_extract_cal_points()`. |
| 4.7 | `core/calibration.py` `_compute_calibration()` | ~45 | Further split if needed. |
| 4.8 | `core/calibration.py` `_read_cal_file()` | — | DONE (uses shared parsing) |

---

## Summary

| Category | Done | Remaining |
|----------|------|-----------|
| Duplicate functionality | 4 | 0 |
| Dead functionality | 11 | 2 |
| SRP violations | 2 (partial) | 2 |
| Long methods | 2 | 6 |

**Recommended next steps:**
1. Remove or expose dead code (2.10 `calibrate_from_peaks`; 2.12 base mode refs)
2. SRP refactors (3.2–3.3)
3. Split long methods (4.1, 4.3–4.7)
