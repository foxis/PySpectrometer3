# PySpectrometer3 Code Quality Report

**Generated:** 2026-03-09  
**Focus:** Duplicate functionality, dead code, SRP violations, long methods  
**Priority:** Duplicate/dead first, then SRP, then long methods

---

## 1. Duplicate Functionality (Priority: HIGH) — DONE

| # | Location | Description | Status |
|---|----------|-------------|--------|
| 1.1 | `gui/control_bar.py` vs `modes/*.py` | Button definitions duplicated. | FIXED: mode.get_buttons() is single source; ControlBar accepts buttons from mode_instance. |
| 1.2 | renderer, extraction, picamera | uint16→uint8 display scaling repeated. | FIXED: utils.display.scale_to_uint8(). |
| 1.3 | `modes/calibration.py` | Sensitivity np.interp duplicated. | FIXED: _interp_sensitivity(wavelengths) helper. |
| 1.4 | `core/calibration.py` | _read_cal_file and _read_pixels_wavelengths_only share parsing. | FIXED: _parse_pixels_wavelengths(lines) shared; _read_cal_file uses it. |

---

## 2. Dead Functionality (Priority: HIGH)

| # | Location | Description | Fix |
|---|----------|-------------|-----|
| 2.1 | `spectrometer.py:310` | `_on_load_reference()` — never registered. | Remove or wire to "Load Ref" button if feature is planned. |
| 2.2 | `spectrometer.py:314` | `_on_use_as_reference()` — never registered. | Remove or wire to button. |
| 2.3 | `spectrometer.py:323` | `_on_capture_dark()` — never registered; set_dark exists and works. | Remove (redundant with set_dark flow). |
| 2.4 | `spectrometer.py:398` | `_on_fit_xyz()` — never registered. | Remove or implement if Color Science XYZ fit is planned. |
| 2.5 | `spectrometer.py:457` | `_on_toggle_auto_gain_meas()` — never registered; `_on_toggle_auto_gain` handles both modes. | Remove. |
| 2.6 | `spectrometer.py:530` | `_on_toggle_auto_gain_cal()` — never registered. | Remove. |
| 2.7 | `capture/picamera.py:396` | `capture_normalized()` — never called. | Remove or use for display path if needed. |
| 2.8 | `display/renderer.py:562` | `_draw_sampling_lines()` — never called. | Remove (superseded by `_draw_rotated_crop_box`). |
| 2.9 | `display/renderer.py:634` | `_draw_sampling_lines_full()` — marked DEPRECATED, never called. | Remove. |
| 2.10 | `modes/calibration.py:427` | `calibrate_from_peaks()` — never called (doc: "for later use"). | Remove or expose via UI. |
| 2.11 | `modes/calibration.py:188` | `cycle_source()` — never called; source chosen via buttons. | Remove or add "next source" shortcut. |
| 2.12 | `modes/base.py:174–206` | `set_black_reference`, `set_white_reference`, `apply_references`, `calculate_auto_gain_adjustment` — MeasurementMode uses its own dark/white API and AutoGainController. | Remove or keep for future modes; document if kept. |
| 2.13 | `display/renderer.py:180` | `set_preview_mode()` — never called; only `cycle_preview_mode` used. | Remove or use for programmatic control. |

---

## 3. SRP Violations (Priority: MEDIUM)

| # | Location | Description | Fix |
|---|----------|-------------|-----|
| 3.1 | `spectrometer.py` | Orchestrator + init + key callbacks + button callbacks + auto gain/exposure + main loop + save — ~1000 lines. | Extract `CallbackRegistry`, `MainLoop`, `SnapshotExporter`. |
| 3.2 | `display/renderer.py` DisplayManager | Window setup + mouse handling + graph/peak/cursor/overlay/status rendering + slider + control bar. | Extract `GraphRenderer`, `PreviewComposer`, `StatusRenderer`. |
| 3.3 | `modes/calibration.py` CalibrationMode | Source choice + overlay + sensitivity + auto-cal (peak + correlation) + calibrate_from_peaks + CMOS load. | Extract `AutoCalibrator`, `SensitivityCorrection`. |
| 3.4 | `core/calibration.py` Calibration | Load, save, recalibrate, compute, graticule params, extraction params. | Extract `CalibrationFileIO`, keep Calibration for logic. |

---

## 4. Long Methods (Priority: LOW)

| # | Location | Approx. Lines | Fix |
|---|----------|---------------|-----|
| 4.1 | `display/renderer.py:231` `render()` | ~155 | Split into `_render_graph()`, `_compose_preview()`, `_compose_window()`. |
| 4.2 | `spectrometer.py:631` `run()` main loop | ~115 | Extract `_process_frame()`, `_update_display()`. |
| 4.3 | `capture/picamera.py:261` `_process_raw_frame()` | ~70 | Extract `_unpack_packed_uint8()`, `_handle_uint16()`. |
| 4.4 | `processing/extraction.py:320` `detect_angle()` | ~105 | Extract `_build_visualization()`. |
| 4.5 | `modes/calibration.py:243` `_auto_calibrate_peak_ordered()` | ~68 | Extract `dispersion_score`, `_try_shift_mapping()`. |
| 4.6 | `modes/calibration.py:314` `_auto_calibrate_correlation()` | ~95 | Extract `_build_loss_fn()`, `_extract_cal_points()`. |
| 4.7 | `core/calibration.py:351` `_compute_calibration()` | ~45 | Already has helpers; further split if needed. |
| 4.8 | `core/calibration.py:303` `_read_cal_file()` | ~50 | Reuse `_read_pixels_wavelengths_only()` for pixel/wavelength parsing. |

---

## Summary

| Category | Count |
|----------|-------|
| Duplicate functionality | 4 |
| Dead functionality | 13 |
| SRP violations | 4 |
| Long methods | 8 |

**Recommended order:**
1. Remove dead code (2.1–2.13)
2. Unify button definitions (1.1)
3. Extract shared uint16→uint8 helper (1.2)
4. Address remaining duplicates (1.3, 1.4)
5. SRP refactors (3.x)
6. Split long methods (4.x)
