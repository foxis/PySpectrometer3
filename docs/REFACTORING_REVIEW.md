# Review: Phase 1.4 — scale_intensity_to_graph Refactor

## Summary

The refactor extracts a central `scale_intensity_to_graph()` helper and replaces 5 ad-hoc scaling sites. Review covers correctness, best practices, architecture alignment, and testing.

---

## What Changed

| File | Change |
|------|--------|
| `utils/graph_scale.py` | **New** — pure function `scale_intensity_to_graph(intensity, graph_height, margin=10)` |
| `modes/calibration.py` | `get_overlay`, `get_sensitivity_curve` use helper (2 sites) |
| `modes/measurement.py` | `get_overlay` uses helper (1 site) |
| `display/renderer.py` | `_render_reference_spectrum`, `_render_spectrum`, `_render_peaks` use helper (3 sites) |

---

## Correctness

### Behavior

- **Calibration overlay:** `ref_intensity * (graph_height - 10)` → `scale_intensity_to_graph(ref_intensity, graph_height)` — equivalent; margin=10 is default.
- **Measurement overlay:** `(ref / max_val) * (graph_height - 10)` → `scale_intensity_to_graph(ref / max_val, graph_height)` — equivalent.
- **Renderer (reference spectrum):** Previously `scale = height - 1`, now `margin=1` — full height preserved.
- **Renderer (spectrum line, peaks):** Same mapping (scale by `height - 1`) — equivalent.

### Edge Case

- **Bug:** If `graph_height <= margin` (e.g. `height=5`, `margin=10`), `max_val = -5` and `np.clip(result, 0, -5)` clamps values to -5. Unusual in practice but incorrect.
- **Fix:** Use `max_val = max(0, graph_height - margin)` and `scale = max(0.0, float(graph_height - margin))` or equivalent guard.

---

## Best Practices

| Practice | Status |
|----------|--------|
| **SRP** | `scale_intensity_to_graph` has a single responsibility. |
| **DRY** | Five call sites replaced with one shared implementation. |
| **Pure function** | No side effects; only numpy dependency. |
| **Typing** | `Union[np.ndarray, float]` for input; docstring for output. |
| **Docstring** | Args, return, and margin semantics documented. |

---

## Architecture Alignment

| Aspect | Status |
|--------|--------|
| **Layer boundaries** | Modes (data/logic) and display (viewer) both use `utils`. No modes → display dependency. |
| **Module map** | ARCHITECTURE §8 lists `utils/color.py`; `utils/graph_scale.py` fits the same pattern. |
| **REFACTORING_GUIDE** | Task 1.4.1 marked done; correctly uses `utils` (not `display`). |

---

## Testing Strategy

### Gaps

1. **No unit test for `scale_intensity_to_graph`** — Extracted logic is untested. REFACTORING_GUIDE says: “add or adjust tests for extracted components.”
2. **No integration coverage of overlay paths** — `get_overlay` and `get_sensitivity_curve` are not exercised in tests. Calibration tests focus on `auto_calibrate`.

### Recommendations

1. Add a unit test for `scale_intensity_to_graph`:
   - Array: `[0, 0.5, 1]` with `graph_height=320`, `margin=10` → `[0, 155, 310]`.
   - Scalar: `0.5` with `graph_height=320`, `margin=1` → `159.5`.
   - Edge: `graph_height < margin` → safe fallback (no negative clip).
2. Optionally add a minimal integration test for `CalibrationMode.get_overlay()` to confirm overlay is produced and scaled.

---

## Verdict

- **Functionality:** Preserved; behavior matches previous implementation.
- **Best practices:** SRP, DRY, pure function, clear API.
- **Architecture:** Aligned with Controller/Viewer/Data split and module map.
- **Testing:** Missing unit test for the new helper and no regression coverage for overlay paths.

---

*Review date: 2026-03-09*
