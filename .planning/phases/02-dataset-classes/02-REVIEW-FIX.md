---
phase: 02-dataset-classes
fixed_at: "2026-05-11T00:30:00Z"
review_path: .planning/phases/02-dataset-classes/02-REVIEW.md
iteration: 1
findings_in_scope: 15
fixed: 15
skipped: 0
status: all_fixed
---

# Phase 02: Code Review Fix Report

**Fixed at:** 2026-05-11T00:30:00Z
**Source review:** .planning/phases/02-dataset-classes/02-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 15
- Fixed: 15
- Skipped: 0

## Fixed Issues

### CR-01: ForecastingStrategySingleFile.get_num_sequences -- off-by-one (range excludes index 0)

**Files modified:** `src/tscollection/datasets/datasets/classes/strategies.py`
**Commit:** 14ebfd0
**Applied fix:** Changed `range(..., 0, -step)` to `range(..., -1, -step)` to include start index 0 in valid window count.

### CR-02: ClassificationStrategySingleFile.get_num_sequences -- double off-by-one

**Files modified:** `src/tscollection/datasets/datasets/classes/strategies.py`
**Commit:** 38b7678
**Applied fix:** Changed range end from `0` to `-1` AND filter from `<` to `<=` for correct boundary inclusion.

### CR-03: ClassificationStrategyMultipleFiles.get_num_sequences_per_file -- same double off-by-one

**Files modified:** `src/tscollection/datasets/datasets/classes/strategies.py`
**Commit:** ae9dcdc
**Applied fix:** Applied same two corrections as CR-02 to per-file counting method.

### CR-04: FlexibleTimeSeriesDatasetMultipleFiles._go_to_idx -- wrong file at boundary indices

**Files modified:** `src/tscollection/datasets/datasets/classes/flexible.py`
**Commit:** 284493f
**Applied fix:** Removed the broken `if idx in accumulated` branch with `.index()`, using bisect alone for correct file-boundary mapping. Also resolves IN-02.

### CR-05: ETTDataset default transforms produce numpy arrays instead of tensors

**Files modified:** `src/tscollection/datasets/datasets/ett.py`
**Commit:** 8a380b7
**Applied fix:** Removed `convert_data_to_np_array` from default `transformations_sequence` tuple. Also updated type hint to `tuple[Callable, ...]`.

### WR-01: Flexible datasets do not handle negative indices

**Files modified:** `src/tscollection/datasets/datasets/classes/flexible.py`
**Commit:** 08bf09f
**Applied fix:** Added negative index normalization (`idx = len(self) + idx`) and bounds check (`idx < 0 or idx >= len(self)`) in both SingleFile and MultipleFiles `_go_to_idx` methods.

### WR-02: _get_sample_3 crashes when label is None

**Files modified:** `src/tscollection/datasets/datasets/classes/fixed.py`
**Commit:** e51b4c1
**Applied fix:** Added None guard before calling `self._transform(label)` in `_get_sample_3`, raising informative RuntimeError.

### WR-03: convert_numpy_to_tensor raises unhelpful KeyError for invalid dtype

**Files modified:** `src/tscollection/datasets/datasets/transformations.py`
**Commit:** 1be6e7d
**Applied fix:** Replaced bare `KeyError` with `ValueError` that lists valid dtype options.

### WR-04: Tests validate wrong expected values, masking bugs

**Files modified:** `tests/test_strategies.py`, `tests/test_ett_dataset.py`
**Commit:** e004a6b
**Applied fix:** Updated all assertion values to match mathematically correct counts:
- `test_forecasting_num_sequences`: 80 -> 81
- `test_classification_num_sequences`: 14 -> 16
- `test_multifile_num_sequences`: 18 -> 22
- `test_multifile_per_file_counts`: [4, 14] -> [6, 16]
- `test_ett_length`: `<= 81` -> `== 81`

### WR-05: expand_data_dimensionality always converts tensor to numpy

**Files modified:** `src/tscollection/datasets/datasets/transformations.py`
**Commit:** b725571
**Applied fix:** Track `was_tensor` flag and convert result back to `torch.Tensor` when input was a tensor. Updated return type hint to `np.ndarray | torch.Tensor`.

### WR-06: Redundant attribute assignment in FlexibleTimeSeriesDatasetMultipleFiles

**Files modified:** `src/tscollection/datasets/datasets/classes/flexible.py`
**Commit:** dd89dad
**Applied fix:** Removed redundant `self._seq_len = seq_len`, `self._step = step`, and `self._n = 0` assignments already handled by parent `__init__`.

### IN-01: Bare `tuple` type hint in ucr.py and uea.py

**Files modified:** `src/tscollection/datasets/datasets/ucr.py`, `src/tscollection/datasets/datasets/uea.py`
**Commit:** c7f8675
**Applied fix:** Changed `transformations_sequence: tuple` to `tuple[Callable, ...]` with TYPE_CHECKING import.

### IN-02: O(n) list membership check in _go_to_idx

**Resolved by:** CR-04 (commit 284493f)
**Applied fix:** No separate change needed; the broken `if idx in ...` branch with `.index()` was removed by CR-04.

### IN-03: Missing test for FlexibleTimeSeriesDatasetMultipleFiles

**Files modified:** `tests/test_flexible_dataset.py`
**Commit:** aae76cb (initial), 97495e1 (correction)
**Applied fix:** Added `test_flexible_multifile_boundary_indices` that verifies file boundary indices map to the correct file using bisect logic.

### IN-04: Missing test for UEAClassificationMultivariateDataset

**Files modified:** `tests/test_uea_dataset.py` (new file)
**Commit:** 5f7ccea
**Applied fix:** Created test file with four tests covering WITH_LABELS mode, WITHOUT_LABELS mode, dataset length, and default expand_dims_axis=None behavior.

## Skipped Issues

None — all findings were resolved.

## Test Results

All 40 tests pass after fixes:
```
======================== 40 passed, 1 warning in 0.74s ========================
```

---

_Fixed: 2026-05-11T00:30:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
