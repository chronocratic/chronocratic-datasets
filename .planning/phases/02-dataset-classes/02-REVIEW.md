---
phase: 02-dataset-classes
reviewed: 2026-05-11T00:00:00Z
depth: deep
files_reviewed: 14
files_reviewed_list:
  - src/tscollection/datasets/datasets/classes/strategies.py
  - src/tscollection/datasets/datasets/classes/__init__.py
  - tests/test_strategies.py
  - src/tscollection/datasets/datasets/classes/fixed.py
  - src/tscollection/datasets/datasets/classes/flexible.py
  - tests/test_fixed_dataset.py
  - tests/test_flexible_dataset.py
  - src/tscollection/datasets/datasets/ucr.py
  - src/tscollection/datasets/datasets/uea.py
  - src/tscollection/datasets/datasets/ett.py
  - src/tscollection/datasets/datasets/__init__.py
  - tests/test_ucr_dataset.py
  - tests/test_ett_dataset.py
  - tests/test_transformations.py
findings:
  critical: 5
  warning: 6
  info: 4
  total: 15
status: all_fixed
---

# Phase 02: Code Review Report

**Reviewed:** 2026-05-11T00:00:00Z
**Depth:** deep
**Files Reviewed:** 14
**Status:** issues_found

## Summary

Deep review of the dataset class hierarchy (Strategy pattern, FixedDataset, FlexibleDataset, domain wrappers) and associated tests. Five critical bugs were found, all verified empirically. The most severe are off-by-one errors in every strategy's `get_num_sequences` implementation, causing datasets to silently skip valid samples. A boundary-mapping bug in `FlexibleTimeSeriesDatasetMultipleFiles` returns data from the wrong file. ETTDataset's transform pipeline converts tensors back to numpy arrays, breaking PyTorch conventions. All existing tests for the strategies validate the incorrect expected values, masking the bugs entirely.

## Critical Issues

### CR-01: ForecastingStrategySingleFile.get_num_sequences -- off-by-one (range excludes index 0)

**File:** `src/tscollection/datasets/datasets/classes/strategies.py:129`
**Issue:** `range(num_samples_ts - seq_len - self._forecast_horizon + 1, 0, -step)` stops at 1, never including 0. This means the first valid sliding window (starting at index 0) is never counted. For a (200, 7) array with seq_len=96, step=1, forecast_horizon=24, the code returns 80 instead of the correct 81.

Verified: `range(81, 0, -1)` produces `[81, 80, ..., 1]` -- 81 values. After filtering out 81 (since 81+96+24=201 > 200), only 80 remain. The correct count includes start index 0.

**Test impact:** `tests/test_strategies.py:31` asserts `count == 80`, validating the wrong value.

**Fix:**
```python
possible_steps = list(
    range(num_samples_ts - seq_len - self._forecast_horizon + 1, -1, -step)
)
```

### CR-02: ClassificationStrategySingleFile.get_num_sequences -- double off-by-one

**File:** `src/tscollection/datasets/datasets/classes/strategies.py:158-160`
**Issue:** Two bugs compound:
1. `range(num_samples_ts - seq_len, 0, -step)` excludes 0 (same root cause as CR-01).
2. The filter `e < num_samples_ts` should be `e <= num_samples_ts`. Start index 150 with seq_len=50 produces a valid window `data[150:200]` (exactly 50 elements), but `150+50=200` fails the `< 200` check.

For a (200,) array with seq_len=50, step=10, the code returns 14 instead of the correct 16 (valid start indices: 0, 10, 20, ..., 140, 150).

**Test impact:** `tests/test_strategies.py:64` asserts `count == 14`, validating the wrong value.

**Fix:**
```python
possible_steps = list(range(num_samples_ts - seq_len, -1, -step))
possible_ends = [x + seq_len for x in possible_steps]
return len([e for e in possible_ends if e <= num_samples_ts])
```

### CR-03: ClassificationStrategyMultipleFiles.get_num_sequences_per_file -- same double off-by-one

**File:** `src/tscollection/datasets/datasets/classes/strategies.py:193-195`
**Issue:** Identical bugs as CR-02, affecting per-file counting in multi-file classification datasets. Each file loses 2 windows (one at the start due to range excluding 0, one at the end due to `<` instead of `<=`).

For files of length 100 and 200 (seq_len=50, step=10): code returns [4, 14] instead of the correct [6, 16].

**Test impact:** `tests/test_strategies.py:129` asserts `counts == [4, 14]`.

**Fix:** Apply the same two corrections as CR-02 to lines 193-195.

### CR-04: FlexibleTimeSeriesDatasetMultipleFiles._go_to_idx -- wrong file at boundary indices

**File:** `src/tscollection/datasets/datasets/classes/flexible.py:218-220`
**Issue:** When `idx` equals an accumulated boundary value, the special-case branch uses `.index(idx)` which returns the position of that value in the accumulated list, NOT the file it belongs to. The `bisect` function alone handles all cases correctly (it returns the right insertion point, which IS the correct file index).

Example: accumulated = [4, 18] (file 0 has 4 sequences, file 1 has 14). For idx=4 (first sequence of file 1):
- Current code: `4 in [4, 18]` is True, `.index(4)` returns 0, so `_current_file=0`. WRONG -- should be file 1.
- bisect alone: `bisect([4, 18], 4)` returns 1. `_current_file=1`. Correct.

This causes data from the wrong file to be returned for every global index that lands exactly on a file boundary.

**Fix:** Remove the `if idx in ...` branch entirely and use only the `bisect` logic:
```python
def _go_to_idx(self, idx: int) -> None:
    if idx >= len(self):
        msg = 'Index out of range'
        raise IndexError(msg)
    file_num = bisect(self._accumulated_num_sequences_per_file, idx)
    self._current_file = file_num
    self._n = (
        idx - self._accumulated_num_sequences_per_file[file_num - 1]
        if file_num != 0
        else idx
    )
```

### CR-05: ETTDataset default transforms produce numpy arrays instead of tensors

**File:** `src/tscollection/datasets/datasets/ett.py:50`
**Issue:** The default `transformations_sequence` is `(convert_numpy_to_tensor, convert_data_to_np_array)`. The pipeline applies them in order: numpy array -> torch.Tensor -> numpy array. The final output is `np.ndarray`, not `torch.Tensor`. This breaks PyTorch Dataset convention where samples should be tensors for GPU compatibility.

Verified empirically: `np.array(torch_tensor).astype(np.float32)` returns `np.ndarray`, not `torch.Tensor`.

Additionally, `convert_data_to_np_array` has type hint `data: list | tuple`, but at runtime it receives a `torch.Tensor` (the output of `convert_numpy_to_tensor`). This type mismatch triggers a deprecation warning with NumPy 2.0+ regarding `__array__` not accepting a `copy` keyword.

**Fix:** Remove `convert_data_to_np_array` from the default transforms:
```python
transformations_sequence: tuple = (convert_numpy_to_tensor,),
```

## Warnings

### WR-01: Flexible datasets do not handle negative indices

**File:** `src/tscollection/datasets/datasets/classes/flexible.py:138-142` (SingleFile) and line 213-228 (MultipleFiles)

**Issue:** `_go_to_idx` checks `if idx >= len(self)` but does not check for negative indices. Accessing `ds[-1]` sets `_n = -1`, and `_get_current_data` returns `self._data[-1 : -1 + seq_len]`, which wraps around and yields incorrect data silently.

While PyTorch DataLoaders use positive indices internally, negative indexing is standard Python convention and manual testing may rely on it.

**Fix:** Add a negative index check or normalize:
```python
def _go_to_idx(self, idx: int) -> None:
    if idx < 0:
        idx = len(self) + idx
    if idx < 0 or idx >= len(self):
        msg = 'Index out of range'
        raise IndexError(msg)
    self._n = idx
```

### WR-02: _get_sample_3 crashes when label is None

**File:** `src/tscollection/datasets/datasets/classes/fixed.py:122-126`

**Issue:** In FORECASTING mode, `_get_sample_3` calls `self._transform(self._get_current_label())`. If `_get_current_label()` returns `None` (which happens in `FixedTimeSeriesDataset` when `labels` is not provided), then `self._transform(None)` passes `None` to `convert_numpy_to_tensor`, which raises `TypeError` because `None` is not `np.ndarray`.

Although FORECASTING mode with no labels is unusual for fixed datasets, the crash is uninformative.

**Fix:** Guard against None in `_get_sample_3`:
```python
def _get_sample_3(self) -> tuple[object, object]:
    sample = self._transform(self._get_current_data())
    label = self._get_current_label()
    if label is None:
        msg = 'FORECASTING mode requires labels; _get_current_label returned None'
        raise RuntimeError(msg)
    return (sample, self._transform(label))
```

### WR-03: convert_numpy_to_tensor raises unhelpful KeyError for invalid dtype

**File:** `src/tscollection/datasets/datasets/transformations.py:32-38`

**Issue:** `dtype_map[dtype]` raises a bare `KeyError` if `dtype` is not one of the recognized strings. The error message does not indicate which values are valid, making debugging harder.

**Fix:**
```python
if dtype not in dtype_map:
    msg = f'Unsupported dtype "{dtype}". Choose from {list(dtype_map.keys())}.'
    raise ValueError(msg)
return torch.from_numpy(data).to(dtype=dtype_map[dtype])
```

### WR-04: Tests validate wrong expected values, masking bugs

**File:** `tests/test_strategies.py:31, 64, 129` and `tests/test_ett_dataset.py:46`

**Issue:** The strategy tests assert incorrect expected counts that match the buggy implementation rather than the correct mathematical answer:
- `test_forecasting_num_sequences` asserts 80 (correct: 81)
- `test_classification_num_sequences` asserts 14 (correct: 16)
- `test_multifile_per_file_counts` asserts [4, 14] (correct: [6, 16])
- `test_ett_length` uses `<= 81` which passes with the buggy value 80 instead of asserting `== 81`

These tests give false confidence that the implementation is correct.

**Fix:** Update all expected values to the mathematically correct counts after fixing CR-01 through CR-03.

### WR-05: expand_data_dimensionality always converts tensor to numpy

**File:** `src/tscollection/datasets/datasets/transformations.py:70-71`

**Issue:** When `expand_dims_axis` is set (e.g., UCRClassificationUnivariateDataset defaults to axis=1), the transform pipeline applies `convert_numpy_to_tensor` first, then `expand_data_dimensionality`, which calls `data.numpy()` on the tensor. The final output is `np.ndarray`, not `torch.Tensor`. The UCR test comment on line 27 acknowledges this as "Pitfall 3 in research docs."

This means any dataset using `expand_dims_axis` returns numpy arrays instead of tensors, which is unexpected for a PyTorch Dataset.

**Fix:** Make `expand_data_dimensionality` preserve torch.Tensor type:
```python
def expand_data_dimensionality(
    data: np.ndarray | torch.Tensor | list | tuple, expand_dims_axis: int
) -> np.ndarray | torch.Tensor:
    was_tensor = isinstance(data, torch.Tensor)
    if was_tensor:
        data = data.numpy()
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    result = np.expand_dims(data, axis=expand_dims_axis)
    return torch.from_numpy(result) if was_tensor else result
```

### WR-06: Redundant attribute assignment in FlexibleTimeSeriesDatasetMultipleFiles

**File:** `src/tscollection/datasets/datasets/classes/flexible.py:199-201`

**Issue:** `_seq_len`, `_step`, and `_n` are already set by the parent `FlexibleTimeSeriesDataset.__init__` (lines 77-80), then reassigned identically in the child class (lines 199-201). This is harmless redundancy but indicates the parent-child contract is unclear about which attributes each level owns.

**Fix:** Remove the redundant lines from `FlexibleTimeSeriesDatasetMultipleFiles.__init__`.

## Info

### IN-01: Bare `tuple` type hint in ucr.py and uea.py

**File:** `src/tscollection/datasets/datasets/ucr.py:43` and `src/tscollection/datasets/datasets/uea.py:44`

**Issue:** `transformations_sequence: tuple = (convert_numpy_to_tensor,)` uses an unparameterized `tuple`, which is `tuple[Any, ...]`. The parent classes use `list[Callable] | tuple[Callable, ...]`.

**Fix:** Use `tuple[Callable, ...]` for consistency (with `from __future__ import annotations` or `TYPE_CHECKING` guard).

### IN-02: O(n) list membership check in _go_to_idx

**File:** `src/tscollection/datasets/datasets/classes/flexible.py:218`

**Issue:** `if idx in self._accumulated_num_sequences_per_file` uses list `__contains__` which is O(n) per call. Combined with `.index()` on the next line (another O(n)), this makes boundary-case lookups O(n). In practice, file counts are small so this is not measurable, but a set or direct bisect would be cleaner.

**Fix:** Remove the special case entirely (see CR-04 fix), which also eliminates this inefficiency.

### IN-03: Missing test for FlexibleTimeSeriesDatasetMultipleFiles

**File:** `tests/test_flexible_dataset.py`

**Issue:** No test exercises `FlexibleTimeSeriesDatasetMultipleFiles` or its `_go_to_idx` bisect logic. The CR-04 bug exists because this class is untested. The strategy tests cover `ClassificationStrategyMultipleFiles` but not the dataset wrapper that uses it.

**Fix:** Add a test that creates a multi-file flexible dataset and verifies that indices at file boundaries return data from the correct file.

### IN-04: Missing test for UEAClassificationMultivariateDataset

**File:** `tests/` (no test_uea_dataset.py)

**Issue:** UCR and ETT have dedicated test files, but UEA does not. The `test_multivariate_get_current_data` in `test_fixed_dataset.py` exercises the base class but not the UEA wrapper specifically.

**Fix:** Add `tests/test_uea_dataset.py` with wrapper-specific tests (default expand_dims_axis=None, 3D data shape, etc.).

---

_Reviewed: 2026-05-11T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_

## Fixed Issues

All 15 findings have been resolved:

### Critical Issues (5/5 fixed)

- **CR-01**: Fixed `ForecastingStrategySingleFile.get_num_sequences` — changed `range(..., 0, -step)` to `range(..., -1, -step)` to include index 0.
- **CR-02**: Fixed `ClassificationStrategySingleFile.get_num_sequences` — changed range end to `-1` AND filter `<` to `<=`.
- **CR-03**: Fixed `ClassificationStrategyMultipleFiles.get_num_sequences_per_file` — same two corrections as CR-02.
- **CR-04**: Fixed `FlexibleTimeSeriesDatasetMultipleFiles._go_to_idx` — removed broken `if idx in accumulated` branch with `.index()`, using bisect alone.
- **CR-05**: Fixed ETTDataset default transforms — removed `convert_data_to_np_array` from the pipeline.

### Warning Issues (6/6 fixed)

- **WR-01**: Added negative index handling in `_go_to_idx` for both SingleFile and MultipleFiles classes in flexible.py.
- **WR-02**: Added guard against None label in `_get_sample_3` in fixed.py, raising informative RuntimeError.
- **WR-03**: Improved `convert_numpy_to_tensor` KeyError to ValueError with valid dtype options listed.
- **WR-04**: Updated test expected values in test_strategies.py and test_ett_dataset.py to match corrected implementations.
- **WR-05**: Fixed `expand_data_dimensionality` to preserve torch.Tensor type when input is a tensor.
- **WR-06**: Removed redundant attribute assignments (`_seq_len`, `_step`, `_n`) in `FlexibleTimeSeriesDatasetMultipleFiles.__init__`.

### Info Issues (4/4 fixed)

- **IN-01**: Fixed bare `tuple` type hints in ucr.py and uea.py to `tuple[Callable, ...]`.
- **IN-02**: Already resolved by CR-04 fix (removed the O(n) list `.index()` call).
- **IN-03**: Added `test_flexible_multifile_boundary_indices` test in test_flexible_dataset.py.
- **IN-04**: Added `tests/test_uea_dataset.py` with wrapper-specific tests.

**Test results:** All 40 tests pass after fixes.
