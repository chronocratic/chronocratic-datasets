---
phase: 06-lightning-lifecycle
reviewed: 2026-05-28T00:00:00Z
depth: deep
files_reviewed: 13
files_reviewed_list:
  - src/tscollection/datasets/modules/_base/base.py
  - src/tscollection/datasets/modules/_base/classification.py
  - src/tscollection/datasets/modules/_base/forecasting.py
  - src/tscollection/datasets/utils/features.py
  - src/tscollection/datasets/utils/__init__.py
  - src/tscollection/datasets/datatypes/__init__.py
  - tests/test_modules_base.py
  - tests/test_modules_classification_forecasting.py
  - tests/test_modules_forecasting.py
  - tests/test_modules_ucr.py
  - tests/test_modules_uea.py
  - tests/test_utils_features.py
findings:
  critical: 2
  warning: 4
  info: 4
  total: 10
status: issues_found
---

# Phase 6: Lightning Lifecycle Purity -- Code Review Report

**Reviewed:** 2026-05-28T00:00:00Z
**Depth:** deep
**Files Reviewed:** 13
**Status:** issues_found

## Summary

This phase implements Lightning lifecycle purity across the base module hierarchy: setup() stage gating via `_setup_completed_stages` sentinel, scaler caching (classification uses a `create_data_scaler` closure; forecasting uses sklearn scalers directly), the `prepare_dimensions()` API with `_compute_dimensions()` subclass hook, collate regression tests, and the `datatypes/__init__.py` package restructuring.

Deep cross-file analysis reveals two critical bugs in the forecasting `setup()` guard logic that block test/predict stages after `setup(None)`, a missing re-export violating the phase plan, and several warnings around silent fallback paths and test quality gaps. The classification branch and base module are mostly sound, but the forecasting override introduces structural asymmetry in how `None` is treated as a sentinel value.

## Critical Issues

### CR-01: Forecasting `setup()` guard blocks test/predict after `setup(None)`

**File:** `src/tscollection/datasets/modules/_base/forecasting.py:191`
**Severity:** BLOCKER

The stage guard on line 191 includes `or None in self._setup_completed_stages`, which is overly broad. When `setup(None)` runs first (which Lightning may do explicitly), it adds `None` to the sentinel set. A subsequent `setup('test')` or `setup('predict')` then hits this guard and returns early without executing any of the test/predict-specific scaling or data-splitting logic.

Trace:
1. `setup(None)` -- passes guard, runs fit branch, adds `None` to `_setup_completed_stages`
2. `setup('test')` -- `'test' not in stages` (True), `None in stages` (True) -- returns early
3. Test/predict setup never runs; `_data_scaler_cache` is never populated for test-only scenarios

This differs from the base class (`base.py:252`), which does NOT have the `or None in ...` guard. The base class uses a more targeted check at lines 255-258 that only blocks fit/None equivalence, not test/predict.

**Fix:**
```python
# Replace line 191:
if stage in self._setup_completed_stages or None in self._setup_completed_stages:
    return

# With:
if stage in self._setup_completed_stages:
    return
# fit and None are equivalent -- skip if the other already ran
if stage in ('fit', None) and (
    'fit' in self._setup_completed_stages or None in self._setup_completed_stages
):
    return
```

### CR-02: Missing `TIME_FEATURE_COUNT` re-export from `utils/__init__.py`

**File:** `src/tscollection/datasets/utils/__init__.py`
**Severity:** BLOCKER

Phase plan 06-01 explicitly requires `TIME_FEATURE_COUNT` (value: 7) to be re-exported from `utils/__init__.py`. The constant is defined in `utils/features.py:8` and is imported directly by `forecasting.py:18`, but `utils/__init__.py` does not import or re-export it. This violates the documented contract and means callers following the established convention of importing from `tscollection.datasets.utils` cannot access the constant.

The current `__all__` in `utils/__init__.py` includes 12 names (lines 19-32), all from submodules, but `TIME_FEATURE_COUNT` is absent.

**Fix:**
```python
# Add to imports in utils/__init__.py:
from tscollection.datasets.utils.features import TIME_FEATURE_COUNT, extract_time_features

# Add 'TIME_FEATURE_COUNT' to __all__:
__all__ = [
    'FunctionComposer',
    'TIME_FEATURE_COUNT',  # Add this entry (alphabetical order)
    'centralize_variable_length_series',
    ...
]
```

## Warnings

### WR-01: Silent no-scaling fallback when `scale_data=True` but no scaler cache

**File:** `src/tscollection/datasets/modules/_base/forecasting.py:270-274`
**Severity:** WARNING

When `scale_data=True` and the user calls `setup('test')` without a prior `setup('fit')`, AND `_data_scaler_cache` is `None` (no external cache was set), the code falls through to the `else` branch at line 270. This branch calls `_transform_data()`, `_calculate_num_features()`, and `_split_data()` -- but does NOT apply any scaling, despite `scale_data=True`. No warning or error is raised. The user's explicit scaling preference is silently ignored.

This is reachable if a user creates the module, injects `_full_data` and slices manually (for testing), and calls `setup('test')` without setting `_data_scaler_cache`.

**Fix:**
```python
else:
    # No cached scaler and no prior fit
    if self.scale_data:
        import logging
        logging.warning(
            'scale_data=True but no fitted scaler cache available. '
            'Data will not be scaled. Call setup(stage="fit") first or '
            'provide a pre-fitted _data_scaler_cache.'
        )
    self._transform_data()
    self._calculate_num_features()
    self._split_data()
```

### WR-02: `_setup_completed_stages` is never reset -- prevents re-using DataModule instances

**File:** `src/tscollection/datasets/modules/_base/base.py:87` and `forecasting.py:191`
**Severity:** WARNING

The `_setup_completed_stages` sentinel set is populated during `setup()` but never cleared. If a user reuses the same DataModule instance across multiple `trainer.fit()` or `trainer.test()` calls (a valid Lightning pattern for hyperparameter sweeps), the sentinel will prevent `setup()` from re-running on subsequent uses.

While the common pattern is to create a fresh DataModule per run, this is not enforced by Lightning and the class does not document this limitation. A `reset()` or `_clear_sentinels()` method would improve reusability.

**Fix:** Add a public reset method:
```python
def reset(self) -> None:
    """Clear lifecycle sentinels to allow re-use of this DataModule."""
    self._setup_completed_stages.clear()
    self._prepare_data_called = False
```

### WR-03: Test for `setup(None)` coverage is misleading -- does not test `None` first

**File:** `tests/test_modules_base.py:289-309`
**Severity:** WARNING

The test `test_setup_none_covers_all_stages` calls `setup(stage=None)` then `setup(stage='fit')` and verifies `create_data_scaler` was called once. The docstring claims "None covers all" but the test only verifies that `None` blocks a subsequent `'fit'` call. It does NOT test the reverse: `setup('fit')` then `setup(None)`.

More importantly, it does not test that `setup(None)` followed by `setup('test')` correctly allows the test stage to run. This is precisely the scenario where CR-01 manifests in the base class (though the base class happens to be correct for this case -- the forecasting class is not).

**Fix:** Add a test:
```python
def test_setup_none_then_test_runs(self, concrete_module_class) -> None:
    """setup(None) should NOT block setup('test') from running."""
    mod = concrete_module_class(...)
    mod._train_data_samples = pd.DataFrame({'a': [1.0]})
    mod._valid_data_samples = pd.DataFrame({'a': [2.0]})
    mod._test_data_samples = pd.DataFrame({'a': [3.0]})

    with patch('...create_data_scaler', ...) as scaler_spy:
        scaler_spy.return_value = lambda t, v, te: (t, v, te)
        mod.setup(stage=None)
        mod.setup(stage='test')
        # test stage should have run (scaler called on test data)
        assert scaler_spy.call_count == 1  # cached, not recreated
        assert 'test' in mod._setup_completed_stages
```

### WR-04: `extract_time_features` relies on pandas `isocalendar().week` accessor pattern

**File:** `src/tscollection/datasets/utils/features.py:32`
**Severity:** WARNING

Line 32 uses `series.dt.isocalendar().week.to_numpy()`. The `isocalendar()` method returns a DataFrame (since pandas 1.1.0), and `.week` accesses the 'week' column. This works in pandas >= 1.1, but the code has no version guard. If pandas changes the return type of `isocalendar()` in a future major release (e.g., deprecating DataFrame column access in favor of a dedicated accessor), this could break silently.

The test (`test_utils_features.py`) verifies the output shape (N, 7) but does not assert the actual week values, so a regression in the week computation would not be caught.

**Fix:** Add an explicit assertion for week values in the test:
```python
def test_extract_time_features_week() -> None:
    """Seventh column is ISO week number."""
    dti = pd.date_range('2020-01-01', periods=1, freq='D')
    result = extract_time_features(dti)
    # 2020-01-01 is in ISO week 1
    assert result[0, 6] == 1.0
```

## Info

### IN-01: Dead code in test -- unused `TensorDataset(MagicMock())` assignment

**File:** `tests/test_modules_base.py:112`
**Severity:** INFO

Line 112 creates a `TensorDataset` from a `MagicMock()` object and assigns it via the walrus operator to `_torch_rand`, but neither the assignment result nor the variable is used. The actual dataset is created on line 116 with real tensors. This dead line adds noise.

**Fix:** Remove line 112 entirely.

### IN-02: `_scaler_cache` type uses `Any` for tuple elements

**File:** `src/tscollection/datasets/modules/_base/base.py:89`
**Severity:** INFO

The type annotation `Callable[..., tuple[Any, Any, Any]]` for `_scaler_cache` uses `Any` for all three tuple elements. Since the scaler always returns `(train_data, valid_data, test_data)` which are `np.ndarray | pd.DataFrame | None`, the type could be more precise:
```python
from typing import Callable, Any

ScalerOutput = tuple[Any, Any, Any]  # or use Protocol for tighter typing
```

This is not a bug but reduces the value of static type checking for callers.

### IN-03: `prepare_dimensions` docstring claims "setup() is NOT required" but classification needs `prepare_data()`

**File:** `src/tscollection/datasets/modules/_base/base.py:175-180`
**Severity:** INFO

The docstring for `prepare_dimensions()` says: "Caller must invoke prepare_data() first. setup() is NOT required." For the forecasting branch, this is true because `_compute_dimensions()` derives feature count from `_full_data`. For the classification branch, `_compute_dimensions()` checks `_train_data_samples` (which is set by `_do_prepare_data()` in the concrete module, not by `prepare_data()` itself).

The docstring is technically correct but could be clearer about the prerequisite state: `prepare_data()` must have populated either `_full_data` (forecasting) or `_train_data_samples` (classification) before `prepare_dimensions()` works.

**Fix:** Update the docstring to clarify branch-specific prerequisites.

### IN-04: `datatypes/__init__.py` exports are incomplete relative to `_base/__init__.py`

**File:** `src/tscollection/datasets/datatypes/__init__.py`
**Severity:** INFO

The `datatypes/__init__.py` re-exports 7 classes from `_base` and concrete implementations (`ETTDataset`, `UCRClassificationUnivariateDataset`, `UEAClassificationMultivariateDataset`). However, `datatypes/_base/__init__.py` exports 15 names (including `FlexibleTimeSeriesDataset`, `SequenceHandlingStrategy`, various strategy classes). The top-level `datatypes/__init__.py` does not re-export these, meaning users cannot import them from `tscollection.datasets.datatypes` directly.

This may be intentional (keeping the public API minimal), but the discrepancy is not documented. If these are internal-only classes, consider prefixing their module paths with `_` or adding a note in the docstring.
