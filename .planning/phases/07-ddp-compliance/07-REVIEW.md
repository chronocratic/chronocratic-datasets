---
phase: 07-ddp-compliance
reviewed: 2026-06-01T00:00:00Z
depth: deep
files_reviewed: 14
files_reviewed_list:
  - src/tscollection/datasets/modules/_base/classification.py
  - src/tscollection/datasets/modules/_base/forecasting.py
  - src/tscollection/datasets/modules/electricity.py
  - src/tscollection/datasets/modules/ett.py
  - src/tscollection/datasets/modules/ucr.py
  - src/tscollection/datasets/modules/uea.py
  - src/tscollection/datasets/modules/weather.py
  - src/tscollection/datasets/utils/cache.py
  - src/tscollection/datasets/utils/general.py
  - tests/test_ddp_compliance.py
  - tests/test_modules_classification_forecasting.py
  - tests/test_modules_forecasting.py
  - tests/test_modules_ucr.py
  - tests/test_modules_uea.py
findings:
  critical: 1
  warning: 1
  info: 3
  total: 5
status: issues_found
---

# Phase 7: Code Review Re-Review Report

**Reviewed:** 2026-06-01T00:00:00Z
**Depth:** deep
**Files Reviewed:** 14
**Status:** issues_found

## Summary

Re-review of the DDP-compliance phase after 12 of 15 previous findings were fixed (5 critical, 7 warning). All 12 applied fixes are verified correct and do not introduce new bugs. Three info items from the previous review remain unfixed. One new critical issue was found: metadata `n_features` unconditionally adds `TIME_FEATURE_COUNT` regardless of the `scale_data` flag, causing dimension mismatches. One new warning: the WR-06 assert-to-validation fix was incomplete, with 8 additional assert statements serving the same purpose still present.

## Previous Fix Verification

All 12 previously identified fixes (CR-01 through CR-05, WR-01 through WR-07) verified correct:

- **CR-01** (UEA `allow_pickle=True`): Removed. `np.load()` without `allow_pickle` works because UEA saves numeric arrays (float32 samples, int64 labels) -- not object dtype. Verified at `uea.py:334`.
- **CR-02** (None in `all_data_labels`/`all_data_samples`): Both properties filter `None` before concatenation and raise `RuntimeError` when no splits exist. Applied consistently at `classification.py:130-138` and `base.py:161-173`.
- **CR-03/WR-07** (TOCTOU in `_save_scaler_to_cache`): Existence pre-check removed. Relies on `save_scaler()` internal atomic write with OSError handling. Verified at `forecasting.py:367-371`.
- **CR-04** (Hardcoded `version: 1`): All three forecasting modules now import and use `CACHE_SCHEMA_VERSION`. Consistent with classification modules. Verified at `ett.py:182`, `electricity.py:172`, `weather.py:164`.
- **CR-05** (In-place mutation in `custom_collate_fn`): Works on `padded = list(batch)` copy. Original batch is never mutated. Verified at `general.py:31-36`.
- **WR-01** (UEA singleton-class warning): `logger.warning()` added at `uea.py:235-240`. Dropped count and dataset name included.
- **WR-03** (Duplicated TS feature scaling): Extracted to `_apply_ts_features()` helper at `forecasting.py:339-350`. Both fit and test/predict branches call it.
- **WR-04** (grep subprocess): Replaced with Python-native file scanning using `rglob` and `splitlines`. Cross-platform safe. Verified at `test_ddp_compliance.py:293-304`.
- **WR-05** (Hardcoded DDP ports): `_get_free_port()` uses socket binding. Port passed to `mp.spawn()` args. Both workers updated.
- **WR-06** (assert in `_split_data`): Replaced with explicit `RuntimeError` checks at `forecasting.py:426-437`. **Incomplete** -- see WR-08.

## Critical Issues

### CR-06: Metadata `n_features` unconditionally adds TIME_FEATURE_COUNT regardless of `scale_data`

**Files:**
- `src/tscollection/datasets/modules/ett.py:180`
- `src/tscollection/datasets/modules/electricity.py:170`
- `src/tscollection/datasets/modules/weather.py:162`

**Issue:** All three forecasting modules compute `n_features` for the metadata cache file by unconditionally adding `TIME_FEATURE_COUNT`:

```python
n_features = data.shape[1] + TIME_FEATURE_COUNT
metadata = {
    'version': CACHE_SCHEMA_VERSION,
    ...
    'n_features': n_features,
    ...
}
```

However, time features are only extracted during `setup()` when `scale_data=True`. When `scale_data=False`, the entire scaling and time feature extraction block is skipped (`forecasting.py:325-331`), so the actual data contains no time features.

This causes `prepare_dimensions()` to return different values depending on whether it reads from the metadata cache or computes from raw data:

- **Pre-setup path** (`_compute_dimensions` at `forecasting.py:187`): Correctly checks `self.scale_data`:
  ```python
  has_time_features = self._time_index is not None and self.scale_data
  n_features = raw_cols + TIME_FEATURE_COUNT if has_time_features else raw_cols
  ```
- **Post-setup path** (`prepare_dimensions` at `base.py:236-240`): Reads `n_features` from metadata, which is always inflated by `TIME_FEATURE_COUNT` even when `scale_data=False`.

Concrete example: ETT with `scale_data=False`, 1 raw column (univariate OT):
- Metadata writes `n_features = 8` (1 + 7 time features)
- Actual data after setup has 1 feature (no time features extracted)
- `prepare_dimensions()` post-setup returns `(8, seq_len)` -- off by factor of 8x

This silently corrupts model dimension calculations. If a model uses `n_features` to size its input layer, it allocates for 8 features but receives 1, causing shape mismatches at runtime.

The existing test `test_pre_setup_matches_post_setup` (`test_modules_forecasting.py:1693-1710`) does NOT catch this bug because it uses `scale_data=True` (default), where time features are actually added and both paths agree.

**Fix:** Conditionally add `TIME_FEATURE_COUNT` only when time features will actually be extracted:

```python
n_features = data.shape[1]
if self.scale_data and self._time_index is not None:
    n_features += TIME_FEATURE_COUNT
metadata = {
    'version': CACHE_SCHEMA_VERSION,
    'n_features': n_features,
    ...
}
```

The `_time_index` check is needed because `scale_data=True` with no DatetimeIndex still produces 0 time features.

## Warnings

### WR-08: WR-06 fix incomplete -- assert statements remain for input validation in 8 locations

**Files:**
- `src/tscollection/datasets/modules/_base/forecasting.py:252-253` (setup() preconditions)
- `src/tscollection/datasets/modules/_base/forecasting.py:347` (_apply_ts_features() precondition)
- `src/tscollection/datasets/modules/_base/forecasting.py:414` (_calculate_num_features() precondition)
- `src/tscollection/datasets/modules/electricity.py:103` (_set_data_slices() precondition)
- `src/tscollection/datasets/modules/electricity.py:115` (_transform_data() precondition)
- `src/tscollection/datasets/modules/ett.py:136` (_transform_data() precondition)
- `src/tscollection/datasets/modules/weather.py:103` (_set_data_slices() precondition)
- `src/tscollection/datasets/modules/weather.py:115` (_transform_data() precondition)

**Issue:** WR-06 identified that `assert` statements are stripped by Python `-O` (optimize) flag and replaced `assert` with explicit `RuntimeError` in `_split_data()`. However, 8 additional `assert` statements serving the same input validation purpose remain.

In `setup()` at `forecasting.py:252-253`:
```python
assert self._full_data_raw is not None, 'Full data not set; call prepare_data() first'
assert self._train_slice is not None, 'Train slice not set; call _set_data_slices() first'
```

In `_apply_ts_features()` at `forecasting.py:347`:
```python
assert self._full_data_scaled is not None
```

In `_calculate_num_features()` at `forecasting.py:414`:
```python
assert self._full_data_scaled is not None
```

If the module is deployed with `python -O`, these checks silently disappear. A missing `_full_data_scaled` would cause `TypeError: 'NoneType' object is not subscriptable` rather than a descriptive error identifying which precondition failed.

**Fix:** Apply the same `RuntimeError` pattern used in `_split_data()`:

```python
if self._full_data_scaled is None:
    msg = '_apply_ts_features requires _full_data_scaled. Ensure scaling completed.'
    raise RuntimeError(msg)
```

## Info

### IN-01: Duplicated cache-write boilerplate across forecasting modules

**Files:** `src/tscollection/datasets/modules/ett.py:162-191`, `electricity.py:149-181`, `weather.py:142-173`

**Issue:** All three forecasting modules follow the identical pattern of: convert data to numpy, resolve cache directory, create cache path, mkdir, save npz with `atomic_save_npz`, build metadata dict with `atomic_save_metadata`. Approximately 30-40 lines of duplicated code per module. The concrete differences (CSV parsing, column selection, split computation) are module-specific, but the cache-write sequence could be extracted to the base class.

**Suggestion:** Add a `_write_forecasting_cache()` helper to `BaseForecastingTimeSeriesDataModule` that accepts data, index, and metadata override fields as parameters.

### IN-02: `_compute_dimensions` error message references wrong method name

**File:** `src/tscollection/datasets/modules/_base/classification.py:159`

**Issue:** The RuntimeError message says `prepare_dimensions()` when the actual failing method is `_compute_dimensions()`:

```python
msg = 'prepare_dimensions() requires prepare_data() to have run first'
```

**Suggestion:** Use the internal method name:
```python
msg = '_compute_dimensions() requires prepare_data() to have run first'
```

### IN-03: `test_setup_idempotent_with_cache` manually clears state instead of using `reset()`

**File:** `tests/test_ddp_compliance.py:365-371`

**Issue:** The idempotency test manually clears individual attributes:

```python
module._setup_completed_stages.clear()
module._train_data_samples = None
module._valid_data_samples = None
module._test_data_samples = None
module._full_data_scaled = None
module._data_scaler_cache = None
module._ts_feature_scaler_cache = None
```

This duplicates `reset()` logic (`base.py:356-366`) and will silently diverge if `reset()` adds new attributes. No comment explains why `reset()` is not used (it clears `_cache_key`, which would prevent the cache re-read this test verifies).

**Suggestion:** Add a comment explaining the design decision:

```python
# Do NOT call reset() -- it clears _cache_key, preventing the cache
# re-read that this test is designed to verify. Only clear setup state:
```

---

_Reviewed: 2026-06-01T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
