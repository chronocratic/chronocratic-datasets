---
phase: 07-ddp-compliance
reviewed: 2026-06-01T12:00:00Z
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
  critical: 0
  warning: 0
  info: 3
  total: 3
status: clean
---

# Phase 7: Code Review Re-Re-Review Report

**Reviewed:** 2026-06-01T12:00:00Z
**Depth:** deep
**Files Reviewed:** 14
**Status:** clean

## Summary

Deep re-review of the DDP-compliance phase after all previous CR (CR-01 through CR-06) and WR (WR-01 through WR-08) fixes have been applied. All 14 fixes are verified correct with no regressions introduced. Cross-file analysis of the import graph, error propagation chains, and DDP data flow confirms architectural consistency. Three pre-existing info items remain (cosmetic/style only). No new critical or warning issues found.

## Previous Fix Verification

All 14 previously identified fixes verified correct:

- **CR-01** (UEA `allow_pickle=True`): Removed. `np.load()` at `uea.py:334` loads only numeric arrays (float32 samples, int64 labels). No object dtype.
- **CR-02** (`None` in `all_data_labels`/`all_data_samples`): Both properties filter `None` before `pd.concat`/`np.concatenate` and raise `RuntimeError` when no splits exist. Verified at `classification.py:130-138` and `base.py:161-173`.
- **CR-03/WR-07** (TOCTOU in `_save_scaler_to_cache`): Existence pre-check removed. Relies on `save_scaler()` internal atomic write with `OSError` handling. Verified at `forecasting.py:367-377`.
- **CR-04** (Hardcoded `version: 1`): All three forecasting modules import and use `CACHE_SCHEMA_VERSION`. Verified at `ett.py:188`, `electricity.py:178`, `weather.py:170`.
- **CR-05** (In-place mutation in `custom_collate_fn`): Works on `padded = list(batch)` copy. Original batch never mutated. Verified at `general.py:31-36`.
- **CR-06** (Metadata `n_features` ignores `scale_data`): All three forecasting modules now conditionally add `TIME_FEATURE_COUNT` based on both `scale_data` and `_time_index` presence. Verified at `ett.py:185-187`, `electricity.py:175-177`, `weather.py:167-169`. No regression: the `if` guard matches the actual time feature extraction logic in `forecasting.py:276-294`.
- **WR-01** (UEA singleton-class warning): `logger.warning()` reports dropped count and dataset name. Verified at `uea.py:235-240`.
- **WR-03** (Duplicated TS feature scaling): Extracted to `_apply_ts_features()` helper at `forecasting.py:343-356`. Both fit and test/predict branches call it. Shape alignment verified: `np.repeat` uses `self._full_data_scaled.shape[0]` which matches the post-transform first dimension.
- **WR-04** (grep subprocess): Replaced with Python-native `Path.rglob()` scanning. Cross-platform safe. Verified at `test_ddp_compliance.py:293-304`.
- **WR-05** (Hardcoded DDP ports): `_get_free_port()` uses socket binding. Port passed to `mp.spawn()`. Verified at `test_ddp_compliance.py:24-33`.
- **WR-06** (`assert` in `_split_data`): Replaced with explicit `RuntimeError` checks. Verified at `forecasting.py:434-445`.
- **WR-08** (Remaining `assert` statements): All 8 remaining `assert`-as-validation replaced with `if ... is None: raise RuntimeError(...)` pattern. Verified at `forecasting.py:252-257`, `forecasting.py:351-353`, `forecasting.py:420-422`, `forecasting.py:434-436`, `electricity.py:103-105`, `electricity.py:117-119`, `ett.py:136-138`, `weather.py:103-105`, `weather.py:117-119`.

## Cross-File Analysis (Deep Review)

### Import Graph

The module dependency chain is acyclic and well-structured:

```
base.py --import--> cache.py, scaling.py, general.py
classification.py --import--> base.py, common.py, general.py
forecasting.py --import--> base.py, cache.py, features.py
ucr.py --import--> classification.py, cache.py, arff.py
uea.py --import--> classification.py, cache.py
ett.py --import--> forecasting.py, cache.py, features.py
electricity.py --import--> forecasting.py, cache.py, features.py
weather.py --import--> forecasting.py, cache.py, features.py
```

No circular dependencies detected.

### Error Propagation

- `_do_prepare_data()` in all concrete modules raises `FileNotFoundError` for missing paths. This propagates through `BaseTimeSeriesDataModule.prepare_data()` without suppression.
- `setup()` in both classification and forecasting branches validates preconditions with descriptive `RuntimeError` messages. No bare `except` blocks catch these errors silently.
- Cache I/O errors (`FileNotFoundError` in `_load_cached_data`, `OSError` in `save_scaler`) are handled with appropriate fallbacks or re-raises.

### DDP Data Flow

The DDP-safe pattern (rank-0 writes cache, all ranks read) is correctly implemented:

1. `prepare_data_per_node = True` ensures each node's rank-0 writes cache.
2. `_prepare_data_called` sentinel prevents duplicate I/O.
3. `save_scaler()` handles `OSError` from concurrent writes (DDP race condition).
4. `_setup_completed_stages` prevents duplicate scaling in `setup()`.
5. Cache files use atomic writes (`Path.replace()`) for POSIX consistency.

### Shape Consistency Across Transform Chain

Traced the `_full_data_scaled` shape through the setup pipeline for each forecasting module:

**ETT:** `(samples, features)` -> scale -> `(samples, features)` -> `expand_dims(0)` -> `(1, samples, features)` -> concat ts on axis=-1 -> `(1, samples, features+7)` -> split on axis=1 -> `(1, split_size, features+7)`. Correct.

**Electricity:** `(samples, features)` -> scale -> `(samples, features)` -> `.T` -> `(features, samples)` -> `expand_dims(-1)` -> `(features, samples, 1)` -> concat ts on axis=-1 -> `(features, samples, 1+7)`. `_apply_ts_features` repeats on axis=0 (features dimension). Correct.

**Weather:** `(samples, features)` -> scale -> `(samples, features)` -> `expand_dims(0)` -> `(1, samples, features)` -> concat ts on axis=-1 -> `(1, samples, features+7)`. Same pattern as ETT. Correct.

### Type Consistency at Module Boundaries

- Classification modules: `_train_data_samples` is `pd.DataFrame` (UCR) or `np.ndarray` (UEA). Both types are handled by the scaling pipeline in `base.py:166-173`.
- Forecasting modules: `_full_data_raw` is always `np.ndarray` (float32) after cache read. Time features are `np.ndarray` (float32) from `features.py:35`.
- DataLoader outputs: Forecasting modules use `TensorDataset` with `torch.float32` tensors. Classification modules use custom dataset wrappers.

## Info

### IN-01: Duplicated cache-write boilerplate across forecasting modules

**Files:** `src/tscollection/datasets/modules/ett.py:162-198`, `electricity.py:142-187`, `weather.py:142-179`

**Issue:** All three forecasting modules follow the identical pattern of: convert data to numpy, resolve cache directory, create cache path, mkdir, save npz with `atomic_save_npz`, build metadata dict with `atomic_save_metadata`. Approximately 30-40 lines of duplicated code per module.

**Suggestion:** Add a `_write_forecasting_cache()` helper to `BaseForecastingTimeSeriesDataModule` that accepts data, index, and metadata override fields as parameters.

### IN-02: `_compute_dimensions` error message references `prepare_dimensions()` instead of `_compute_dimensions()`

**File:** `src/tscollection/datasets/modules/_base/classification.py:159`

**Issue:** The `RuntimeError` message in `_compute_dimensions()` says `prepare_dimensions()` when the actual failing method is `_compute_dimensions()`:

```python
msg = 'prepare_dimensions() requires prepare_data() to have run first'
```

**Suggestion:** Use the internal method name:
```python
msg = '_compute_dimensions() requires prepare_data() to have run first'
```

### IN-03: `test_setup_idempotent_with_cache` manually clears state instead of using `reset()`

**File:** `tests/test_ddp_compliance.py:365-371`

**Issue:** The idempotency test manually clears individual attributes rather than calling `reset()`. This duplicates `reset()` logic (`base.py:344-366`) and risks divergence.

**Suggestion:** Add a comment explaining the design decision:
```python
# Do NOT call reset() -- it clears _cache_key, preventing the cache
# re-read that this test is designed to verify. Only clear setup state:
```

---

_Info items IN-01 through IN-03 are cosmetic/maintainability suggestions from the previous review. They remain unfixed but do not affect correctness or security._

_All critical (CR-01 through CR-06) and warning (WR-01 through WR-08) findings have been resolved with no regressions. Code is clean for this phase._

_Reviewed: 2026-06-01T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
