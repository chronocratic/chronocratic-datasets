---
phase: 07-ddp-compliance
reviewed: 2026-05-29T00:00:00Z
depth: deep
files_reviewed: 13
files_reviewed_list:
  - src/tscollection/datasets/modules/_base/classification.py
  - src/tscollection/datasets/modules/_base/forecasting.py
  - src/tscollection/datasets/modules/electricity.py
  - src/tscollection/datasets/modules/ett.py
  - src/tscollection/datasets/modules/ucr.py
  - src/tscollection/datasets/modules/uea.py
  - src/tscollection/datasets/modules/weather.py
  - src/tscollection/datasets/utils/cache.py
  - tests/test_ddp_compliance.py
  - tests/test_modules_classification_forecasting.py
  - tests/test_modules_forecasting.py
  - tests/test_modules_ucr.py
  - tests/test_modules_uea.py
findings:
  critical: 5
  warning: 7
  info: 3
  total: 15
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-05-29T00:00:00Z
**Depth:** deep
**Files Reviewed:** 13
**Status:** issues_found

## Summary

Deep review of the DDP-compliance phase covering cache infrastructure, forecasting and classification base modules, concrete implementations (ETT, Electricity, Weather, UCR, UEA), and associated tests. Five critical issues found: unsafe pickle deserialization in UEA cache loading, None-dereference crashes in aggregate properties, redundant scaler fitting across DDP ranks with TOCTOU race, inconsistent cache schema versioning across forecasting modules, and in-place mutation of batch lists in the collate function. Seven warnings cover silent data loss in UEA validation splits, lack of DistributedSampler for real DDP training, duplicated scaling logic, brittle static analysis tests, and test port-hardcoding. Three informational items note code duplication in cache-write paths, misleading error messages, and suboptimal test patterns.

## Critical Issues

### CR-01: UEA `_load_cached_data` uses `allow_pickle=True` -- arbitrary code execution via crafted cache

**File:** `src/tscollection/datasets/modules/uea.py:325`

**Issue:** The UEA cache loader passes `allow_pickle=True` to `np.load()`:

```python
loaded = np.load(str(cache_path), allow_pickle=True)
```

This enables arbitrary Python object deserialization via pickle. If an attacker places a malicious `.npz` file in the cache directory (shared network filesystem, compromised CI cache, or adversarial dataset distribution), code execution occurs on load. The UCR module at `ucr.py:312` correctly omits `allow_pickle`, demonstrating the team knows the safe pattern.

The root cause is that UEA stores 3-D structured arrays (shape `(samples, timesteps, features)`) which numpy saves as object dtype, requiring pickle for round-trip. The fix should be to flatten the 3-D array to a regular numeric dtype rather than accepting pickle as the solution.

No comment or docstring justifies the `allow_pickle=True` flag, making it appear as oversight rather than intentional trade-off.

**Fix:** Save the 3-D array with explicit shape metadata and a flattened numeric array, then reconstruct on load without pickle:

```python
# In _do_prepare_data when saving:
atomic_save_npz(
    path=cache_path,
    train_samples=self._train_data_samples.ravel(),
    train_shape=np.array(self._train_data_samples.shape),
    # ... other arrays similarly
)

# In _load_cached_data:
loaded = np.load(str(cache_path))
self._train_data_samples = loaded['train_samples'].reshape(loaded['train_shape'])
```

### CR-02: `all_data_labels` crashes when `_valid_data_labels` is None

**File:** `src/tscollection/datasets/modules/_base/classification.py:130-132`

**Issue:** The `all_data_labels` property passes `None` to `pd.concat()` when `_valid_data_labels` is unset:

```python
@property
def all_data_labels(self) -> pd.Series:
    return pd.concat(
        [self._train_data_labels, self._test_data_labels, self._valid_data_labels], axis=0
    )
```

`_valid_data_labels` is `None` when `valid_size=0.0`, before cache load completes, or after the UEA validation split drops all samples (singleton-class filtering at `uea.py:227-231`). `pd.concat()` with a `None` element raises `TypeError: cannot concatenate object of type 'NoneType'; only pd.DataFrame, pd.Series, and pd.Index are valid`.

The identical bug exists in `all_data_samples` at `src/tscollection/datasets/modules/_base/base.py:162-182`. Both the numpy and pandas branches pass `_valid_data_samples` (which may be `None`) directly to the concatenation function.

**Fix:** Filter out `None` values before concatenation:

```python
@property
def all_data_labels(self) -> pd.Series:
    splits = [
        s for s in (
            self._train_data_labels, self._test_data_labels, self._valid_data_labels
        )
        if s is not None
    ]
    if not splits:
        msg = 'No data loaded. Call prepare_data() and setup() first.'
        raise RuntimeError(msg)
    return pd.concat(splits, axis=0)
```

Apply the same pattern to `all_data_samples` in `base.py`.

### CR-03: Forecasting `setup()` fits scaler on every DDP rank -- redundant computation and race condition

**File:** `src/tscollection/datasets/modules/_base/forecasting.py:281-285`

**Issue:** When `scale_data=True` and `stage='fit'`, every DDP rank fits its own sklearn scaler independently and attempts to persist it:

```python
if stage in ('fit', None):
    data_scaler = self._prepare_data_scaler()
    data_scaler.fit(full_array[self._train_slice])
    self._data_scaler_cache = data_scaler
    self._save_scaler_to_cache(data_scaler, 'data')
```

In the DDP flow established by the tests (`test_ddp_compliance.py`), rank 0 writes the cache via `prepare_data()`, then all ranks call `setup(stage='fit')`. Each rank fits its own scaler. While `_save_scaler_to_cache` has an existence-check guard at lines 389-390, this is a TOCTOU race: two ranks can both pass `scaler_path.exists()` before either writes.

More importantly, each rank fits the scaler independently on identical raw data. Floating-point results could differ across ranks on different hardware (CPU instruction set variations, different BLAS backends), violating the DDP invariant that all ranks see identical data. The test verifies shapes match but not that scaler parameters (scale_, min_) are bitwise identical across ranks.

**Fix:** Save the fitted scaler state (numpy arrays for `scale_`, `min_`, etc.) in the cache during `_do_prepare_data()`, then restore from those arrays in `setup()`. This ensures deterministic results regardless of which rank fits:

```python
# In _do_prepare_data (rank 0 only):
scaler.fit(train_data)
atomic_save_npz(scaler_cache_path, scale=scaler.scale_, min_=scaler.min_)

# In setup():
scaler_state = np.load(scaler_cache_path)
data_scaler = MinMaxScaler()
data_scaler.scale_ = scaler_state['scale']
data_scaler.min_ = scaler_state['min_']
```

### CR-04: Forecasting metadata hardcodes `version: 1` instead of using `CACHE_SCHEMA_VERSION`

**Files:**
- `src/tscollection/datasets/modules/ett.py:179`
- `src/tscollection/datasets/modules/electricity.py:167`
- `src/tscollection/datasets/modules/weather.py:165`

**Issue:** All three forecasting modules write `"version": 1` as a literal integer in their metadata:

```python
metadata = {
    "version": 1,
    "dataset_name": self._dataset_name,
    ...
}
```

Meanwhile, UCR (`ucr.py:295`) and UEA (`uea.py:308`) correctly use the `CACHE_SCHEMA_VERSION` constant imported from `cache.py`. If `CACHE_SCHEMA_VERSION` is incremented to 2 for any future schema change (e.g., adding a new field to the metadata JSON), forecasting modules will still write `1`. This causes `load_metadata()` at `cache.py:148` to reject valid cache files with:

```
ValueError: Cache version 1 does not match expected version 2.
Delete cache dir and re-run prepare_data().
```

The error message would be confusing because the cache files are actually valid -- only the version constant is stale. Classification modules would work fine while forecasting modules fail, creating an inconsistent user experience.

**Fix:** Import and use `CACHE_SCHEMA_VERSION` in all three forecasting modules:

```python
from tscollection.datasets.utils.cache import (
    atomic_save_metadata,
    atomic_save_npz,
    build_cache_key,
    CACHE_SCHEMA_VERSION,
)

metadata = {
    "version": CACHE_SCHEMA_VERSION,
    ...
}
```

### CR-05: `custom_collate_fn` mutates input batch list in-place

**File:** `src/tscollection/datasets/utils/general.py:29-34`

**Issue:** The collate function modifies the caller's `batch` list by appending samples:

```python
def custom_collate_fn(batch: list[Any], *, desired_batch_size: int) -> Any:
    current_batch_size = len(batch)
    if current_batch_size < desired_batch_size:
        additional_needed = desired_batch_size - current_batch_size
        for i in range(additional_needed):
            sample_index = current_batch_size - 1 - (i % current_batch_size)
            batch.append(batch[sample_index])
    return default_collate(batch)
```

When `prefetch_factor > 1` (PyTorch default is 2) and `num_workers > 0`, the data loader pipeline may hold references to batches that get mutated before `default_collate()` consumes them. Even in single-worker mode, in-place mutation of the batch list is a side effect violating the principle that collate functions should be pure transformers.

If the dataset's `__getitem__` returns references to shared or cached objects, appended references could cause the same sample to appear multiple times within a batch, introducing data leakage in training.

**Fix:** Work on a copy of the batch list:

```python
def custom_collate_fn(batch: list[Any], *, desired_batch_size: int) -> Any:
    current_batch_size = len(batch)
    if current_batch_size < desired_batch_size:
        padded = list(batch)
        additional_needed = desired_batch_size - current_batch_size
        for i in range(additional_needed):
            sample_index = current_batch_size - 1 - (i % current_batch_size)
            padded.append(padded[sample_index])
        batch = padded
    return default_collate(batch)
```

## Warnings

### WR-01: UEA validation split silently drops singleton classes from training data

**File:** `src/tscollection/datasets/modules/uea.py:227-231`

**Issue:** When creating the validation split, UEA filters out all samples belonging to classes with fewer than 2 training samples:

```python
label_counts = np.bincount(self._train_data_labels)
valid_mask = np.isin(self._train_data_labels, np.where(label_counts > 1)[0])
filtered_samples = self._train_data_samples[valid_mask]
filtered_labels = self._train_data_labels[valid_mask]
```

This is necessary to prevent `train_test_split` from failing when `stratify` requires at least 2 samples per class. However, singleton-class samples are permanently removed from training data without any warning or logging. If a dataset has rare classes with exactly one sample, those classes vanish from the model entirely.

The UCR module uses the same pattern (`ucr.py:220`) with `groupby('label').filter(lambda x: len(x) > 1)`, which is functionally equivalent but at least more transparent about the filtering operation. Neither module emits a warning.

**Fix:** Log a warning when samples are dropped:

```python
dropped_count = len(self._train_data_samples) - len(filtered_samples)
if dropped_count > 0:
    logger.warning(
        'Dropped %d samples from singleton classes in dataset %s. '
        'These classes will not be present in training data.',
        dropped_count, self._dataset_name,
    )
```

### WR-02: No DistributedSampler support -- dataloaders load full splits on every rank

**Files:** All dataloader methods across `ucr.py`, `uea.py`, `ett.py`, `electricity.py`, `weather.py`

**Issue:** The dataloader methods wrap full train/valid/test splits in Dataset objects without any `DistributedSampler`. In real DDP training, each rank would see all samples, causing each sample to be processed `world_size` times per epoch. This effectively multiplies the batch size by the number of GPUs, altering training dynamics.

The cache infrastructure (rank-0 writes, all-ranks read) is correct for data preparation, but the dataloader layer lacks the sampler needed to shard data across ranks. The existing DDP tests verify cache consistency but do not test distributed training loops.

**Fix:** Add optional DistributedSampler support:

```python
def train_dataloader(
    self,
    *,
    mode: TimeSeriesDatasetMode = ...,
    shuffle: bool | None = None,
    strict_batch_size: bool = False,
    extra_args: dict[str, Any] | None = None,
    distributed_sampler: torch.utils.data.DistributedSampler | None = None,
) -> DataLoader:
    dataset = ...
    dl_kwargs = {...}
    if distributed_sampler is not None:
        dl_kwargs['sampler'] = distributed_sampler
        dl_kwargs['shuffle'] = False  # sampler handles shuffling
    return self._process_train_dataloader(**dl_kwargs)
```

### WR-03: Duplicated time-series feature scaling logic in `setup()`

**File:** `src/tscollection/datasets/modules/_base/forecasting.py:292-306` (fit branch) and `319-340` (test/predict branch)

**Issue:** The time-series feature scaling block is duplicated nearly verbatim between the `fit` and `test`/`predict` branches:

```python
# fit branch
if num_time_series_features > 0:
    ts_feature_scaler = self._prepare_data_scaler()
    ts_feature_scaler.fit(time_series_features[self._train_slice])
    self._ts_feature_scaler_cache = ts_feature_scaler
    self._save_scaler_to_cache(ts_feature_scaler, 'ts')
    scaled_ts_features = ts_feature_scaler.transform(time_series_features)
    scaled_ts_features = np.expand_dims(scaled_ts_features, axis=0)
    repeated_ts = np.repeat(scaled_ts_features, self._full_data_scaled.shape[0], axis=0)
    self._full_data_scaled = np.concatenate(
        [repeated_ts, self._full_data_scaled], axis=-1
    )

# test/predict branch -- same pattern, different scaler retrieval
if num_time_series_features > 0:
    if self._ts_feature_scaler_cache is None:
        self._ts_feature_scaler_cache = self._load_scaler_from_cache('ts')
    if self._ts_feature_scaler_cache is not None:
        scaled_ts_features = self._ts_feature_scaler_cache.transform(time_series_features)
        scaled_ts_features = np.expand_dims(scaled_ts_features, axis=0)
        repeated_ts = np.repeat(scaled_ts_features, self._full_data_scaled.shape[0], axis=0)
        self._full_data_scaled = np.concatenate(
            [repeated_ts, self._full_data_scaled], axis=-1
        )
```

Both blocks perform: expand_dims(axis=0), repeat along axis 0, concatenate along axis=-1. A bug fix or feature change applied to one block is likely to miss the other.

**Fix:** Extract to a shared helper method:

```python
def _apply_time_features(
    self,
    time_series_features: np.ndarray,
    fit: bool,
) -> None:
    if time_series_features.shape[-1] == 0:
        return
    if fit:
        ts_scaler = self._prepare_data_scaler()
        ts_scaler.fit(time_series_features[self._train_slice])
        self._ts_feature_scaler_cache = ts_scaler
        self._save_scaler_to_cache(ts_scaler, 'ts')
    else:
        ts_scaler = self._ts_feature_scaler_cache
    scaled = np.expand_dims(ts_scaler.transform(time_series_features), axis=0)
    repeated = np.repeat(scaled, self._full_data_scaled.shape[0], axis=0)
    self._full_data_scaled = np.concatenate([repeated, self._full_data_scaled], axis=-1)
```

### WR-04: `test_isinstance_branch_elimination` uses fragile subprocess grep

**File:** `tests/test_ddp_compliance.py:287-296`

**Issue:** The isinstance elimination test runs `grep` via `subprocess.run()` to scan for `isinstance.*_full_data` patterns. This approach has three weaknesses:

1. **Cross-platform fragility:** `grep` is not available on Windows by default. If tests run on Windows CI, this test crashes with `FileNotFoundError: [Errno 2] No such file or directory: 'grep'`.

2. **False positive risk:** The regex `isinstance.*_full_data` could match comments, docstrings, or variable names containing `_full_data` in non-isinstance contexts (e.g., `self._full_data_raw` contains the substring `_full_data`).

3. **No line context:** When the test fails, it reports the raw grep output without indicating which files are legitimate matches versus actual regressions.

**Fix:** Use Python-native file scanning:

```python
def test_isinstance_branch_elimination(self) -> None:
    modules_dir = Path(__file__).parents[1] / 'src' / 'tscollection' / 'datasets' / 'modules'
    matches = []
    for py_file in modules_dir.rglob('*.py'):
        for lineno, line in enumerate(py_file.read_text().splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            if 'isinstance' in stripped and 'self._full_data' in stripped:
                matches.append(f'{py_file.relative_to(modules_dir)}:{lineno}: {stripped}')
    assert not matches, f'Found isinstance(self._full_data) branches:\n' + '\n'.join(matches)
```

### WR-05: DDP test workers use hardcoded ports without randomization

**File:** `tests/test_ddp_compliance.py:44` and `104`

**Issue:** Both DDP worker functions hardcode `MASTER_PORT`:

```python
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'  # forecasting worker
```

and:

```python
os.environ['MASTER_PORT'] = '29501'  # classification worker
```

If tests run in parallel (common on CI with concurrent jobs) or the ports are already in use by other processes, `init_process_group()` fails with `OSError: [Errno 98] Address already in use`. The DDP compliance tests themselves become flaky, undermining their value as reliability gates.

**Fix:** Use socket-based port discovery:

```python
def _get_free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(('localhost', 0))
        return s.getsockname()[1]

os.environ['MASTER_PORT'] = str(_get_free_port())
```

Note: each rank must use the SAME port, so `_get_free_port()` should be called once before `mp.spawn()` and passed as an argument.

### WR-06: `_split_data` uses `assert` for input validation -- stripped with Python `-O`

**File:** `src/tscollection/datasets/modules/_base/forecasting.py:443-446`

**Issue:** The `_split_data` method uses `assert` statements to validate preconditions:

```python
def _split_data(self) -> None:
    assert self._full_data_scaled is not None
    assert self._train_slice is not None
    assert self._valid_slice is not None
    assert self._test_slice is not None
```

Python's `-O` (optimize) flag strips all assert statements at compile time. If the module is deployed with `python -O` (common in production to reduce overhead), these checks disappear silently. The subsequent indexing operations (`self._full_data_scaled[:, self._train_slice]`) would then raise `TypeError: 'NoneType' object is not subscriptable` -- a confusing error that does not indicate which precondition failed.

**Fix:** Replace assert with explicit validation:

```python
def _split_data(self) -> None:
    if self._full_data_scaled is None:
        msg = '_split_data requires _full_data_scaled. Ensure scaling completed.'
        raise RuntimeError(msg)
    if self._train_slice is None:
        msg = '_split_data requires _train_slice. Ensure _set_data_slices() was called.'
        raise RuntimeError(msg)
    # ... same for valid_slice and test_slice
```

### WR-07: `_save_scaler_to_cache` existence-check is redundant with `save_scaler` internal handling

**File:** `src/tscollection/datasets/modules/_base/forecasting.py:389-391`

**Issue:** The `_save_scaler_to_cache` method checks `scaler_path.exists()` before calling `save_scaler()`:

```python
def _save_scaler_to_cache(self, scaler: object, kind: str) -> None:
    ...
    if scaler_path.exists():
        return
    save_scaler(scaler=scaler, path=scaler_path)
```

The `save_scaler()` function in `cache.py:171-184` already handles the case where the target file exists -- it writes to a `.tmp` file and catches `OSError` from `tmp.replace()`. The existence pre-check is redundant and creates a TOCTOU window where the file could appear between the check and the write.

**Fix:** Remove the pre-check and rely on `save_scaler()`'s internal race handling:

```python
def _save_scaler_to_cache(self, scaler: object, kind: str) -> None:
    if self._cache_key is None:
        return
    cache_dir = self._resolve_cache_dir()
    scaler_path = cache_dir / f'{self._cache_key}_{kind}_scaler.pt'
    save_scaler(scaler=scaler, path=scaler_path)
```

## Info

### IN-01: Duplicated cache-write boilerplate across forecasting modules

**Files:** `src/tscollection/datasets/modules/ett.py:162-192`, `electricity.py:150-184`, `weather.py:143-176`

**Issue:** All three forecasting modules follow the identical pattern of: convert data to numpy, resolve cache directory, create cache path, save npz, build metadata dict, save metadata, store time index. This is approximately 30-40 lines of duplicated code per module. The concrete differences (CSV parsing, column selection, split computation) are module-specific, but the cache-write sequence could be extracted to the base class.

**Suggestion:** Add a `_write_forecasting_cache()` helper to `BaseForecastingTimeSeriesDataModule` that handles the npz + metadata write, accepting data, index, and metadata fields as parameters.

### IN-02: `_compute_dimensions` error message references wrong method name

**File:** `src/tscollection/datasets/modules/_base/classification.py:153`

**Issue:** The RuntimeError message in `_compute_dimensions()` references the public method name rather than the internal one:

```python
msg = 'prepare_dimensions() requires prepare_data() to have run first'
```

This is technically correct since `_compute_dimensions()` is called by `prepare_dimensions()`, but it omits that the actual failing method is `_compute_dimensions()`. In tracebacks, developers would see both method names and the message provides minimal additional context.

**Suggestion:** Include the internal method name for clarity:

```python
msg = '_compute_dimensions() requires prepare_data() to have run first'
```

### IN-03: `test_setup_idempotent_with_cache` manually clears state instead of using `reset()`

**File:** `tests/test_ddp_compliance.py:357-363`

**Issue:** The idempotency test manually clears individual attributes to simulate a fresh process:

```python
module._setup_completed_stages.clear()
module._train_data_samples = None
module._valid_data_samples = None
module._test_data_samples = None
module._full_data_scaled = None
module._data_scaler_cache = None
module._ts_feature_scaler_cache = None
```

This duplicates the logic in `reset()` (`base.py:365-375`) and will silently diverge if `reset()` adds new attributes. The test intentionally avoids calling `reset()` because it wants to preserve `_full_data_raw` and `_time_index` (loaded from cache), but there is no documentation explaining this design decision.

**Suggestion:** Add a comment explaining why `reset()` is not used, or create a `reset_setup_only()` helper method that clears setup state while preserving cache-loaded state:

```python
# Do NOT call reset() -- it would clear _cache_key, preventing
# the cache re-read that this test is designed to verify.
# Instead, clear only the setup-specific state:
module._setup_completed_stages.clear()
# ...
```

---

_Reviewed: 2026-05-29T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
