---
phase: 04-data-modules
reviewed: 2026-05-13T00:00:00Z
depth: deep
files_reviewed: 22
files_reviewed_list:
  - src/tscollection/datasets/__init__.py
  - src/tscollection/datasets/enums/__init__.py
  - src/tscollection/datasets/enums/data.py
  - src/tscollection/datasets/modules/__init__.py
  - src/tscollection/datasets/modules/classes/__init__.py
  - src/tscollection/datasets/modules/classes/base.py
  - src/tscollection/datasets/modules/classes/classification.py
  - src/tscollection/datasets/modules/classes/forecasting.py
  - src/tscollection/datasets/modules/electricity.py
  - src/tscollection/datasets/modules/ett.py
  - src/tscollection/datasets/modules/ucr.py
  - src/tscollection/datasets/modules/uea.py
  - src/tscollection/datasets/modules/weather.py
  - src/tscollection/datasets/utils/__init__.py
  - src/tscollection/datasets/utils/common.py
  - tests/test_modules_base.py
  - tests/test_modules_classification_forecasting.py
  - tests/test_modules_forecasting.py
  - tests/test_modules_ucr.py
  - tests/test_modules_uea.py
  - tests/test_package.py
  - tests/test_utils_common_separate_target.py
findings:
  critical: 3
  warning: 6
  info: 5
  total: 14
status: issues_found
---

# Phase 4: Code Review Report

**Reviewed:** 2026-05-13T00:00:00Z
**Depth:** deep
**Files Reviewed:** 22
**Status:** issues_found

## Summary

Deep review of the LightningDataModule layer for time series datasets, including the base class hierarchy (`BaseTimeSeriesDataModule`, `BaseClassificationTimeSeriesDataModule`, `BaseForecastingTimeSeriesDataModule`), concrete dataset modules (ETT, Electricity, Weather, UCR, UEA), and supporting utilities. Three critical issues were found: a data leakage bug in the forecasting scaler caused by wrong-axis slicing, a silently ignored configuration parameter, and a fragile hardcoded index offset. Six warnings and five informational findings cover edge cases, metadata loss, test reliability gaps, and code quality issues.

---

## Critical Issues

### CR-01: Forecaster scaler fits on wrong axis, causing data leakage

**File:** `src/tscollection/datasets/modules/classes/forecasting.py:171`

**Issue:** In `setup()`, the scaler is fit on `full_array[:, self._train_slice]`. At this point, `_full_data` has shape `(time_steps, features)` (raw CSV: rows are time, columns are features). The `_train_slice` values (e.g., `slice(None, 8640)` for ETTh1) are designed as time boundaries, not column indices. Slicing axis 1 with a time-based `slice` means:

- For multivariate data (more columns than train boundary, e.g., 7 features with slice `:8640`): numpy silently clips, returning all columns. The scaler is fit on ALL time steps, not just the training period -- this is data leakage, violating the documented intent of "Fits scaler on train slice only" (line 140-141).
- For univariate data (single column, e.g., shape `(T, 1)` with slice `:8640`): numpy clips to column 0, returning the full array. Again, all time steps are used for fitting, leaking validation/test statistics into the training scaler.

The same axis error applies to the time feature scaler at line 180: `ts_feature_scaler.fit(time_series_features[:, self._train_slice])`.

After `_transform_data()` (called at line 175), the data is reshaped to `(1, time_steps, features)` (ETT, Weather) or `(features, time_steps, 1)` (Electricity). The subsequent `_split_data()` then uses `[:, self._train_slice]` correctly -- but the damage is already done since the scaler was fit on the pre-transform shape.

**Fix:**
```python
# Line 171: Fit scaler on training TIME period, not columns
# _full_data before _transform_data has shape (time_steps, features)
# The train slice should select rows (time), not columns (features)
data_scaler.fit(full_array[self._train_slice])  # Shape: (train_time_steps, features)

# Line 180: Same fix for time feature scaler
ts_feature_scaler.fit(time_series_features[self._train_slice])
```

Note: After this fix, the scaler expects 2D input `(samples, features)`, which `full_array[self._train_slice]` provides. The transform at line 172 (`data_scaler.transform(full_array)`) should also remain consistent: it transforms the full `(time_steps, features)` array using the scaler fitted on training time steps only.

---

### CR-02: `scale_data` parameter is ignored in forecasting branch

**File:** `src/tscollection/datasets/modules/classes/forecasting.py:135-193`

**Issue:** The `setup()` method in `BaseForecastingTimeSeriesDataModule` unconditionally creates a scaler, fits it, and transforms the data -- even when `self.scale_data` is `False`. The `scale_data` flag is checked in the classification branch via `create_data_scaler()` (which returns a no-op when `scale=False`), but the forecasting branch completely bypasses this mechanism and always scales.

Cross-referencing: The docstring at lines 138-141 explicitly states "Fits scaler on train slice only to prevent data leakage (T-04-02-04)" but never mentions the `scale_data` guard. The base class `__init__` at line 57 defines `scale_data: bool = True` as a public API parameter. Users passing `scale_data=False` would expect no scaling to occur, but their data gets scaled anyway.

**Fix:**
```python
def setup(self, stage: str) -> None:
    assert self._full_data is not None, 'Full data not set; call prepare_data() first'
    assert self._train_slice is not None, 'Train slice not set; call _set_data_slices() first'

    # Extract time features from DataFrame index if applicable
    if isinstance(self._full_data, pd.DataFrame):
        time_index = self._full_data.index
        full_array = self._full_data.to_numpy()
    else:
        time_index = None
        full_array = self._full_data

    # Time feature extraction (independent of scaling)
    if time_index is not None:
        from tscollection.datasets.utils.features import extract_time_features

        time_series_features = extract_time_features(
            pd.DatetimeIndex(time_index)
        )
        num_time_series_features = time_series_features.shape[-1]
    else:
        time_series_features = np.empty((0, 0))
        num_time_series_features = 0

    # Only scale when requested
    if self.scale_data:
        data_scaler = self._prepare_data_scaler()
        data_scaler.fit(full_array[self._train_slice])
        full_array = data_scaler.transform(full_array)
        self._full_data = full_array

        # Scale time features if present
        if num_time_series_features > 0:
            ts_feature_scaler = self._prepare_data_scaler()
            ts_feature_scaler.fit(time_series_features[self._train_slice])
            scaled_ts_features = ts_feature_scaler.transform(time_series_features)
            scaled_ts_features = np.expand_dims(scaled_ts_features, axis=0)
            repeated_ts = np.repeat(
                scaled_ts_features, self._full_data.shape[0], axis=0
            )
            self._full_data = np.concatenate(
                [repeated_ts, self._full_data], axis=-1
            )
    else:
        # When not scaling, still need to handle time features (unscaled)
        if num_time_series_features > 0:
            scaled_ts_features = np.expand_dims(time_series_features, axis=0)
            repeated_ts = np.repeat(
                scaled_ts_features, self._full_data.shape[0], axis=0
            )
            self._full_data = np.concatenate(
                [repeated_ts, self._full_data], axis=-1
            )

    # Apply module-specific transform
    self._transform_data()

    self._num_time_series_features = num_time_series_features
    self._calculate_num_features()
    self._split_data()
```

---

### CR-03: Hardcoded `.iloc[8920]` crashes on smaller electricity datasets

**File:** `src/tscollection/datasets/modules/electricity.py:144`

**Issue:** The column filtering logic `df.cumsum(axis=0).iloc[8920]` accesses row index 8920 (the 8921st row) without verifying the DataFrame has that many rows. If `df` has fewer than 8921 rows after resampling (e.g., datasets shorter than ~372 days at hourly resolution), this raises `IndexError: single positional indexer is out-of-bounds`.

The number 8920 is a magic constant -- it appears to correspond to approximately 6 months of hourly data (6 * 30 * 24 = 4320, so not an exact match; 8920 hours is roughly 13.7 days). Without documentation of what this index represents, it is unclear whether the logic is intentional or a copy-paste artifact from a specific dataset.

This is a correctness bug: the module silently assumes the input data exceeds a specific size, and the error message from the crash (`IndexError: single positional indexer is out-of-bounds`) does not tell the user that the dataset is too small.

**Fix:**
```python
# Add bounds check before accessing the hardcoded index
MIN_ROWS_FOR_FILTERING = 8921  # Document the magic number
if len(df) < MIN_ROWS_FOR_FILTERING:
    raise ValueError(
        f'Electricity dataset too short ({len(df)} rows); '
        f'minimum {MIN_ROWS_FOR_FILTERING} rows required for column filtering. '
        f'Expected hourly data spanning multiple years.'
    )
df = df.loc[:, df.cumsum(axis=0).iloc[8920] != 0]
```

Alternatively, if the intent is to remove zero-valued columns, a more robust approach:
```python
# Remove columns that are all zeros (or have zero cumulative sum)
df = df.loc[:, (df != 0).any(axis=0)]
```

---

## Warnings

### WR-01: UEA labels computed as numpy array, then converted to pd.Series after state use

**File:** `src/tscollection/datasets/modules/uea.py:276-285`

**Issue:** The `_train_data_labels`, `_test_data_labels`, and `_valid_data_labels` are set as numpy arrays by `_process_stacked_data()` (line 165: `return output_data, np.array(encoded_labels)`). Module state (`_num_classes` at line 284, `_seq_len` at line 285) is computed after the labels are converted to `pd.Series` (lines 276-281), so there is no immediate type mismatch. However, the conversion happens AFTER `_process_data_with_varying_sequence_lengths()` is called (line 273), and if any code path accesses `train_data_labels` property between `prepare_data()` and `setup()`, the labels would be numpy arrays rather than the `pd.Series` type expected by `UEAClassificationMultivariateDataset` (which accepts `pd.Series | pd.DataFrame | None` per the type hint in `datasets/uea.py:43`).

The LightningDataModule lifecycle calls `prepare_data()` once, then `setup()`, then dataloader methods. Between `prepare_data()` and dataloader methods, `setup()` runs. The labels are converted to Series at the end of `prepare_data()`, so by the time dataloaders run, labels are correct. Still, the ordering is fragile: any refactoring that inserts code between the variable-length processing (line 273) and the Series conversion (line 276) would reintroduce the type mismatch.

**Fix:** Move the Series conversion earlier, immediately after `_process_stacked_data()` returns, and document the invariant.

---

### WR-02: `_split_data` assertion fails when `valid_size=0.0`

**File:** `src/tscollection/datasets/modules/classes/forecasting.py:234`

**Issue:** The `_split_data()` method asserts `self._valid_slice is not None` at line 234. If a user configures `valid_size=0.0` and a subclass does not explicitly set `_valid_slice` to a valid slice (rather than None), this assertion crashes. The `_set_data_slices()` implementations in Electricity and Weather always create a valid slice (since they use fixed 60/20/20 splits), so this may not trigger in practice for current subclasses. However, it creates a brittle contract: any future subclass that supports `valid_size=0.0` by setting `_valid_slice=None` would hit this assertion.

Contrast with `_process_valid_dataloader()` in `base.py` (line 295), which correctly handles the `valid_size == 0.0` case by returning `None` before attempting to use the data.

**Fix:** Guard against None in `_split_data()`:
```python
def _split_data(self) -> None:
    assert self._full_data is not None
    assert self._train_slice is not None
    assert self._test_slice is not None

    self._train_data_samples = self._full_data[:, self._train_slice]
    self._test_data_samples = self._full_data[:, self._test_slice]

    if self._valid_slice is not None:
        self._valid_data_samples = self._full_data[:, self._valid_slice]
    else:
        self._valid_data_samples = None
```

---

### WR-03: Scaling functions drop DataFrame index and column metadata

**File:** `src/tscollection/datasets/utils/scaling.py:146-163`

**Issue:** The `_scale_regular_data_and_return_same_type()` function preserves column names for train data (line 155: `pd.DataFrame(scaled_train, columns=train_data.columns)`), but discards the DataFrame index entirely. The original `train_data.index` is lost after scaling. For validation and test data, columns are preserved (lines 156-161), but indices are also dropped.

This matters because `process_data_with_varying_sequence_lengths_single()` in `utils/general.py:82-103` also drops column names when converting back from numpy to DataFrame (line 101: `pd.DataFrame(data)` with no columns argument). After the full pipeline (scaling + variable-length processing), DataFrames lose both their index and column names, making debugging harder and potentially breaking code that relies on named columns.

**Fix:** Preserve index when reconstructing DataFrames:
```python
# In _scale_regular_data_and_return_same_type():
if isinstance(train_data, pd.DataFrame):
    scaled_train = pd.DataFrame(
        scaled_train, columns=train_data.columns, index=train_data.index
    )
```

---

### WR-04: ETT univariate mode depends on implicit column name 'OT'

**File:** `src/tscollection/datasets/modules/ett.py:163`

**Issue:** When `mode=ForecastingMode.UNIVARIATE`, the ETT module selects `df = df[['OT']]`. This hardcodes the target column name to 'OT' (outdoor temperature), which is specific to the ETT dataset schema. If a user provides a CSV with different column names, or if the standard ETT dataset uses a different column for the target in some variant, this silently fails with `KeyError: 'OT'`.

The error message would be a raw pandas KeyError without context about the ForecastingMode setting, making it harder to debug.

**Fix:** Either document the 'OT' requirement or add validation:
```python
if self._mode == ForecastingMode.UNIVARIATE:
    if 'OT' not in df.columns:
        raise KeyError(
            f"Column 'OT' not found in ETT data. "
            f"Available columns: {list(df.columns)}. "
            f"For UNIVARIATE mode, the dataset must contain an 'OT' column."
        )
    df = df[['OT']]
```

---

### WR-05: ETT hardcoded slice boundaries with no data-length validation

**File:** `src/tscollection/datasets/modules/ett.py:117-124`

**Issue:** The `_set_data_slices()` method defines fixed boundaries (`12 * 30 * 24 = 8640` for hourly, `8640 * 4 = 34560` for 15-min) without checking that the actual data length covers these boundaries. The docstring claims "16-month / 4-month / 4-month splits", but the code uses 12 months for training (not 16), which is inconsistent with the documented split.

If a user loads a truncated ETT file (e.g., only 6 months of data), the train slice `slice(None, 8640)` would consume all available data, the valid slice `slice(8640, 11520)` would produce an empty array, and the test slice `slice(11520, 17280)` would also be empty. This silent data loss would only surface later when training fails or metrics look wrong.

Additionally, the split is described as "16/4/4" in the module docstring (line 4) and comments, but the actual code splits at 12 months (train), 4 months (valid), 4 months (test) -- that is 12/4/4, not 16/4/4. The 16-month reference appears to be the total training period in the original paper, but the code uses 12 months.

**Fix:** Add data-length validation:
```python
def _set_data_slices(self) -> None:
    assert self._full_data is not None
    num_samples = len(self._full_data)

    if self.variant in {'ETTh1', 'ETTh2'}:
        train_end = 12 * 30 * 24
        valid_end = 16 * 30 * 24
        test_end = 20 * 30 * 24
    else:
        train_end = 12 * 30 * 24 * 4
        valid_end = 16 * 30 * 24 * 4
        test_end = 20 * 30 * 24 * 4

    if num_samples < test_end:
        logger.warning(
            'ETT data has %d samples, less than expected %d for variant %s. '
            'Train/valid/test splits may be truncated.',
            num_samples, test_end, self.variant,
        )

    self._train_slice = slice(None, train_end)
    self._valid_slice = slice(train_end, valid_end)
    self._test_slice = slice(valid_end, test_end)
```

---

### WR-06: `test_val_dataloader_returns_dataloader_or_none` has loose assertion

**File:** `tests/test_modules_ucr.py:185-186`

**Issue:** The assertion `assert dl is None or isinstance(dl, DataLoader)` passes regardless of whether `dl` is a `DataLoader` or `None`. With `valid_size=0.1` (non-zero), the test should verify that a `DataLoader` is actually returned. The current assertion accepts `None` as a valid result, meaning the test does not verify the positive case (that validation data exists when `valid_size > 0`).

Combined with the synthetic test data (only 14 samples, 2 classes), the `train_test_split` with `stratify` might fail or return edge-case results that produce `None` for validation. The test silently passes even in failure scenarios.

**Fix:**
```python
def test_val_dataloader_returns_dataloader_or_none(
    self,
    module_class: type,
    synthetic_ucr_folder: Path,
) -> None:
    mod = module_class(
        dataset_folder_path=synthetic_ucr_folder,
        target_column_name='class',
        valid_size=0.1,
    )
    mod.prepare_data()
    mod.setup('fit')

    dl = mod.val_dataloader(mode=TimeSeriesDatasetMode.WITH_LABELS)
    # With valid_size > 0 and sufficient data, should return a DataLoader
    if mod._valid_data_samples is not None:
        assert isinstance(dl, DataLoader), (
            'Expected DataLoader when valid_size > 0 and data is available'
        )
    else:
        assert dl is None

    # Test with valid_size=0
    mod_no_val = module_class(
        dataset_folder_path=synthetic_ucr_folder,
        target_column_name='class',
        valid_size=0.0,
    )
    mod_no_val.prepare_data()
    mod_no_val.setup('fit')
    assert mod_no_val.val_dataloader(mode=TimeSeriesDatasetMode.WITH_LABELS) is None
```

---

## Info

### IN-01: Unused `abstractmethod` import in test fixture

**File:** `tests/test_modules_base.py:26`

**Issue:** Inside the `concrete_module_class` fixture, `from abc import abstractmethod` is imported but never used. The `ConcreteTestModule` class simply overrides `prepare_data` without using `abstractmethod`.

**Fix:** Remove the unused import:
```python
@pytest.fixture
def concrete_module_class(self):
    """Create a minimal concrete subclass for testing."""
    from tscollection.datasets.modules.classes.base import (
        BaseTimeSeriesDataModule,
    )

    class ConcreteTestModule(BaseTimeSeriesDataModule):
        """Minimal concrete subclass for testing."""

        def prepare_data(self) -> None:
            pass

    return ConcreteTestModule
```

---

### IN-02: Test reads source files as strings to verify implementation

**File:** `tests/test_modules_forecasting.py:322-347`

**Issue:** The `TestForecastingModulesUseTensorDataset` class opens source files with bare `open()` calls (not using context managers) and searches for `'TensorDataset'` in the raw text. This tests implementation details (source code content) rather than behavior (actual DataLoader output). If the code is refactored (e.g., TensorDataset imported via alias), these tests would break. Additionally, files are opened but never closed (no `with` statement), leaking file descriptors.

**Fix:** Replace with behavioral tests:
```python
class TestForecastingModulesUseTensorDataset:
    """Tests that all three modules produce TensorDataset-backed DataLoaders."""

    def test_ett_dataloader_uses_tensordataset(self, synthetic_csv_file: Path) -> None:
        from tscollection.datasets.modules.ett import ETTDataModule
        module = ETTDataModule(dataset_file_path=synthetic_csv_file, variant='ETTh1')
        # Verify the dataloader methods exist and reference TensorDataset
        import inspect
        source = inspect.getsource(module.train_dataloader)
        assert 'TensorDataset' in source
```

Or, prefer closing the file handle:
```python
def test_ett_uses_tensordataset_in_source(self) -> None:
    import tscollection.datasets.modules.ett as ett_module
    source_path = Path(ett_module.__file__).parent / 'ett.py'
    with open(source_path) as f:
        assert 'TensorDataset' in f.read()
```

---

### IN-03: Redundant `{}` argument in `defaultdict` constructor

**File:** `src/tscollection/datasets/modules/classes/base.py:180-182`

**Issue:** `defaultdict(lambda: lambda x: x, {})` passes an empty dict as the initial data. Since no key-value pairs are provided, the second argument has no effect. It also adds unnecessary noise to the code, making it look like the empty dict is intentional when it is not.

The same pattern appears in `ucr.py:111-119` where the defaultdict is initialized with actual key-value pairs -- that one is correct. But the base class initialization is redundant.

**Fix:**
```python
def _initiate_datatypes_handling_functions_map(self) -> None:
    self._datatype_handling_functions_map = defaultdict(
        lambda: lambda x: x
    )
```

---

### IN-04: Magic number `8920` lacks documentation

**File:** `src/tscollection/datasets/modules/electricity.py:144`

**Issue:** Beyond the crash risk identified in CR-03, the value 8920 has no comment explaining its origin. Is it the number of rows expected in a specific electricity dataset version? A date offset? A filtering heuristic? Without documentation, future maintainers cannot determine whether this number should change when the dataset format evolves.

**Fix:** Add a comment explaining the intent:
```python
# Filter columns: at row 8920 (approx. month X of the original dataset),
# remove columns whose cumulative sum is still zero (all-zero columns)
df = df.loc[:, df.cumsum(axis=0).iloc[8920] != 0]
```

---

### IN-05: Empty `if TYPE_CHECKING: pass` blocks

**File:** `src/tscollection/datasets/modules/electricity.py:29-30`, `src/tscollection/datasets/modules/ett.py:30-31`, `src/tscollection/datasets/modules/weather.py:28-29`, `src/tscollection/datasets/modules/uea.py:36-37`

**Issue:** Four module files contain `if TYPE_CHECKING: pass` blocks with no actual type-checking-only imports. These are vestigial -- likely added as scaffolding but never populated. They add noise to the source files without serving a purpose.

**Fix:** Remove the empty blocks entirely. The `TYPE_CHECKING` import can also be removed if it becomes unused.

---

_Reviewed: 2026-05-13T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
