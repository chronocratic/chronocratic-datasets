# Phase 4: Data Modules - Research

**Researched:** 2026-05-13
**Domain:** PyTorch Lightning DataModule hierarchy for time series datasets
**Confidence:** HIGH

## Summary

This phase implements the `LightningDataModule` hierarchy that wraps Phase 2 dataset classes and Phase 3 utility functions into ready-to-use dataloaders for Lightning Trainer. The hierarchy has two branches: classification (UCR univariate, UEA multivariate) and forecasting (ETT, Electricity, Weather). Each branch has a base class defining shared dataloader construction and scaling, plus concrete subclasses handling dataset-specific file reading, split boundaries, and transforms.

The primary source code is `_sources/rbspaper/src/rbspaper/data/modules/` with 8 source files (1 abstract base, 2 classification, 3 forecasting modules, plus 1 electricity module). Key deviations from source: (1) replace JSON config with explicit constructor parameters, (2) use `ScalingMethod`/`ForecastingMode` enums instead of strings, (3) rename `SplittingStrategy` to `ClassificationSplittingStrategy`, (4) port `separate_target_feature_from_df` utility from rbspaper's bundled `common.py`, (5) follow Lightning best practice of validating in `prepare_data()` and loading/splitting/scaling in `setup()`.

**Primary recommendation:** Implement three base class files (`base.py`, `classification.py`, `forecasting.py`) and five concrete module files (`ucr.py`, `uea.py`, `ett.py`, `electricity.py`, `weather.py`), plus wire all `__init__.py` exports. Port `separate_target_feature_from_df` to `utils/common.py` before building classification modules.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Classification modules accept explicit params instead of `dataset_config_path` — `target_col_name`, `data_form` as constructor kwargs. ARFF file patterns hardcoded in each concrete subclass (UCR: `{dataset_name}_TRAIN.arff`, `{dataset_name}_TEST.arff`; UEA: same pattern). No JSON configs per v1 constraint.
- **D-02:** `data_form` hardcoded as `DataForm` enum per subclass — UCR: `DataForm.REGULAR`, UEA: `DataForm.NESTED`. Not user-configurable; intrinsic dataset fact.
- **D-03:** `ScalingMethod` enum (not string) for `data_scaling_method` constructor param. Consistent with Phase 3 enum wiring.
- **D-04:** `SplittingStrategy` in enums/data.py renamed to `ClassificationSplittingStrategy` — it's classification-only, forecasting uses intrinsic time slices.
- **D-05:** `ForecastingMode` enum from enums/data.py for univariate/multivariate mode.
- **D-06:** ETT module accepts explicit `variant` param (`"ETTh1"`, `"ETTh2"`, `"ETTm1"`, `"ETTm2"`) to determine split boundaries. No filename auto-detection.
- **D-07:** File path params (`dataset_file_path`, `dataset_folder_path`) typed as `Path` only. No str acceptance.
- **D-08:** Three separate files: `modules/classes/base.py` (BaseTimeSeriesDataModule), `modules/classes/classification.py` (BaseClassificationTimeSeriesDataModule), `modules/classes/forecasting.py` (BaseForecastingTimeSeriesDataModule). Follows Phase 3 file-per-concern pattern.
- **D-09:** Follow Lightning's recommended pattern: `prepare_data()` validates file paths and performs lightweight checks only. `setup()` handles data loading, splitting, scaling, and sets module state (`_train_data_samples`, `_seq_len`, etc.). This deviates from rbspaper source which loads data in `prepare_data()`.
- **D-10:** Classification base uses `create_data_scaler()` from utils in `setup()`. Forecasting base uses sklearn scalers directly (`_prepare_data_scaler()`) — different data shapes require different scaling approaches.
- **D-11:** Full names: `sequence_length`, `num_features`, `num_classes` — matches ROADMAP.md MOD-05 verbatim. Internal attributes: `_seq_len`, `_num_features`, `_num_classes`.
- **D-12:** UEA's nested ARFF processing (`_process_stacked_data()`) stays internal to UEA module. Does not use `arff.py` utility — raw `scipy.io.arff.loadarff()` with manual byte-decoding, reshaping, and LabelEncoder.
- **D-13:** Keep `TensorDataset` for forecasting module dataloaders (defer proper dataset class integration to a future phase).
- **D-14:** Keep `extra_args`, `mode`, `strict_batch_size` params on `train_dataloader()`, `val_dataloader()`, `test_dataloader()` — preserves flexibility for ad-hoc scripting beyond Lightning calls.
- **D-15:** Wire all `modules/__init__.py` and `modules/classes/__init__.py` exports in Phase 4. Phase 5 tests need clean imports.
- **D-16:** Fail fast with descriptive errors in `prepare_data()` — `FileNotFoundError` for missing files, `ValueError` for format issues. No silent degradation.

### Claude's Discretion

- Internal implementation of `_set_data_slices()`, `_transform_data()` abstract methods per concrete forecasting module follows rbspaper source logic.
- Weather and Electricity modules share 60/20/20 fractional split pattern. Climate's module-specific transform (`.T` + `expand_dims(axis=-1)` vs Weather's `expand_dims(axis=0)`) is preserved as-is.
- Validation split fallback for small datasets (stratify error handling) follows rbspaper pattern.

### Deferred Ideas (OUT OF SCOPE)

- **Forecasting proper dataset classes** — D-13 defers using `FlexibleTimeSeriesDataset` in forecasting dataloaders. TensorDataset used for now.
- **Nested ARFF utility** — D-12 keeps UEA `_process_stacked_data()` internal. If additional nested-ARFF datasets emerge, extracting to `arff.py` becomes worthwhile.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MOD-01 | Module loads data from provided file paths in `prepare_data()` | `prepare_data()` in concrete classes reads ARFF/CSV files; per D-09, validation happens in `prepare_data()`, actual loading in `setup()` |
| MOD-02 | User passes module to Lightning Trainer with explicit file paths | Constructor accepts `dataset_file_path` or `dataset_folder_path` as `Path` kwargs (D-07) |
| MOD-03 | Classification modules support `AS_DEFINED` and `MANUAL` splitting strategies | `ClassificationSplittingStrategy` enum (D-04); `train_test_split` logic in classification base (D-01) |
| MOD-04 | Forecasting modules use dataset-intrinsic split boundaries | `_set_data_slices()` abstract method; ETT: 16/4/4 month slices, Weather/Electricity: 60/20/20 fractional |
| MOD-05 | Modules expose `sequence_length`, `num_classes`, `num_features` as read-only properties | Property methods in base class (D-11); `_seq_len`, `_num_features`, `_num_classes` internal attrs |
| MOD-06 | Dataloader methods return `DataLoader` instances | `_process_train/val/test_dataloader()` base methods; concrete modules implement `train/val/test_dataloader()` |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| File validation (existence, format) | Classification/Forecasting concrete modules | — | Dataset-specific formats (ARFF vs CSV) |
| Data loading (ARFF, CSV) | Classification/Forecasting concrete modules | — | Format parsing is dataset-specific |
| Train/val/test splitting | Classification base, Forecasting base | — | Split logic differs by branch |
| Data scaling | Classification base (`create_data_scaler`), Forecasting base (sklearn directly) | — | Per D-10; different data shapes |
| Variable-length handling | Classification base | — | Only applies to classification |
| Time feature extraction | Forecasting base | — | Only applies to forecasting |
| Dataloader construction | BaseTimeSeriesDataModule | — | Shared `_process_*_dataloader()` methods |
| Label encoding | UEA module only | — | Per D-12; internal to UEA |

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| lightning | 2.5.6 [VERIFIED: pip] | `LightningDataModule` base class, `Trainer` integration | Required by project; provides dataloader lifecycle |
| torch | 2.x [VERIFIED: pyproject.toml] | `TensorDataset`, `DataLoader` | Fundamental ML framework |
| numpy | 2.x [VERIFIED: pyproject.toml] | Array operations, splitting, reshaping | Universal numeric computation |
| pandas | 2.2+ [VERIFIED: pyproject.toml] | CSV reading, DataFrame operations, time index | Tabular data handling |
| scipy | 1.13+ [VERIFIED: pyproject.toml] | `scipy.io.arff.loadarff()` for ARFF files | Only standard ARFF reader for Python |
| scikit-learn | 1.6+ [VERIFIED: pyproject.toml] | `MinMaxScaler`, `StandardScaler`, `train_test_split`, `LabelEncoder` | Standard scaling and splitting |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `functools.partial` | stdlib | Bind `custom_collate_fn` with `desired_batch_size` | Strict batch size enforcement in dataloaders |
| `sklearn.preprocessing.LabelEncoder` | 1.6+ | Encode string labels to integers for UEA | UEA module only |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `scipy.io.arff.loadarff()` | `liac-arff` library | `liac-arff` supports more ARFF variants but adds dependency; `scipy.io.arff` sufficient for UCR/UEA format |
| `TensorDataset` (D-13) | `FlexibleTimeSeriesDataset` | Proper dataset classes are deferred; `TensorDataset` is simpler and sufficient for v1 |

**Installation:** No new packages needed. All dependencies already in `pyproject.toml`.

## Architecture Patterns

### System Architecture Diagram

```
User (file paths + params)
  |
  v
+------------------------------------------+
|  Concrete Module (UCR/UEA/ETT/etc.)      |
|                                          |
|  __init__()                              |
|    - Store file paths, batch_size, etc.  |
|    - Set data_form, scaling defaults     |
|                                          |
|  prepare_data()  [Lightning calls this]  |
|    - Validate file paths exist           |
|    - Read ARFF/CSV into memory           |
|    - Parse, clean, split train/test/val  |
|    - Set _seq_len, _num_features, etc.   |
+------------------------------------------+
  |
  v  (setup() is called by Lightning Trainer)
+------------------------------------------+
|  Base Module (Classification/Forecasting)|
|                                          |
|  setup()  [Lightning calls this]         |
|    Classification:                       |
|      - create_data_scaler() -> scale()   |
|      - Variable-length centering         |
|    Forecasting:                          |
|      - Fit scaler on train slice only    |
|      - Transform full data               |
|      - Extract time features             |
|      - Apply per-module _transform_data()|
|      - Slice into train/val/test         |
+------------------------------------------+
  |
  v  (dataloader() methods called by Trainer)
+------------------------------------------+
|  BaseTimeSeriesDataModule                |
|                                          |
|  _process_train_dataloader()             |
|  _process_val_dataloader()               |
|  _process_test_dataloader()              |
|    - Build DataLoader with collate_fn    |
|    - Handle persistent_workers           |
|    - Apply strict_batch_size logic       |
+------------------------------------------+
  |
  v
PyTorch DataLoader (yields batches)
```

### Recommended Project Structure

```
src/tscollection/datasets/modules/
├── __init__.py              # Public exports for all module classes (D-15)
├── classes/
│   ├── __init__.py          # Base class exports (D-15)
│   ├── base.py              # BaseTimeSeriesDataModule (D-08)
│   ├── classification.py    # BaseClassificationTimeSeriesDataModule (D-08)
│   └── forecasting.py       # BaseForecastingTimeSeriesDataModule (D-08)
├── ucr.py                   # UCRClassificationDataModule (D-01, D-07)
├── uea.py                   # UEAClassificationDataModule (D-12)
├── ett.py                   # ETTDataModule (D-06)
├── electricity.py           # ElectricityLoadModule
└── weather.py               # WeatherModule
```

### Pattern 1: Classification Module Lifecycle

**What:** Two-phase data preparation — `prepare_data()` loads and splits raw data, `setup()` scales and centers.

**When to use:** All classification modules (UCR, UEA).

**Flow:**
1. `prepare_data()`: Read ARFF files, separate data from labels, create train/val/test splits, compute `_seq_len`, `_num_features`, `_num_classes`, apply variable-length processing.
2. `setup()`: Invoke `create_data_scaler()` callable on (train, valid, test) tuples.

```python
# Classification base setup() pattern
def setup(self, stage: str) -> None:
    scaler = create_data_scaler(
        scale=self.scale_data,
        scaling_range=self.data_scaling_range,
        scaling_method=self.data_scaling_method,
        data_form=self._data_form,
    )
    (
        self._train_data_samples,
        self._valid_data_samples,
        self._test_data_samples,
    ) = scaler(
        self._train_data_samples,
        self._valid_data_samples,
        self._test_data_samples,
    )
```

**Source:** Adapted from `_sources/rbspaper/src/rbspaper/data/modules/abstract.py` `BaseTimeSeriesDataModule.setup()`.

### Pattern 2: Forecasting Module Lifecycle

**What:** Two-phase data preparation — `prepare_data()` reads CSV and sets slices, `setup()` scales, extracts time features, transforms, and splits.

**When to use:** All forecasting modules (ETT, Electricity, Weather).

**Flow:**
1. `prepare_data()`: Read CSV file, select columns (univariate vs multivariate), call `_post_prepare_data()` which invokes `_set_data_slices()`.
2. `setup()`: Extract DataFrame index (if any), fit scaler on train slice, transform full data, apply `_transform_data()`, extract time features, calculate feature count, slice into train/val/test.

```python
# Forecasting base setup() pattern (simplified)
def setup(self, stage: str) -> None:
    # Handle DataFrame vs ndarray input
    if isinstance(self._full_data, pd.DataFrame):
        time_index = self._full_data.index
        full_array = self._full_data.to_numpy()
    else:
        time_index = None
        full_array = self._full_data

    # Extract cyclical time features
    if time_index is not None:
        time_series_features = extract_time_features(pd.DatetimeIndex(time_index))
        num_time_series_features = time_series_features.shape[-1]
    else:
        time_series_features = np.empty((0, 0))
        num_time_series_features = 0

    # Scale on train slice only
    data_scaler = self._prepare_data_scaler()
    data_scaler.fit(full_array[:, self._train_slice])
    self._full_data = data_scaler.transform(full_array)

    # Module-specific transform
    self._transform_data()

    # Add time features back
    if num_time_series_features > 0:
        ts_feature_scaler = self._prepare_data_scaler()
        ts_feature_scaler.fit(time_series_features[:, self._train_slice])
        scaled_ts_features = ts_feature_scaler.transform(time_series_features)
        scaled_ts_features = np.expand_dims(scaled_ts_features, axis=0)
        repeated_ts = np.repeat(scaled_ts_features, self._full_data.shape[0], axis=0)
        self._full_data = np.concatenate([repeated_ts, self._full_data], axis=-1)

    self._num_time_series_features = num_time_series_features
    self._calculate_num_features()
    self._split_data()
```

**Source:** `_sources/rbspaper/src/rbspaper/data/modules/abstract.py` `BaseForecastingTimeSeriesDataModule.setup()`.

### Pattern 3: Strict Batch Size Collation

**What:** Custom collate function that pads the last batch by cycling samples to maintain exact batch size.

**When to use:** Classification training dataloaders (`strict_batch_size=True` default).

```python
# From utils/general.py
def custom_collate_fn(batch: list[Any], *, desired_batch_size: int) -> Any:
    current_batch_size = len(batch)
    if current_batch_size < desired_batch_size:
        additional_needed = desired_batch_size - current_batch_size
        for i in range(additional_needed):
            sample_index = current_batch_size - 1 - (i % current_batch_size)
            batch.append(batch[sample_index])
    return default_collate(batch)
```

Applied via `functools.partial`:
```python
def _get_custom_collate_fn(self, desired_batch_size: int | None = None) -> Any:
    if desired_batch_size is None:
        desired_batch_size = self.batch_size
    return partial(custom_collate_fn, desired_batch_size=desired_batch_size)
```

### Anti-Patterns to Avoid

- **Loading data in `prepare_data()` when `setup()` should handle scaling** — Per D-09, `prepare_data()` validates and loads raw data; `setup()` applies scaling. rbspaper source does everything in `prepare_data()`. Do not copy that pattern.
- **Using string literals for `ScalingMethod`** — The rbspaper source passes `'min_max'` and `'standardization'` as strings. Our project uses `ScalingMethod` enum with `'minmax'`/`'standard'` values. Passing strings will cause comparison failures in `_get_scaler()`. Always use enum values.
- **Sharing `data_scaling_method` as `str` type** — Constructor params should be `ScalingMethod` (D-03), not `str`. The type hint must reflect the enum type.
- **Putting `_process_stacked_data()` in `arff.py`** — Per D-12, UEA's nested ARFF processing stays internal to the UEA module. The `arff.py` utility (`read_arff_as_df` + `process_df_according_to_dtypes`) is only for regular (UCR) ARFF files.
- **Accepting `str` for file paths** — Per D-07, file path params are `Path` only. Do not add `str | Path` union types.
- **Auto-detecting ETT variant from filename** — Per D-06, ETT module requires explicit `variant` parameter. Do not parse filenames to infer splits.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ARFF file parsing | Custom ARFF reader | `scipy.io.arff.loadarff()` | Handles all ARFF syntax variants; well-tested |
| Train/val/test splitting | Manual index calculation | `sklearn.model_selection.train_test_split` | Handles stratification, edge cases, reproducibility |
| Label encoding | Manual dict mapping | `sklearn.preprocessing.LabelEncoder` | Standard integer encoding with inverse transform |
| Data scaling | Manual min-max/standard formula | `sklearn.preprocessing.MinMaxScaler` / `StandardScaler` | Handles shape preservation, fit/transform separation |
| Time feature extraction | Manual datetime parsing | `utils.features.extract_time_features()` | Phase 3 utility already implements this |
| Custom collation | Manual batch padding | `utils.general.custom_collate_fn()` | Phase 3 utility already implements this |
| Variable-length centering | Manual NaN detection | `utils.general.process_data_with_varying_sequence_lengths_single()` | Phase 3 utility already implements this |

**Key insight:** This phase is primarily about orchestration — wiring existing Phase 2 dataset classes and Phase 3 utility functions through Lightning's DataModule lifecycle. The value is in correct lifecycle integration, not in new data processing logic.

## Common Pitfalls

### Pitfall 1: Enum Value Mismatch Between Source and Project

**What goes wrong:** Copying rbspaper source that uses `'min_max'` and `'standardization'` string literals directly into our code, where `ScalingMethod` enum defines `'minmax'` and `'standard'` values.

**Why it happens:** The rbspaper source predates our enum-based refactoring. String comparisons in `_get_scaler()` will fail.

**How to avoid:** Always use `ScalingMethod` enum members (`ScalingMethod.MINMAX`, `ScalingMethod.STANDARD`) in constructor defaults and comparisons. The `_get_scaler()` function in `scaling.py` checks `scaling_method == ScalingMethod.MINMAX`, not string equality.

**Warning signs:** `ValueError: Unsupported scaling method: min_max` at runtime.

### Pitfall 2: Missing `separate_target_feature_from_df` Utility

**What goes wrong:** Classification base class imports `separate_target_feature_from_df` from utils but it does not exist in our project.

**Why it happens:** The function lives in rbspaper's `common.py` but was not ported to our `utils/common.py` in Phase 2 or 3.

**How to avoid:** Port `separate_target_feature_from_df` to `utils/common.py` as the first task. It is a simple 4-line function:

```python
def separate_target_feature_from_df(
    df: pd.DataFrame, target_feature_name: str
) -> tuple[pd.DataFrame, pd.Series]:
    target_feature = df[target_feature_name]
    features = df.drop(target_feature_name, axis=1)
    return features, target_feature
```

Also add it to `utils/__init__.py` exports and `utils/common.py` `__all__`.

**Warning signs:** `ImportError` when running classification modules.

### Pitfall 3: `SplittingStrategy` vs `ClassificationSplittingStrategy` Rename

**What goes wrong:** Code references `SplittingStrategy` but the enum was renamed to `ClassificationSplittingStrategy` (D-04).

**Why it happens:** The rename affects `enums/data.py`, `enums/__init__.py`, root `__init__.py`, and all classification module imports.

**How to avoid:** Perform the rename in `enums/data.py` first, update `enums/__init__.py` re-export, update root `__init__.py`, and then update classification module imports. Use `grep -rn "SplittingStrategy"` to find all references.

**Warning signs:** `ImportError: cannot import name 'ClassificationSplittingStrategy'` or stale references to `SplittingStrategy` causing confusion.

### Pitfall 4: Nested Data Scaling Type Mismatch

**What goes wrong:** `create_data_scaler()` with `DataForm.NESTED` calls `_scale_nested_data_all_dimensions()` which expects `np.ndarray` inputs, but classification data starts as `pd.DataFrame`.

**Why it happens:** UCR (regular) data is `pd.DataFrame` -> scales fine. UEA (nested) data is already `np.ndarray` from `_process_stacked_data()` -> scales fine. The issue would only arise if UCR data accidentally reached the nested scaler.

**How to avoid:** Ensure `data_form` is correctly set per subclass (D-02): UCR uses `DataForm.REGULAR`, UEA uses `DataForm.NESTED`. The enum values are hardcoded in constructors, not passed by the user.

**Warning signs:** `AttributeError: 'DataFrame' object has no attribute 'reshape'` in scaling code.

### Pitfall 5: Lightning `prepare_data()` Called Once, Not Per-Process

**What goes wrong:** Putting data loading in `prepare_data()` but expecting in-memory data to be available in `setup()`. In Lightning, `prepare_data()` is called once on the global process (for distributed training), while `setup()` is called on each process.

**Why it happens:** This is standard Lightning behavior. rbspaper source works because it does everything in `prepare_data()` before the Trainer calls it.

**How to avoid:** Per D-09, `prepare_data()` should load and store data on `self`. `setup()` then operates on that stored data. This pattern works because `prepare_data()` is called before `setup()`, and in single-GPU mode both run on the same process.

**Warning signs:** `AttributeError: '_train_data_samples' not found` in `setup()` during distributed training.

### Pitfall 6: `persistent_workers=True` with `num_workers=0`

**What goes wrong:** Setting `persistent_workers=True` when `num_workers=0` causes a PyTorch warning or error.

**Why it happens:** The rbspaper source guards this: `if self.num_workers > 0: dataloader_args['persistent_workers'] = True`.

**How to avoid:** Copy the rbspaper guard pattern exactly. Do not add `persistent_workers` to the default dataloader args dict.

**Warning signs:** `UserWarning: 'persistent_workers' was True, but 'num_workers' was 0` at runtime.

### Pitfall 7: ETT Variant Detection from Filename

**What goes wrong:** Parsing the filename stem to determine ETT variant (hourly vs 15-min) instead of accepting an explicit parameter.

**Why it happens:** rbspaper source sets `self._dataset_name = self.dataset_file_path.stem` and then branches on it. This works but is fragile.

**How to avoid:** Per D-06, accept explicit `variant` parameter (`"ETTh1"`, `"ETTh2"`, `"ETTm1"`, `"ETTm2"`). Use it directly in `_set_data_slices()` without filename parsing.

**Warning signs:** Incorrect slice boundaries when filenames don't match expected patterns.

## Code Examples

### Classification Module `prepare_data()` Pattern (UCR)

```python
def prepare_data(self) -> None:
    # Validate folder path
    if not self.dataset_folder_path.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {self.dataset_folder_path}"
        )

    self._dataset_name = self.dataset_folder_path.name

    # Construct ARFF file paths
    arff_train = self.dataset_folder_path / f"{self._dataset_name}_TRAIN.arff"
    arff_test = self.dataset_folder_path / f"{self._dataset_name}_TEST.arff"

    # Read and process ARFF files
    train_data = self._read_arff_file_as_df(arff_train)
    test_data = self._read_arff_file_as_df(arff_test)
    train_data = self._clean_data_of_missing_values(train_data)
    test_data = self._clean_data_of_missing_values(test_data)

    # Apply splitting strategy
    if self.splitting_strategy == ClassificationSplittingStrategy.MANUAL:
        combined = pd.concat([train_data, test_data], axis=0, ignore_index=True)
        train_data, test_data = train_test_split(
            combined,
            test_size=self.test_size,
            stratify=combined[self.target_column_name],
            random_state=42,
        )

    # Separate data from labels
    self._train_data_samples, self._train_data_labels = self._separate_target_feature(train_data)
    self._test_data_samples, self._test_data_labels = self._separate_target_feature(test_data)

    # Compute module state
    self._num_classes = len(self._train_data_labels.unique())
    self._seq_len = len(self._train_data_samples.columns)
    self._num_features = 1

    # Create validation split
    if self.valid_size > 0.0:
        # ... train_test_split with stratify fallback ...

    # Variable-length processing
    self._process_data_with_varying_sequence_lengths()
```

### Forecasting Module `prepare_data()` Pattern (ETT)

```python
def prepare_data(self) -> None:
    if not self.dataset_file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {self.dataset_file_path}"
        )

    self._dataset_name = self.variant  # Explicit variant, not filename
    df = pd.read_csv(self.dataset_file_path, parse_dates=True, index_col="date")

    if self._mode == ForecastingMode.UNIVARIATE:
        df = df[["OT"]]

    self._full_data = df
    self._post_prepare_data()  # Calls _set_data_slices()
```

### ETT `_set_data_slices()` Implementation

```python
def _set_data_slices(self) -> None:
    if self.variant in {"ETTh1", "ETTh2"}:
        self._train_slice = slice(None, 12 * 30 * 24)        # 8640
        self._valid_slice = slice(12 * 30 * 24, 16 * 30 * 24) # 8640:11520
        self._test_slice = slice(16 * 30 * 24, 20 * 30 * 24)  # 11520:14400
    elif self.variant in {"ETTm1", "ETTm2"}:
        self._train_slice = slice(None, 12 * 30 * 24 * 4)     # 34560
        self._valid_slice = slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
        self._test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
```

### Weather `_transform_data()` vs Electricity `_transform_data()`

```python
# Weather: expand along axis 0 -> shape (1, samples)
def _transform_data(self) -> None:
    if isinstance(self._full_data, pd.DataFrame):
        self._full_data = self._full_data.to_numpy()
    self._full_data = np.expand_dims(self._full_data, axis=0)

# Electricity: transpose + expand along last axis -> shape (features, samples, 1)
def _transform_data(self) -> None:
    if isinstance(self._full_data, pd.DataFrame):
        self._full_data = self._full_data.to_numpy()
    self._full_data = self._full_data.T
    self._full_data = np.expand_dims(self._full_data, axis=-1)
```

### Electricity CSV Parsing

```python
def prepare_data(self) -> None:
    self._dataset_name = "ElectricityLoad"
    df = pd.read_csv(
        self.dataset_file_path,
        parse_dates=True,
        sep=";",
        decimal=",",
        index_col=[0],
    )
    df = df.resample("1h", closed="right").sum()
    df = df.loc[:, df.cumsum(axis=0).iloc[8920] != 0]
    df.index = df.index.rename("date")
    df = df["2012":]

    if self._mode == ForecastingMode.UNIVARIATE:
        df = df[["MT_001"]]

    self._full_data = df
    self._post_prepare_data()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| rbspaper: All data loading in `prepare_data()` | Lightning best practice: validate in `prepare_data()`, scale in `setup()` | D-09 decision | Better distributed training compatibility; `setup()` is called per-process |
| rbspaper: `data_scaling_method` as `str` | `ScalingMethod` enum (StrEnum) | Phase 3 (D-03) | Type-safe; IDE autocomplete; compile-time checking |
| rbspaper: `dataset_config_path` JSON config | Explicit constructor parameters | v1 scope (D-01) | Simpler API; no config file dependency |
| rbspaper: Bundled `utils/__init__.py` | Separate files (arff.py, scaling.py, features.py, general.py) | Phase 3 | Better modularity; easier testing |
| rbspaper: `SplittingStrategy` for both branches | `ClassificationSplittingStrategy` for classification only | D-04 | Clearer naming; forecasting uses intrinsic slices |

### Deprecated/Outdated

- **`data_scaling_method` as string** — Use `ScalingMethod` enum. rbspaper source uses `'min_max'`/`'standardization'` strings. Our project uses `ScalingMethod.MINMAX`/`ScalingMethod.STANDARD`.
- **`dataset_config_path` parameter** — Not used in v1. Configuration comes from explicit constructor kwargs.
- **`load_json` utility for module config** — Not needed in v1 (no JSON configs). Only relevant if porting for other purposes.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | ETT CSV always has `date` column as index and `OT` column for univariate mode | Code Examples (ETT `prepare_data()`) | Module would crash on datasets with different column names; mitigated by D-06 explicit variant param |
| A2 | Electricity CSV always uses `;` separator and `,` decimal | Code Examples (Electricity `prepare_data()`) | Parsing failure on differently formatted files; mitigated by file path validation |
| A3 | UCR/UEA ARFF files always follow `{dataset_name}_TRAIN.arff` / `{dataset_name}_TEST.arff` naming | D-01 | FileNotFoundError if naming convention differs; acceptable for v1 scope |
| A4 | `separate_target_feature_from_df` function signature matches rbspaper source exactly | Don't Hand-Roll | Minor import break; easily fixable |
| A5 | Lightning 2.5.6 `setup()` signature is `setup(self, stage: str)` — not `stage: str | None` | Architecture Patterns | Verified: `setup: (self, stage: str) -> None`; no issue |

## Open Questions (RESOLVED)

1. **Should `BaseTimeSeriesDataModule` store `data_scaling_method` as `ScalingMethod` type or `str`?**
   - What we know: D-03 says use `ScalingMethod` enum. rbspaper source uses `str`.
   - What's unclear: Whether internal storage should be enum or cast to enum value.
   - Recommendation: Store as `ScalingMethod` enum. Since it's a StrEnum, it compares equal to its string value, ensuring compatibility with any code that checks `== 'minmax'`.
   - **RESOLVED:** Plans 01-03 all use `ScalingMethod` enum type for `data_scaling_method` per D-03.

2. **How should the `valid_size == 0.0` edge case be handled in classification?**
   - What we know: rbspaper source skips validation split creation when `valid_size <= 0.0`. `val_dataloader()` returns `None`.
   - What's unclear: Whether `_valid_data_labels` should be `None` or an empty Series when no validation split exists.
   - Recommendation: Set `_valid_data_labels` to `None` and handle `None` checks in `val_dataloader()`. The base `_process_valid_dataloader()` already returns `None` when `valid_size == 0.0`.
   - **RESOLVED:** Plan 01 Task 2 implements `None` handling in `_process_valid_dataloader()` and `_valid_data_labels`. Base class sets to `None` by default.

3. **Should the root `tscollection.datasets.__init__.py` export module classes?**
   - What we know: D-15 says wire `modules/__init__.py` and `modules/classes/__init__.py`. Root `__init__.py` currently only exports enums.
   - What's unclear: Whether modules should be part of the public API surface at the root level.
   - Recommendation: Wire `modules/__init__.py` with all concrete module classes and base class re-exports. Leave root `__init__.py` unchanged unless ROADMAP explicitly requires it. Phase 5 tests import from `tscollection.datasets.modules`.
   - **RESOLVED:** Plan 04 wires only `modules/__init__.py` and `modules/classes/__init__.py`. Root `__init__.py` left unchanged per D-15.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | All code | Verified | 3.12 | — |
| lightning | DataModule base class | Verified | 2.5.6 | — |
| torch | TensorDataset, DataLoader | Verified (pyproject) | 2.x | — |
| numpy | Array operations | Verified (pyproject) | 2.x | — |
| pandas | CSV/ARFF data handling | Verified (pyproject) | 2.2+ | — |
| scipy | `scipy.io.arff.loadarff()` | Verified (pyproject) | 1.13+ | — |
| scikit-learn | Scaling, splitting, encoding | Verified (pyproject) | 1.6+ | — |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** None.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.x (verified: `pyproject.toml` dependency) |
| Config file | `pyproject.toml` [tool.pytest.ini_options] — `testpaths = ["tests"]`, `pythonpath = ["."]` |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ --cov=tscollection.datasets -v` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MOD-01 | Module loads data from file paths | unit (mock files) | `pytest tests/test_modules.py::test_prepare_data_loads_files -x` | Gap — Wave 0 |
| MOD-02 | Module works with Lightning Trainer | integration | `pytest tests/test_modules.py::test_trainer_integration -x` | Gap — Wave 0 |
| MOD-03 | Classification supports AS_DEFINED and MANUAL splitting | unit | `pytest tests/test_modules.py::test_classification_splitting_strategies -x` | Gap — Wave 0 |
| MOD-04 | Forecasting uses intrinsic split boundaries | unit | `pytest tests/test_modules.py::test_forecasting_split_boundaries -x` | Gap — Wave 0 |
| MOD-05 | Modules expose sequence_length, num_classes, num_features | unit | `pytest tests/test_modules.py::test_module_properties -x` | Gap — Wave 0 |
| MOD-06 | Dataloader methods return DataLoader instances | unit | `pytest tests/test_modules.py::test_dataloader_returns -x` | Gap — Wave 0 |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/ -x -q` — existing tests still pass
- **Per wave merge:** `uv run pytest tests/ -v` — full suite
- **Phase gate:** All tests green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/test_modules.py` — covers MOD-01 through MOD-06
- [ ] `tests/conftest.py` — needs synthetic ARFF fixtures for classification module tests; existing fixtures (`synthetic_classification_df`, `synthetic_classification_labels`, `synthetic_forecast_data`, `synthetic_multivariate_data`) provide numpy/pandas data but not file-based ARFF
- [ ] Framework: No additional install needed — pytest already in dev dependencies

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | N/A — data-only package |
| V3 Session Management | No | N/A — no sessions |
| V4 Access Control | No | N/A — no authorization layer |
| V5 Input Validation | Yes | `Path` existence checks (D-16); type validation via `Path`-only params (D-07) |
| V6 Cryptography | No | N/A — no encryption |

### Known Threat Patterns for this Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via `dataset_folder_path` | Spoofing | `Path.exists()` check; user controls input, not external |
| ARFF file injection (malformed ARFF causing crashes) | Repudiation | `scipy.io.arff.loadarff()` raises descriptive errors; no custom parser |
| Data poisoning via unvalidated CSV | Tampering | Out of scope — user provides trusted data files |

## Sources

### Primary (HIGH confidence)
- `_sources/rbspaper/src/rbspaper/data/modules/abstract.py` — Full base class hierarchy source code [VERIFIED: local file read]
- `_sources/rbspaper/src/rbspaper/data/modules/ucr_datamodule.py` — UCR module source [VERIFIED: local file read]
- `_sources/rbspaper/src/rbspaper/data/modules/uea_datamodule.py` — UEA module source [VERIFIED: local file read]
- `_sources/rbspaper/src/rbspaper/data/modules/ett_datamodule.py` — ETT module source [VERIFIED: local file read]
- `_sources/rbspaper/src/rbspaper/data/modules/electricity_load_datamodule.py` — Electricity module source [VERIFIED: local file read]
- `_sources/rbspaper/src/rbspaper/data/modules/weather_datamodule.py` — Weather module source [VERIFIED: local file read]
- `_sources/rbspaper/src/rbspaper/data/utils/common.py` — `separate_target_feature_from_df`, `load_json` [VERIFIED: local file read]
- LightningDataModule API (lightning 2.5.6) — `setup(self, stage: str)`, `prepare_data(self)` [VERIFIED: runtime check]
- `pyproject.toml` — Dependencies: lightning>=2.5,<3.0, torch>=2.4,<3.0, numpy>=2.1, pandas>=2.2, scipy>=1.13, scikit-learn>=1.6 [VERIFIED: local file read]

### Secondary (MEDIUM confidence)
- `src/tscollection/datasets/utils/scaling.py` — `create_data_scaler()` with `ScalingMethod` enum [VERIFIED: local file read]
- `src/tscollection/datasets/enums/data.py` — `ScalingMethod`, `ForecastingMode`, `SplittingStrategy` (needs D-04 rename) [VERIFIED: local file read]
- `.planning/phases/04-data-modules/04-CONTEXT.md` — All D-01 through D-16 decisions [VERIFIED: local file read]

### Tertiary (LOW confidence)
- Electricity CSV column filtering logic (`df.loc[:, df.cumsum(axis=0).iloc[8920] != 0]`) — dataset-specific hardcoded index; correctness depends on the specific electricity CSV structure [ASSUMED: copied from rbspaper source]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — All dependencies verified in `pyproject.toml` and runtime-checked
- Architecture: HIGH — Directly derived from rbspaper source code with explicit deviation decisions (D-01 through D-16)
- Pitfalls: HIGH — Identified from comparing rbspaper source patterns against our enum-wired utilities

**Research date:** 2026-05-13
**Valid until:** 2026-06-12 (30 days — stable ML ecosystem; Lightning 2.5.x is mature)
