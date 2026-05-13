# Phase 04: Data Modules - Context

**Gathered:** 2026-05-13
**Status:** Ready for planning

## Phase Boundary

`LightningDataModule` hierarchy for time series datasets — classification and forecasting base classes, then family-specific modules (UCR, UEA, ETT, Electricity, Weather) that accept file paths, manage data loading/splitting/scaling, and return ready-to-use dataloaders. Modules integrate with Phase 2 dataset classes and Phase 3 utility functions. Follow Lightning's recommended lifecycle: `prepare_data()` for validation, `setup()` for loading/splitting/scaling.

## Implementation Decisions

### Constructor API
- **D-01:** Classification modules accept explicit params instead of `dataset_config_path` — `target_col_name`, `data_form` as constructor kwargs. ARFF file patterns hardcoded in each concrete subclass (UCR: `{dataset_name}_train.arff`, UEA: same). No JSON configs per v1 constraint.
- **D-02:** `data_form` hardcoded as `DataForm` enum per subclass — UCR: `DataForm.REGULAR`, UEA: `DataForm.NESTED`. Not user-configurable; intrinsic dataset fact.
- **D-03:** `ScalingMethod` enum (not string) for `data_scaling_method` constructor param. Consistent with Phase 3 enum wiring.
- **D-04:** `SplittingStrategy` in enums/data.py renamed to `ClassificationSplittingStrategy` — it's classification-only, forecasting uses intrinsic time slices.
- **D-05:** `ForecastingMode` enum from enums/data.py for univariate/multivariate mode.
- **D-06:** ETT module accepts explicit `variant` param (`"ETTh1"`, `"ETTh2"`, `"ETTm1"`, `"ETTm2"`) to determine split boundaries. No filename auto-detection.
- **D-07:** File path params (`dataset_file_path`, `dataset_folder_path`) typed as `Path` only. No str acceptance.

### Base Class Organization
- **D-08:** Three separate files: `modules/classes/base.py` (BaseTimeSeriesDataModule), `modules/classes/classification.py` (BaseClassificationTimeSeriesDataModule), `modules/classes/forecasting.py` (BaseForecastingTimeSeriesDataModule). Follows Phase 3 file-per-concern pattern.

### Lightning Lifecycle
- **D-09:** Follow Lightning's recommended pattern: `prepare_data()` validates file paths and performs lightweight checks only. `setup()` handles data loading, splitting, scaling, and sets module state (`_train_data_samples`, `_seq_len`, etc.). This deviates from rbspaper source which loads data in `prepare_data()`.
- **D-10:** Classification base uses `create_data_scaler()` from utils in `setup()`. Forecasting base uses sklearn scalers directly (`_prepare_data_scaler()`) — different data shapes require different scaling approaches.

### Property Naming
- **D-11:** Full names: `sequence_length`, `num_features`, `num_classes` — matches ROADMAP.md MOD-05 verbatim. Internal attributes: `_seq_len`, `_num_features`, `_num_classes`.

### UEA ARFF Handling
- **D-12:** UEA's nested ARFF processing (`_process_stacked_data()`) stays internal to UEA module. Does not use `arff.py` utility — raw `scipy.io.arff.loadarff()` with manual byte-decoding, reshaping, and LabelEncoder. Nested multivariate ARFF doesn't fit DataFrame-based `read_arff_as_df`.

### Forecasting Dataloader
- **D-13:** Keep `TensorDataset` for forecasting module dataloaders (defer proper dataset class integration to a future phase). Note: PROJECT.md lists "Forecasting modules use proper dataset classes instead of raw TensorDataset" as an active requirement — this decision temporarily defers that.

### Dataloader Methods
- **D-14:** Keep `extra_args`, `mode`, `strict_batch_size` params on `train_dataloader()`, `val_dataloader()`, `test_dataloader()` — preserves flexibility for ad-hoc scripting beyond Lightning calls.

### Exports
- **D-15:** Wire all `modules/__init__.py` and `modules/classes/__init__.py` exports in Phase 4. Phase 5 tests need clean imports.

### Error Handling
- **D-16:** Fail fast with descriptive errors in `prepare_data()` — `FileNotFoundError` for missing files, `ValueError` for format issues. No silent degradation.

### Claude's Discretion
- Internal implementation of `_set_data_slices()`, `_transform_data()` abstract methods per concrete forecasting module follows rbspaper source logic.
- Weather and Electricity modules share 60/20/20 fractional split pattern — Climate's module-specific transform (`.T` + `expand_dims(axis=-1)` vs Weather's `expand_dims(axis=0)`) is preserved as-is.
- Validation split fallback for small datasets (stratify error handling) follows rbspaper pattern.

## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Code (primary — rbspaper)
- `_sources/rbspaper/src/rbspaper/data/modules/abstract.py` — Full base class hierarchy: `BaseTimeSeriesDataModule`, classification and forecasting bases. Note: rbspaper loads data in `prepare_data()`, which we'll restructure per D-09.
- `_sources/rbspaper/src/rbspaper/data/modules/ucr_datamodule.py` — UCR classification module with ARFF reading, splitting, validation logic
- `_sources/rbspaper/src/rbspaper/data/modules/uea_datamodule.py` — UEA module with nested ARFF, scipy loading, LabelEncoder
- `_sources/rbspaper/src/rbspaper/data/modules/ett_datamodule.py` — ETT forecasting module with intrinsic 16/4/4 month splits
- `_sources/rbspaper/src/rbspaper/data/modules/electricity_load_datamodule.py` — Electricity forecasting with CSV parsing, resampling, transpose transform
- `_sources/rbspaper/src/rbspaper/data/modules/weather_datamodule.py` — Weather forecasting with 60/20/20 fractional splits

### Source Code (secondary — autotsrc)
- `_sources/autotsrc/src/autotsrc/datasets/modules/abstract/abstract.py` — Reference for comparison; has slightly different setup flow for forecasting

### Planning Documents
- `.planning/ROADMAP.md` §Phase 4: Data Modules — deliverables, success criteria, key changes from source
- `.planning/REQUIREMENTS.md` — MOD-01 through MOD-06
- `.planning/PROJECT.md` — Package structure, v1 constraints (no JSON configs, no Pydantic, file paths from user)

### Existing Code (Phase 1-3 output)
- `src/tscollection/datasets/modules/__init__.py` — Current stub, `__all__ = []`
- `src/tscollection/datasets/enums/data.py` — `ScalingMethod`, `DataForm`, `SplittingStrategy` (rename to `ClassificationSplittingStrategy`), `ForecastingMode`, `DatasetFamily`
- `src/tscollection/datasets/utils/scaling.py` — `create_data_scaler()` — enum-wired, used by classification base
- `src/tscollection/datasets/utils/general.py` — `custom_collate_fn`, `process_data_with_varying_sequence_lengths_single`
- `src/tscollection/datasets/utils/arff.py` — `read_arff_as_df`, `process_df_according_to_dtypes` (UCR only, not UEA)
- `src/tscollection/datasets/datasets/ucr.py` — `UCRClassificationUnivariateDataset` — consumed by UCR module dataloaders
- `src/tscollection/datasets/datasets/uea.py` — `UEAClassificationMultivariateDataset` — consumed by UEA module dataloaders
- `src/tscollection/datasets/datasets/ett.py` — `ETTDataset` — NOT used by forecasting modules (D-13: TensorDataset)

## Existing Code Insights

### Reusable Assets
- Phase 2 dataset classes (`UCRClassificationUnivariateDataset`, `UEAClassificationMultivariateDataset`) — used by classification module dataloaders
- Phase 3 `create_data_scaler()` — classification base setup scaling
- Phase 3 `custom_collate_fn` — variable-length batch collation
- Phase 3 `process_data_with_varying_sequence_lengths_single` — sequence centering
- `SplittingStrategy` enum → rename to `ClassificationSplittingStrategy` (D-04)
- `DataForm` enum — hardcoded per subclass (D-02)

### Cross-dependencies
- Classification modules → `arff.py` (UCR only), dataset classes, `create_data_scaler`, `custom_collate_fn`, `process_data_with_varying_sequence_lengths_single`
- Forecasting modules → `extract_time_features`, sklearn scalers, pandas CSV loading
- All modules → `Phase 2` enums (`TimeSeriesDatasetMode`), Lightning `DataLoader`

### Integration Points
- `modules/__init__.py` — needs `__all__` exports for all module classes (D-15)
- `modules/classes/__init__.py` — needs to be created with base class exports (D-08)
- `enums/data.py` — needs `SplittingStrategy` → `ClassificationSplittingStrategy` rename (D-04)
- `enums/__init__.py` — needs to re-export renamed enum
- Root package `__init__.py` — may need module exports in public API

## Specific Ideas

- ETT split boundaries (from rbspaper source, concrete):
  - ETTh1/ETTh2: train=slice(0, 12\*30\*24), valid=slice(12\*30\*24, 16\*30\*24), test=slice(16\*30\*24, 20\*30\*24)
  - ETTm1/ETTm2: train=slice(0, 12\*30\*24\*4), valid=slice(12\*30\*24\*4, 16\*30\*24\*4), test=slice(16\*30\*24\*4, 20\*30\*24\*4)
- Weather: 60/20/20 fractional split of total samples
- Electricity: 60/20/20 fractional split + unique transform (`.T` + `expand_dims(axis=-1)`)
- UCR ARFF patterns: `{dataset_name}_TRAIN.arff`, `{dataset_name}_TEST.arff` (from rbspaper config)
- UEA ARFF patterns: `{dataset_name}_TRAIN.arff`, `{dataset_name}_TEST.arff`

## Deferred Ideas

- **Forecasting proper dataset classes** — D-13 defers using `FlexibleTimeSeriesDataset` in forecasting dataloaders. TensorDataset used for now. Future phase can integrate Phase 2 flexible datasets with forecasting strategy.
- **Nested ARFF utility** — D-12 keeps UEA `_process_stacked_data()` internal. If additional nested-ARFF datasets emerge, extracting to `arff.py` becomes worthwhile.

---

*Phase: 04-Data Modules*
*Context gathered: 2026-05-13*
