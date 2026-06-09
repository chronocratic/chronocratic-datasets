---
name: rbspaper-phase-8-migration
description: Update rbspaper codebase to use new tsdatasets enums and APIs from phase 08
metadata:
  type: project
  priority: high
  status: pending
---

# Rbspaper Migration: tsdatasets Phase 08 Enum Restructuring

## Context

tsdatasets completed phase 08 (forecasting-loader-modes), which restructures the output-mode enums, adds task-specific loader modes, and introduces new dataset classes. Rbspaper depends on tsdatasets and needs to be updated to use the new APIs.

**tsdatasets branch:** `gsd/phase-08-forecasting-loader-modes`
**tsdatasets commit:** (check latest on branch)

## What Changed in tsdatasets

### 1. TimeSeriesDatasetMode Renamed

| Old (broken)          | New (task-agnostic) |
|-----------------------|---------------------|
| `WITHOUT_LABELS`      | `SAMPLE_ONLY`       |
| `WITH_LABELS`         | `SAMPLE_LABEL`      |
| `FORECASTING`         | `INPUT_OUTPUT`      |

String values also changed:
- `'without_labels'` → `'sample_only'`
- `'with_labels'` → `'sample_label'`
- `'forecasting'` → `'input_output'`

### 2. New Task-Specific Loader Enums

**ClassificationLoaderMode** — replaces `TimeSeriesDatasetMode` for classification dataloaders:
- `SAMPLE_ONLY`, `SAMPLE_LABEL`

**ForecastingLoaderMode** — new enum for forecasting dataloaders:
- `RAW_SERIES` (default) — TensorDataset(full_data), existing behavior
- `INPUT_TARGET` — sliding-window (input, target) pairs
- `INPUT_ONLY` — sliding-window input only

### 3. Central Mode Mapping

New `maps/loader_to_dataset.py`:
- `CLASSIFICATION_LOADER_MAP` — maps ClassificationLoaderMode → TimeSeriesDatasetMode
- `FORECASTING_LOADER_MAP` — maps ForecastingLoaderMode → TimeSeriesDatasetMode | None

### 4. Forecasting Dataloaders Accept loader_mode

All forecasting datamodules now accept `loader_mode: ForecastingLoaderMode` in `train_dataloader()`, `val_dataloader()`, `test_dataloader()`:
- `ETTDataModule`, `WeatherModule`, `ElectricityLoadModule`
- Default: `ForecastingLoaderMode.RAW_SERIES` (preserves existing behavior)
- New: `ForecastingLoaderMode.INPUT_TARGET` returns sliding-window `(input, target)` pairs

### 5. New Dataset Classes

- `FlexibleTimeSeriesDatasetSingleFileMultipleSeries` — handles 3D (series, T, features) from single source
- `WeatherDataset` — uses SingleFile
- `ElectricityDataset` — uses SingleFileMultipleSeries (370 independent clients)

### 6. DataPartition Enum (rbspaper-adjacent)

New `DataPartition` enum (TRAIN/VAL/TEST) used internally by tsdatasets forecasting modules. Rbspaper's `_collect_forecasting_partition_tensors` could use it instead of string keys.

## Rbspaper Migration Tasks

### A. Enum Migration (breaking — must be done)

**Search for:**
```bash
grep -rn 'WITHOUT_LABELS\|WITH_LABELS\|TimeSeriesDatasetMode\.FORECASTING\|"without_labels"\|"with_labels"\|"forecasting"' src/
```

**Update to:**
```python
# Before
from tscollection.datasets.enums import TimeSeriesDatasetMode
TimeSeriesDatasetMode.WITH_LABELS

# After
from tscollection.datasets.enums import TimeSeriesDatasetMode
TimeSeriesDatasetMode.SAMPLE_LABEL
```

**Critical files to check:**
- `src/rbspaper/pipeline/core.py` — uses `TimeSeriesDatasetMode` in dataloader calls
- Any test files importing from `tscollection.datasets.enums`
- Any configuration that uses string values

### B. Forecasting Dataloader Upgrade (optional but recommended)

**Current:** rbspaper bypasses dataloaders for forecasting via `_collect_forecasting_partition_tensors`, reading `full_data` directly and generating sliding windows manually.

**New option:** Use the sliding-window dataloaders directly:
```python
# Before
partition_tensors = _collect_forecasting_partition_tensors(
    data_module=data_module, forecast_horizon=96
)

# After (if you want to migrate away from the workaround)
dataloader = data_module.train_dataloader(
    loader_mode=ForecastingLoaderMode.INPUT_TARGET
)
for batch in dataloader:
    inputs, targets = batch  # (input_window, target_horizon) pairs
```

**Trade-offs:**
| Approach | Pros | Cons |
|----------|------|------|
| Keep current | Works, proven | Couples rbspaper to tsdatasets internals |
| Migrate to dataloaders | Clean API, no coupling | Requires rewriting `_collect_forecasting_partition_tensors` |

**If migrating:** Remove `_collect_forecasting_partition_tensors` and the branching in `_collect_partition_tensors`. Use `_collect_via_dataloader` for all tasks.

### C. _full_data Property Cleanup (if relevant)

tsdatasets removed the legacy `_full_data` setter. Rbspaper currently uses `data_module.full_data` (the property), which still works. Verify:
```bash
grep -rn '\.full_data' src/rbspaper/
```
- `full_data` property returns `_full_data_scaled` after setup, `_full_data_raw` before — still works
- Any code doing `module._full_data = ...` (setter) is broken — must use typed attributes

### D. DataPartition Enum (optional cleanup)

rbspaper's `_collect_forecasting_partition_tensors` uses string keys `('train', 'valid', 'test')`. Could use `DataPartition` enum for type safety:
```python
partitions = [
    (DataPartition.TRAIN, data_module.train_slice),
    (DataPartition.VAL, data_module.valid_slice),
    (DataPartition.TEST, data_module.test_slice),
]
```

## Verification

After migration:
```bash
# Run rbspaper tests that depend on tsdatasets
pytest tests/ -v -k "forecasting or pipeline"

# Verify imports work
python -c "from tscollection.datasets.enums import TimeSeriesDatasetMode, ClassificationLoaderMode, ForecastingLoaderMode; print('OK')"

# Verify old names are gone
python -c "from tscollection.datasets.enums import TimeSeriesDatasetMode; TimeSeriesDatasetMode.WITHOUT_LABELS"  # Should raise AttributeError
```

## Priority Order

1. **Enum migration** — blocking, old names are removed
2. **full_data check** — verify rbspaper reads work, fix any setter usage
3. **Dataloader upgrade** — optional, decouples rbspaper from tsdatasets internals
4. **DataPartition cleanup** — optional, improves type safety

## Notes

- `ForecastingLoaderMode.RAW_SERIES` is the default, so existing rbspaper code that relies on `full_data` continues working without changes.
- The `maps/loader_to_dataset.py` module is for tsdatasets internal use. Rbspaper should use `ForecastingLoaderMode` directly with dataloaders.
- tsdatasets `forecast_horizon` and `step` are constructor params on forecasting modules, not dataloader params. Rbspaper needs to set these when creating the data module.
