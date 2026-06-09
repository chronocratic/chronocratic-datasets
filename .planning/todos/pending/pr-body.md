## Phase 08: Forecasting Loader Modes + Enum Restructuring

### What Changed

**Enum Refactoring:**
- Renamed `TimeSeriesDatasetMode` members: `WITHOUT_LABELS` -> `SAMPLE_ONLY`, `WITH_LABELS` -> `SAMPLE_LABEL`, `FORECASTING` -> `INPUT_OUTPUT`
- Added `ClassificationLoaderMode` - task-specific enum for classification dataloaders
- Added `ForecastingLoaderMode` - task-specific enum for forecasting dataloaders (`RAW_SERIES`, `INPUT_TARGET`, `INPUT_ONLY`)
- Added `DataPartition` - explicit TRAIN/VAL/TEST enum replacing callback pattern
- Added `maps/loader_to_dataset.py` - central mode mapping

**Forecasting Dataloaders:**
- `loader_mode` moved from constructor to per-call dataloader param
- `RAW_SERIES` (default) preserves existing TensorDataset behavior
- `INPUT_TARGET` / `INPUT_ONLY` return sliding-window datasets via `_build_sliding_dataset()`
- `forecast_horizon` and `step` added as constructor params with per-dataset defaults (ETT=96, Electricity=24, Weather=96)

**New Dataset Classes:**
- `FlexibleTimeSeriesDatasetSingleFileMultipleSeries` - handles 3D (series, T, features) from single source
- `WeatherDataset` - uses SingleFile
- `ElectricityDataset` - uses SingleFileMultipleSeries (370 independent clients)
- `ETTDataset` - now accepts `mode` param (was hardcoded)

**Classification Modules:**
- UCR/UEA dataloaders migrated to `ClassificationLoaderMode`

**Cleanup:**
- Removed legacy `_full_data` property/setter (dead code)
- Moved data shape reference tables from base class to each concrete module
- Added TS2Vec-style inline comments in Electricity._transform_data
- Removed cross-module "Different from..." references in docstrings

### Data Shape Reference

| Dataset | Raw CSV Shape | Post-Transform | Notes |
|---------|---------------|----------------|-------|
| ETT | Variant-dep. | (1, T, F) | F=7 multi, F=2 univar |
| Weather | (52696, 22) | (1, 52696, 22) | Hourly, 7 years |
| Electricity | (27340, 370) | (370, 27340, 1) | 370 independent clients |

### Verification

- 296 tests pass (4 pre-existing failures unrelated)
- ty check - clean
- ruff check - clean

### Migration Note

rbspaper needs to update `TimeSeriesDatasetMode` enum usage. See `.planning/todos/pending/rbspaper-phase-8-migration.md` for the migration guide.
