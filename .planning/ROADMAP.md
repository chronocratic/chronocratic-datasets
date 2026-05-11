# Roadmap: tsdatasets

**Defined:** 2026-05-08
**Phases:** 7
**Dependencies:** PyTorch, Lightning, Pydantic v2

## Phase Overview

```
Phase 1: Package Foundation    [==== PKG-01..03 ====]
Phase 2: Dataset Classes       [==== DST-01..05 ====]  depends on Phase 1
Phase 3: Pydantic Registry     [==== CFG-01..03 ====]  depends on Phase 1
Phase 4: Download & Caching    [==== DL-01..04 ====]   depends on Phase 3
Phase 5: Data Modules          [==== MOD-01..06 ====]  depends on Phases 2, 4
Phase 6: Factory API           [==== FCT-01..05 ====]  depends on Phases 3, 5
Phase 7: Tests                 [==== TST-01..04 ====]  depends on Phases 1-6
```

## Phase 1: Package Foundation

**Goal:** Installable package with proper `__init__.py` exports at all levels.

**Requirements:** PKG-01, PKG-02, PKG-03

**Deliverables:**
- `src/tscollection/datasets/__init__.py` — public API surface
- `src/tscollection/datasets/datasets/__init__.py`
- `src/tscollection/datasets/datasets/classes/__init__.py`
- `src/tscollection/datasets/modules/__init__.py`
- `src/tscollection/datasets/modules/classes/__init__.py`
- `src/tscollection/datasets/download/__init__.py`
- `src/tscollection/datasets/config/__init__.py`
- `src/tscollection/datasets/enums/__init__.py`
- `src/tscollection/datasets/utils/__init__.py`
- `pyproject.toml` — name, version, dependencies (torch, lightning, pydantic, numpy, pandas, scipy)

**Success Criteria:**
- `pip install -e .` resolves all dependencies
- `import tscollection.datasets` works and exposes the public API

## Phase 2: Dataset Classes

**Goal:** PyTorch `Dataset` hierarchy — fixed (classification) and flexible (forecasting with sliding windows) — decoupled via strategy pattern.

**Requirements:** DST-01, DST-02, DST-03, DST-04, DST-05

**Plans:** 4 plans

**Plan list:**
- [x] 02-00-PLAN.md — Utility modules (transformations, common) and test infrastructure
- [x] 02-01-PLAN.md — Strategy pattern (SequenceHandlingStrategy + 3 concrete)
- [x] 02-02-PLAN.md — Fixed and flexible dataset ABCs with seq_len property
- [x] 02-03-PLAN.md — Concrete wrappers (UCR, UEA, ETT) and export wiring

**Deliverables:**
- `src/tscollection/datasets/datasets/classes/fixed.py`
  - `FixedTimeSeriesDataset` (abstract base)
  - `FixedTimeSeriesDatasetUnivariate`
  - `FixedTimeSeriesDatasetMultivariate`
- `src/tscollection/datasets/datasets/classes/flexible.py`
  - `FlexibleTimeSeriesDataset` (abstract base)
  - `FlexibleTimeSeriesDatasetSingleFile`
  - `FlexibleTimeSeriesDatasetMultipleFiles`
- `src/tscollection/datasets/datasets/classes/strategies.py`
  - `SequenceHandlingStrategy` (abstract)
  - `ForecastingStrategySingleFile`
  - `ClassificationStrategySingleFile`
  - `ClassificationStrategyMultipleFiles`
- `src/tscollection/datasets/datasets/ucr.py` — `UCRClassificationUnivariateDataset`
- `src/tscollection/datasets/datasets/uea.py` — `UEAClassificationMultivariateDataset`
- `src/tscollection/datasets/datasets/ett.py` — `ETTDataset`

**Source:** Primarily `_sources/rbspaper/src/rbspaper/data/datasets/` (better docstrings, defensive code). Reference `_sources/autotsrc/` for generic-type patterns.

**Success Criteria:**
- Classification dataset yields `(data, label)` pairs with correct shapes
- Forecasting dataset yields sliding-window sequences with configurable `seq_len` and `step`
- Fixed datasets expose `seq_len` as read-only property (computed from data)
- Flexible datasets accept user-configurable `seq_len` and `step`

## Phase 3: Pydantic Registry

**Goal:** Typed configuration — one class per family, frozen instances per dataset, with enums for all parameters.

**Requirements:** CFG-01, CFG-02, CFG-03

**Plans:** 4 plans

**Plan list:**
- [ ] 03-01-PLAN.md — DatasetFamily/SplitMode enums, config hierarchy (DatasetConfig, ClassificationConfig, ForecastingConfig), test fixtures
- [ ] 03-02-PLAN.md — Classification configs: UCRConfig (3 instances) + UEAConfig (2 instances)
- [ ] 03-03-PLAN.md — Forecasting configs: ETTConfig (4 instances) + ElectricityConfig + WeatherConfig
- [ ] 03-04-PLAN.md — Factory registry (CONFIGS dict, get_config, list_configs) + export chain

**Deliverables:**
- `src/tscollection/datasets/enums/data.py`
  - `TimeSeriesDatasetMode` (without_labels / with_labels / forecasting)
  - `ScalingMethod` (none / minmax / standard)
  - `SplittingStrategy` (as_defined / manual)
  - `ForecastingMode` (univariate / multivariate)
  - `DatasetFamily` (ucr / uea / ett / electricity / weather / exchange / traffic / illness)
  - `SplitMode` (indexed / fractional)
- `src/tscollection/datasets/config/base.py` — abstract `DatasetConfig` base class with common fields (name, family, url, num_classes, data_form, etc.)
- `src/tscollection/datasets/config/ucr.py` — `UCRConfig` class + instances (`UCR_COFFEE`, `UCR_ECG200`, `UCR_FACE_FOUR`)
- `src/tscollection/datasets/config/uea.py` — `UEAConfig` class + instances (`UEA_BASIC_MOTIONS`, `UEA_ATRIAL_FIBRILLATION`)
- `src/tscollection/datasets/config/ett.py` — `ETTConfig` class + `ETT_H1`, `ETT_H2`, `ETT_M1`, `ETT_M2`
- `src/tscollection/datasets/config/electricity.py` — `ElectricityConfig` + `ELECTRICITY_LOAD`
- `src/tscollection/datasets/config/weather.py` — `WeatherConfig` + `WEATHER`
- `src/tscollection/datasets/config/factory.py` — `get_config(name)`, `list_configs(family)`

**Source:** `_sources/rbspaper/src/rbspaper/data/registry.py` for metadata structure. Split from JSON configs into Pydantic.

**Success Criteria:**
- Config instances are frozen (immutable)
- All params typed with enums, no raw strings
- `HttpUrl` validation on download URLs
- `Field(ge=1)` constraints on numeric fields like `num_classes`

## Phase 4: Download and Caching

**Goal:** Auto-download to `~/.cache/tscollection/` with SHA256 validation — no re-download on subsequent runs.

**Requirements:** DL-01, DL-02, DL-03, DL-04

**Deliverables:**
- `src/tscollection/datasets/download/cache.py`
  - `get_cache_dir()` — returns `~/.cache/tscollection/`
  - `download_file(url, sha256, cache_dir, overwrite_cache)`
  - `file_exists_in_cache(dataset_name, sha256)`
  - `extract_archive(archive_path, extract_to)`
- `src/tscollection/datasets/download/ucr_uea.py` — download from `timeseriesclassification.com/Downloads/`
- `src/tscollection/datasets/download/forecasting.py` — download ETT, electricity, weather CSVs

**Source:** Pattern from torchtime (`/tmp/torchtime-extract/src/torchtime/utils.py`) — `_download_archive`, `_cache_data`, `_validate_cache`.

**Success Criteria:**
- First run downloads + validates SHA256
- Subsequent runs use cache without re-downloading
- `overwrite_cache=True` forces fresh download

## Phase 5: Data Modules

**Goal:** `LightningDataModule` hierarchy — classification and forecasting bases, then family-specific modules that auto-download in `prepare_data()`.

**Requirements:** MOD-01, MOD-02, MOD-03, MOD-04, MOD-05, MOD-06

**Deliverables:**
- `src/tscollection/datasets/modules/classes/classification.py`
  - `BaseClassificationTimeSeriesDataModule` (abstract)
  - Handles ARFF parsing, label extraction, variable-length padding, splitting
- `src/tscollection/datasets/modules/classes/forecasting.py`
  - `BaseForecastingTimeSeriesDataModule` (abstract)
  - Handles CSV loading, scaling, time-feature extraction, train/val/test slices
- `src/tscollection/datasets/modules/ucr.py`
  - `UCRClassificationDataModule` — generic UCR loader, uses config to download + load
- `src/tscollection/datasets/modules/uea.py`
  - `UEAClassificationDataModule`
- `src/tscollection/datasets/modules/ett.py`
  - `ETTDataModule` — intrinsic 16/4/4 month splits
- `src/tscollection/datasets/modules/electricity.py`
  - `ElectricityLoadModule` — 60/20/20 split
- `src/tscollection/datasets/modules/weather.py`
  - `WeatherModule` — 60/20/20 split

**Source:** `_sources/rbspaper/src/rbspaper/data/modules/` — primary base. Better defensive code, `prepare_data()` pattern, split logic.

**Key Changes from Source:**
- Remove `dataset_folder_path` and `dataset_config_path` — replaced with auto-download
- Config comes from Pydantic registry, not JSON
- Modules expose `sequence_length`, `num_classes`, `num_features` as read-only properties
- Classification: `seq_len` computed from data in `prepare_data()`
- Forecasting: `seq_len` user-configurable with registry default

**Success Criteria:**
- Module can be instantiated with just `batch_size` and passed to `Trainer.fit()`
- `prepare_data()` downloads and caches data
- `train_dataloader()` returns a `DataLoader` instance
- Module itself is a `LightningDataModule`

## Phase 6: Factory API

**Goal:** Convenience functions — `get_module()`, `get_dataset()`, `list_modules()` — and dynamic family-prefixed imports.

**Requirements:** FCT-01, FCT-02, FCT-03, FCT-04, FCT-05

**Deliverables:**
- `src/tscollection/datasets/factory.py`
  - `get_module(name, /, family=None, **kwargs)` — returns configured `LightningDataModule`
  - `get_dataset(name, /, family=None, **kwargs)` — returns configured `Dataset`
  - `list_modules(family=None)` — returns available module names
  - Resolution: bare name ("Coffee"), qualified name ("UCRCoffee"), or name + family
- Dynamic imports in `src/tscollection/datasets/modules/__init__.py`
  - `from tscollection.datasets.modules import UCRCoffeeModule` — auto-generated per registry entry

**Success Criteria:**
- `get_module("Coffee")` returns a ready-to-use `LightningDataModule`
- `get_module("Coffee", scaling="minmax", valid_size=0.2)` passes kwargs through
- `list_modules(family="ucr")` returns all UCR dataset names
- Family-prefixed imports work: `from tscollection.datasets.modules import UCRCoffeeModule`

## Phase 7: Tests

**Goal:** Verify dataset shapes, module properties, factory resolution, and download caching.

**Requirements:** TST-01, TST-02, TST-03, TST-04

**Deliverables:**
- `tests/test_datasets.py` — dataset classes yield correct shapes/types
- `tests/test_modules.py` — module properties return expected values after `prepare_data()`
- `tests/test_factory.py` — factory resolves names correctly (bare, qualified, with family)
- `tests/test_download.py` — cache download + SHA256 validation
- `tests/conftest.py` — shared fixtures

**Success Criteria:**
- All tests pass with `pytest`
- Download test verifies SHA256 validation rejects bad checksums

## Utility Modules (cross-phase)

These are small, self-contained extracts from rbspaper and can be landed early:

| Module | Source | Phase |
|--------|--------|-------|
| `utils/arff.py` | `rbspaper/data/utils/arff.py` | Phase 5 |
| `utils/scaling.py` | `rbspaper/data/utils/scaling.py` | Phase 5 |
| `utils/features.py` | `rbspaper/data/utils/features.py` | Phase 5 |
| `utils/general.py` | `rbspaper/data/utils/general.py` | Phase 5 |

## Dependency Graph

```
Phase 1 ──┐
          ├──> Phase 2 ──┐
          ├──> Phase 3 ──┤
                         ├──> Phase 5 ──┐
          ├──> Phase 4 ──┘              ├──> Phase 6 ──> Phase 7
                                        │
                                        └──> Phase 7 (also tests Phases 1-6)
```

Phases 2 and 3 can run in parallel after Phase 1.
Phase 4 depends on Phase 3 (uses config for URLs/checksums).
Phase 5 depends on Phases 2, 4 (uses datasets + download).
Phase 6 depends on Phase 5 (factory creates modules).
Phase 7 depends on everything.

---
*Roadmap defined: 2026-05-08*
