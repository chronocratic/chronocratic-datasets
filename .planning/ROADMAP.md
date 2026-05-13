# Roadmap: tsdatasets

**Defined:** 2026-05-08
**Revised:** 2026-05-13 — v1 scope simplified: minimal working classes, no downloading, no pydantic
**Phases:** 5 (v1), 2+ (v2)
**Dependencies:** PyTorch, Lightning, numpy, pandas, scipy, scikit-learn

## v1 Scope

v1 delivers the foundational classes and utilities from `_sources/` with improved style
(file separation, type hints, docstrings). No auto-downloading, no Pydantic config registry,
no factory API — users pass file paths and instantiate classes directly.

## Phase Overview (v1)

```
Phase 1: Package Foundation     [==== PKG-01..03 ====]  DONE
Phase 2: Dataset Classes        [==== DST-01..05 ====]  DONE  depends on Phase 1
Phase 3: Utility Modules        [==== UTI-01..05 ====]         depends on Phase 1
Phase 4: Data Modules           [==== MOD-01..06 ====]         depends on Phases 2, 3
Phase 5: Tests                  [==== TST-01..03 ====]         depends on Phases 1-4
```

## v2 Scope (Deferred)

Features moved to v2 — archived on `archive/v2-full-implementation` branch:

```
Phase 3: Pydantic Registry      [==== CFG-01..03 ====]  v2  — Typed config classes per family
Phase 4: Download & Caching     [==== DL-01..04 ====]   v2  — Auto-download to ~/.cache/
Phase 6: Factory API            [==== FCT-01..05 ====]  v2  — get_module(), get_dataset()
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
- `src/tscollection/datasets/enums/__init__.py`
- `src/tscollection/datasets/utils/__init__.py`
- `pyproject.toml` — name, version, dependencies (torch, lightning, numpy, pandas, scipy)

**Success Criteria:**
- `pip install -e .` resolves all dependencies
- `import tscollection.datasets` works and exposes the public API

**Status:** DONE

## Phase 2: Dataset Classes

**Goal:** PyTorch `Dataset` hierarchy — fixed (classification) and flexible (forecasting with sliding windows) — decoupled via strategy pattern.

**Requirements:** DST-01, DST-02, DST-03, DST-04, DST-05

**Plans:** 4 plans (executed)

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

**Status:** DONE

## Phase 3: Utility Modules

**Goal:** Port utility modules from `_sources/` with improved file separation and style.

**Requirements:** UTI-01, UTI-02, UTI-03, UTI-04, UTI-05

**Plans:** 3 plans

**Plan list:**
- [x] 03-01-PLAN.md — DataForm enum, flatten_list_of_np_arrays, arff.py, features.py
- [x] 03-02-PLAN.md — scaling.py (enum-wired), general.py
- [x] 03-03-PLAN.md — __init__.py exports, test suite

**Deliverables:**
- `src/tscollection/datasets/utils/arff.py` — ARFF reading and dtype processing
- `src/tscollection/datasets/utils/scaling.py` — `create_data_scaler`, MinMax/Standard scaling
- `src/tscollection/datasets/utils/features.py` — `extract_time_features`
- `src/tscollection/datasets/utils/general.py` — `custom_collate_fn`, `centralize_variable_length_series`, `process_data_with_varying_sequence_lengths_single`
- `src/tscollection/datasets/utils/__init__.py` — public exports

**Source:**
- `_sources/rbspaper/src/rbspaper/data/utils/arff.py`
- `_sources/rbspaper/src/rbspaper/data/utils/scaling.py`
- `_sources/rbspaper/src/rbspaper/data/utils/features.py`
- `_sources/rbspaper/src/rbspaper/data/utils/general.py`
- `_sources/rbspaper/src/rbspaper/data/utils/common.py` (already ported as `utils/common.py` in Phase 2)

**Key Changes from Source:**
- Separate files per concern (was bundled in rbspaper `utils/__init__.py`)
- Proper `__all__` exports
- Functional style with type hints
- Enum-wired `create_data_scaler` (ScalingMethod, DataForm instead of strings)

**Success Criteria:**
- Each utility module is importable and self-contained
- `create_data_scaler()` returns callable that scales train/valid/test splits
- ARFF reader handles nominal/numeric dtypes correctly

## Phase 4: Data Modules

**Goal:** `LightningDataModule` hierarchy — classification and forecasting bases, then family-specific modules that accept file paths.

**Requirements:** MOD-01, MOD-02, MOD-03, MOD-04, MOD-05, MOD-06

**Plans:** 4 plans

**Plan list:**
- [x] 04-01-PLAN.md — Enum rename, utility port, BaseTimeSeriesDataModule
- [x] 04-02-PLAN.md — Classification/Forecasting base classes, UCRClassificationDataModule
- [x] 04-03-PLAN.md — UEA, ETT, Electricity, Weather concrete modules
- [x] 04-04-PLAN.md — Export wiring and import verification

**Deliverables:**
- `src/tscollection/datasets/modules/classes/base.py`
  - `BaseTimeSeriesDataModule` (shared base — dataloader construction, scaling setup)
- `src/tscollection/datasets/modules/classes/classification.py`
  - `BaseClassificationTimeSeriesDataModule` (abstract)
  - Handles ARFF parsing, label extraction, variable-length padding, splitting
- `src/tscollection/datasets/modules/classes/forecasting.py`
  - `BaseForecastingTimeSeriesDataModule` (abstract)
  - Handles CSV loading, scaling, time-feature extraction, train/val/test slices
- `src/tscollection/datasets/modules/ucr.py`
  - `UCRClassificationDataModule` — generic UCR loader, accepts file path
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
- Remove `dataset_config_path` — replaced with explicit parameters
- Config comes from user parameters, not JSON/Pydantic
- Modules expose `sequence_length`, `num_classes`, `num_features` as read-only properties
- Classification: `seq_len` computed from data in `prepare_data()`
- Forecasting: `seq_len` user-configurable with sensible default

**Success Criteria:**
- Module can be instantiated with file path + batch_size and passed to `Trainer.fit()`
- `prepare_data()` loads data from provided file paths
- `train_dataloader()` returns a `DataLoader` instance
- Module itself is a `LightningDataModule`

## Phase 5: Tests

**Goal:** Verify dataset shapes, module properties, and utility functions.

**Requirements:** TST-01, TST-02, TST-03

**Plans:** 3 plans

**Plan list:**
- [ ] 05-01-PLAN.md — ETT golden-path integration and setup() edge-case tests
- [ ] 05-02-PLAN.md — Weather/Electricity dataloader smoke tests (fractional-split path)
- [ ] 05-03-PLAN.md — transformations.py error-path unit tests

**Deliverables:**
- `tests/test_modules_forecasting.py` — ETT/Weather/Electricity integration + edge-case tests
- `tests/test_transformations.py` — error-path unit tests
- Coverage: 92% total, no module below 85%

**Success Criteria:**
- All tests pass with `pytest`
- Coverage includes dataset loading, module setup, and utility functions

## Dependency Graph

```
Phase 1 ──┐
          ├──> Phase 2 ──┐
          ├──> Phase 3 ──┤
                         ├──> Phase 4 ──> Phase 5
```

Phases 2 and 3 can run in parallel after Phase 1.
Phase 4 depends on Phases 2, 3 (uses datasets + utilities).
Phase 5 depends on everything.

---
*Roadmap defined: 2026-05-08*
*Revised: 2026-05-13 — v1 scope simplified*
