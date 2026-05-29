# Roadmap: tsdatasets

**Defined:** 2026-05-08
**Revised:** 2026-05-13 — v1 scope simplified: minimal working classes, no downloading, no pydantic
**Revised:** 2026-05-18 — Phase 6 added: Lightning lifecycle fix (setup idempotency)
**Revised:** 2026-05-27 — Phase 6 plans created (8 plans, 6 waves)
**Revised:** 2026-05-29 — Phases 7-8 added to v1: DDP compliance + forecasting mode wiring
**Revised:** 2026-05-29 — Phase 7 plans created (8 plans, 5 waves)
**Phases:** 8 (v1), 2+ (v2)
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
Phase 6: Lightning Lifecycle    [==== LIF-01..02 ====]        depends on Phase 4 — Fix setup idempotency
Phase 7: DDP Compliance         [==== DDP-01..04 ====]        depends on Phase 6 — DDP-safe prepare_data/setup
Phase 8: Forecasting Mode Wiring [==== FOR-01..03 ====]      depends on Phase 7 — split_mode + dataloader mode
```

## Phase 6: Data Module Lightning Lifecycle Fix

**Goal:** Add stage guards and proper `prepare_data()` / `setup()` separation per Lightning conventions, stopping double-normalization.

**Problem:** `BaseTimeSeriesDataModule.setup()` and `BaseForecastingTimeSeriesDataModule.setup()` apply normalization without `stage` guards. Lightning calls `setup()` once per trainer stage (`"fit"`, `"validate"`, `"test"`); each call re-normalizes already-normalized data.

**Fix:**
- Move one-time data loading, scaling, and splitting to `prepare_data()` (idempotent by Lightning calling it once per trainer)
- Guard `setup()` with `stage` checks so scaling runs only on `"fit"`
- Add `_setup_completed_stages` idempotency sentinel as safety net
- Verify with tests that calling `setup()` multiple times does not alter data values

**Success Criteria:**
- Calling `setup("fit")` then `setup("test")` produces identical data (no double normalization)
- `prepare_data()` handles all one-time I/O and heavy computation
- `setup()` only constructs dataset instances and dataloaders
- All existing tests pass

**Plans:** 6/8 plans executed

**Plan list:**
- [x] 06-01-PLAN.md — TIME_FEATURE_COUNT constant and export (A2, per D4)
- [x] 06-02-PLAN.md — Setup idempotency sentinel (B1, per D1)
- [x] 06-03-PLAN.md — Idempotent prepare_data wrapper + rename chain (B3+B2, per D3)
- [x] 06-04-PLAN.md — Stage gating + fitted scaler caches (B4, per D1, D2)
- [x] 06-05-PLAN.md — prepare_dimensions interim API (A1, per D4)
- [x] 06-06-PLAN.md — custom_collate_fn padding regression test (C2)
- [ ] 06-07-PLAN.md — Integration tests for all concrete modules
- [ ] 06-08-PLAN.md — Full regression suite + lint verification

**Status:** PLANNED

## Phase 7: DDP Compliance + `_full_data` Split

**Goal:** Make the package safe under Lightning Distributed Data Parallel (multi-GPU) and fix `_full_data` type drift.

**Problem:** `prepare_data()` runs rank-0 only; state on `self` is invisible to ranks 1+. Additionally `_full_data` drifts from `pd.DataFrame` -> `np.ndarray` during pipeline, causing isinstance branches and silent double-scaling.

**Fix:**
1. Move state assignment from `prepare_data()` -> `setup(stage)`. `prepare_data()` becomes I/O-only: writes cache files + `metadata.json` to disk.
2. Split `_full_data` into typed attrs: `_full_data_raw` (ndarray, immutable), `_time_index` (DatetimeIndex), `_full_data_scaled` (ndarray, rebuilt each setup).

**Success Criteria:**
- DDP smoke test (gloo, 2 ranks): identical state across ranks after `setup('fit')`
- No `isinstance` branches on `_full_data` consumers
- `prepare_dimensions()` reads metadata without loading arrays
- Second `setup()` call produces identical output (idempotent without sentinel)

**Plans:** 8/8 plans executed

**Plan list:**
- [x] 07-01-PLAN.md — Cache utility module (build_cache_key, resolve_cache_dir, atomic save/load, scaler persistence) — TDD
- [x] 07-02-PLAN.md — synthetic_cache_dir fixture for tests
- [x] 07-03-PLAN.md — Base module: cache_dir param, prepare_data_per_node, metadata-based prepare_dimensions, extended reset
- [x] 07-04-PLAN.md — Forecasting base: split _full_data into typed attrs, cache-read setup, scaler persistence
- [x] 07-05-PLAN.md — ETT module: cache-aware _do_prepare_data, typed _transform_data
- [x] 07-06-PLAN.md — Weather + Electricity modules: cache-aware _do_prepare_data, typed _transform_data
- [x] 07-07-PLAN.md — UCR + UEA classification modules: cache-aware _do_prepare_data, cache-read setup
- [x] 07-08-PLAN.md — DDP smoke test (gloo, 2 ranks), isinstance elimination verification, full regression

**Status:** PLANNED (on `phase-07` branch)

## Phase 8: Forecasting Mode Wiring

**Goal:** Wire forecasting split_mode dispatch and dataloader mode correctness.

**Problem:** `_set_data_slices` and dataloader construction need proper split_mode handling across forecasting modules.

**Status:** QUEUED

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

**Plans:** 5 plans (executed)

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

**Status:** DONE

## Phase 4: Data Modules

**Goal:** `LightningDataModule` hierarchy — classification and forecasting bases, then family-specific modules that accept file paths.

**Requirements:** MOD-01, MOD-02, MOD-03, MOD-04, MOD-05, MOD-06

**Plans:** 5 plans

**Plan list:**
- [x] 04-01-PLAN.md — Enum rename, utility port, BaseTimeSeriesDataModule
- [x] 04-02-PLAN.md — Classification/Forecasting base classes, UCRClassificationDataModule
- [x] 04-03-PLAN.md — UEA, ETT, Electricity, Weather concrete modules
- [x] 04-04-PLAN.md — Export wiring and import verification
- [x] 04-05-PLAN.md — Critical bug fixes: scaler axis (CR-01), scale_data flag (CR-02), electricity iloc (CR-03)

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

**Status:** DONE

## Phase 5: Tests

**Goal:** Verify dataset shapes, module properties, and utility functions.

**Requirements:** TST-01, TST-02, TST-03

**Plans:** 3 plans

**Plan list:**
- [x] 05-01-PLAN.md — ETT golden-path integration and setup() edge-case tests
- [x] 05-02-PLAN.md — Weather/Electricity dataloader smoke tests (fractional-split path)
- [x] 05-03-PLAN.md — transformations.py error-path unit tests

**Deliverables:**
- `tests/test_modules_forecasting.py` — ETT/Weather/Electricity integration + edge-case tests
- `tests/test_transformations.py` — error-path unit tests
- Coverage: 92% total, no module below 85%

**Success Criteria:**
- All tests pass with `pytest`
- Coverage includes dataset loading, module properties, and utility functions

**Status:** DONE

## Dependency Graph

```
Phase 1 ─┐
         ├──> Phase 2 ─┐
         ├──> Phase 3 ─┤
                        ├──> Phase 4 ─> Phase 5
                        └─────────────> Phase 6 ─> Phase 7 ─> Phase 8
```

Phases 2 and 3 can run in parallel after Phase 1.
Phase 4 depends on Phases 2, 3 (uses datasets + utilities).
Phase 5 depends on everything.
Phase 6 depends on Phase 4 (fixes module lifecycle; no new dependencies).
Phase 7 depends on Phase 6 (DDP compliance + full_data split).
Phase 8 depends on Phase 7 (forecasting mode wiring after DDP rewrite).

---
*Roadmap defined: 2026-05-08*
*Revised: 2026-05-13 — v1 scope simplified*
*Revised: 2026-05-27 — Phase 6 plans created*
*Revised: 2026-05-29 — Phases 7-8 added to v1*
*Revised: 2026-05-29 — Phase 7 plans created (8 plans, 5 waves)*
