---
phase: 08-forecasting-loader-modes
plan: 01
subsystem: enums, data-models
tags: [enum, refactoring, mode-mapping, tdd]

# Dependency graph
requires:
  - phase: 07-ddp-compliance
    provides: dataset and module infrastructure
provides:
  - Renamed TimeSeriesDatasetMode values (SAMPLE_ONLY, SAMPLE_LABEL, INPUT_OUTPUT)
  - New ClassificationLoaderMode and ForecastingLoaderMode enums
  - Mode mapping dictionaries (CLASSIFICATION_LOADER_MAP, FORECASTING_LOADER_MAP)
  - Updated base dataset sample map
affects: [08-02, 08-03, 08-04, classification, forecasting]

# Tech tracking
tech-stack:
  added: []
  patterns: [loader-to-dataset mode mapping, StrEnum for all modes]

key-files:
  created:
    - src/tscollection/datasets/maps/__init__.py
    - src/tscollection/datasets/maps/loader_to_dataset.py
    - tests/test_enum_refactoring.py
  modified:
    - src/tscollection/datasets/enums/data.py
    - src/tscollection/datasets/enums/__init__.py
    - src/tscollection/datasets/datatypes/_base/base.py
    - src/tscollection/datasets/datatypes/ett.py
    - src/tscollection/datasets/modules/ett.py
    - src/tscollection/datasets/modules/weather.py
    - src/tscollection/datasets/modules/electricity.py
    - src/tscollection/datasets/modules/ucr.py
    - src/tscollection/datasets/modules/uea.py
    - tests/test_fixed_dataset.py
    - tests/test_flexible_dataset.py
    - tests/test_ucr_dataset.py
    - tests/test_uea_dataset.py
    - tests/test_modules_ucr.py
    - tests/test_modules_uea.py
    - tests/test_package.py

key-decisions:
  - "RAW_SERIES maps to None in FORECASTING_LOADER_MAP (no dataset mode needed)"
  - "ForecastingMode.UNIVARIATE/MULTIVARIATE left unchanged (D-03)"
  - "Enum renaming is a single atomic migration — no backward compat aliases"

requirements-completed: [D-01, D-02, D-04, D-14]

# Metrics
duration: ~10min
completed: 2026-06-09
---

# Phase 08 Plan 01: Enum Refactoring and Mode Mapping Foundation Summary

**Rename TimeSeriesDatasetMode enum values to task-agnostic semantics, add loader-mode enums, and create mode mapping module using TDD RED-GREEN-REFACTOR cycle.**

## Performance

- **Duration:** ~10 min
- **Tasks:** 6 (RED tests + GREEN implementation)
- **Files modified:** 19 (10 source, 8 test, 2 new modules)
- **Tests added:** 21

## Accomplishments

- TDD RED phase: 21 tests covering all enum refactoring requirements
- TimeSeriesDatasetMode values renamed (WITHOUT_LABELS→SAMPLE_ONLY, WITH_LABELS→SAMPLE_LABEL, FORECASTING→INPUT_OUTPUT)
- New ClassificationLoaderMode enum (SAMPLE_ONLY, SAMPLE_LABEL)
- New ForecastingLoaderMode enum (RAW_SERIES, INPUT_TARGET, INPUT_ONLY)
- Mode mapping module with CLASSIFICATION_LOADER_MAP and FORECASTING_LOADER_MAP
- Complete source and test migration across 19 files
- ForecastingMode left unchanged (D-03)

## Task Commits

1. **RED: Failing tests for enum refactoring** - `8641408` (test)
2. **GREEN: Implementation + migration** - `4d7fcdc` (feat)

**Plan metadata:** inline (docs committed with GREEN)

## Files Created/Modified

- `src/tscollection/datasets/maps/loader_to_dataset.py` - Mode mapping dictionaries
- `src/tscollection/datasets/maps/__init__.py` - Maps package exports
- `src/tscollection/datasets/enums/data.py` - Renamed TimeSeriesDatasetMode, added loader enums
- `src/tscollection/datasets/enums/__init__.py` - Updated exports for new enums
- `src/tscollection/datasets/datatypes/_base/base.py` - Updated sample fun map keys
- `src/tscollection/datasets/datatypes/ett.py` - FORECASTING→INPUT_OUTPUT
- `src/tscollection/datasets/modules/ett.py` - FORECASTING→INPUT_OUTPUT (3 sites)
- `src/tscollection/datasets/modules/weather.py` - FORECASTING→INPUT_OUTPUT (3 sites)
- `src/tscollection/datasets/modules/electricity.py` - FORECASTING→INPUT_OUTPUT (3 sites)
- `src/tscollection/datasets/modules/ucr.py` - WITHOUT_LABELS→SAMPLE_ONLY (3 sites)
- `src/tscollection/datasets/modules/uea.py` - WITHOUT_LABELS→SAMPLE_ONLY (3 sites)

## Deviations from Plan

Plan 08-01 Task 6 specified migrating source file imports. During implementation, test files also referenced old enum values and were migrated proactively (7 test files) to prevent cascading failures in subsequent plans. This is within Rule 1 (auto-fix) scope — tests would crash without migration.

## Verification

- `ruff check` passes on all modified files
- `pytest tests/` 299 passed (4 pre-existing failures unrelated)
- `grep -r 'WITHOUT_LABELS\|WITH_LABELS\|FORECASTING' src/` returns 0 hits

## Self-Check: PASSED
