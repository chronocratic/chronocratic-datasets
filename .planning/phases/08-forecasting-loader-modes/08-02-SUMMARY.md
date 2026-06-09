---
phase: 08-forecasting-loader-modes
plan: 02
subsystem: datatypes
tags: [dataset, sliding-window, forecasting, multi-series, tdd]

# Dependency graph
requires:
  - phase: 08
    plan: 01
    provides: [renamed TimeSeriesDatasetMode, loader mode enums]
provides:
  - FlexibleTimeSeriesDatasetSingleFileMultipleSeries (3D multi-series base class)
  - WeatherDataset (concrete single-file dataset)
  - ElectricityDataset (concrete multi-series dataset)
affects: [08-03, 08-04]

# Tech tracking
tech-stack:
  added: []
  patterns: [bisect+accumulate for multi-series indexing, strategy pattern for sliding windows]

key-files:
  created:
    - src/tscollection/datasets/datatypes/weather.py
    - src/tscollection/datasets/datatypes/electricity.py
    - tests/test_new_dataset_classes.py
  modified:
    - src/tscollection/datasets/datatypes/_base/flexible.py
    - src/tscollection/datasets/datatypes/__init__.py
    - src/tscollection/datasets/datatypes/_base/__init__.py

key-decisions:
  - "Pre-compute per-series windows before super().__init__ to avoid AttributeError"
  - "WeatherDataset uses FlexibleTimeSeriesDatasetSingleFile (2D squeezed data)"
  - "ElectricityDataset uses FlexibleTimeSeriesDatasetSingleFileMultipleSeries (3D data)"

requirements-completed: [D-08, D-09, D-14]

# Metrics
duration: ~8min
completed: 2026-06-09
---

# Phase 08 Plan 02: New Dataset Classes for Forecasting Sliding-Window Mode Summary

**Add FlexibleTimeSeriesDatasetSingleFileMultipleSeries for multi-series data, WeatherDataset, and ElectricityDataset concrete classes using TDD RED-GREEN cycle.**

## Performance

- **Duration:** ~8 min
- **Tasks:** 4 (RED tests + GREEN implementation)
- **Files modified:** 5 (3 source, 2 test)
- **Tests added:** 14

## Accomplishments

- FlexibleTimeSeriesDatasetSingleFileMultipleSeries handles 3D (series, T, features) data
- WeatherDataset for single-file forecasting (22 features, hourly)
- ElectricityDataset for multi-series forecasting (370 clients)
- Correct bisect+accumulate indexing for cross-series window mapping

## Verification

- `ruff check` passes on all modified files
- `pytest tests/test_new_dataset_classes.py` 14 passed
- Full suite: 313 passed (4 pre-existing failures unrelated)

## Self-Check: PASSED
