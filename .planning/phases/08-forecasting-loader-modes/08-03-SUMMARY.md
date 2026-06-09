---
phase: 08-forecasting-loader-modes
plan: 03
subsystem: modules
tags: [forecasting, lightning-datamodule, mode-dispatch, sliding-window]

# Dependency graph
requires:
  - phase: 08
    plan: 01
    provides: [ForecastingLoaderMode enum, FORECASTING_LOADER_MAP]
  - phase: 08
    plan: 02
    provides: [WeatherDataset, ElectricityDataset]
provides:
  - Mode dispatch in base forecasting module (RAW_SERIES vs sliding-window)
  - _build_sliding_dataset abstract method
  - WeatherDataModule with sliding-window support
  - ETTDataModule and ElectricityLoadModule with sliding-window support
affects: [08-04]

# Tech tracking
tech-stack:
  added: []
  patterns: [mode-dispatch dataloader, _build_dataloader helper]

key-files:
  created: []
  modified:
    - src/tscollection/datasets/modules/_base/forecasting.py
    - src/tscollection/datasets/modules/ett.py
    - src/tscollection/datasets/modules/weather.py
    - src/tscollection/datasets/modules/electricity.py
    - tests/test_modules_classification_forecasting.py
    - tests/test_modules_forecasting.py

key-decisions:
  - "RAW_SERIES default preserves existing TensorDataset behavior"
  - "forecast_horizon defaults: ETT=96, Weather=96, Electricity=24 (D-06)"
  - "strict_batch_size=False, shuffle=False for sliding-window (D-17)"
  - "ValueError when seq_len + forecast_horizon > partition_length (D-12)"
  - "forecast_horizon and step do NOT affect cache key (D-13)"

requirements-completed: [D-05, D-06, D-12, D-13, D-15, D-16, D-17]

# Metrics
duration: ~15min
completed: 2026-06-09
---

# Phase 08 Plan 03: Base Forecasting Module Dispatch Summary

**Add loader_mode dispatch to BaseForecastingTimeSeriesDataModule with RAW_SERIES backward compatibility and sliding-window support for INPUT_TARGET/INPUT_ONLY modes.**

## Accomplishments

- Base module accepts loader_mode, forecast_horizon, step params
- Mode dispatch: RAW_SERIES -> TensorDataset (existing), INPUT_TARGET/INPUT_ONLY -> sliding dataset
- D-12 validation gate (ValueError for invalid window sizes)
- D-13 cache key documentation
- D-15 shape reference table
- D-16 forecast_horizon defaults per dataset
- D-17 strict_batch_size=False for forecasting
- WeatherDataModule alias created
- All concrete modules updated (ETT, Weather, Electricity)

## Verification

- `ruff check` passes (with pre-existing warnings)
- `pytest tests/` 313 passed (4 pre-existing failures unrelated)

## Self-Check: PASSED
