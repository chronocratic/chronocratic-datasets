---
phase: 03-pydantic-registry
plan: 03
subsystem: config
tags: [pydantic, frozen-models, forecasting, ett, electricity, weather, strenum, httpurl]

requires:
  - phase: 03-pydantic-registry
    provides: ForecastingConfig base class from plan 03-01 (base.py) with split_mode, split_bounds, default_seq_len, default_horizon
provides:
  - ETTConfig with 4 frozen instances (ETTh1, ETTh2, ETTm1, ETTm2) using indexed split boundaries
  - ElectricityConfig with 1 frozen instance (ELECTRICITY_LOAD) using fractional splits
  - WeatherConfig with 1 frozen instance (WEATHER) using fractional splits
affects: [04-download, 05-modules, 06-factory]

tech-stack:
  added: []
  patterns:
    - "Family-specific config class with class-level defaults for split_mode and family"
    - "Computed split constants (_HOURLY_SPLIT_BOUNDS, _15MIN_SPLIT_BOUNDS) avoid magic numbers"

key-files:
  created:
    - src/tscollection/datasets/config/ett.py
    - src/tscollection/datasets/config/electricity.py
    - src/tscollection/datasets/config/weather.py
    - tests/test_config_ett.py
    - tests/test_config_electricity.py
    - tests/test_config_weather.py
  modified: []

key-decisions:
  - "split_mode as class-level default (not property) to satisfy ForecastingConfig's required field"
  - "ETTh1/ETTh2 default_horizon=24 (1 day hourly), ETTm1/ETTm2 default_horizon=96 (24h at 15-min intervals)"
  - "ElectricityConfig uses explicit csv_sep and csv_decimal fields (not generic csv_kwargs dict)"

requirements-completed: [CFG-02]

duration: 3min
completed: 2026-05-11
---

# Phase 3 Plan 03: Forecasting Family Configs Summary

**ETT, Electricity, and Weather forecasting configs with frozen Pydantic instances — indexed splits for ETT (rbspaper-verified boundaries), fractional 60/20/20 splits for Electricity/Weather, family-specific CSV parsing parameters.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-05-11T12:08:23Z
- **Completed:** 2026-05-11T12:11:19Z
- **Tasks:** 2 (TDD: RED/GREEN for both)
- **Files modified:** 6

## Accomplishments

- ETTConfig with 4 frozen instances (ETTh1/ETTh2 hourly, ETTm1/ETTm2 15-min) — indexed split boundaries verified against rbspaper source
- ElectricityConfig with 1 frozen instance — fractional 60/20/20 splits, CSV params (sep=';', decimal=',')
- WeatherConfig with 1 frozen instance — fractional 60/20/20 splits, univariate column='last'
- 45 new tests (21 ETT + 10 Electricity + 14 Weather) — all passing, full regression suite green (112 tests)

## Task Commits

Each task was committed atomically:

1. **Task 1: ETT forecasting config with 4 instances** (TDD)
   - `7f5e741` — test: add failing tests for ETT forecasting config
   - `ad0c1c9` — feat: implement ETT forecasting config with 4 instances

2. **Task 2: Electricity and Weather forecasting configs** (TDD)
   - `2cf8bed` — test: add failing tests for Electricity and Weather configs
   - `9617f90` — feat: implement Electricity and Weather forecasting configs

## Files Created/Modified

- `src/tscollection/datasets/config/ett.py` — ETTConfig class + 4 frozen instances (ETTh1, ETTh2, ETTm1, ETTm2) with indexed split boundaries from rbspaper
- `src/tscollection/datasets/config/electricity.py` — ElectricityConfig class + ELECTRICITY_LOAD frozen instance with fractional splits, CSV params
- `src/tscollection/datasets/config/weather.py` — WeatherConfig class + WEATHER frozen instance with fractional splits, univariate column
- `tests/test_config_ett.py` — 21 tests for ETT config structure, split bounds, frequency, frozen behavior, model_copy
- `tests/test_config_electricity.py` — 10 tests for Electricity config structure, CSV params, frozen behavior
- `tests/test_config_weather.py` — 14 tests for Weather config structure, univariate column, frozen behavior

## Decisions Made

- **split_mode as class-level default:** Since `ForecastingConfig` declares `split_mode` as a required field, using a class-level default (`SplitMode.INDEXED` for ETT, `SplitMode.FRACTIONAL` for Electricity/Weather) satisfies the parent's field requirement while providing the correct default. This is simpler than overriding with a property.
- **Explicit CSV fields on ElectricityConfig:** Used `csv_sep: str = ';'` and `csv_decimal: str = ','` as explicit fields (RESEARCH.md Open Question 6 recommendation) rather than a generic `csv_kwargs` dict. Cleaner types, no mutation risk.
- **Computed split constants:** ETT split bounds use `_HOURLY_SPLIT_BOUNDS = (12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24)` and `_15MIN_SPLIT_BOUNDS = (12 * 30 * 24 * 4, 16 * 30 * 24 * 4, 20 * 30 * 24 * 4)` to match rbspaper's exact arithmetic.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Worktree initially at `ecd3be8` (before wave 1's 03-01 commits). Merged `worktree-agent-ab3f959825022c155` to bring in config/base.py, enums, and test infrastructure needed as dependencies.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All 6 forecasting config instances (ETT x4, Electricity x1, Weather x1) are frozen, validated, and importable.
- Plan 03-02 (classification configs: UCR + UEA) must complete for phase 03 to be ready for factory integration (Plan 03-04).
- Phase 4 (download) can use `url`, `sha256`, and `cache_key` from any of these configs.
- Phase 5 (modules) can use `split_bounds`, `forecast_column`, `csv_sep`, `csv_decimal`, `univariate_column`, `frequency`, `num_features`.

## Known Stubs

None — all implementations are complete for this plan's scope.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: T-03-07 | src/tscollection/datasets/config/ett.py | HttpUrl on all ETT/Electricity/Weather instances validates dataset CSV URLs at model construction |
| threat_flag: T-03-08 | src/tscollection/datasets/config/ett.py, electricity.py, weather.py | frozen=True on all instances prevents runtime tampering of split_bounds |

---
*Phase: 03-pydantic-registry*
*Completed: 2026-05-11*
