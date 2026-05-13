---
phase: 04-data-modules
plan: 01
subsystem: data-modules
tags: [lightning, pytorch, enums, data-scaling, dataloader, tdd]

# Dependency graph
requires:
  - phase: 03-utility-modules
    provides: create_data_scaler, custom_collate_fn, ScalingMethod enum, DataForm enum
provides:
  - ClassificationSplittingStrategy enum (renamed from SplittingStrategy)
  - separate_target_feature_from_df utility function
  - BaseTimeSeriesDataModule shared base class with dataloader construction
affects: [04-02, 04-03, 04-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "BaseTimeSeriesDataModule inherits pl.LightningDataModule + ABC"
    - "Keyword-only constructor args with enum types (ScalingMethod, DataForm)"
    - "persistent_workers guarded behind num_workers > 0 check"
    - "setup() handles scaling via create_data_scaler; prepare_data() is abstract"

key-files:
  created:
    - src/tscollection/datasets/modules/classes/base.py
    - tests/test_utils_common_separate_target.py
    - tests/test_modules_base.py
  modified:
    - src/tscollection/datasets/enums/data.py
    - src/tscollection/datasets/enums/__init__.py
    - src/tscollection/datasets/__init__.py
    - src/tscollection/datasets/utils/common.py
    - src/tscollection/datasets/utils/__init__.py
    - tests/test_package.py

key-decisions:
  - "Renamed SplittingStrategy to ClassificationSplittingStrategy per D-04 (classification-only enum)"
  - "Added KeyError validation to separate_target_feature_from_df per T-04-01-02 threat mitigation"
  - "BaseTimeSeriesDataModule uses ScalingMethod enum (not str) for data_scaling_method per D-03"
  - "Properties named sequence_length and num_features (full names) per D-11"

requirements-completed: [MOD-01, MOD-05, MOD-06]

# Metrics
duration: 5min
completed: 2026-05-13
---

# Phase 4 Plan 1: Data Module Foundation Summary

**ClassificationSplittingStrategy enum rename, separate_target_feature_from_df utility port, and BaseTimeSeriesDataModule with enum-typed constructor, dataloader construction with persistent_workers guard, and create_data_scaler integration**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-13T11:27:12Z
- **Completed:** 2026-05-13T11:32:32Z
- **Tasks:** 2 completed (both TDD: RED/GREEN)
- **Files modified:** 8 files created/modified, 256 lines of tests added

## Accomplishments

- Renamed `SplittingStrategy` to `ClassificationSplittingStrategy` across enums, re-exports, and root package (D-04)
- Ported `separate_target_feature_from_df` from rbspaper source with KeyError validation (T-04-01-02)
- Created `BaseTimeSeriesDataModule` with full dataloader construction, scaling setup, and property definitions

## Task Commits

Each task was committed atomically:

1. **Task 1: Rename SplittingStrategy and port separate_target_feature_from_df** (TDD)
   - `d7b90f6` (test): Add failing tests for ClassificationSplittingStrategy and separate_target_feature_from_df
   - `cbb4e02` (feat): Rename enum, port utility, update all __init__.py exports, fix test_package.py

2. **Task 2: Create BaseTimeSeriesDataModule** (TDD)
   - `fb47ba5` (test): Add failing tests for BaseTimeSeriesDataModule (15 tests)
   - `86c4186` (feat): Implement BaseTimeSeriesDataModule with constructor, properties, setup, dataloaders

## Files Created/Modified

- `src/tscollection/datasets/modules/classes/base.py` - BaseTimeSeriesDataModule with LightningDataModule inheritance, keyword-only constructor using ScalingMethod/DataForm enums, setup() with create_data_scaler, _process_train/test/valid_dataloader with persistent_workers guard
- `src/tscollection/datasets/enums/data.py` - Renamed SplittingStrategy to ClassificationSplittingStrategy (D-04)
- `src/tscollection/datasets/enums/__init__.py` - Updated re-export to ClassificationSplittingStrategy
- `src/tscollection/datasets/__init__.py` - Updated root package export
- `src/tscollection/datasets/utils/common.py` - Added separate_target_feature_from_df with KeyError validation (T-04-01-02)
- `src/tscollection/datasets/utils/__init__.py` - Added separate_target_feature_from_df export
- `tests/test_utils_common_separate_target.py` - Tests for enum rename and utility port (10 tests)
- `tests/test_modules_base.py` - Tests for BaseTimeSeriesDataModule (15 tests)
- `tests/test_package.py` - Updated to use ClassificationSplittingStrategy (deviation fix)

## Decisions Made

- Used `ScalingMethod` enum (not `str`) for `data_scaling_method` per D-03
- Used `DataForm` enum (not `str`) for `data_form` per D-02
- Full property names: `sequence_length`, `num_features` per D-11
- `prepare_data()` is abstract; subclasses implement file validation per D-09
- `setup()` calls `create_data_scaler()` for classification branch per D-10

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated test_package.py to use renamed enum**
- **Found during:** Task 1 (enum rename)
- **Issue:** `test_package.py` imported `SplittingStrategy` which no longer exists after the rename
- **Fix:** Updated import and assertion to use `ClassificationSplittingStrategy`
- **Files modified:** `tests/test_package.py`
- **Verification:** Full test suite (86 tests) passes
- **Committed in:** `cbb4e02` (Task 1 commit)

**2. [Rule 1 - Bug] Fixed DataLoader.shuffle attribute access in test**
- **Found during:** Task 2 (GREEN phase)
- **Issue:** `DataLoader.shuffle` is not exposed as a public attribute in newer PyTorch versions
- **Fix:** Changed test to use `patch('...DataLoader', wraps=DataLoader)` to capture kwargs
- **Files modified:** `tests/test_modules_base.py`
- **Verification:** All 15 module tests pass
- **Committed in:** `86c4186` (Task 2 commit)

**3. [Rule 2 - Missing Critical] Added KeyError validation to separate_target_feature_from_df**
- **Found during:** Task 1 (utility port)
- **Issue:** Threat model T-04-01-02 requires input validation; rbspaper source has no validation
- **Fix:** Added `KeyError` with descriptive message when target column not in DataFrame
- **Files modified:** `src/tscollection/datasets/utils/common.py`
- **Verification:** Test `test_missing_column_raises_keyerror` confirms behavior
- **Committed in:** `cbb4e02` (Task 1 commit)

---

**Total deviations:** 3 (2 bug fixes, 1 missing critical per threat model)
**Impact on plan:** All deviations strengthen correctness. No scope creep.

## Issues Encountered

- None — execution followed TDD flow cleanly with expected RED failures.

## Next Phase Readiness

- `BaseTimeSeriesDataModule` provides shared base for classification and forecasting modules (Plans 04-02, 04-03, 04-04)
- `ClassificationSplittingStrategy` enum is importable from `tscollection.datasets.enums`
- `separate_target_feature_from_df` is importable from `tscollection.datasets.utils`
- All 101 tests pass (86 existing + 15 new module tests)

## User Setup Required

None — no external service configuration required.

---
*Phase: 04-data-modules*
*Completed: 2026-05-13*
