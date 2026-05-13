---
phase: 04-data-modules
plan: 02
subsystem: data-modules
tags: [lightning, pytorch, classification, forecasting, ucr, arff, dataloader, tdd]

# Dependency graph
requires:
  - phase: 04-data-modules
    plan: 04-01
    provides: BaseTimeSeriesDataModule, ClassificationSplittingStrategy, separate_target_feature_from_df, create_data_scaler
provides:
  - BaseClassificationTimeSeriesDataModule with label handling and target column separation
  - BaseForecastingTimeSeriesDataModule with time slicing and sklearn-based scaling
  - UCRClassificationDataModule with ARFF reading, splitting, and dataloader methods
affects: [04-03, 04-04, 04-05]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "BaseClassificationTimeSeriesDataModule inherits BaseTimeSeriesDataModule; uses create_data_scaler in inherited setup()"
    - "BaseForecastingTimeSeriesDataModule overrides setup() with sklearn direct scaling (fit on train slice only)"
    - "_prepare_data_scaler() compares ScalingMethod enum members (not strings)"
    - "UCRClassificationDataModule hardcodes DataForm.REGULAR, uses Path-only folder param"
    - "Variable-length processing skips None validation splits"

key-files:
  created:
    - src/tscollection/datasets/modules/classes/classification.py
    - src/tscollection/datasets/modules/classes/forecasting.py
    - src/tscollection/datasets/modules/ucr.py
    - tests/test_modules_classification_forecasting.py
    - tests/test_modules_ucr.py
  modified:
    - src/tscollection/datasets/modules/classes/classification.py (None-check fix)

key-decisions:
  - "Classification base forwards data_form to grandparent (BaseTimeSeriesDataModule) for subclass flexibility"
  - "Forecasting base checks isinstance(self._full_data, pd.DataFrame) (not ndarray-first) for clarity"
  - "_process_data_with_varying_sequence_lengths handles None valid_data_samples gracefully"

requirements-completed: [MOD-01, MOD-02, MOD-03, MOD-05, MOD-06]

# Metrics
duration: 8min
completed: 2026-05-13
---

# Phase 4 Plan 2: Classification and Forecasting Base Classes + UCR Module Summary

**Classification and forecasting base DataModule classes with enum-wired constructors, sklearn-based forecasting scaling, and UCR concrete module with ARFF loading, manual splitting, and variable-length processing**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-13T11:36:20Z
- **Completed:** 2026-05-13T11:44:20Z
- **Tasks:** 2 completed (both TDD: RED/GREEN)
- **Files modified:** 5 files created, 1 file modified, 24 test cases added

## Accomplishments

- Created `BaseClassificationTimeSeriesDataModule` with target column separation, label properties, splitting strategy support, and variable-length processing
- Created `BaseForecastingTimeSeriesDataModule` with sklearn-based scaling (fit on train slice only), time feature extraction, abstract slice/transform hooks
- Created `UCRClassificationDataModule` with ARFF reading via `utils/arff.py`, FileNotFoundError validation, manual splitting support, and stratify fallback for validation

## Task Commits

Each task was committed atomically:

1. **Task 1: Create classification and forecasting base classes** (TDD)
   - `78b3cc9` (test): Add failing tests for classification and forecasting base modules (14 tests)
   - `2b42bbb` (feat): Implement classification and forecasting base DataModule classes

2. **Task 2: Create UCRClassificationDataModule** (TDD)
   - `4f97d21` (test): Add failing tests for UCRClassificationDataModule (10 tests)
   - `7a27fd4` (feat): Implement UCRClassificationDataModule with ARFF loading

## Files Created/Modified

- `src/tscollection/datasets/modules/classes/classification.py` - BaseClassificationTimeSeriesDataModule with target_column_name param (D-01), ClassificationSplittingStrategy enum (D-04), _separate_target_feature partial, num_classes property (D-11), label properties, varying-sequence-length processing, abstract prepare_data/dataloader methods
- `src/tscollection/datasets/modules/classes/forecasting.py` - BaseForecastingTimeSeriesDataModule with ForecastingMode enum (D-05), ScalingMethod-based _prepare_data_scaler (D-03), overridden setup() with time feature extraction (D-10), abstract _set_data_slices/_transform_data, _split_data, _post_prepare_data, train-slice-only scaler fitting (T-04-02-04)
- `src/tscollection/datasets/modules/ucr.py` - UCRClassificationDataModule with Path-only folder param (D-07), hardcoded DataForm.REGULAR (D-02), ARFF file reading via utils/arff.py, manual splitting support, validation split with stratify fallback, variable-length processing, dataloader methods returning DataLoader instances, FileNotFoundError for missing paths (D-16)
- `tests/test_modules_classification_forecasting.py` - 14 tests for both base classes
- `tests/test_modules_ucr.py` - 10 tests for UCR module with synthetic ARFF fixtures

## Decisions Made

- Classification base accepts `data_form` parameter (forwarded to grandparent) so subclasses can set it explicitly
- Forecasting base checks `isinstance(self._full_data, pd.DataFrame)` first (not ndarray-first) for clarity
- `_process_data_with_varying_sequence_lengths` handles `None` valid_data_samples (no crash when valid_size=0)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Classification base missing data_form parameter**
- **Found during:** Task 2 (UCR module construction)
- **Issue:** `BaseClassificationTimeSeriesDataModule.__init__()` did not accept `data_form`, but UCR needs to pass `DataForm.REGULAR` through the chain
- **Fix:** Added `data_form: DataForm = DataForm.REGULAR` to classification base constructor, forwarded to grandparent
- **Files modified:** `src/tscollection/datasets/modules/classes/classification.py`
- **Verification:** All 125 tests pass
- **Committed in:** `7a27fd4` (Task 2 commit)

**2. [Rule 1 - Bug] Variable-length processing crashes on None validation data**
- **Found during:** Task 2 (test with valid_size=0.0)
- **Issue:** `_process_data_with_varying_sequence_lengths()` calls `process_data_with_varying_sequence_lengths_single(None)` when no validation split exists
- **Fix:** Added `if self._valid_data_samples is not None` guard before processing
- **Files modified:** `src/tscollection/datasets/modules/classes/classification.py`
- **Verification:** Test `test_val_dataloader_returns_dataloader_or_none` passes with valid_size=0.0
- **Committed in:** `7a27fd4` (Task 2 commit)

**3. [Rule 3 - Blocking] Test fixture mismatch for synthetic ARFF folder naming**
- **Found during:** Task 2 (test implementation)
- **Issue:** `synthetic_ucr_folder` fixture created files named `synthetic_TRAIN.arff` but the tmp_path folder had a pytest-generated name, so the module looked for `{tmp_path.name}_TRAIN.arff`
- **Fix:** Created a named subdirectory (`synthetic`) inside tmp_path, so folder name matches file prefixes
- **Files modified:** `tests/test_modules_ucr.py`
- **Verification:** All 10 UCR tests pass
- **Committed in:** `7a27fd4` (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (2 bug fixes, 1 blocking test fixture)
**Impact on plan:** All deviations strengthen correctness. No scope creep.

## Issues Encountered

- None beyond the auto-fixed deviations above.

## Threat Surface Scan

- T-04-02-01 (Spoofing): Mitigated via `Path.exists()` check raising `FileNotFoundError` in `prepare_data()`
- T-04-02-03 (Repudiation): Mitigated via `_separate_target_feature` partial wrapping `separate_target_feature_from_df` (which has KeyError validation from Plan 01)
- T-04-02-04 (Information Disclosure): Mitigated via scaler fitting on `full_array[:, self._train_slice]` only in forecasting setup()

## Known Stubs

None - all implementations are complete and wired through.

## Next Phase Readiness

- `BaseClassificationTimeSeriesDataModule` ready for UEA subclass (Plan 04-03)
- `BaseForecastingTimeSeriesDataModule` ready for ETT/Electricity/Weather subclasses (Plan 04-04)
- All 125 tests pass (101 existing + 14 classification/forecasting + 10 UCR)
- `_process_data_with_varying_sequence_lengths` handles edge cases (None validation)

---
*Phase: 04-data-modules*
*Completed: 2026-05-13*
