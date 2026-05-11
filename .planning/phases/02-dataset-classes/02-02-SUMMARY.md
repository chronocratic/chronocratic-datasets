---
phase: 02-dataset-classes
plan: 02
subsystem: dataset-core
tags:
  - pytorch-dataset
  - abstract-base-classes
  - template-method
  - strategy-pattern
  - sliding-window
  - classification
  - forecasting
dependency_graph:
  requires:
    - phase: 02-dataset-classes
      plan: 02-00
      provides: "Transform utilities (convert_numpy_to_tensor, expand_data_dimensionality), compose/get_num_samples_from_ts"
    - phase: 02-dataset-classes
      plan: 02-01
      provides: "SequenceHandlingStrategy ABC and concrete strategies (strategies.py)"
  provides:
    - TimeSeriesDataset ABC with mode-based __getitem__ dispatch (Template Method)
    - FixedTimeSeriesDataset hierarchy with seq_len read-only property (4 classes)
    - FlexibleTimeSeriesDataset hierarchy with strategy injection (3 classes)
    - classes/__init__.py exporting all 13 classes
    - 7 verification tests across fixed and flexible datasets
  affects:
    - 02-03 (concrete wrappers: ucr.py, uea.py, ett.py — inherit from these ABCs)
    - Phase 5 (modules — consume datasets)
tech-stack:
  added: []
  patterns:
    - Template Method: TimeSeriesDataset.__getitem__ dispatches via _get_sample_fun_map
    - Strategy: FlexibleTimeSeriesDataset injects SequenceHandlingStrategy
    - Read-only property: FixedTimeSeriesDataset.seq_len derived from data shape
key-files:
  created:
    - src/tscollection/datasets/datasets/classes/fixed.py
    - src/tscollection/datasets/datasets/classes/flexible.py
  modified:
    - src/tscollection/datasets/datasets/classes/__init__.py
    - tests/test_fixed_dataset.py
    - tests/test_flexible_dataset.py
key-decisions:
  - "Added seq_len property on FixedTimeSeriesDataset (not in rbspaper) per DST-03"
  - "Type-checked data in FixedTimeSeriesDataset.__init__ per threat model T-02-02-01"
  - "Validated data.ndim >= 2 in __init__ per threat model T-02-02-02"
  - "Used from __future__ import annotations for forward references (rbspaper convention)"
  - "Split rbspaper abstract.py into fixed.py + flexible.py for clarity"
requirements-completed:
  - DST-01
  - DST-02
  - DST-03
  - DST-04
metrics:
  duration: 3min
  completed_date: "2026-05-11"
  tasks_completed: 3
  files_created: 2
  files_modified: 3
  tests_added: 7
---

# Phase 2 Plan 02: Fixed and Flexible Dataset ABCs Summary

Ported 7 dataset ABC classes from rbspaper's abstract.py into the tscollection.datasets namespace, adding the seq_len read-only property for fixed datasets and strategy injection for flexible sliding-window datasets, with 7 verification tests covering DST-01 through DST-04.

## Performance

- **Duration:** ~3 min
- **Started:** 2026-05-11T09:49:26Z
- **Completed:** 2026-05-11T09:53:00Z
- **Tasks:** 3 (Task 1: TDD fixed.py, Task 2: TDD flexible.py, Task 3: __init__.py + tests)
- **Files created:** 2 (fixed.py, flexible.py)
- **Files modified:** 3 (__init__.py, test_fixed_dataset.py, test_flexible_dataset.py)

## Accomplishments
- TimeSeriesDataset ABC with mode-based dispatch (Template Method via `_get_sample_fun_map`)
- FixedTimeSeriesDataset hierarchy (4 classes) with read-only `seq_len` property derived from data shape
- FlexibleTimeSeriesDataset hierarchy (3 classes) with Strategy pattern injection for window counting
- All 13 classes (6 strategies + 7 dataset ABCs) exported from `classes/__init__.py`
- 7 passing tests covering DST-01 (classification tuples), DST-02 (forecasting windows), DST-03 (seq_len read-only), DST-04 (configurable seq_len/step)

## Task Commits

Each task was committed atomically:

1. **Task 1: Port fixed dataset ABCs with seq_len property (TDD)** - `d2cc50e` (test: RED) + `cf6ebbd` (feat: GREEN)
2. **Task 2: Port flexible dataset ABCs (TDD)** - `903cc1c` (test: RED) + `6951215` (feat: GREEN)
3. **Task 3: Update classes/__init__.py and implement dataset tests** - `82e84ec` (feat)

## Files Created/Modified
- `src/tscollection/datasets/datasets/classes/fixed.py` - 267 lines, 4 classes (TimeSeriesDataset, FixedTimeSeriesDataset, FixedTimeSeriesDatasetUnivariate, FixedTimeSeriesDatasetMultivariate)
- `src/tscollection/datasets/datasets/classes/flexible.py` - 225 lines, 3 classes (FlexibleTimeSeriesDataset, FlexibleTimeSeriesDatasetSingleFile, FlexibleTimeSeriesDatasetMultipleFiles)
- `src/tscollection/datasets/datasets/classes/__init__.py` - 13-entry __all__ (6 strategies + 7 dataset ABCs), alphabetically sorted
- `tests/test_fixed_dataset.py` - 92 lines, 4 tests (DST-01, DST-03)
- `tests/test_flexible_dataset.py` - 85 lines, 3 tests (DST-02, DST-04)

## Decisions Made
- Followed rbspaper source verbatim for iteration logic (cursor management, transform pipeline, mode dispatch)
- Added `seq_len` as a read-only `@property` on `FixedTimeSeriesDataset` (new, not in rbspaper) using `data.shape[1]` for ndarray and `len(df.iloc[0])` for DataFrame
- Split rbspaper's single `abstract.py` into `fixed.py` + `flexible.py` matching the planned structure
- Applied threat model mitigations: type-check data (T-02-02-01), validate dimensions (T-02-02-02), bounds check in `_go_to_idx` (T-02-02-03)

## Deviations from Plan

### Security Hardening (Threat Model)

**1. [Rule 2 - Missing validation] Added type-check to FixedTimeSeriesDataset.__init__**
- **Threat:** T-02-02-01 (Spoofing)
- **Issue:** Original rbspaper code had no isinstance check on data parameter
- **Fix:** Added TypeError for non-ndarray/non-DataFrame inputs with helpful message
- **Files modified:** src/tscollection/datasets/datasets/classes/fixed.py
- **Commit:** cf6ebbd

**2. [Rule 2 - Missing validation] Added dimension validation to FixedTimeSeriesDataset.__init__**
- **Threat:** T-02-02-02 (Spoofing)
- **Issue:** Original rbspaper code had no guard against 1D data (seq_len would crash)
- **Fix:** Added ValueError when data.ndim < 2 (ndarray) or DataFrame has 0 columns
- **Files modified:** src/tscollection/datasets/datasets/classes/fixed.py
- **Commit:** cf6ebbd

---

**Total deviations:** 2 auto-fixed (both Rule 2: threat model hardening)
**Impact on plan:** Both fixes required for correctness; no scope creep.

## Issues Encountered
None - all tasks completed as specified.

## Known Stubs
None -- all test placeholders replaced with real tests.

## Threat Flags
None -- all threat model items (T-02-02-01 through T-02-02-05) were addressed. T-02-02-04 (mode dispatch safety) is mitigated by enum type; T-02-02-05 (transformations_sequence injection) is marked accept.

## Next Phase Readiness
- Fixed and flexible dataset ABCs are complete and tested (DST-01 through DST-04 satisfied)
- Ready for Plan 02-03 (concrete wrappers: ucr.py, uea.py, ett.py) which will inherit from these ABCs
- Full test suite passes (28 tests): `uv run pytest tests/ -q --tb=short`

## Self-Check

### Created Files
- [PASS] fixed.py exists with 4 classes in __all__ (267 lines, minimum 140)
- [PASS] flexible.py exists with 3 classes in __all__ (225 lines, minimum 140)

### Tests
- [PASS] test_fixed_dataset.py has 4 test functions (92 lines, minimum 60)
- [PASS] test_flexible_dataset.py has 3 test functions (85 lines, minimum 50)
- [PASS] All 7 tests pass: `uv run pytest tests/test_fixed_dataset.py tests/test_flexible_dataset.py -x --tb=short -v`

### Imports
- [PASS] All 13 classes importable from `tscollection.datasets.datasets.classes`
- [PASS] Full test suite passes (28 tests): `uv run pytest tests/ -q --tb=short`

### Commits
- [PASS] d2cc50e: test(02-02): add failing tests for fixed dataset ABCs
- [PASS] cf6ebbd: feat(02-02): port fixed dataset ABCs with seq_len property
- [PASS] 903cc1c: test(02-02): add failing tests for flexible dataset ABCs
- [PASS] 6951215: feat(02-02): port flexible dataset ABCs with strategy injection
- [PASS] 82e84ec: feat(02-02): update classes/__init__.py exports and implement dataset tests

### TDD Gate Compliance
- [PASS] Task 1: RED commit (d2cc50e) before GREEN commit (cf6ebbd)
- [PASS] Task 2: RED commit (903cc1c) before GREEN commit (6951215)

---
*Phase: 02-dataset-classes*
*Completed: 2026-05-11*
