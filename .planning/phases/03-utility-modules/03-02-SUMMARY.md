---
phase: 03-utility-modules
plan: 02
subsystem: utility
tags: [numpy, pandas, sklearn, scipy, torch, scaling, collation, variable-length, strenum]

# Dependency graph
requires:
  - phase: 03-01
    provides: DataForm enum in enums/data.py, flatten_list_of_np_arrays in common.py
provides:
  - create_data_scaler factory with enum-typed parameters in scaling.py
  - custom_collate_fn, centralize_variable_length_series, process_data_with_varying_sequence_lengths_single in general.py
affects: 04-data-modules

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Callable/Any must be top-level imports when used in runtime-evaluated annotations (no future annotations)
    - ScalingMethod enum members compared directly (not string values) per D-06
    - DataForm enum compared directly (not .value) per D-07

key-files:
  created:
    - src/tscollection/datasets/utils/scaling.py
    - src/tscollection/datasets/utils/general.py
    - tests/test_utils_scaling.py
    - tests/test_utils_general.py
  modified: []

key-decisions:
  - "Callable moved out of TYPE_CHECKING in scaling.py to avoid NameError (same pattern as arff.py in 03-01)"

patterns-established:
  - "Runtime type imports (Callable, Any) must be top-level when no future annotations"

requirements-completed: [UTI-02, UTI-04]

# Metrics
duration: 6min
completed: 2026-05-13
---

# Phase 3 Plan 02: Scaling and General Utilities Summary

**Enum-wired create_data_scaler factory with MinMax/Standard scaling across REGULAR, NESTED, and MULTI_FILES data forms; general.py with keyword-only collation and variable-length centering utilities.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-13T09:38:23Z
- **Completed:** 2026-05-13T09:44:31Z
- **Tasks:** 2/2 completed
- **Files created:** 4 (2 source + 2 test files)
- **Files modified:** 0

## Accomplishments

- scaling.py created with enum-wired create_data_scaler factory (ScalingMethod, DataForm)
- general.py created with custom_collate_fn (keyword-only), centralize_variable_length_series, process_data_with_varying_sequence_lengths_single
- 24 new unit tests added across 2 test files; full suite (76 tests) passes

## Task Commits

Each task was committed atomically (TDD):

1. **Task 1: Create scaling.py with enum-wired create_data_scaler** (TDD)
   - `3116600` (test: add failing tests for scaling.py utility module)
   - `bfbbff7` (feat: create scaling.py with enum-wired create_data_scaler)

2. **Task 2: Create general.py utility module** (TDD)
   - `e87034e` (test: add failing tests for general.py utility module)
   - `99aff2f` (feat: create general.py utility module)

_Note: TDD tasks have two commits each (RED test commit, GREEN implementation commit)_

## Files Created/Modified

- `src/tscollection/datasets/utils/scaling.py` (265 lines) — create_data_scaler factory with ScalingMethod/DataForm enums, private helpers (_get_scaler, _scale_regular_data, _scale_regular_data_and_return_same_type, _scale_multi_file_data, _scale_nested_data_all_dimensions)
- `src/tscollection/datasets/utils/general.py` (103 lines) — custom_collate_fn (keyword-only desired_batch_size), centralize_variable_length_series, process_data_with_varying_sequence_lengths_single
- `tests/test_utils_scaling.py` — 12 tests covering enum wiring, all data forms, scale=False, __all__
- `tests/test_utils_general.py` — 12 tests covering batch padding, centering, 2D/3D/DF processing, __all__

## Decisions Made

- Callable moved out of TYPE_CHECKING in scaling.py to avoid NameError (without `from __future__ import annotations`, type hints are evaluated at runtime). Same pattern as arff.py deviation in 03-01.
- TYPE_CHECKING import removed from general.py (no types used only for annotations).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Callable behind TYPE_CHECKING caused NameError in scaling.py**
- **Found during:** Task 1 (scaling.py creation)
- **Issue:** `Callable` was imported inside `TYPE_CHECKING` but used in the return type `-> Callable` of `create_data_scaler`. Without `from __future__ import annotations`, type annotations are evaluated at runtime, causing `NameError: name 'Callable' is not defined`.
- **Fix:** Moved `Callable` to a top-level import: `from collections.abc import Callable`
- **Files modified:** `src/tscollection/datasets/utils/scaling.py`
- **Verification:** Tests pass; imports resolve without errors
- **Committed in:** `bfbbff7` (part of Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Fix was necessary for correctness. No scope creep.

## Issues Encountered

- Test for StandardScaler used `ddof=1` (sample std) but StandardScaler computes population std (`ddof=0`). Fixed test assertion.
- Test for cycling pattern in custom_collate_fn had incorrect expected values due to batch growth shifting negative indices. Fixed test to match actual behavior.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- scaling.py provides create_data_scaler factory with enum wiring for Phase 4 data modules
- general.py provides custom_collate_fn (via functools.partial in Phase 4), centralize_variable_length_series, process_data_with_varying_sequence_lengths_single
- All 76 tests pass; utilities are ready for Phase 4 consumption

---

*Phase: 03-utility-modules*
*Completed: 2026-05-13*
