---
phase: 03-utility-modules
plan: 01
subsystem: utility
tags: [numpy, pandas, scipy, arff, enums, strenum, str]

# Dependency graph
requires:
  - phase: 02-dataset-classes
    provides: common.py with compose/FunctionComposer/get_num_samples_from_ts, enums/data.py with existing enums
provides:
  - DataForm enum (StrEnum) in enums/data.py
  - flatten_list_of_np_arrays in utils/common.py
  - read_arff_as_df and process_df_according_to_dtypes in utils/arff.py
  - extract_time_features in utils/features.py
  - Export wiring through enums/__init__.py, datasets/__init__.py
affects: 03-02, 04-data-modules

# Tech tracking
tech-stack:
  added: []
  patterns:
    - StrEnum without "Enum" suffix for new enums (D-07)
    - Lazy scipy import inside read_arff_as_df (avoid top-level dependency)
    - No from __future__ import annotations in new files (D-10)
    - Runtime numpy import must be top-level, not TYPE_CHECKING (Pitfall 3)

key-files:
  created:
    - src/tscollection/datasets/utils/arff.py
    - src/tscollection/datasets/utils/features.py
    - tests/test_utils_common.py
    - tests/test_utils_arff.py
    - tests/test_utils_features.py
  modified:
    - src/tscollection/datasets/enums/data.py
    - src/tscollection/datasets/enums/__init__.py
    - src/tscollection/datasets/__init__.py
    - src/tscollection/datasets/utils/common.py

key-decisions:
  - "DataForm inserted after ForecastingMode in enums/data.py (alphabetical Data-grouping)"
  - "Any moved out of TYPE_CHECKING in arff.py due to runtime annotation evaluation without future annotations"

patterns-established:
  - "Lazy import for optional heavy deps (scipy.io.arff inside read_arff_as_df)"
  - "StrEnum attributes listed in docstring Attributes section"

requirements-completed: [UTI-01, UTI-03]

# Metrics
duration: 10min
completed: 2026-05-13
---

# Phase 3 Plan 01: DataForm Enum, Common Utilities, ARFF and Features Modules Summary

**DataForm StrEnum added to enums package, flatten_list_of_np_arrays wired into common.py, arff.py and features.py created with lazy scipy import and pandas 3.0-compatible time extraction.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-05-13T09:28:07Z
- **Completed:** 2026-05-13T09:38:00Z
- **Tasks:** 3/3 completed
- **Files created:** 5 (3 source + 2 test files for new modules; arff.py and features.py are new)
- **Files modified:** 4 (enums/data.py, enums/__init__.py, datasets/__init__.py, utils/common.py)

## Accomplishments

- DataForm enum (StrEnum) with REGULAR, NESTED, MULTI_FILES members wired through package exports
- flatten_list_of_np_arrays added to common.py with top-level numpy import (Pitfall 3 fix)
- arff.py created with lazy scipy.io.arff import and nominal-bytes documentation (Pitfall 2)
- features.py created with pandas 3.0-compatible isocalendar().week pattern (Pitfall 4)
- 16 new tests added across 3 test files; all 52 tests in suite pass

## Task Commits

Each task was committed atomically:

1. **Task 1: DataForm enum and flatten_list_of_np_arrays** (TDD)
   - `de91a38` (test: add failing tests for DataForm and flatten_list_of_np_arrays)
   - `8038e29` (feat: add DataForm enum and flatten_list_of_np_arrays)

2. **Task 2: Create arff.py utility module** (TDD)
   - `4fd73ca` (test: add failing tests for arff.py utility module)
   - `5f8105e` (feat: create arff.py utility module)

3. **Task 3: Create features.py utility module** (TDD)
   - `70fb467` (test: add failing tests for features.py utility module)
   - `a4aa3e9` (feat: create features.py utility module)

_Note: TDD tasks have two commits each (RED test commit, GREEN implementation commit)_

## Files Created/Modified

- `src/tscollection/datasets/enums/data.py` — Added DataForm(StrEnum) with REGULAR, NESTED, MULTI_FILES
- `src/tscollection/datasets/enums/__init__.py` — Export wiring for DataForm
- `src/tscollection/datasets/__init__.py` — Root package export for DataForm
- `src/tscollection/datasets/utils/common.py` — Moved numpy import to top-level; added flatten_list_of_np_arrays to __all__ and implementation
- `src/tscollection/datasets/utils/arff.py` — NEW: read_arff_as_df (lazy scipy) and process_df_according_to_dtypes
- `src/tscollection/datasets/utils/features.py` — NEW: extract_time_features returning (N,7) float32
- `tests/test_utils_common.py` — 7 tests for DataForm enum and flatten_list_of_np_arrays
- `tests/test_utils_arff.py` — 4 tests for read_arff_as_df and process_df_according_to_dtypes
- `tests/test_utils_features.py` — 5 tests for extract_time_features

## Decisions Made

- DataForm inserted after ForecastingMode in enums/data.py to maintain alphabetical grouping within the "Data" prefix range
- `Any` moved out of TYPE_CHECKING in arff.py because without `from __future__ import annotations`, annotations are evaluated at runtime and `Any` must be defined

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Any type behind TYPE_CHECKING caused NameError in arff.py**
- **Found during:** Task 2 (arff.py creation)
- **Issue:** `Any` was imported inside `TYPE_CHECKING` but used in the return type `tuple[pd.DataFrame, Any]` of `read_arff_as_df`. Without `from __future__ import annotations`, type annotations are evaluated at runtime, causing `NameError: name 'Any' is not defined`.
- **Fix:** Moved `Any` to a top-level import: `from typing import TYPE_CHECKING, Any`
- **Files modified:** `src/tscollection/datasets/utils/arff.py`
- **Verification:** Tests pass; imports resolve without errors
- **Committed in:** `5f8105e` (part of Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Fix was necessary for correctness. No scope creep.

## Issues Encountered

None — all planned work executed smoothly after the one auto-fix.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- DataForm enum is available for scaling.py (Wave 2, Plan 03-02) which needs it for the `create_data_scaler` factory
- flatten_list_of_np_arrays is available in common.py for scaling.py imports
- arff.py and features.py are self-contained and ready for Phase 4 data module consumption
- Plan 03-02 (scaling.py + general.py) can proceed independently in the next wave

---

*Phase: 03-utility-modules*
*Completed: 2026-05-13*
