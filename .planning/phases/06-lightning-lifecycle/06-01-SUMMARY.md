---
phase: 06-lightning-lifecycle
plan: 01
subsystem: testing
tags: [pytest, tdd, constants, time-features]

requires:
  - phase: 03-utility-modules
    provides: extract_time_features function producing 7 columns
provides:
  - TIME_FEATURE_COUNT module-level constant (value 7) in features.py
  - TIME_FEATURE_COUNT re-exported from utils/__init__.py
  - Tests verifying constant value, package import, and shape alignment
affects: [06-05, 06-06]

tech-stack:
  added: []
  patterns: [Single source of truth for feature dimension count]

key-files:
  created: []
  modified:
    - src/tscollection/datasets/utils/features.py
    - src/tscollection/datasets/utils/__init__.py
    - tests/test_utils_features.py

key-decisions:
  - "TIME_FEATURE_COUNT = 7 derived from extract_time_features column list (minute, hour, dayofweek, day, dayofyear, month, week)"
  - "No refactor needed -- constant is trivial; structure follows existing patterns"

requirements-completed: [LIF-02]

duration: 2min
completed: 2026-05-28
---

# Phase 6 Plan 01: TIME_FEATURE_COUNT Constant Summary

**Add TIME_FEATURE_COUNT = 7 as a module-level constant in features.py and re-export from utils package, verified via TDD.**

## Performance

- **Duration:** 2 min
- **Started:** 2026-05-28T08:39:47Z
- **Completed:** 2026-05-28T08:41:46Z
- **Tasks:** 2 (RED test + GREEN implementation)
- **Files modified:** 3

## Accomplishments

- TIME_FEATURE_COUNT = 7 added as a typed module-level constant in features.py
- TIME_FEATURE_COUNT re-exported from utils/__init__.py via import and __all__ entry
- Three tests verifying constant value, package-level import, and shape alignment with extract_time_features

## Task Commits

1. **RED: Failing tests for TIME_FEATURE_COUNT** - `fc2f411` (test)
2. **GREEN: Implement TIME_FEATURE_COUNT and export** - `7bd7f9e` (feat)

## Files Created/Modified

- `src/tscollection/datasets/utils/features.py` - Added `TIME_FEATURE_COUNT: int = 7` constant, exported in __all__
- `src/tscollection/datasets/utils/__init__.py` - Imported TIME_FEATURE_COUNT from features module, added to __all__
- `tests/test_utils_features.py` - Added test_time_feature_count_value, test_time_feature_count_exported, test_time_feature_count_matches_extractor

## Decisions Made

None - followed plan as specified.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- TIME_FEATURE_COUNT is available for forecasting `_compute_dimensions` in subsequent plans
- All 8 tests in test_utils_features.py passing (5 existing + 3 new)

---
*Phase: 06-lightning-lifecycle*
*Completed: 2026-05-28*
