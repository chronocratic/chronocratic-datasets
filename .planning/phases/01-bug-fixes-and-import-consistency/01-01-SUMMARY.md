---
phase: 01-bug-fixes-and-import-consistency
plan: 01
subsystem: imports
tags: [circular-import, lazy-import, augmentation, ts2vec]

# Dependency graph
requires: []
provides:
  - "AugmentationMethod importable without circular ImportError"
  - "TS2Vec importable without circular ImportError"
  - "test_pipeline_core.py collects without import errors"
affects: [01-02, 01-03, 01-04]

# Tech tracking
tech-stack:
  added: []
  patterns: [lazy-import-to-break-circular-dependency]

key-files:
  created: []
  modified: [src/rbspaper/models/augmentation/strategies.py]

key-decisions:
  - "Used lazy import inside CropShiftAugmentation.augment() instead of TYPE_CHECKING — the function is needed at runtime, just not at module load time"

patterns-established:
  - "Lazy import for breaking circular dependencies when the import is runtime-only, not load-time"

requirements-completed: [BUG-01]

# Metrics
duration: 5min
completed: 2026-05-05
---

# Plan 01-01: Circular Import Fix Summary

**Broke augmentation/ts2vec circular import chain by moving extract_subsequences_per_row to a lazy import inside CropShiftAugmentation.augment()**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-05T00:00:00Z
- **Completed:** 2026-05-05T00:05:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Removed top-level import of `extract_subsequences_per_row` from `strategies.py`
- Added lazy import inside `CropShiftAugmentation.augment()` method
- All augmentation and ts2vec imports resolve without circular dependency
- test_pipeline_core.py collects 8 tests successfully

## Task Commits

Each task was committed atomically:

1. **Task 1: Break circular import via lazy import in CropShiftAugmentation** - TBD (fix)

**Plan metadata:** TBD (docs: complete plan)

## Files Created/Modified
- `src/rbspaper/models/augmentation/strategies.py` - Removed top-level import of `extract_subsequences_per_row`; added local import inside `CropShiftAugmentation.augment()` with `# noqa: PLC0415`

## Decisions Made
- Lazy import over TYPE_CHECKING: the function is used at runtime in `augment()`, so a compile-time guard would not suffice. The lazy import breaks the chain because strategies.py no longer eagerly pulls in ts2vec/ during module initialization.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Ruff flagged PLC0415 (import should be at top level) for the lazy import. Resolved with inline `# noqa: PLC0415` directive since this is an intentional circular import break.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Circular import resolved. Phase 01 plans 02-04 can proceed.

---
*Phase: 01-bug-fixes-and-import-consistency*
*Completed: 2026-05-05*
