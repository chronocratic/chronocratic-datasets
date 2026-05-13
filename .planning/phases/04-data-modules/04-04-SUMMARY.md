---
phase: 04-data-modules
plan: 04
subsystem: data-modules
tags: [lightning, pytorch, modules, exports, wiring, packaging]

# Dependency graph
requires:
  - phase: 04-data-modules
    plan: 04-03
    provides: UEAClassificationDataModule, ETTDataModule, ElectricityLoadModule, WeatherModule
provides:
  - modules/__init__.py with all 8 class exports (3 bases + 5 concrete)
  - modules/classes/__init__.py with 3 base class exports
  - Full import chain: tscollection.datasets -> modules -> classes
affects: [05-tests]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Alphabetically sorted __all__ lists (ASCII sort, uppercase-first)"
    - "Base classes re-exported through modules/__init__.py"
    - "Concrete modules imported via relative submodule paths"

key-files:
  created: []
  modified:
    - src/tscollection/datasets/modules/__init__.py
    - src/tscollection/datasets/modules/classes/__init__.py

key-decisions:
  - "modules/__init__.py re-exports base classes from .classes (no duplication)"
  - "Concrete modules imported via relative paths (.ucr, .uea, .ett, .electricity, .weather)"
  - "__all__ follows ASCII alphabetical sort matching utils/ and enums/ conventions"

requirements-completed: [MOD-02]

# Metrics
duration: 3min
completed: 2026-05-13
---

# Phase 4 Plan 4: Export Wiring Summary

**Wired modules/__init__.py and modules/classes/__init__.py with all 8 DataModule class exports (3 bases + 5 concrete) using ASCII-sorted __all__ lists, verified full import chain across 163 existing tests**

## Performance

- **Duration:** 3 min
- **Started:** 2026-05-13T12:02:05Z
- **Completed:** 2026-05-13T12:05:05Z
- **Tasks:** 2 completed (1 implementation + 1 verification)
- **Files modified:** 2 files

## Accomplishments

- Wired `modules/classes/__init__.py` with 3 base class exports (BaseTimeSeriesDataModule, BaseClassificationTimeSeriesDataModule, BaseForecastingTimeSeriesDataModule)
- Wired `modules/__init__.py` with all 8 class exports (3 bases re-exported + 5 concrete: UCRClassificationDataModule, UEAClassificationDataModule, ETTDataModule, ElectricityLoadModule, WeatherModule)
- Verified full import chain: root package loads, all module classes importable, no circular dependencies
- Confirmed full 163-test suite passes with no regressions

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire modules/classes/__init__.py and modules/__init__.py exports** - `8a3bf0c` (feat)
2. **Task 2: Verify full package import chain and existing tests** - verification only, no code changes

## Files Created/Modified

- `src/tscollection/datasets/modules/classes/__init__.py` - Replaces empty `__all__ = []` stub with imports from .base, .classification, .forecasting submodules; alphabetically sorted `__all__` list with 3 entries
- `src/tscollection/datasets/modules/__init__.py` - Replaces empty `__all__ = []` stub with re-exports from .classes and imports of 5 concrete modules (.ucr, .uea, .ett, .electricity, .weather); alphabetically sorted `__all__` list with 8 entries

## Decisions Made

- Re-export base classes through `modules/__init__.py` (not just `modules/classes/__init__.py`) for single-import convenience
- Import order: classes first (base), then concrete modules — verified to avoid circular imports per threat model T-04-04-02
- `__all__` sorting follows ASCII alphabetical order (uppercase before lowercase), matching existing `utils/__init__.py` and `enums/__init__.py` conventions

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Threat Surface Scan

- T-04-04-01 (Spoofing): Accepted — internal module wiring; exports are class references, not executable code
- T-04-04-02 (Repudiation): Mitigated — import order verified (classes/__init__.py loads before modules/__init__.py); full 163-test chain validates no circular imports

## Known Stubs

None - all exports are wired to real implementations from Plans 04-01 through 04-03.

## Next Phase Readiness

- `from tscollection.datasets.modules import *` loads all 8 classes without errors
- All module classes are importable via clean paths for Phase 5 test writing
- Full 163-test suite passes (no regressions)

---
*Phase: 04-data-modules*
*Completed: 2026-05-13*
