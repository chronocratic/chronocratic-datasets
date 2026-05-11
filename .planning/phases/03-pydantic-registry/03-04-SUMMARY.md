---
phase: 03-pydantic-registry
plan: 04
subsystem: config
tags: [pydantic, factory, registry, export-chain, config-lookup]

requires:
  - phase: 03-pydantic-registry
    plan: 01
    provides: DatasetConfig base, ClassificationConfig, ForecastingConfig, DatasetFamily enum
  - phase: 03-pydantic-registry
    plan: 02
    provides: UCRConfig (3 instances), UEAConfig (2 instances)
  - phase: 03-pydantic-registry
    plan: 03
    provides: ETTConfig (4 instances), ElectricityConfig (1), WeatherConfig (1)
provides:
  - CONFIGS dict with 11 frozen config entries keyed by dataset name
  - get_config(name) lookup function with keyword-only args
  - list_configs(family) filtering function with keyword-only args
  - Full export chain: root -> config -> factory -> all instances
  - DatasetFamily and SplitMode exported from tscollection.datasets (root)
affects: [04-download, 05-modules, 06-factory]

tech-stack:
  added: []
  patterns:
    - "Explicit registry dict + lookup functions (RESEARCH.md Pattern 4)"
    - "Keyword-only factory API (rbspaper registry.py mirror)"
    - "Alphabetical __all__ exports (established convention)"

key-files:
  created:
    - src/tscollection/datasets/config/factory.py
    - tests/test_config_factory.py
    - tests/test_config_init.py
  modified:
    - src/tscollection/datasets/config/__init__.py
    - src/tscollection/datasets/__init__.py

key-decisions:
  - "CONFIGS dict built from explicit tuple, not auto-discovery"
  - "get_config uses keyword-only args, raises KeyError with descriptive message"
  - "list_configs accepts optional family filter, returns list"

patterns-established:
  - "Factory API: keyword-only args, descriptive errors, sorted availability list"
  - "Export chain: base types -> config classes -> instances -> factory functions in __init__.py"

requirements-completed: ['CFG-01', 'CFG-02']

duration: 3min
completed: 2026-05-11
---

# Phase 3 Plan 04: Config Factory Registry and Export Chain Summary

**CONFIGS registry dict with 11 frozen Pydantic config entries, get_config/list_configs factory API, and complete export chain from root package to all config instances.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-05-11T12:18:47Z
- **Completed:** 2026-05-11T12:21:13Z
- **Tasks:** 2 (Task 1: TDD RED/GREEN, Task 2: direct implementation)
- **Files modified:** 5

## Accomplishments

- factory.py with CONFIGS dict (11 entries), get_config(name), list_configs(family) — keyword-only APIs
- config/__init__.py exports all 11 instances, base types, config classes, and factory functions
- Root __init__.py exports DatasetFamily and SplitMode alongside existing enums
- 27 new tests (19 factory + 8 export chain) — all passing, full regression suite green (184 tests)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create factory.py with CONFIGS dict and lookup functions** (TDD)
   - `756a66c` — test: add failing tests for config factory registry
   - `44c8a85` — feat: implement config factory registry with CONFIGS dict and lookup functions

2. **Task 2: Update config/__init__.py, root __init__.py, and export chain**
   - `5956c7f` — feat: complete config export chain and root package updates

## Files Created/Modified

- `src/tscollection/datasets/config/factory.py` — NEW: CONFIGS dict (11 entries), get_config(name), list_configs(family) with keyword-only APIs
- `src/tscollection/datasets/config/__init__.py` — Full exports: base types, config classes, all 11 instances, factory functions
- `src/tscollection/datasets/__init__.py` — Added DatasetFamily and SplitMode to root exports
- `tests/test_config_factory.py` — 19 tests for CONFIGS structure, get_config lookup, list_configs filtering
- `tests/test_config_init.py` — 8 tests for export chain completeness and root package imports

## Decisions Made

- **Explicit tuple for CONFIGS:** Followed RESEARCH.md Pattern 4 recommendation — build registry from explicit tuple `_ALL_CONFIGS`, then create dict comprehension. No auto-discovery.
- **Keyword-only factory API:** `get_config(*, name)` and `list_configs(*, family=None)` enforce explicit argument names, consistent with established package conventions.
- **Descriptive KeyError:** `get_config` raises `KeyError('unknown dataset: {name!r}. Available: {sorted keys}')` to help users find correct dataset names.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Initial test file was written to the main repo path instead of the worktree path due to path resolution. Corrected by writing to the worktree root directly.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All 11 config instances are importable from `tscollection.datasets.config` and discoverable via `get_config()` and `list_configs()`.
- `DatasetFamily` and `SplitMode` are importable from `tscollection.datasets` (root).
- Phase 4 (download) can use `get_config(name)` to look up URL, sha256, and cache_key.
- Phase 6 (factory API) can iterate over `CONFIGS` values for `list_modules()` and dynamic lookup.

## Known Stubs

None — all implementations are complete for this plan's scope.

## Threat Flags

None — threat model entries (T-03-09, T-03-10) were dispositioned as "Accept" in the plan; factory.py introduces no new trust boundary surfaces.

---
*Phase: 03-pydantic-registry*
*Completed: 2026-05-11*
