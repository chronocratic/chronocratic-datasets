---
phase: 01-bug-fixes-and-import-consistency
plan: 03
subsystem: data
tags: [python, imports, ruff, consistency]

# Dependency graph
requires: []
provides:
  - All imports in data/ and adapters/ use consistent src.rbspaper.* prefix
  - Zero bare rbspaper.* imports across the entire codebase
affects: [01-bug-fixes-and-import-consistency, all future phases]

# Tech tracking
tech-stack:
  added: []
  patterns: [src.rbspaper.* import convention]

key-files:
  created: []
  modified:
    - src/rbspaper/adapters/__init__.py
    - src/rbspaper/data/__init__.py
    - src/rbspaper/data/data_setup.py
    - src/rbspaper/data/preparation.py
    - src/rbspaper/data/datasets/__init__.py
    - src/rbspaper/data/datasets/abstract.py
    - src/rbspaper/data/datasets/strategies.py
    - src/rbspaper/data/datasets/ett_dataset.py
    - src/rbspaper/data/datasets/weather_dataset.py
    - src/rbspaper/data/datasets/uea_dataset.py
    - src/rbspaper/data/datasets/electricity_load_dataset.py
    - src/rbspaper/data/datasets/ucr_dataset.py
    - src/rbspaper/data/utils/__init__.py
    - src/rbspaper/data/utils/scaling.py
    - src/rbspaper/data/modules/__init__.py
    - src/rbspaper/data/modules/abstract.py
    - src/rbspaper/data/modules/weather_datamodule.py
    - src/rbspaper/data/modules/electricity_load_datamodule.py
    - src/rbspaper/data/modules/ucr_datamodule.py
    - src/rbspaper/data/modules/ett_datamodule.py
    - src/rbspaper/data/modules/uea_datamodule.py

key-decisions:
  - Used sed for bulk replacement across 21 files for efficiency
  - Handled indented import in data_setup.py separately (line 90)
  - Auto-fixed ruff I001 import sorting issues in 4 files

patterns-established:
  - "All internal imports use 'from src.rbspaper.*' prefix consistently"

requirements-completed: [BUG-05]

# Metrics
duration: 5min
completed: 2026-05-05
---

# Phase 1 Plan 03: Import Consistency Summary

**Replaced 70 bare rbspaper.* imports with src.rbspaper.* across 21 files in data/ and adapters/ packages**

## Performance

- **Duration:** 5 min
- **Tasks:** 1
- **Files modified:** 21

## Accomplishments
- Replaced all bare `from rbspaper.` imports with `from src.rbspaper.` across data/ and adapters/
- Fixed indented import in data_setup.py (line 90) that sed missed
- Auto-fixed ruff I001 import sorting in 4 files (ett_dataset.py, weather_dataset.py, electricity_load_dataset.py, modules/__init__.py)
- Verified zero bare rbspaper.* imports remain across the entire src/ directory

## Task Commits

1. **Unify imports to src.rbspaper.* across data and adapters** - single commit (refactor)

## Files Modified
- `src/rbspaper/adapters/__init__.py` - 3 import lines updated
- `src/rbspaper/data/__init__.py` - 3 import lines updated
- `src/rbspaper/data/data_setup.py` - 4 import lines updated (3 top-level, 1 indented)
- `src/rbspaper/data/preparation.py` - 2 import lines updated
- `src/rbspaper/data/datasets/__init__.py` - 10 import lines updated
- `src/rbspaper/data/datasets/abstract.py` - 4 import lines updated
- `src/rbspaper/data/datasets/strategies.py` - 1 import line updated
- `src/rbspaper/data/datasets/ett_dataset.py` - 4 import lines updated
- `src/rbspaper/data/datasets/weather_dataset.py` - 4 import lines updated
- `src/rbspaper/data/datasets/uea_dataset.py` - 3 import lines updated
- `src/rbspaper/data/datasets/electricity_load_dataset.py` - 4 import lines updated
- `src/rbspaper/data/datasets/ucr_dataset.py` - 3 import lines updated
- `src/rbspaper/data/utils/__init__.py` - 6 import lines updated
- `src/rbspaper/data/utils/scaling.py` - 1 import line updated
- `src/rbspaper/data/modules/__init__.py` - 7 import lines updated
- `src/rbspaper/data/modules/abstract.py` - 3 import lines updated
- `src/rbspaper/data/modules/weather_datamodule.py` - 2 import lines updated
- `src/rbspaper/data/modules/electricity_load_datamodule.py` - 2 import lines updated
- `src/rbspaper/data/modules/ucr_datamodule.py` - 4 import lines updated
- `src/rbspaper/data/modules/ett_datamodule.py` - 2 import lines updated
- `src/rbspaper/data/modules/uea_datamodule.py` - 3 import lines updated

## Deviations from Plan

None - plan executed exactly as written.

## Verification

- `grep -rn "from rbspaper\." src/` returns 0 matches (excluding string literals)
- `uv run ruff check src/rbspaper/data/ src/rbspaper/adapters/ --select I001` passes (0 errors)
- `uv run python -c "from src.rbspaper.data.registry import get_dataset_metadata"` succeeds

## Issues Encountered
None

## Next Phase Readiness
- Import consistency across data/ and adapters/ is now complete
- Combined with previous plans, the entire codebase now uses consistent src.rbspaper.* imports

---
*Phase: 01-bug-fixes-and-import-consistency*
*Completed: 2026-05-05*
