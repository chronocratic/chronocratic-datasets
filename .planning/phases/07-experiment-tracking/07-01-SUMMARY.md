---
phase: 07-experiment-tracking
plan: 01
subsystem: experiment tracking
tags: [wandb, tensorboard, lightning, logging, tracking]

requires:
  - phase: 06-attacks-extended
    provides: ExperimentPipelineConfig dataclass structure
provides:
  - tracking dependency group in pyproject.toml (wandb, tensorboard)
  - loggers field on ExperimentPipelineConfig
  - create_loggers factory (dual W&B + TensorBoard)
  - W&B helper functions (_find_wandb_logger, _flatten_dict, _log_config_to_wandb, _log_results_to_wandb)
affects: [07-experiment-tracking, pipeline runner, result analysis]

tech-stack:
  added: [wandb>=0.18.0, tensorboard>=2.17.0]
  patterns: [Factory Method for logger creation, lazy import for optional deps]

key-files:
  created: [src/rbspaper/pipeline/loggers.py]
  modified: [pyproject.toml, src/rbspaper/pipeline/config.py]

key-decisions:
  - "TensorBoardLogger always created when persist_artifacts=True (local-only, no network)"
  - "WandbLogger lazily imported with graceful ImportError fallback (T-07-04 mitigation)"
  - "log_model=False to prevent checkpoint upload to W&B cloud (D-07, T-07-01 mitigation)"
  - "No wandb.init() calls - use Lightning's WandbLogger.experiment property"
  - "Path moved to TYPE_CHECKING block (only used in annotations with from __future__)"

patterns-established:
  - "Lazy import pattern: try/except ImportError with logger.warning for optional deps"
  - "Keyword-only args for all public/private functions"
  - "type(logger).__name__ check instead of isinstance to avoid module-level wandb import"

requirements-completed: [D-01, D-02, D-07]

duration: 6min
completed: 2026-05-08
---

# Phase 07 Plan 01: Logger Foundation Summary

**Logger factory with dual W&amp;B + TensorBoard tracking, lazy WandbLogger import with graceful fallback, and config/results logging helpers.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-05-08T09:16:01Z
- **Completed:** 2026-05-08T09:22:00Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- tracking dependency group with wandb>=0.18.0 and tensorboard>=2.17.0
- loggers tuple field on ExperimentPipelineConfig with empty default
- loggers.py module with create_loggers factory and 4 W&amp;B helper functions

## Task Commits

Each task was committed atomically:

1. **Task 1: Add tracking dependency group to pyproject.toml** - `b81af2c` (feat)
2. **Task 2: Add loggers field to ExperimentPipelineConfig** - `d6c4184` (feat)
3. **Task 3: Create loggers.py module with factory and W&B helpers** - `bdb687d` (feat)

## Files Created/Modified
- `pyproject.toml` - Added tracking dependency group between attacks_extended and notebooks
- `src/rbspaper/pipeline/config.py` - Added loggers field after attack_scope in ExperimentPipelineConfig
- `src/rbspaper/pipeline/loggers.py` - New module with create_loggers factory and W&amp;B helpers

## Decisions Made
- TensorBoardLogger always created (local-only, no network dependency)
- WandbLogger lazily imported with graceful ImportError fallback
- log_model=False prevents checkpoint upload (D-07 security mitigation)
- No wandb.init() calls -- Lightning's WandbLogger.experiment holds the active run

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] F404: future import ordering**
- **Found during:** Task 3 (linting)
- **Issue:** `from __future__ import annotations` was after `__all__`, causing F404
- **Fix:** Moved `from __future__ import annotations` before `__all__`
- **Files modified:** src/rbspaper/pipeline/loggers.py
- **Verification:** ruff check passes
- **Committed in:** bdb687d (Task 3 commit)

**2. [Rule 2 - Missing Critical] TC003: Path import optimization**
- **Found during:** Task 3 (linting)
- **Issue:** `Path` imported at runtime but only used in type annotations
- **Fix:** Moved `Path` into TYPE_CHECKING block (safe with `from __future__ import annotations`)
- **Files modified:** src/rbspaper/pipeline/loggers.py
- **Verification:** ruff check passes
- **Committed in:** bdb687d (Task 3 commit)

**3. [Rule 2 - Missing Critical] PLC0415: import placement**
- **Found during:** Task 3 (linting)
- **Issue:** ruff flagged function-level imports for TensorBoardLogger, WandbLogger, wandb
- **Fix:** Moved TensorBoardLogger to top-level import. Added `# noqa: PLC0415` for WandbLogger (intentional lazy import with ImportError catch) and wandb (intentional lazy import inside _log_results_to_wandb)
- **Files modified:** src/rbspaper/pipeline/loggers.py
- **Verification:** ruff check passes
- **Committed in:** bdb687d (Task 3 commit)

---

**Total deviations:** 3 auto-fixed (all Rule 2 lint compliance)
**Impact on plan:** All fixes improve code quality without changing plan intent. Lazy imports preserved for W&amp;B graceful fallback.

## Issues Encountered
- None beyond linting adjustments documented above.

## User Setup Required

**Optional: W&amp;B online mode requires authentication.**
- For online tracking: set `WANDB_API_KEY` environment variable (obtain from W&amp;B Dashboard -> Settings -> API Keys)
- For offline HPC mode: no authentication needed, set `tracking_mode='offline'`
- For disabled mode: `tracking_mode='disabled'` uses TensorBoardLogger only

## Known Stubs
None -- all functions are fully implemented with real logic.

## Threat Flags
None -- all threat surfaces from the plan's threat model are mitigated:
- T-07-01: log_model=False enforced
- T-07-02: Table contains only aggregated metrics
- T-07-04: try/except ImportError around WandbLogger
- T-07-05: Pinned minimum versions in pyproject.toml

## Next Phase Readiness
- create_loggers factory ready for use by pipeline runner (Plan 02)
- W&amp;B helpers ready for config/results logging integration (Plan 03)
- ExperimentPipelineConfig.loggers field ready for injection

---
*Phase: 07-experiment-tracking*
*Completed: 2026-05-08*
