---
phase: 08-code-quality-audit
plan: 01
subsystem: code_quality
tags: [ty, ruff, type_annotations, circular_import, lightning, __getattr__]

requires:
  - phase: 07-experiment-tracking
    provides: loggers.py with LightningLogger references and ts2vec/augmentation lazy imports
provides:
  - Zero ty errors on encoding.py, setup/model.py, ts2vec/__init__.py, augmentation/__init__.py, loggers.py
  - Direct TS2Vec and AugmentationMethod imports work at runtime
  - Correct pl.loggers.Logger type references in loggers.py
affects: [08-code-quality-audit, all plans importing TS2Vec or AugmentationMethod]

tech-stack:
  added: []
  patterns: [TYPE_CHECKING + __getattr__ hybrid for circular import resolution]

key-files:
  created: []
  modified:
    - src/rbspaper/models/ts2vec/__init__.py
    - src/rbspaper/models/augmentation/__init__.py
    - src/rbspaper/pipeline/loggers.py

key-decisions:
  - "Use TYPE_CHECKING + __getattr__ hybrid in ts2vec/__init__.py to satisfy both ty (actual type) and runtime (no circular import)"
  - "Keep augmentation/__init__.py as direct import (works because ts2vec hybrid breaks the circular chain)"
  - "encoding.py _ModelType remains as original union — Task 1 fix resolves all ty errors without widening"

requirements-completed: []

duration: 45min
completed: 2026-05-08
---

# Phase 08 Plan 01: Remove Lazy Imports and Fix Type Annotations Summary

**TYPE_CHECKING + __getattr__ hybrid pattern resolves cascading ty errors while avoiding circular imports, plus LightningLogger -> Logger API fix**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-05-08T14:05:00Z
- **Completed:** 2026-05-08T14:50:00Z
- **Tasks:** 3 (Task 2 required no file changes)
- **Files modified:** 3

## Accomplishments

- ts2vec/__init__.py uses TYPE_CHECKING + __getattr__ hybrid — ty resolves TS2Vec as actual class type, runtime avoids circular import chain
- augmentation/__init__.py uses direct import — works because ts2vec's hybrid approach breaks the circular dependency
- loggers.py references correct pl.loggers.Logger (Lightning 2.x API) — zero possibly-missing-submodule warnings

## Task Commits

Each task was committed atomically:

1. **Task 1: Remove lazy imports from ts2vec and augmentation __init__.py** — `410fe59` (feat) — ts2vec hybrid fix; augmentation direct import re-applied in `6a19985` (feat) after rebase
2. **Task 2: Widen _ModelType in encoding.py** — no file changes needed; Task 1's hybrid fix resolved all ty errors on encoding.py and setup/model.py
3. **Task 3: Fix LightningLogger -> Logger in loggers.py** — `5f507d0` (fix)

## Files Created/Modified

- `src/rbspaper/models/ts2vec/__init__.py` — TYPE_CHECKING import for ty, __getattr__ for runtime; resolves TS2Vec as actual type
- `src/rbspaper/models/augmentation/__init__.py` — Direct import of AugmentationMethod; circular chain broken by ts2vec hybrid
- `src/rbspaper/pipeline/loggers.py` — Replaced 3x pl.loggers.LightningLogger with pl.loggers.Logger; added submodule import for ty resolution

## Decisions Made

- **Hybrid ts2vec/__init__.py:** Full direct import was impossible due to circular dependency (strategies.py -> ts2vec/utils.py -> ts2vec/__init__.py -> model.py -> augmentation/factories.py -> strategies.py). TYPE_CHECKING + __getattr__ hybrid satisfies both ty and runtime.
- **encoding.py unchanged:** Task 1's fix cascaded — with TS2Vec resolving as actual type, the original `_ModelType = TS2Vec | AutoTCL | CoST` union passes ty cleanly. Widening to `pl.LightningModule` was not needed and would have introduced `call-non-callable` errors (pl.LightningModule.encode is a property returning Tensor | Module).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 4 - Architectural] ts2vec/__init__.py uses TYPE_CHECKING + __getattr__ hybrid instead of full direct import**
- **Found during:** Task 1 (runtime verification)
- **Issue:** Plan assumed circular dependency was resolved (D-01). It was not — the import chain strategies.py -> ts2vec/utils.py -> ts2vec/__init__.py -> model.py -> augmentation/factories.py -> strategies.py causes ImportError when both __init__.py files use direct imports.
- **Fix:** Used TYPE_CHECKING block to provide actual type to ty (solves unsupported-operator, invalid-type-form errors) while keeping __getattr__ for runtime (avoids circular import). This satisfies both the type-checking goal and runtime correctness.
- **Files modified:** src/rbspaper/models/ts2vec/__init__.py
- **Verification:** `uv run ty check` passes zero errors; `uv run python -c "from src.rbspaper.models.ts2vec import TS2Vec"` succeeds.
- **Committed in:** 410fe59

**2. [Rule 1 - Bug] encoding.py _ModelType remains as original union instead of widened to pl.LightningModule**
- **Found during:** Task 2 (ty verification)
- **Issue:** With TS2Vec resolving correctly (Task 1 hybrid fix), the original `_ModelType = TS2Vec | AutoTCL | CoST` passes ty. Widening to `pl.LightningModule` introduces new `call-non-callable` errors because pl.LightningModule.encode is a property returning Tensor | Module, not a callable method.
- **Fix:** Kept original union type — it is more precise and ty-clean with the Task 1 fix.
- **Files modified:** None (encoding.py unchanged)
- **Verification:** `uv run ty check src/rbspaper/models/encoding.py` — zero errors.
- **Committed in:** N/A (no changes)

**3. [Rule 1 - Bug] loggers.py needed additional submodule import and ty-specific ignore comment**
- **Found during:** Task 3 (ty verification)
- **Issue:** After replacing LightningLogger with Logger, ty reported `possibly-missing-submodule` because `pl.loggers` was not explicitly imported. Also, `# type: ignore[return-value]` (mypy syntax) did not suppress ty's `invalid-return-type` diagnostic.
- **Fix:** Added `import lightning.pytorch.loggers` in TYPE_CHECKING block with `# noqa: F401`. Changed `# type: ignore[return-value]` to `# ty: ignore[invalid-return-type]`.
- **Files modified:** src/rbspaper/pipeline/loggers.py
- **Verification:** `uv run ty check src/rbspaper/pipeline/loggers.py` — zero diagnostics.
- **Committed in:** 5f507d0

---

**Total deviations:** 3 (2 auto-fixed per Rules 1/4, 1 no file change needed)
**Impact on plan:** All deviations improve correctness. The hybrid ts2vec approach achieves the same ty-clean result the plan intended (removing __getattr__ as the root cause of type resolution issues) while respecting the runtime circular dependency constraint.

## Issues Encountered

- Worktree was created from a commit (`1a8207c docs: map existing codebase`) that predates loggers.py. Required rebase onto `gsd_fixes_and_updates` to bring in the file, with conflict resolution for ts2vec/__init__.py.
- The plan's assumption that the circular import was resolved (D-01: "verified at runtime") was incorrect — the circular dependency still exists in the codebase.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- ty is clean on all files targeted by this plan (encoding.py, setup/model.py, ts2vec/__init__.py, augmentation/__init__.py, loggers.py)
- Runtime imports for TS2Vec, AugmentationMethod, and encode_data all succeed
- Subsequent plans in phase 08 can proceed with the type annotation fixes in place

---
*Phase: 08-code-quality-audit*
*Completed: 2026-05-08*
