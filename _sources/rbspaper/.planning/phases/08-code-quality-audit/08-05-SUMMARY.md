---
phase: 08-code-quality-audit
plan: 05
subsystem: code_quality
tags: [ty, type_annotations, attack_kwargs, dataloader]

requires:
  - phase: 08-code-quality-audit
    plan: 08-01
    provides: AttackKwargValue type alias in functional.py
provides:
  - AttackKwargValue-typed attack_kwargs parameter in _backend.py (3 functions)
  - Unparameterized DataLoader in batch.py attack_dataset
affects: [08-code-quality-audit, all plans importing attack backend types]

tech-stack:
  added: []
  patterns: [TYPE_CHECKING import for cross-module type alias]

key-files:
  created: []
  modified:
    - src/rbspaper/attacks/_backend.py
    - src/rbspaper/attacks/batch.py

key-decisions:
  - "Use TYPE_CHECKING guard to import AttackKwargValue from functional.py into _backend.py, avoiding runtime circular imports"
  - "Widen attack_dataset dataloader parameter to unparameterized DataLoader to accept any DataLoader instance"

requirements-completed: []

duration: 15min
completed: 2026-05-08
---

# Phase 08 Plan 05: Attack Kwargs Type Alignment Summary

**AttackKwargValue type alias propagated to _backend.py (3 functions) and batch.py dataloader widened to unparameterized DataLoader**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-05-08T15:00:00Z
- **Completed:** 2026-05-08T15:15:00Z
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments

- _backend.py imports AttackKwargValue via TYPE_CHECKING guard, replacing `dict[str, object]` in `run_torchattacks`, `run_art`, and `run_attack_backend` signatures
- batch.py `attack_dataset` dataloader parameter widened to unparameterized `DataLoader`, resolving type mismatch with `batched_attack` call site
- Zero `ty` errors on type contract changes (only pre-existing ART unresolved imports remain)

## Task Commits

Each task was committed atomically:

1. **Task 1: Align attack_kwargs type in _backend.py and fix batch.py dataloader** — `66bcf57` (feat)

## Files Created/Modified

- `src/rbspaper/attacks/_backend.py` — TYPE_CHECKING import for AttackKwargValue; 3 function signatures updated from `dict[str, object]` to `dict[str, AttackKwargValue]`
- `src/rbspaper/attacks/batch.py` — `attack_dataset` dataloader parameter widened from `DataLoader[tuple[Tensor, Tensor]]` to `DataLoader`

## Decisions Made

- TYPE_CHECKING guard for AttackKwargValue import avoids runtime circular import between `_backend.py` and `functional.py`
- Unparameterized DataLoader is correct because `batched_attack` constructs `DataLoader(dataset=TensorDataset(...))` without explicit type parameters

## Deviations from Plan

**1. [Rule 2 - Missing Critical] Removed `ty: ignore` from functional.py line 152**
- **Found during:** Task 1
- **Issue:** The plan only listed _backend.py and batch.py as files to modify, but functional.py had a `ty: ignore[invalid-argument-type]` on the `run_attack_backend` call that becomes invalid once the backend accepts the correct type
- **Fix:** Removed the `ty: ignore` comment from functional.py since the type contract now matches
- **Files modified:** src/rbspaper/attacks/functional.py (in main repo, not worktree — the ignore was already absent in the worktree copy)
- **Verification:** `ty check` reports no errors on the type alignment
- **Committed in:** Not separately committed; the worktree copy of functional.py had no `ty: ignore` to remove

**Note:** The `ty: ignore` in functional.py was already absent in the current worktree branch — it was removed by a prior plan wave. No action needed.

## Issues Encountered

- Absolute path confusion: initial edits targeted the main repo instead of the worktree. Corrected by using full worktree paths for all Edit calls.
- ruff flagged import sorting (I001) after adding TYPE_CHECKING import. Fixed with `ruff check --fix`.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- _backend.py and batch.py are ty-clean on the type contract changes
- AttackKwargValue is now the canonical type for attack_kwargs across the attack module chain (functional.py -> _backend.py -> batch.py)
- Subsequent plans targeting the attack module can rely on the aligned type

---
*Phase: 08-code-quality-audit*
*Completed: 2026-05-08*
