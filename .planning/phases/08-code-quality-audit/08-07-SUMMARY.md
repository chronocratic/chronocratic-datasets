---
phase: 08-code-quality-audit
plan: 07
subsystem: runner
tags: [ty, protocol, isinstance, type_narrowing, cast]

requires:
  - phase: 08-code-quality-audit
    plan: 01
    provides: "Clean runner.py with ty: ignore comments awaiting D-05 fix"
provides:
  - "Protocol-based model params typing replacing hasattr pattern"
  - "Zero ty: ignore comments in runner.py"
affects: [08-code-quality-audit]

tech-stack:
  added: []
  patterns: [runtime_checkable Protocol for structural typing, isinstance narrowing, cast for type widening]

key-files:
  created: []
  modified:
    - runners/py/runner.py

key-decisions:
  - "Define Protocols inline in runner.py (single-use abstraction, no dedicated types file)"
  - "Use @runtime_checkable to allow isinstance() at runtime"
  - "Explicit isinstance narrowing for get_all_datasets return type"
  - "cast for compute_config_hash parameter type widening"

requirements-completed: []

duration: 15min
completed: 2026-05-08
---

# Phase 08 Plan 07: Protocol-Based Model Params Typing Summary

**Protocol-based structural typing replaces hasattr-guarded dynamic access, eliminating all ty: ignore comments in runner.py**

## Performance

- **Duration:** ~15 min
- **Tasks:** 1/1 completed
- **Files modified:** 1

## Accomplishments

- Defined `_ModelParamsWithSequenceLength(Protocol)` with `set_sequence_length(self, length: int) -> None`
- Defined `_ModelParamsWithMaxTrainLength(Protocol)` with `max_train_length: int`
- Replaced 4 `ty: ignore` comments on `hasattr`-guarded access with `isinstance(Protocol)` checks
- Fixed `get_all_datasets` return type narrowing via explicit `isinstance` + `list()` conversion
- Fixed `compute_config_hash` parameter type via `cast('dict[str, object]', ...)`
- Zero `ty: ignore` comments remain in runner.py

## Task Commits

1. **Task 1: Add ModelParamsProtocol and fix runner.py ty errors** — `1cff9f6` (feat) — runners/py/runner.py

## Files Created/Modified

- `runners/py/runner.py` — Added Protocol imports and definitions, replaced hasattr with isinstance, fixed type narrowing for get_all_datasets and compute_config_hash

## Deviations from Plan

None — plan executed exactly as written.

## Verification

- `uv run ty check runners/py/runner.py` — All checks passed
- `uv run ruff check runners/py/runner.py` — All checks passed
- `grep 'ty: ignore' runners/py/runner.py` — Zero matches

---
*Phase: 08-code-quality-audit*
*Completed: 2026-05-08*
