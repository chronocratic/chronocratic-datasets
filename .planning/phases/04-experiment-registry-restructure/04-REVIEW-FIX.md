---
phase: 04-experiment-registry-restructure
fixed_at: 2026-05-07T12:35:00Z
review_path: .planning/phases/04-experiment-registry-restructure/04-REVIEW.md
iteration: 1
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase 04: Code Review Fix Report

**Fixed at:** 2026-05-07T12:35:00Z
**Source review:** .planning/phases/04-experiment-registry-restructure/04-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 3 (CR-01, CR-02, IN-01 + IN-02)
- Fixed: 3
- Skipped: 0

## Fixed Issues

### CR-01: Preflight warn-and-drop does not actually drop attacks

**Files modified:** `src/rbspaper/pipeline/core.py`
**Commit:** 73ed510
**Applied fix:** Replaced the try/except warn-and-drop block in `_preflight_pipeline_config` with a direct `validate_attack_support()` call that raises `ValueError` on unsupported attacks. Removed the `valid_attacks` list and the `if attack_config in valid_attacks` membership test. Query budget check now runs unconditionally after strict validation.

### CR-02: Duplicate _logger variable

**Files modified:** `src/rbspaper/pipeline/core.py`
**Commit:** c6df598
**Applied fix:** Removed `_logger = logging.getLogger(__name__)` at line 54. Only the pre-existing `logger = logging.getLogger(__name__)` remains.

### IN-01 + IN-02: Unused imports in preflight test; test updated for strict validation

**Files modified:** `test/test_preflight_compat.py`
**Commit:** 02f3351
**Applied fix:** Removed unused `AttackMethod` import and unused `pytest` import. Renamed `test_prefilter_warns_on_unsupported_attack` to `test_preflight_raises_on_unsupported_attack` and updated it to verify that `ValueError` is raised for unsupported attacks instead of checking for logged warnings.

## Verification

- `uv run ruff check` — passed on both modified files
- `uv run ruff format --check` — passed (files already formatted)
- `uv run pytest test/test_preflight_compat.py -v` — 2 tests passed
- `uv run pytest -q` — all 111 tests passed

## Skipped Issues

None.

---

_Fixed: 2026-05-07T12:35:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
