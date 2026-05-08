---
phase: 05-local-test-runners
fixed_at: "2026-05-07T15:00:00Z"
review_path: .planning/phases/05-local-test-runners/05-REVIEW.md
iteration: 1
findings_in_scope: 8
fixed: 7
skipped: 1
status: all_fixed
---

# Phase 05: Code Review Fix Report

**Fixed at:** 2026-05-07T15:00:00Z
**Source review:** .planning/phases/05-local-test-runners/05-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 8 (CR-01, CR-02, WR-01 through WR-05, IN-01)
- Fixed: 7
- Skipped: 1 (IN-02: pre-existing pattern, out of scope)

## Fixed Issues

### CR-01: local_batch.sh always uses --dataset_index, breaking name-based lookups

**Files modified:** `runners/bash/local_batch.sh`
**Commit:** 137bfa6
**Applied fix:** Replaced hardcoded `--dataset_index` in the execution loop with `--dataset_name` by resolving each index to its dataset name via `uv run python -c "from src.rbspaper.data.data_setup import get_all_datasets; ..."` before each run. Added validation that all expanded indices are non-negative integers within the registry range (0 to TOTAL_DATASETS-1) before entering the loop. Updated the summary report to display dataset names instead of indices.

### CR-02: Missing option values cause unbound variable crash (set -u)

**Files modified:** `runners/bash/local_single.sh`, `runners/bash/local_batch.sh`
**Commit:** 4a2c45c
**Applied fix:** Added `$# -lt 2` guards with clear error messages before every `$2` reference in the argument parser case arms. Covered options: `--seed`, `--max_epochs`, `--data_root`, `--output_dir` in both scripts, plus `--fraction` in local_batch.sh.

### WR-01: config.sh.example lost SEED option in 05-03 (regression)

**Files modified:** None
**Commit:** f6d0da5
**Applied fix:** Verified that `config.sh.example` already contains the SEED line (`# SEED=42` at line 19). This was fixed during the merge process. No code changes required.

### WR-02: expand_dataset_spec does not validate reverse ranges

**Files modified:** `runners/bash/local_batch.sh`
**Commit:** 742ae7c
**Applied fix:** Added validation in the range case of `expand_dataset_spec()` that checks `start_val` does not exceed `end_val`. When a reverse range like `5-2` is detected, the function emits a clear error message and returns 1, rather than silently producing empty output from `seq`.

### WR-03: apply_fraction spawns raw python without uv

**Files modified:** `runners/bash/local_batch.sh`
**Commit:** b402dc0
**Applied fix:** Changed `python -c` to `uv run python -c` in the `apply_fraction()` function to respect the project's uv-based environment management, ensuring the correct Python interpreter is used.

### WR-04: --fraction accepts values outside [0, 1] without validation

**Files modified:** `runners/bash/local_batch.sh`
**Commit:** 03f9f7c
**Applied fix:** Added fraction range validation after argument parsing using `uv run python -c "print('yes' if 0 <= FRACTION <= 1 else 'no')"` that rejects values outside [0, 1] with a clear error message.

### WR-05: --list_experiments uses different logging config than normal path

**Files modified:** `runners/py/runner.py`
**Commit:** 8369e75
**Applied fix:** Replaced `print()` calls in the `--list_experiments` code path with `logger.info()` using `logging.basicConfig()` with a format that matches the `setup_logging()` configuration. Converted f-string log message to lazy `%` formatting per ruff G004 rule.

### IN-01: Unnecessary `from __future__ import annotations` in runner.py

**Files modified:** `runners/py/runner.py`
**Commit:** 7b43d1d
**Applied fix:** Removed the `from __future__ import annotations` import, which is redundant in Python 3.12 where PEP 563 (postponed annotation evaluation) is the default behavior.

## Skipped Issues

### IN-02: Repeated ArgumentParser instantiation in _resolve_dataset

**File:** `runners/py/runner.py:309, 314, 320`
**Reason:** Pre-existing pattern not introduced by Phase 05. Fixing would require restructuring `_resolve_dataset` to accept a parser reference from `_parse_args`, which is too invasive for an informational finding. The pattern is documented and does not cause runtime issues.

## Verification

- **Shell scripts:** `bash -n` passed for both `local_single.sh` and `local_batch.sh`
- **Python:** `ruff check runners/py/runner.py` passed with no errors
- **Test suite:** `uv run pytest -x -q --tb=short` — all 111 tests passed

---

_Fixed: 2026-05-07T15:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
