---
phase: 05-local-test-runners
verified: 2026-05-07T14:30:00Z
status: gaps_found
score: 7/8 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 2/8
  gaps_closed:
    - "runners/__init__.py exists and enables package resolution"
    - "runner.py has no print() calls (all converted to logging)"
    - "local_single.sh exists, is executable, and runs single experiments"
    - ".gitignore excludes config.sh"
  gaps_remaining:
    - "config.sh.example has unresolved git merge conflict markers"
  regressions: []
gaps:
  - truth: "config.sh.example is a valid bash template (free of merge conflicts)"
    status: failed
    reason: "Unresolved git merge conflict markers (<<<<<<< HEAD, =======, >>>>>>> 417a127) in config.sh.example. bash -n syntax check fails with exit code 2. When auto-copied to config.sh by local_single.sh or local_batch.sh, the resulting file will also be invalid and will fail when sourced."
    artifacts:
      - path: "runners/bash/config.sh.example"
        issue: "Lines 1, 17, 31 contain merge conflict markers. File fails bash -n syntax check."
    missing:
      - "Resolve merge conflict in config.sh.example — keep one version of the content (recommended: the planned version from lines 18-30, which includes Classification/Forecasting path docs and SEED default)"
human_verification:
  - test: "Run local_single.sh ts2vec 0 with valid DATA_ROOT"
    expected: "Script detects project root, sources config.sh, runs experiment on dataset 0, reports success/failure"
    why_human: "Requires actual dataset files and DATA_ROOT configuration to exercise the full flow"
  - test: "Run local_batch.sh ts2vec 0-2 with valid DATA_ROOT"
    expected: "Script expands dataset spec to [0, 1, 2], runs 3 experiments sequentially, prints BATCH RUN SUMMARY table"
    why_human: "Requires actual dataset files and DATA_ROOT configuration to exercise the full flow"
---

# Phase 5: Local Test Runners Verification Report

**Phase Goal:** Local validation before HPC — Single experiment runs end-to-end locally
**Verified:** 2026-05-07T14:30:00Z
**Status:** gaps_found
**Re-verification:** Yes — after gap closure (all 3 plans now executed and merged)

## Execution Context

Phase 05 had 3 planned waves. The previous verification (2026-05-07T14:20:00Z) found that only Plan 03 was executed, leaving 4 gaps. All 3 plans have now been executed and merged. This re-verification confirms gap closure and checks for regressions.

## Goal Achievement

### Observable Truths

Derived from ROADMAP.md Phase 5 Success Criteria (6 items), Plan must_haves (Plans 01, 02, 03), and deliverables:

| #   | Truth                                                               | Status     | Evidence                                                                 |
|-----|---------------------------------------------------------------------|------------|--------------------------------------------------------------------------|
| 1   | runners/__init__.py exists for package resolution                   | ✓ VERIFIED | File exists (52 bytes), valid Python, docstring-only, import resolves    |
| 2   | runner.py has no print() calls (all logging)                        | ✓ VERIFIED | AST check: zero print() calls. _log_summary exists. Module logger at line 49. basicConfig on 3 exception paths. |
| 3   | local_single.sh runs single experiment from any directory           | ✓ VERIFIED | File exists (159 lines), executable, BASH_SOURCE-based root detection, forwards to runner.py |
| 4   | local_batch.sh runs multiple datasets sequentially with report      | ✓ VERIFIED | File exists (280 lines), executable, expand_dataset_spec, BATCH RUN SUMMARY table |
| 5   | Config file auto-created from template on first run                 | ✓ VERIFIED | Both scripts implement auto-copy from config.sh.example when config.sh missing |
| 6   | Dataset spec expansion handles range/list/all                       | ✓ VERIFIED | expand_dataset_spec() handles all three formats                          |
| 7   | --fraction flag samples subset of datasets                          | ✓ VERIFIED | apply_fraction() exists, uses python -c for float math (Bash 3.2 compatible) |
| 8   | config.sh.example is a valid bash template                          | ✗ FAILED   | Unresolved merge conflict markers. bash -n fails with exit code 2.       |

**Score:** 7/8 truths verified

### Previously Failed Items (Gap Closure)

All 4 gaps from the previous verification have been addressed:

1. **`runners/__init__.py`** — CLOSED. File created with docstring `"""Entry point package for the rbspaper-run CLI."""`. Import test confirms resolution.

2. **`runner.py` print() to logging** — CLOSED. All 15 print() calls converted. Module-level logger added. `_print_summary` renamed to `_log_summary` with 8 logger.info() calls using `%s` formatting. `logging.basicConfig()` present on 3 paths that skip `setup_logging()`.

3. **`local_single.sh`** — CLOSED. 159-line Bash 3.2-compatible script created. Project root detection via BASH_SOURCE, config management, argument forwarding via CMD array, DATA_ROOT validation, numeric dataset detection.

4. **`.gitignore` excludes config.sh** — CLOSED. Entry `runners/bash/config.sh` confirmed present.

### New Gap (Not in Previous Verification)

**`config.sh.example` has unresolved git merge conflict markers.** The file contains `<<<<<<< HEAD` (line 1), `=======` (line 17), `>>>>>>> 417a127` (line 31). This was not flagged in the previous verification because the file existed but was not read for content quality. The merge conflict causes `bash -n` syntax check to fail. Since both `local_single.sh` and `local_batch.sh` auto-copy this template to `config.sh` on first run, the resulting config file would be invalid.

### Required Artifacts

| Artifact | Expected    | Status | Details |
|----------|-------------|--------|---------|
| `runners/__init__.py` | Package marker with docstring | ✓ VERIFIED | 52 bytes, single-line docstring, valid Python |
| `runners/py/runner.py` | CLI runner with logging output | ✓ VERIFIED | Module logger, _log_summary, zero print() calls, basicConfig on exception paths |
| `runners/bash/config.sh.example` | Template config with DATA_ROOT | ⚠️ FLAWED | File exists but has unresolved merge conflict markers (lines 1, 17, 31). bash -n fails. |
| `.gitignore` | Excludes `runners/bash/config.sh` | ✓ VERIFIED | Entry confirmed present |
| `runners/bash/local_single.sh` | Single experiment runner (80+ lines) | ✓ VERIFIED | 159 lines, executable, syntax-clean, all key patterns present |
| `runners/bash/local_batch.sh` | Batch runner (150+ lines) | ✓ VERIFIED | 280 lines, executable, syntax-clean, expand_dataset_spec, apply_fraction, BATCH RUN SUMMARY |

### Key Link Verification

| From | To | Via | Status | Details |
|------|---|-----|--------|---------|
| `pyproject.toml` → `runners.py.runner:main` | Python package resolution | `import runners.py.runner` | ✓ VERIFIED | Import test succeeds |
| `local_single.sh` → `runner.py` | `uv run python runners/py/runner.py` | CMD array | ✓ VERIFIED | Line 126: `"uv" "run" "python" "$PROJECT_ROOT/runners/py/runner.py"` |
| `local_single.sh` → `config.sh` | `source $CONFIG_FILE` | shell source | ✓ VERIFIED | Line 47: `source "$CONFIG_FILE"` |
| `local_batch.sh` → `runner.py` | `uv run python runners/py/runner.py` | CMD array | ✓ VERIFIED | Line 222: `CMD=(uv run python runners/py/runner.py ...)` |
| `local_batch.sh` → `config.sh` | `source $CONFIG_FILE` | shell source | ✓ VERIFIED | Line 53: `source "$CONFIG_FILE"` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `local_batch.sh` | `TOTAL_DATASETS` | `uv run python -c "from src.rbspaper.data.data_setup import get_all_datasets; print(len(...))"` | ✓ Real registry query | ✓ FLOWING |
| `local_batch.sh` | `INDICES` | `expand_dataset_spec()` → `seq`/`tr`/`grep` | ✓ POSIX utilities | ✓ FLOWING |
| `local_single.sh` | `CMD` | Array construction from args | ✓ Real arguments | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| local_single.sh syntax valid | `bash -n local_single.sh` | Exit 0 | ✓ PASS |
| local_single.sh executable | `test -x local_single.sh` | true | ✓ PASS |
| local_batch.sh syntax valid | `bash -n local_batch.sh` | Exit 0 | ✓ PASS |
| local_batch.sh executable | `test -x local_batch.sh` | true | ✓ PASS |
| config.sh.example syntax valid | `bash -n config.sh.example` | Exit 2 | ✗ FAIL |
| No Bash 4+ features in single | `grep -c "declare -A\|(( " local_single.sh` | 0 matches | ✓ PASS |
| No Bash 4+ features in batch | `grep -c "declare -A\|(( " local_batch.sh` | 0 matches | ✓ PASS |
| runner.py has _log_summary | `grep "_log_summary" runner.py` | Found at lines 331, 420, 423 | ✓ PASS |
| runner.py has module logger | `grep "logger = logging.getLogger" runner.py` | Found at line 49 | ✓ PASS |
| runner.py zero print() calls | AST-based check | No print() in executable code | ✓ PASS |
| runners/__init__.py valid Python | `python -c "import ast; ast.parse(...)"` | Valid | ✓ PASS |
| Package import resolves | `python -c "import runners.py.runner"` | Success | ✓ PASS |
| ruff check runner.py | `uv run ruff check runner.py` | All checks passed | ✓ PASS |
| Full test suite | `uv run pytest` | 111 passed | ✓ PASS |

### Requirements Coverage

All 3 plans had empty `requirements:` fields — no specific requirement IDs were mapped to Phase 5 plans. Phase 5's scope was defined by ROADMAP.md success criteria and deliverables.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `runners/bash/config.sh.example` | 1, 17, 31 | Unresolved git merge conflict markers | 🛑 Blocker | File fails `bash -n`. Auto-copied config.sh will be invalid. |
| `runners/bash/local_batch.sh` | 95 | `python -c` for float math in `apply_fraction()` | ℹ️ Info | Uses system python, not `uv run`. Acceptable for simple arithmetic. |

### Human Verification Required

1. **Test local_single.sh end-to-end**
   - **Test:** Run `./runners/bash/local_single.sh ts2vec 0 --data_root /path/to/data`
   - **Expected:** Script detects project root, sources config.sh, validates DATA_ROOT, runs experiment on dataset 0, reports success/failure
   - **Why human:** Requires valid DATA_ROOT with actual dataset files; cannot be verified with static analysis

2. **Test local_batch.sh end-to-end**
   - **Test:** Run `./runners/bash/local_batch.sh ts2vec 0-2 --data_root /path/to/data`
   - **Expected:** Script expands dataset spec to [0, 1, 2], runs 3 experiments sequentially, prints BATCH RUN SUMMARY table with pass/fail per dataset
   - **Why human:** Requires valid DATA_ROOT with actual dataset files; cannot be verified with static analysis

### Gaps Summary

Phase 05 is substantially complete: all 3 plans were executed and merged, closing all 4 gaps from the previous verification. The `runners/__init__.py` package marker is in place, `runner.py` has been fully converted from `print()` to structured logging, and `local_single.sh` provides the convenient bash wrapper for single-experiment runs.

One blocker remains: `config.sh.example` contains unresolved git merge conflict markers (`<<<<<<< HEAD`, `=======`, `>>>>>>> 417a127`) that cause `bash -n` to fail. Since both `local_single.sh` and `local_batch.sh` auto-copy this template to `config.sh` on first run, a broken template propagates to every new user's first run. The recommended resolution is to keep the planned version (lines 18-30), which includes Classification/Forecasting path documentation and a SEED default.

### Deferred Items

The previous verification had `print() to logging conversion` deferred to Phase 7. This deferral is **no longer applicable** — Plan 01 executed the conversion as part of Phase 5. Phase 7's `NOTE-print-to-logging.md` should be updated or removed.

---

_Verified: 2026-05-07T14:30:00Z_
_Verifier: Claude (gsd-verifier)_
