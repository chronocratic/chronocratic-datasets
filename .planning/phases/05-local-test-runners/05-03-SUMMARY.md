---
phase: 05-local-test-runners
plan: 03
subsystem: infra
tags: [bash, local-testing, batch-runner, dataset-expansion, smoke-tests]

# Dependency graph
requires:
  - phase: 05-local-test-runners
    provides: runner.py (Python entry point), config.sh.example pattern
provides:
  - Batch experiment bash runner with dataset expansion and aggregate reporting
  - Config file template for DATA_ROOT management
affects: [05-local-test-runners, 06-hpc-runners]

# Tech tracking
tech-stack:
  added: []
  patterns: [bash-array-command-construction, dataset-spec-expansion, fraction-sampling]

key-files:
  created:
    - runners/bash/local_batch.sh
    - runners/bash/config.sh.example
  modified: []

key-decisions:
  - "Deterministic first-N fraction sampling (no seed needed for quick smoke tests)"
  - "Array-based CMD construction for safe argument passing (no eval, no string interpolation)"
  - "Bash 3.2 compatible: python -c for float arithmetic, no declare -A or (( )) syntax"
  - "Config.sh auto-creation from template blocks until DATA_ROOT is set"

patterns-established:
  - "Config template: commit config.sh.example, gitignore config.sh, auto-copy on first run"
  - "Dataset spec: range (0-20), list (0,3,7), all — expanded by POSIX utilities"
  - "Aggregate report: table format with index | PASS/FAIL columns, exit 1 on any failure"

requirements-completed: []

# Metrics
duration: 2min
completed: 2026-05-07
---

# Phase 5 Plan 3: Local Batch Runner Summary

**Sequential batch bash runner with flexible dataset spec expansion (range/list/all), fraction sampling, and aggregate pass/fail reporting for local smoke testing**

## Performance

- **Duration:** 2 min
- **Started:** 2026-05-07T12:11:29Z
- **Completed:** 2026-05-07T12:13:14Z
- **Tasks:** 1/1
- **Files modified:** 2

## Accomplishments
- Created `local_batch.sh` for running one experiment across multiple datasets sequentially
- Dataset spec expansion supporting range (`0-20`), list (`0,3,7`), and `all` keywords
- Fraction sampling (`--fraction 0.25`) for quick smoke tests on subsets
- Aggregate pass/fail report with per-run index and exit code tracking
- Config file template (`config.sh.example`) with DATA_ROOT validation

## Task Commits

Each task was committed atomically:

1. **Task 1: Create local_batch.sh with dataset expansion and aggregate reporting** - `e1cc18c` (feat)

## Files Created/Modified
- `runners/bash/local_batch.sh` — Batch runner script (280 lines, executable). Expands dataset specs, runs sequential loop collecting exit codes, prints aggregate summary table.
- `runners/bash/config.sh.example` — Template config file. Copied to `config.sh` on first run; requires DATA_ROOT to be set before proceeding.

## Decisions Made
- Fraction sampling uses deterministic first-N approach (no random seed needed), matching CONTEXT.md discretion guidance
- CMD array constructed with individual elements for shell safety; first line consolidated for grep pattern matching
- All `cd` commands carry `|| exit 1` guards (no `set -e` used, following reference patterns)
- Config template includes only DATA_ROOT (required) and OUTPUT_DIR (optional) to keep it minimal

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- CMD array construction spread across multiple lines did not match the verification grep pattern `uv run python.*runner.py`. Consolidated first CMD line into a single-line array element while preserving safe argument separation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- `local_batch.sh` is ready for testing with real experiments
- Requires `config.sh` to be created with a valid `DATA_ROOT` before use
- Depends on `runner.py` and `local_single.sh` patterns for full batch workflow

## Threat Surface Assessment
- T-05-06 (Injection via dataset spec): Mitigated — `expand_dataset_spec` uses `grep -q` for pattern detection, `seq` for range generation, `tr` for list splitting. No `eval` or string interpolation of untrusted input.
- T-05-07 (Injection via loop command): Mitigated — CMD array construction ensures proper quoting per argument.
- T-05-08 (DoS via fraction edge case): Mitigated — Zero-dataset check before loop entry. `apply_fraction` uses `max(1, ...)` preventing empty results.

---
*Phase: 05-local-test-runners*
*Completed: 2026-05-07*
