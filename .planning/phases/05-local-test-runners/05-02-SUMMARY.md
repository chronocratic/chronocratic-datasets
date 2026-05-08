---
phase: 05-local-test-runners
plan: 02
subsystem: infra
tags: [bash, shell, runner, config-template, uv]

requires:
  - phase: 03-pipeline-hardening
    provides: runners/py/runner.py CLI interface with argparse flags
provides:
  - config.sh.example template for local runner configuration
  - .gitignore entry excluding config.sh (T-05-05 mitigation)
  - local_single.sh for single-experiment local runs
affects:
  - 05-03-local-batch-runner
  - 06-hpc-runners

tech-stack:
  added: []
  patterns:
    - Bash 3.2 compatible scripts (macOS default)
    - Array-based command construction for injection safety
    - Config auto-creation from template on first run

key-files:
  created:
    - runners/bash/config.sh.example
    - runners/bash/local_single.sh
  modified:
    - .gitignore

key-decisions:
  - "Bash array CMD=() for argument forwarding prevents shell injection (T-05-03)"
  - "Config auto-creation blocks with DATA_ROOT validation, not silent defaults"
  - "Bash 3.2 compatibility: no declare -A, no (( )), no [[ =~ ]]"

requirements-completed: []

metrics:
  duration: 3min
  completed: 2026-05-07
---

# Phase 5 Plan 02: Local Single Experiment Runner Summary

**Bash config template with DATA_ROOT validation and single-experiment runner forwarding to Python CLI via uv**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-05-07T12:05:58Z
- **Completed:** 2026-05-07T12:07:48Z
- **Tasks:** 2/2
- **Files modified:** 3

## Accomplishments
- Config template (`config.sh.example`) with documented path patterns for Classification and Forecasting datasets
- Git-ignored `config.sh` via `.gitignore` entry, preventing sensitive paths from being committed (T-05-05)
- Bash 3.2-compatible single-experiment runner with project root detection, auto-config creation, and safe argument forwarding

## Task Commits

1. **Task 1: Create config.sh.example and update .gitignore** - `3e74647` (feat)
2. **Task 2: Create local_single.sh** - `1fb56d4` (feat)

## Files Created/Modified
- `runners/bash/config.sh.example` — Template config with DATA_ROOT placeholder, Classification/Forecasting path docs
- `runners/bash/local_single.sh` — 159-line Bash 3.2-compatible single experiment runner
- `.gitignore` — Added `runners/bash/config.sh` exclusion

## Decisions Made
- Followed plan spec exactly for config.sh.example content (per A-11 config strategy)
- Bash array `CMD=()` used for all argument construction (T-05-03 injection mitigation)
- All `cd` commands carry `|| exit 1` guards for portability safety
- Numeric dataset detection via `grep -qE '^[0-9]+$'` per RESEARCH.md anti-pattern guidance

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## Known Stubs
- `config.sh.example` contains `DATA_ROOT=""` as an intentional template placeholder; users must set it before first real run

## User Setup Required

**DATA_ROOT configuration required before running experiments:**
1. After first `./runners/bash/local_single.sh` run, `config.sh` is auto-created from template
2. Edit `runners/bash/config.sh` and set `DATA_ROOT` to your dataset directory
3. Re-run the script

## Next Phase Readiness
- `local_single.sh` provides the foundation for `local_batch.sh` (plan 05-03)
- Config management pattern (template -> auto-copy -> validate) established for reuse

---

*Phase: 05-local-test-runners*
*Completed: 2026-05-07*
