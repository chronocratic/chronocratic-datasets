---
phase: 03-pipeline-hardening
plan: 11
subsystem: runner
tags: [dataset-index, force-flag, argparse, checkpoint-resume, hpc-array]

requires:
  - phase: 03-pipeline-hardening
    provides: "force parameter on run_experiment_pipeline (03-10), get_all_datasets registry"
provides:
  - "--dataset_index CLI arg for HPC array job compatibility"
  - "--force CLI arg wired to pipeline fresh-start override"
  - "Mutual exclusivity validation for --dataset_index / --dataset_name"
  - "Automatic checkpoint resume when state file exists and not --force"
affects: [03-pipeline-hardening, hpc-runners, local-test-runners]

tech-stack:
  added: []
  patterns: ["_resolve_dataset() helper for arg validation + index resolution"]

key-files:
  created:
    - test/test_runner_cli_args.py
  modified:
    - runners/py/runner.py

key-decisions:
  - "Mutual exclusivity enforced in main() via _resolve_dataset() helper, not argparse add_mutually_exclusive_group, for clearer error messages"
  - "dataset_name variable replaces args.dataset_name throughout main() after resolution"
  - "Checkpoint state loaded before run_experiment_pipeline only when not --force"

patterns-established:
  - "Extract CLI validation into helper functions to keep main() cyclomatic complexity bounded"

requirements-completed: [REQ-07]

duration: 10min
completed: 2026-05-06
---

# Phase 3 Plan 11: Dataset Index and Force CLI Args Summary

**--dataset_index and --force CLI arguments with mutual exclusivity, index-to-name resolution, and automatic checkpoint resume wiring**

## Performance

- **Duration:** 10 min
- **Started:** 2026-05-06T16:31:21Z
- **Completed:** 2026-05-06T16:42:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- `--dataset_index` arg for HPC array job compatibility (D-08), resolved via `get_all_datasets(form='list')`
- `--force` arg wired to `run_experiment_pipeline()` for checkpoint override (D-10)
- Mutual exclusivity enforcement between `--dataset_index` and `--dataset_name`
- Automatic checkpoint resume: state file loaded when it exists and `--force` is not set
- Full test coverage: 8 tests for arg parsing, validation, and bounds checking

## Task Commits

1. **Task 1: Add --dataset_index and --force CLI args** - `e29341a` (test)
   - Added `--dataset_index` (type=int, default=None) and `--force` (store_true) to `_parse_args`
   - 4 parsing tests: index accepted, default None, force False/True

2. **Task 2: Wire dataset resolution and force into main()** - `ed58c48` (feat)
   - `_resolve_dataset()` helper for mutual exclusivity and index-to-name resolution
   - Bounds checking: negative and out-of-range indices rejected
   - State file auto-load before `run_experiment_pipeline()` when not `--force`
   - 4 additional tests: mutual exclusivity, missing arg, out-of-range, negative index

## Files Created/Modified
- `runners/py/runner.py` - New `--dataset_index`, `--force` args; `_resolve_dataset()` helper; state loading before pipeline call
- `test/test_runner_cli_args.py` - 8 tests covering arg parsing, mutual exclusivity, bounds validation

## Decisions Made
- Extracted validation into `_resolve_dataset()` to keep `main()` cyclomatic complexity within ruff C901 limits
- Used explicit type annotation (`list[str]`) for `get_all_datasets(form='list')` return to satisfy `ty` type narrowing
- Mutual exclusivity error uses `parser.error()` (SystemExit) for consistent argparse-style failure

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Runner is now fully HPC-array-job compatible with `--dataset_index`
- `--force` flag provides fresh-start override for checkpoint resume
- State loading is wired through the runner to the pipeline
- Ready for Phase 4 (Local Test Runners) and Phase 5 (HPC Runners)

## Self-Check: PASSED

---
*Phase: 03-pipeline-hardening*
*Completed: 2026-05-06*
