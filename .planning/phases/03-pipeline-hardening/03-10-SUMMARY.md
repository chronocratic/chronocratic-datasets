---
phase: 03-pipeline-hardening
plan: 10
subsystem: pipeline
tags: [force-flag, checkpoint-reset, resume-override]

requires:
  - phase: 03-pipeline-hardening
    provides: "resume gates (03-08), state builder (03-08)"
provides:
  - "force parameter on run_experiment_pipeline for fresh-start override"
affects: [03-pipeline-hardening, hpc-runners]

tech-stack:
  added: []
  patterns: ["three-way state init (force/resume/fresh)"]

key-files:
  created: []
  modified:
    - src/rbspaper/pipeline/core.py

key-decisions:
  - "force=True writes empty state immediately to mark fresh start, preventing stale checkpoint reads"
  - "config_hash fallback to run_name[:8] preserved for backward compatibility"

patterns-established:
  - "if force → elif previous_state → else fresh: three-branch state initialization"

requirements-completed: [REQ-04]

duration: 5min
completed: 2026-05-06
---

# Phase 3 Plan 10: Force Parameter Summary

**`force: bool = False` parameter on `run_experiment_pipeline` with three-way state initialization (force resets, previous_state resumes, fresh start)**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-06T13:34:40Z
- **Completed:** 2026-05-06T13:40:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- `force` parameter added to `run_experiment_pipeline` signature with default `False`
- Three-way state logic: `force=True` resets and writes empty state, `previous_state` resumes, neither creates fresh builder
- Docstring updated with force parameter documentation

## Task Commits

1. **Task 1: Add force parameter and state reset logic** - `4a31c96` (feat)

## Files Created/Modified
- `src/rbspaper/pipeline/core.py` - Force parameter added to signature, state init block replaced with three-way logic

## Decisions Made
- `force=True` immediately writes an empty state file via `save_pipeline_state` to prevent stale checkpoint reads during the fresh run
- Config hash uses `config_hash or config.artifacts.run_name[:8]` pattern consistently across force and fresh branches for backward compatibility

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Force parameter is in place for the runner to wire `--force` CLI flag
- Ready for Wave 3 completion or next plan

## Self-Check: PASSED

---
*Phase: 03-pipeline-hardening*
*Completed: 2026-05-06*
