---
phase: 03-pipeline-hardening
plan: 06
subsystem: pipeline
tags: [experiment-config, metadata, audit-trail, json-persistence]

requires:
  - phase: 03-02
    provides: _save_json helper and PipelineArtifactConfig with persist_artifacts guard
  - phase: 03-05
    provides: run_dir construction and pipeline core structure
provides:
  - _write_experiment_config function persisting experiment metadata before pipeline steps
  - experiment_config.json at run_dir with model_name, seed, downstream_tasks, attack_names
affects: [03-07, 03-08, runner, output-structure, debugging]

tech-stack:
  added: []
  patterns: [pre-execution metadata persistence, keyword-only helper composition]

key-files:
  created: []
  modified:
    - src/rbspaper/pipeline/core.py
    - test/test_pipeline_core.py

key-decisions:
  - "Placed _write_experiment_config immediately after _prepare_run_directory for call-site locality"
  - "Config JSON gated behind persist_artifacts flag to match directory creation semantics"
  - "Model name uses getattr with class name fallback for models lacking model_name attribute"

patterns-established:
  - "experiment_config.json written before any pipeline step (training, encoding, attacks)"

requirements-completed: [REQ-06]

duration: 2min
completed: 2026-05-06
---

# Phase 3 Plan 06: Experiment Config Persistence Summary

**_write_experiment_config function persisting experiment_config.json before any pipeline step runs, containing model_name, seed, downstream_tasks, attack_names, trainer_kwargs, attack_scope, encoding_batch_size**

## Performance

- **Duration:** 2 min
- **Started:** 2026-05-06T12:11:58Z
- **Completed:** 2026-05-06T12:13:25Z
- **Tasks:** 2 (TDD + wire)
- **Files modified:** 2

## Accomplishments
- _write_experiment_config helper writing JSON with full pipeline metadata
- Function wired into _prepare_run_directory behind persist_artifacts guard
- 3 tests covering required keys, file location, and model name fallback
- Smoke test updated to assert experiment_config.json existence

## Task Commits

Each task was committed atomically:

1. **Task 1: Create _write_experiment_config helper** - TDD flow
   - `649d772` (test) — 3 failing tests for _write_experiment_config
   - `4ffd88c` (feat) — _write_experiment_config implementation + tests passing
2. **Task 2: Wire into _prepare_run_directory** - Direct implementation
   - `b2837fb` (feat) — Call _write_experiment_config after mkdir, smoke test assertion
   - `d3ce1ff` (style) — Fix import ordering (ruff I001)

## Files Created/Modified
- `src/rbspaper/pipeline/core.py` — Added _write_experiment_config after _prepare_run_directory (lines 190-213); wired call from _prepare_run_directory
- `test/test_pipeline_core.py` — Added 3 tests for experiment_config persistence; updated smoke test assertion

## Decisions Made
- Placed `_write_experiment_config` immediately after `_prepare_run_directory` to keep call-site local
- Config JSON gated behind `persist_artifacts` flag so it only writes when the run directory is created
- Model name extracted via `getattr(config.model, 'model_name', type(config.model).__name__)` for graceful fallback

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- experiment_config.json is persisted before any pipeline steps execute
- Survives crashes, provides full audit trail of run configuration
- Plans 03-07+ can read experiment_config.json for debugging and reproducibility

## Self-Check: PASSED

---
*Phase: 03-pipeline-hardening*
*Completed: 2026-05-06*
