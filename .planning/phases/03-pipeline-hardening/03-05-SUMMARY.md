---
phase: 03-pipeline-hardening
plan: 05
subsystem: pipeline
tags: [hierarchical-paths, run-name, config-hash, runner]

requires:
  - phase: 03-03
    provides: compute_config_hash for 8-char SHA-256 config drift detection
provides:
  - build_hierarchical_run_name function producing {experiment_id}/{short_hash}/seed_{seed}/{dataset_name}
  - Runner wiring: config hash computation + hierarchical run name construction
affects: [03-06, 03-07, 03-08, runner, output-structure]

tech-stack:
  added: []
  patterns: [hierarchical output paths, keyword-only function composition]

key-files:
  created:
    - test/test_hierarchical_run_name.py
  modified:
    - src/rbspaper/pipeline/config.py
    - runners/py/runner.py

key-decisions:
  - "Placed build_hierarchical_run_name under 'Run Identity' section in config.py"
  - "Runner computes hash before _build_run_name to decouple hash logic from naming"
  - "asdict import moved to module level to satisfy ruff PLC0415"

patterns-established:
  - "Keyword-only pure function for path segment construction"
  - "Factory delegation: _build_run_name delegates to build_hierarchical_run_name"

requirements-completed: [REQ-05]

duration: 8min
completed: 2026-05-06
---

# Phase 3 Plan 05: Hierarchical Run Name Construction Summary

**build_hierarchical_run_name producing {experiment_id}/{short_hash}/seed_{seed}/{dataset_name} paths, wired into runner with compute_config_hash**

## Performance

- **Duration:** 8 min
- **Started:** 2026-05-06T12:06:00Z
- **Completed:** 2026-05-06T12:14:00Z
- **Tasks:** 2 (+ 1 Rule 2 fix)
- **Files modified:** 3

## Accomplishments
- build_hierarchical_run_name function with deterministic path construction
- Runner computes config hash from model_params + seed before building run name
- _build_run_name replaced to accept short_hash and seed, delegating to hierarchical builder
- 4 tests covering format, seed variation, path compatibility, dataset variation

## Task Commits

Each task was committed atomically:

1. **Task 1: Add build_hierarchical_run_name to config.py** - TDD flow
   - `3b772b2` (test) — 4 failing tests for build_hierarchical_run_name
   - `63b3c7c` (feat) — build_hierarchical_run_name implementation + tests passing
2. **Task 2: Wire hierarchical run name into runner** - Direct implementation
   - `355f061` (feat) — Import wiring, _build_run_name rewrite, hash computation in main()
   - `0b57b3f` (fix) — Enum serialization for compute_config_hash (Rule 2)

## Files Created/Modified
- `src/rbspaper/pipeline/config.py` — Added build_hierarchical_run_name under 'Run Identity' section (lines 229-255)
- `runners/py/runner.py` — Imported build_hierarchical_run_name + compute_config_hash, replaced _build_run_name signature, added hash computation before run name building
- `test/test_hierarchical_run_name.py` — 4 new tests for hierarchical run name construction

## Decisions Made
- Placed `build_hierarchical_run_name` between `PipelineArtifactConfig` and `DataConfig` with a section header comment, keeping the file logically organized
- Moved `asdict` import to module level to satisfy ruff PLC0415 rule, avoiding inline function imports
- Runner computes hash from `asdict(experiment.model_params)` before the config assembly, ensuring the hash captures the raw parameter shape without runtime-resolved values

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added Enum serialization for compute_config_hash**
- **Found during:** Task 2 verification
- **Issue:** Model params contain `TS2VecAugmentationMode` enum values that `json.dumps` inside `compute_config_hash` cannot serialize, causing the runner to crash
- **Fix:** Added `_make_json_serializable` helper function that recursively converts Enum values to their string representations before passing to `compute_config_hash`
- **Files modified:** runners/py/runner.py
- **Verification:** Hash computation succeeds for ts2vec_fgsm experiment, ruff + ty clean
- **Committed in:** `0b57b3f`

---

**Total deviations:** 1 auto-fixed (Rule 2 - missing critical functionality)
**Impact on plan:** Essential fix for correctness — without it, the runner crashes when computing config hashes. No scope creep.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Hierarchical path construction complete and tested
- Runner produces deterministic output paths: `outputs/{exp_id}/{hash}/seed_{N}/{dataset}/`
- Plans 03-06+ can rely on `PipelineArtifactConfig.run_name` containing the hierarchical path
- Config hash is already computed in the runner; downstream plans can reference it

## Self-Check: PASSED

---
*Phase: 03-pipeline-hardening*
*Completed: 2026-05-06*
