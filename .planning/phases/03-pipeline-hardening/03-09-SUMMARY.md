---
phase: 03-pipeline-hardening
plan: 09
subsystem: pipeline
tags: [tenacity, retry, backoff, gpu-oom, resilience]

requires:
  - phase: 03-pipeline-hardening
    provides: "tenacity dependency (03-04), resume gates (03-08)"
provides:
  - "retry_step tenacity decorator for transient failure recovery"
  - "Retry-wrapped train and shared_attacks pipeline steps"
affects: [03-pipeline-hardening, hpc-runners]

tech-stack:
  added: []
  patterns: ["tenacity retry decorator", "exponential backoff", "try/except RetryError gating"]

key-files:
  created: []
  modified:
    - src/rbspaper/pipeline/core.py
    - test/test_pipeline_core.py

key-decisions:
  - "Removed reraise=True from retry config: with before_sleep_log, reraise=True causes original exception instead of RetryError, breaking except RetryError handlers"
  - "Used logger.exception in RetryError handlers (ruff TRY400 compliance)"

patterns-established:
  - "retry_step(fn)(args) wrapping pattern for inline retry on pipeline step calls"
  - "mark_complete only after successful retry completion, never after failure"

requirements-completed: [REQ-03]

duration: 15min
completed: 2026-05-06
---

# Phase 3 Plan 09: Tenacity Retry Decorator Summary

**tenacity retry_step decorator with exponential backoff (15-120s) wrapping train and shared_attacks pipeline steps for GPU OOM recovery**

## Performance

- **Duration:** 15 min
- **Started:** 2026-05-06T00:00:00Z
- **Completed:** 2026-05-06T00:15:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- retry_step decorator configured: 3 attempts, exponential backoff (15s, 30s, 60s), retries only RuntimeError/MemoryError
- Train gate wrapped: `retry_step(_train_model)` with RetryError catch/re-raise
- Shared attacks gate wrapped: `retry_step(_generate_attacked_inputs)` per attack with RetryError catch/re-raise
- 3 TDD tests: retry count verification, success-on-retry, non-retry of ValueError

## Task Commits

Each task was committed atomically:

1. **Task 1: Create retry_step tenacity decorator (TDD)** - `391a8c1` (test) + `52c39e0` (feat)
2. **Task 2: Apply retry to train and shared_attacks steps** - `c0f65c4` (feat)

## Files Created/Modified
- `src/rbspaper/pipeline/core.py` - tenacity imports, retry_step decorator, retry-wrapped train and shared_attacks gates
- `test/test_pipeline_core.py` - 3 retry_step tests (retry count, success on retry, no retry for ValueError)

## Decisions Made
- Removed `reraise=True` from retry config: when combined with `before_sleep_log`, tenacity re-raises the original RuntimeError instead of RetryError, which broke the `except RetryError` handlers in the pipeline gates. Without `reraise=True`, RetryError is properly raised after 3 exhausted attempts.
- Used `logger.exception` in RetryError handlers for TRY400 ruff compliance (logging inside except blocks).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed reraise=True from retry config**
- **Found during:** Task 1 (TDD GREEN phase)
- **Issue:** Plan specified `reraise=True`, but with `before_sleep_log`, tenacity re-raises the original RuntimeError instead of RetryError. This broke the `except RetryError` handlers required in Task 2.
- **Fix:** Removed `reraise=True`. The default behavior (without reraise) properly raises RetryError after exhausting 3 attempts.
- **Files modified:** src/rbspaper/pipeline/core.py
- **Verification:** test_retry_step_retries_runtime_error_three_times confirms RetryError is raised after 3 calls
- **Commit:** 52c39e0 (Task 1 feat commit)

**2. [Rule 1 - Bug] Used logger.exception instead of logger.error**
- **Found during:** Task 2 (ruff TRY400 check)
- **Issue:** Plan specified `logger.error` inside `except RetryError` blocks, but ruff TRY400 requires `logger.exception` in exception handlers.
- **Fix:** Changed to `logger.exception` which also includes the traceback in the log output.
- **Files modified:** src/rbspaper/pipeline/core.py
- **Verification:** `uv run ruff check src/rbspaper/pipeline/core.py` passes clean
- **Commit:** c0f65c4 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - bugs discovered during implementation)
**Impact on plan:** Both fixes required for correct operation. No scope creep.

## Issues Encountered
None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Retry infrastructure is in place for train and shared_attacks steps
- Encoding, attacks, evaluate, and analysis gates do not yet have retry wrapping (by design — train and attacks are the GPU-bound steps most likely to hit OOM)
- Ready for Wave 3 completion or next phase planning

---
*Phase: 03-pipeline-hardening*
*Completed: 2026-05-06*
