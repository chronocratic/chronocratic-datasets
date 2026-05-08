---
phase: 06-hpc-runners
fixed_at: "2026-05-07T00:00:00Z"
review_path: .planning/phases/06-hpc-runners/06-REVIEW.md
iteration: 1
findings_in_scope: 9
fixed: 9
skipped: 0
status: all_fixed
---

# Phase 06: Code Review Fix Report

**Fixed at:** 2026-05-07
**Source review:** .planning/phases/06-hpc-runners/06-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 9 (4 Critical, 5 Warning)
- Fixed: 9
- Skipped: 0

## Fixed Issues

### CR-01: Generated SLURM scripts have no error handling — job failures are silent

**Files modified:** `runners/bash/hpc_submit.sh`, `runners/bash/hpc_submit_single.sh`
**Commit:** b185131
**Applied fix:** Added `set -euo pipefail` to generated SLURM scripts. For `hpc_submit.sh`, the dataset_name resolution now has bounds checking and an error handler via `|| { echo "FATAL: ..."; exit 1; }`. Also ensured the logs directory exists before generating job files with `mkdir -p "${HPC_OUTPUT_ROOT}/logs"`.

### CR-02: `run_all.sh` wrappers submit ALL experiments to specific families — no filtering

**Files modified:** `runners/bash/classification/univariate/run_all.sh`, `runners/bash/classification/multivariate/run_all.sh`, `runners/bash/forecasting/univariate/run_all.sh`, `runners/bash/forecasting/multivariate/run_all.sh`
**Commit:** b185131
**Applied fix:** Replaced `get_experiment_list()` with a Python call that filters experiments by their `downstream_tasks` field. Classification wrappers check for `'classification'` in `inst.downstream_tasks`; forecasting wrappers check for `'forecasting'`. The error message on empty results is now specific to the task type.

### CR-03: `get_experiment_list()` suppresses stderr — environment failures become undiagnosable

**Files modified:** `runners/bash/runner_config.sh`
**Commit:** b185131
**Applied fix:** Removed `2>/dev/null` from the `uv run python` command. Added a second error message to guide the user to check their uv installation. The docstring comment was also corrected to reflect newline-separated output.

### CR-04: Retry backoff state leaks across seed iterations in HPC submit scripts

**Files modified:** `runners/bash/hpc_submit.sh`, `runners/bash/hpc_submit_single.sh`
**Commit:** b185131
**Applied fix:** Introduced a `local_backoff` variable initialized to `"$HPC_RETRY_BACKOFF"` at the start of each seed iteration. The retry loop mutates `local_backoff` instead of the global `HPC_RETRY_BACKOFF`, so subsequent seeds start with the original configured backoff.

### WR-01: SLURM script `--output` uses bare filenames without centralized path

**Files modified:** `runners/bash/hpc_submit.sh`, `runners/bash/hpc_submit_single.sh`
**Commit:** b185131
**Applied fix:** Changed `#SBATCH --output` and `#SBATCH --error` to use centralized paths. Array jobs use `${HPC_OUTPUT_ROOT}/logs/%x_%a.out` (%x = job name, %a = array task ID). Single jobs use `${HPC_OUTPUT_ROOT}/logs/%x_%A.out` (%A = job array ID). The logs directory is created via `mkdir -p` before job file generation.

### WR-02: `local_batch.sh` spawns N Python subprocesses to resolve dataset names

**Files modified:** `runners/bash/local_batch.sh`
**Commit:** b185131
**Applied fix:** Pre-resolve the complete index-to-name mapping in a single Python call at startup, storing results in an associative array `INDEX_TO_NAME`. The per-iteration loop now uses the cached mapping instead of spawning a new Python process.

### WR-03: `FORECASTING_MODE_ARG` is subject to shell injection

**Files modified:** `runners/bash/hpc_submit_single.sh`
**Commit:** b185131
**Applied fix:** Added a case-based validation that restricts `FORECASTING_MODE` to the allowlist `univariate|multivariate`. Invalid values cause an immediate error and exit before any SLURM script is generated.

### WR-04: `expand_dataset_spec()` accepts non-integer input silently

**Files modified:** `runners/bash/local_batch.sh`
**Commit:** b185131
**Applied fix:** Added integer validation using `grep -qE '^[0-9]+$'` for range values (start and end) and single-value specs. Invalid input produces a clear error message and returns 1.

### WR-05: `run_all.sh` scripts do not propagate `run_on.sh` failure exit codes

**Files modified:** `runners/bash/classification/univariate/run_all.sh`, `runners/bash/classification/multivariate/run_all.sh`, `runners/bash/forecasting/univariate/run_all.sh`, `runners/bash/forecasting/multivariate/run_all.sh`
**Commit:** b185131
**Applied fix:** Added `total_submitted` and `total_failed` counters. Each `run_on.sh` call is wrapped in an `if` statement to track success/failure. Final log reports submitted/failed counts. Script exits with code 1 if any submissions failed.

## Skipped Issues

None — all in-scope findings were fixed.

---

_Info findings (IN-01 through IN-04) were not in scope for this fix pass._

_Fixed: 2026-05-07_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
