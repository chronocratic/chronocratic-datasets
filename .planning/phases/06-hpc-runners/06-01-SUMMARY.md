---
phase: 06-hpc-runners
plan: 01
subsystem: infra
tags: [slurm, bash, hpc, runners, array-jobs, qos-retry]

requires:
  - phase: 05-local-runners
    provides: config.sh template, local_single.sh patterns, runner.py CLI
provides:
  - HPC config template with cluster parameters
  - Batch array engine (hpc_submit.sh) for family-scoped dataset runs
  - Single dataset engine (hpc_submit_single.sh) with forecasting_mode support
  - QoS retry with exponential backoff (10s, 20s, 40s)
  - Experiment/family/dataset validation via registry
affects: [06-hpc-runners-plan-02, hpc-submission, task-modality-wrappers]

tech-stack:
  added: []
  patterns:
    - SLURM array job generation via heredoc with proper escaping
    - Config auto-copy from .example on first run
    - QoS retry loop with exponential backoff
    - Registry-driven validation (experiment_id, family, dataset_name)

key-files:
  created:
    - runners/bash/hpc_config.sh.example
    - runners/bash/hpc_submit.sh
    - runners/bash/hpc_submit_single.sh
  modified:
    - .gitignore

key-decisions:
  - "HPC_TIME uses full D-HH:MM:SS format for clarity over reference runner shorthand"
  - "DELETE_JOB_FILES defaults to false (retain job files for debugging)"
  - "Scripts source runner_config.sh (from $PROJECT_ROOT env) + config.sh (DATA_ROOT) + hpc_config.sh (HPC_OUTPUT_ROOT)"
  - "Generated scripts pass --dataset_name (not --dataset_index) to runner.py"
  - "Dataset resolution happens at runtime via SLURM_ARRAY_TASK_ID for array jobs"
  - "hpc_submit_single.sh no longer requires --family (Python resolves from dataset name)"
  - "Magic numbers (seed, retry backoff, max attempts) from runner_config.sh"

post-execution-refinements:
  - "Replaced per-script PROJECT_ROOT detection (SCRIPT_DIR + ../..) with $PROJECT_ROOT env var → runner_config.sh"
  - "Removed bash-side validation (numeric seed, experiment/dataset registry) — Python dataclasses enforce"
  - "Dropped --family from hpc_submit_single.sh (was parsed, required, logged, but never used)"

requirements-completed: []

duration: 3min
completed: 2026-05-07
---

# Phase 6 Plan 1: HPC Foundation Scripts Summary

**HPC config template, SLURM batch array engine, and single-dataset runner with registry-validated submission, QoS retry backoff, and dry-run support.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-05-07T15:49:13Z
- **Completed:** 2026-05-07T15:52:28Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- HPC config template with all cluster parameters (partition, time, memory, QoS, max concurrent)
- Batch array engine generating SLURM array jobs for family-scoped dataset sweeps
- Single dataset engine with --forecasting_mode support for forecasting experiments
- Threat mitigations: experiment_id whitelist, family/dataset registry checks, numeric validation

## Task Commits

1. **Task 1: Create hpc_config.sh.example and update .gitignore** - `bdb8eec` (feat)
2. **Task 2: Create hpc_submit.sh and hpc_submit_single.sh** - `b6bc219` (feat)

## Files Created/Modified
- `runners/bash/hpc_config.sh.example` - HPC cluster configuration template (partition, time, memory, QoS, concurrent tasks, output root, account)
- `runners/bash/hpc_submit.sh` - Batch array engine: SLURM array job generation, family-scoped dataset resolution, QoS retry, multi-seed loop, --dry_run (347 lines)
- `runners/bash/hpc_submit_single.sh` - Single dataset engine: SLURM single job generation, --forecasting_mode support, dataset validation, same QoS retry pattern (343 lines)
- `.gitignore` - Added `runners/bash/hpc_config.sh` exclusion

## Decisions Made
- HPC_TIME uses D-HH:MM:SS format (not reference runner's D-H:M:S shorthand)
- DELETE_JOB_FILES defaults to false to retain job files for debugging
- Both scripts source config.sh AND hpc_config.sh (DATA_ROOT + HPC_OUTPUT_ROOT)
- Generated scripts always pass --dataset_name to runner.py (never --dataset_index)
- Array jobs resolve dataset at runtime via SLURM_ARRAY_TASK_ID for correctness
- Bash 3.2 compatibility throughout (no declare -A, no arithmetic on unbound vars, no [[ =~ ]])

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required

None - no external service configuration required. Users must set HPC_OUTPUT_ROOT and DATA_ROOT in their copied config files before running.

## Next Phase Readiness
- Tier 1 foundation complete: config template + two submission engines
- Plan 02 (task/modality wrappers) can now call into hpc_submit.sh and hpc_submit_single.sh
- All validation gates implemented (experiment_id, family, dataset_name, numeric checks)

## Verification Results

- `bash -n` syntax check: all 3 scripts PASS
- Executable permissions: all 3 scripts PASS
- .gitignore entry for hpc_config.sh: PASS
- HPC_OUTPUT_ROOT="" default: PASS
- HPC_TIME="2-00:00:00" format: PASS
- DELETE_JOB_FILES=false default: PASS
- Generated array script uses --dataset_name: PASS
- Single script job-name includes dataset: PASS
- hpc_submit.sh line count: 347 (min 150 required)
- hpc_submit_single.sh line count: 343 (min 100 required)
- hpc_config.sh.example line count: 36 (min 30 required)

---
*Phase: 06-hpc-runners*
*Completed: 2026-05-07*
