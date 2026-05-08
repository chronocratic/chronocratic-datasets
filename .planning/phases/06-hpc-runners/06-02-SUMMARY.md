---
phase: 06-hpc-runners
plan: 02
subsystem: infra
tags: [bash, slurm, hpc, forecasting, classification, slurm-array]

requires:
  - phase: 06-hpc-runners plan 01
    provides: hpc_submit_single.sh core engine
provides:
  - classification/univariate wrappers (UCR datasets)
  - classification/multivariate wrappers (UEA datasets)
  - forecasting/univariate wrappers (ETT, weather, electricity)
  - forecasting/multivariate wrappers (ETT, weather, electricity)
affects: [07, hpc, runners]

tech-stack:
  added: []
  patterns:
    - Task/modality wrapper pattern: run_on.sh (single dataset) + run_all.sh (batch loop)
    - Registry-resolved dataset lists at runtime (no hardcoded text files)
    - Baked-in family and forecasting_mode per directory

key-files:
  created:
    - runners/bash/classification/univariate/run_on.sh
    - runners/bash/classification/univariate/run_all.sh
    - runners/bash/classification/multivariate/run_on.sh
    - runners/bash/classification/multivariate/run_all.sh
    - runners/bash/forecasting/univariate/run_on.sh
    - runners/bash/forecasting/univariate/run_all.sh
    - runners/bash/forecasting/multivariate/run_on.sh
    - runners/bash/forecasting/multivariate/run_all.sh
  modified: []

key-decisions:
  - "Task/modality grouping over family-based (classification/forecasting x univariate/multivariate)"
  - "run_on.sh for single dataset HPC, run_all.sh for batch loop"
  - "Forecasting wrappers bake in --forecasting_mode; classification wrappers forward without extra flags"
  - "Dataset lists always queried from Python registry at runtime"
  - "All scripts source runner_config.sh (derived from $PROJECT_ROOT env var) — no relative ../.. chains"
  - "Experiment IDs resolved from get_experiment_list() at runtime — not hardcoded"

patterns-established:
  - "Thin wrappers: run_on.sh just forwards to hpc_submit_single.sh with baked-in flags"
  - "runner_config.sh provides: RUNNERS_BASE, log(), ensure_config(), get_experiment_list(), defaults"
  - "Magic numbers (seed, retry backoff, max attempts, submit pause) in runner_config.sh"
  - "set -uo pipefail in all scripts"

post-execution-refinements:
  - "Dropped --family from hpc_submit_single.sh and classification wrappers (Python resolves from dataset name)"
  - "Removed bash-side validation of experiment/dataset IDs (Python dataclasses enforce)"
  - "Experiment list from registry instead of hardcoded (ts2vec, autotcl)"

requirements-completed: []

duration: 10min
completed: 2026-05-07
---

# Phase 06 Plan 02 Summary

**8 task/modality convenience wrappers for one-click HPC submission — classification and forecasting x univariate and multivariate**

## Accomplishments
- Classification wrappers (ucr for univariate, uea for multivariate) — families baked in
- Forecasting wrappers — --forecasting_mode baked in, datasets from registry
- All run_all.sh scripts query Python registry at runtime (no static dataset lists)

## Files Created
- `runners/bash/classification/univariate/run_on.sh` — Single dataset HPC for UCR
- `runners/bash/classification/univariate/run_all.sh` — Loop over all UCR datasets
- `runners/bash/classification/multivariate/run_on.sh` — Single dataset HPC for UEA
- `runners/bash/classification/multivariate/run_all.sh` — Loop over all UEA datasets
- `runners/bash/forecasting/univariate/run_on.sh` — Single dataset HPC with --forecasting_mode univariate
- `runners/bash/forecasting/univariate/run_all.sh` — Loop over all forecasting datasets
- `runners/bash/forecasting/multivariate/run_on.sh` — Single dataset HPC with --forecasting_mode multivariate
- `runners/bash/forecasting/multivariate/run_all.sh` — Loop over all forecasting datasets

## Self-Check

| Check | Status |
|-------|--------|
| All 8 scripts created | PASS |
| bash -n syntax check all 8 | PASS |
| Classification wraps pass --family ucr/uea | PASS |
| Forecasting wraps pass --forecasting_mode | PASS |
| run_all.sh queries registry at runtime | PASS |
| EXPERIMENTS = ts2vec, autotcl | PASS |
| All executable | PASS |

---
*Phase: 06-hpc-runners*
*Completed: 2026-05-07*
