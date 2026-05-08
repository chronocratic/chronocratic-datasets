---
status: passed
phase: 06-hpc-runners
source: [06-01-SUMMARY.md, 06-02-SUMMARY.md]
started: "2026-05-07T18:40:00Z"
updated: "2026-05-07T18:48:00Z"
---

## Tests

### 1. runner_config.sh fails clearly without RBS_PROJECT_ROOT
expected: Script fails immediately with clear error message when RBS_PROJECT_ROOT is not defined.
result: **PASS** — exits with "Please set RBS_PROJECT_ROOT environment variable"

### 2. runner_config.sh sources correctly with RBS_PROJECT_ROOT set
expected: Script sets RUNNERS_BASE, exports PYTHONPATH, defines log(), get_experiment_list(), ensure_config() without errors.
result: **PASS** — all variables and functions verified

### 3. local_single.sh --dry_run assembles correct runner.py command
expected: Command includes --experiment_id, --dataset_name, --data_root, --output_dir, --seed with absolute path to runner.py.
result: **PASS** — all flags present in assembled command

### 4. hpc_submit_single.sh generates SLURM script with set -euo pipefail
expected: Generated SLURM script contains error handling (set -euo pipefail) and dataset resolution guards.
result: **PASS** — `set -euo pipefail` confirmed in generated job file

### 5. hpc_submit.sh --dry_run generates array job with correct family scope
expected: SLURM array directive uses family-scoped dataset count, output goes to logs/ directory.
result: **PASS** — `#SBATCH --array=0-2%128` for 3 UCR datasets, `set -euo pipefail` present

### 6. Classification wrappers forward without --family flag
expected: run_on.sh scripts call hpc_submit_single.sh without --family arg (Python resolves it).
result: **PASS** — zero occurrences of `--family` in either classification wrapper

### 7. Forecasting wrappers pass --forecasting_mode correctly
expected: univariate wrapper passes --forecasting_mode univariate, multivariate passes --forecasting_mode multivariate.
result: **PASS** — univariate/run_on.sh passes `--forecasting_mode univariate`, multivariate/run_on.sh passes `--forecasting_mode multivariate`

### 8. run_all.sh resolves experiment list from registry
expected: No hardcoded experiment IDs. Experiments resolved via Python registry at runtime.
result: **PASS** — zero occurrences of `ts2vec` or `autotcl` in any run_all.sh

### 9. Retry backoff is scoped per seed iteration
expected: Backoff variable is local to each retry loop, doesn't leak across seeds.
result: **PASS** — `local_backoff` variable used inside per-seed for loop in both hpc_submit.sh and hpc_submit_single.sh

### 10. All scripts pass bash -n syntax check
expected: No syntax errors in any of the runner scripts.
result: **PASS** — all 14 scripts pass cleanly

## Summary

total: 10
passed: 10
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none]
