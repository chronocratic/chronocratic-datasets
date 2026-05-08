#!/usr/bin/env bash
#
# run_on.sh — Run an experiment on a single forecasting/univariate dataset via HPC.
#
# Usage:
#   RBS_PROJECT_ROOT=/path/to/repo ./runners/bash/forecasting/univariate/run_on.sh <experiment_id> <dataset_name> [options]
#
# Options (forwarded to hpc_submit_single.sh):
#   --seed <N>           Random seed (default: 42)
#   --num_runs <N>       Number of seed runs (default: 1)
#   --force              Ignore existing checkpoint state
#   --dry_run            Generate SLURM script without submitting
#

set -uo pipefail

source "$RBS_PROJECT_ROOT/runners/bash/runner_config.sh"

if [ $# -lt 2 ]; then
  echo "Usage: $0 <experiment_id> <dataset_name> [--seed N] [--num_runs N] [--force] [--dry_run]" >&2
  exit 1
fi

experiment_id="$1"
dataset_name="$2"
shift 2

bash "$RUNNERS_BASE/hpc_submit_single.sh" "$experiment_id" "$dataset_name" --forecasting_mode univariate "$@"
