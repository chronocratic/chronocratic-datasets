#!/usr/bin/env bash
#
# run_all.sh — Run all experiments on classification/multivariate datasets via HPC.
#
# Queries the Python registry at runtime for both experiment and dataset lists.
#
# Usage:
#   RBS_PROJECT_ROOT=/path/to/repo ./runners/bash/classification/multivariate/run_all.sh [options]
#
# Options (forwarded to run_on.sh per experiment):
#   --seed <N>           Random seed (default: 42)
#   --num_runs <N>       Number of seed runs (default: 1)
#   --force              Ignore existing checkpoint state
#   --dry_run            Generate SLURM script without submitting
#

set -uo pipefail

source "$RBS_PROJECT_ROOT/runners/bash/runner_config.sh"

# Resolve classification-compatible experiments from registry
EXPERIMENTS=()
while IFS= read -r line; do
  EXPERIMENTS+=("$line")
done < <(cd "$PROJECT_ROOT" && uv run python -c "
from experiment_instances.instances import list_experiment_ids, EXPERIMENTS_REGISTRY
for eid in list_experiment_ids():
    inst = EXPERIMENTS_REGISTRY[eid]
    if 'classification' in inst.downstream_tasks:
        print(eid)
") || {
  echo "[ERROR] Failed to resolve experiment list from registry" >&2
  exit 1
}

if [ "${#EXPERIMENTS[@]}" -eq 0 ]; then
  echo "[ERROR] No classification experiments found in registry" >&2
  exit 1
fi

# Query registry for UEA dataset list
DATASETS=()
while IFS= read -r line; do
  DATASETS+=("$line")
done < <(cd "$PROJECT_ROOT" && uv run python -c "
from src.rbspaper.data.data_setup import get_datasets_names
for name in get_datasets_names('uea', form='list'):
    print(name)
")

if [ "${#DATASETS[@]}" -eq 0 ]; then
  echo "[ERROR] No classification/multivariate datasets found in registry" >&2
  exit 1
fi

total_submitted=0
total_failed=0
for exp_id in "${EXPERIMENTS[@]}"; do
  log "========================================"
  log "Experiment: $exp_id"
  log "========================================"
  for dataset_name in "${DATASETS[@]}"; do
    log "Submitting: $exp_id on $dataset_name"
    if bash "$RUNNERS_BASE/classification/multivariate/run_on.sh" "$exp_id" "$dataset_name" "$@"; then
      total_submitted=$((total_submitted + 1))
    else
      total_failed=$((total_failed + 1))
    fi
  done
done

log "Submitted: $total_submitted, Failed: $total_failed"
if [ "$total_failed" -gt 0 ]; then
  exit 1
fi
