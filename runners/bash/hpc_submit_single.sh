#!/usr/bin/env bash
#
# hpc_submit_single.sh — HPC single dataset runner.
#
# Generate and submit SLURM single jobs for one specific dataset.
# Useful for forecasting datasets mixed with classification, or ad-hoc runs.
#
# Usage:
#   ./runners/bash/hpc_submit_single.sh <experiment_id> <dataset_name> [options]
#
# Options:
#   --forecasting_mode <mode> Forecasting mode: univariate or multivariate
#   --seed <N>                Random seed (default: 42)
#   --num_runs <N>            Number of seed runs (default: 1)
#   --force                   Ignore existing checkpoint state
#   --dry_run                 Generate SLURM scripts without submitting
#   --delete_job_files        Delete job scripts after submission
#

set -uo pipefail

SCRIPT_ENTRY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_ENTRY/runner_config.sh"

# ======== Config Files ========
ensure_config "$RUNNERS_BASE/config.sh" "$RUNNERS_BASE/config.sh.example" "DATA_ROOT"
ensure_config "$RUNNERS_BASE/hpc_config.sh" "$RUNNERS_BASE/hpc_config.sh.example" "HPC_OUTPUT_ROOT"

# ======== Argument Parsing ========
if [ $# -lt 2 ]; then
  echo "Usage: $0 <experiment_id> <dataset_name> [options]" >&2
  echo "" >&2
  echo "Options:" >&2
  echo "  --forecasting_mode <mode> Forecasting mode: univariate or multivariate" >&2
  echo "  --seed <N>                Random seed (default: $DEFAULT_SEED)" >&2
  echo "  --num_runs <N>            Number of seed runs (default: $DEFAULT_NUM_RUNS)" >&2
  echo "  --force                   Ignore existing checkpoint state" >&2
  echo "  --dry_run                 Generate SLURM scripts without submitting" >&2
  echo "  --delete_job_files        Delete job scripts after submission" >&2
  exit 1
fi

EXP_ID="$1"
DATASET_NAME="$2"
shift 2

SEED="${SEED:-$DEFAULT_SEED}"
NUM_RUNS="$DEFAULT_NUM_RUNS"
FORCE="false"
DRY_RUN="false"
DELETE_JOBS="${DELETE_JOB_FILES:-false}"
FORECASTING_MODE=""

while [ $# -gt 0 ]; do
  case "$1" in
    --forecasting_mode)
      FORECASTING_MODE="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --num_runs)
      NUM_RUNS="$2"; shift 2 ;;
    --force)
      FORCE="true"; shift ;;
    --dry_run)
      DRY_RUN="true"; shift ;;
    --delete_job_files)
      DELETE_JOBS="true"; shift ;;
    *)
      log "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ======== Validate FORECASTING_MODE ========
if [ -n "$FORECASTING_MODE" ]; then
  case "$FORECASTING_MODE" in
    univariate|multivariate) ;;
    *)
      log "ERROR: --forecasting_mode must be 'univariate' or 'multivariate'. Got: '$FORECASTING_MODE'" >&2
      exit 1 ;;
  esac
fi

# ======== Validation ========
if [ -z "$DATA_ROOT" ]; then
  log "ERROR: DATA_ROOT is not set. Configure it in $RUNNERS_BASE/config.sh and re-run." >&2
  exit 1
fi

if [ -z "$HPC_OUTPUT_ROOT" ]; then
  log "ERROR: HPC_OUTPUT_ROOT is not set. Configure it in $RUNNERS_BASE/hpc_config.sh and re-run." >&2
  exit 1
fi

# ======== Ensure Output Directories ========
mkdir -p "$HPC_OUTPUT_ROOT/jobs"

# ======== Build Arguments ========
FORCE_ARG=""
if [ "$FORCE" = "true" ]; then
  FORCE_ARG=" --force"
fi

FORECASTING_MODE_ARG=""
if [ -n "$FORECASTING_MODE" ]; then
  FORECASTING_MODE_ARG=" --forecasting_mode ${FORECASTING_MODE}"
fi

# ======== Per-Seed Submission Loop ========
SUBMITTED=0
FAILED=0
JOB_FILES=()

for s in $(seq 0 "$((NUM_RUNS - 1))"); do
  run_seed=$((SEED + s))

  log "--- Seed run $((s + 1))/$NUM_RUNS (seed=$run_seed) ---"

  job_file="${HPC_OUTPUT_ROOT}/jobs/hpc_${EXP_ID}_${DATASET_NAME}_s${run_seed}.sh"

  mkdir -p "${HPC_OUTPUT_ROOT}/logs"
  cat > "$job_file" <<EOL
#!/bin/bash
set -euo pipefail
#SBATCH --qos=${HPC_QOS}
#SBATCH --job-name=hpc_${EXP_ID}_${DATASET_NAME}_s${run_seed}
#SBATCH --no-requeue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${HPC_CPUS_PER_TASK}
#SBATCH --mem-per-cpu=${HPC_MEM_PER_CPU}
#SBATCH --partition=${HPC_PARTITION}
#SBATCH --time=${HPC_TIME}
#SBATCH --output=${HPC_OUTPUT_ROOT}/logs/%x_%A.out
#SBATCH --error=${HPC_OUTPUT_ROOT}/logs/%x_%A.err

export PYTHONPATH="\${PROJECT_ROOT}:\${PYTHONPATH:-}"

echo "Running experiment ${EXP_ID} on dataset ${DATASET_NAME} (seed=${run_seed})"

uv run python "\${PROJECT_ROOT}/runners/py/runner.py" \
    --experiment_id="${EXP_ID}" \
    --dataset_name="${DATASET_NAME}" \
    --data_root="${DATA_ROOT}" \
    --output_dir="\${HPC_OUTPUT_ROOT}" \
    --seed=${run_seed}${FORCE_ARG}${FORECASTING_MODE_ARG}

EOL

  chmod 755 "$job_file"

  if [ "$DRY_RUN" = "true" ]; then
    log "[DRY RUN] Job file generated: $job_file"
    JOB_FILES+=("$job_file")
    SUBMITTED=$((SUBMITTED + 1))
    continue
  fi

  # ======== QoS Retry Loop ========
  attempt=0
  submit_success=false
  local_backoff="$HPC_RETRY_BACKOFF"

  while [ "$attempt" -lt "$HPC_RETRY_MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    log "  Submission attempt $attempt/$HPC_RETRY_MAX_ATTEMPTS for seed $run_seed..."

    output=$(sbatch "$job_file" 2>&1)
    ret_value=$?

    if [ "$ret_value" -eq 0 ]; then
      log "  Job submitted successfully: $output"
      submit_success=true
      break
    elif echo "$output" | grep -qE 'QOSMaxSubmitJobPerUserLimit|Job violates accounting/QOS policy'; then
      if [ "$attempt" -lt "$HPC_RETRY_MAX_ATTEMPTS" ]; then
        log "  QoS limit reached. Retrying in ${local_backoff}s..."
        sleep "$local_backoff"
        local_backoff=$((local_backoff * 2))
      fi
    else
      log "  ERROR: Failed to submit job for unexpected reason." >&2
      log "  Output: $output" >&2
      break
    fi
  done

  if [ "$submit_success" = "true" ]; then
    SUBMITTED=$((SUBMITTED + 1))
  else
    FAILED=$((FAILED + 1))
    log "  FAILED to submit job for seed $run_seed after $attempt attempts." >&2
  fi

  if [ "$submit_success" = "true" ] && [ "$DELETE_JOBS" = "true" ]; then
    rm -f "$job_file"
    log "  Deleted job file: $job_file"
  else
    JOB_FILES+=("$job_file")
  fi

  sleep "$HPC_SUBMIT_PAUSE"
done

# ======== Submission Report ========
log ""
log "========================================"
log "  HPC SINGLE DATASET SUBMISSION REPORT"
log "========================================"
log "Experiment:    $EXP_ID"
log "Dataset:       $DATASET_NAME"
log "Seed range:    $SEED..$((SEED + NUM_RUNS - 1))"
log "Forecasting:   ${FORECASTING_MODE:-N/A}"
log "Submitted:     $SUBMITTED"
log "Failed:        $FAILED"
log "Dry run:       $DRY_RUN"
log "========================================"

if [ "${#JOB_FILES[@]}" -gt 0 ]; then
  log "Job files:"
  for jf in "${JOB_FILES[@]}"; do
    log "  $jf"
  done
fi

log "========================================"

if [ "$FAILED" -gt 0 ]; then
  log "Submission completed with $FAILED failure(s)." >&2
  exit 1
fi

log "All $SUBMITTED job(s) submitted successfully."
exit 0
