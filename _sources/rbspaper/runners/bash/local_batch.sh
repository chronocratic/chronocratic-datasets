#!/usr/bin/env bash
#
# local_batch.sh — Run a single experiment on multiple datasets sequentially.
#
# Usage:
#   ./runners/bash/local_batch.sh <experiment_id> <dataset_spec> [options]
#
# dataset_spec:
#   0-20           Range of dataset indices
#   0,3,7,10       Comma-separated list of indices
#   all            All registered datasets
#
# Options:
#   --seed <N>           Random seed (default: 42)
#   --force              Ignore existing checkpoint state
#   --max_epochs <N>     Override max training epochs
#   --data_root <path>   Override DATA_ROOT from config.sh
#   --output_dir <path>  Override output directory (default: outputs)
#   --dry_run            Assemble config and print summary without running
#   --fraction <0-1>     Sample a fraction of datasets (e.g. 0.25 = first ~25%)
#

set -uo pipefail

SCRIPT_ENTRY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_ENTRY/runner_config.sh"

# ======== Config Files ========
ensure_config "$RUNNERS_BASE/config.sh" "$RUNNERS_BASE/config.sh.example" "DATA_ROOT"

# ======== Dataset Spec Expansion ========
expand_dataset_spec() {
  local spec="$1"
  local total="$2"

  if [ "$spec" = "all" ]; then
    seq 0 "$((total - 1))"
    return
  fi

  if echo "$spec" | grep -q ','; then
    echo "$spec" | tr ',' '\n'
    return
  fi

  if echo "$spec" | grep -q '-'; then
    local start_val="${spec%-*}"
    local end_val="${spec#*-}"
    if ! echo "$start_val" | grep -qE '^[0-9]+$' || ! echo "$end_val" | grep -qE '^[0-9]+$'; then
      echo "ERROR: Range values must be integers. Got: '$spec'" >&2
      return 1
    fi
    if [ "$start_val" -gt "$end_val" ]; then
      echo "ERROR: Invalid range '$spec' — start ($start_val) exceeds end ($end_val)" >&2
      return 1
    fi
    seq "$start_val" "$end_val"
    return
  fi

  if ! echo "$spec" | grep -qE '^[0-9]+$'; then
    echo "ERROR: Invalid dataset spec: '$spec'" >&2
    return 1
  fi
  echo "$spec"
}

apply_fraction() {
  local fraction="$1"
  shift
  local indices=("$@")
  local count=${#indices[@]}

  if [ "$count" -eq 0 ]; then
    return
  fi

  local target
  target=$(uv run python -c "
import math, sys
count = int(sys.argv[1])
fraction = float(sys.argv[2])
print(max(1, math.floor(count * fraction)))
" "$count" "$fraction")

  local i=0
  for idx in "${indices[@]}"; do
    if [ "$i" -ge "$target" ]; then
      break
    fi
    echo "$idx"
    i=$((i + 1))
  done
}

# ======== Argument Parsing ========
if [ $# -lt 2 ]; then
  echo "Usage: $0 <experiment_id> <dataset_spec> [options]" >&2
  echo "" >&2
  echo "dataset_spec: 0-20 (range), 0,3,7 (list), all (all datasets)" >&2
  echo "" >&2
  echo "Options:" >&2
  echo "  --seed <N>           Random seed (default: $DEFAULT_SEED)" >&2
  echo "  --force              Ignore existing checkpoint state" >&2
  echo "  --max_epochs <N>     Override max training epochs" >&2
  echo "  --data_root <path>   Override DATA_ROOT from config.sh" >&2
  echo "  --output_dir <path>  Override output directory" >&2
  echo "  --dry_run            Assemble config and print summary without running" >&2
  echo "  --fraction <0-1>     Sample a fraction of datasets" >&2
  exit 1
fi

EXP_ID="$1"
DATASET_SPEC="$2"
shift 2

SEED="$DEFAULT_SEED"
FORCE="false"
MAX_EPOCHS=""
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
DATA_ROOT_OVERRIDE=""
DRY_RUN="false"
FRACTION=""

while [ $# -gt 0 ]; do
  case "$1" in
    --seed)
      SEED="$2"; shift 2 ;;
    --force)
      FORCE="true"; shift ;;
    --max_epochs)
      MAX_EPOCHS="$2"; shift 2 ;;
    --data_root)
      DATA_ROOT_OVERRIDE="$2"; shift 2 ;;
    --output_dir)
      OUTPUT_DIR="$2"; shift 2 ;;
    --dry_run)
      DRY_RUN="true"; shift ;;
    --fraction)
      FRACTION="$2"; shift 2 ;;
    *)
      log "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ======== Validation ========
if [ -n "$DATA_ROOT_OVERRIDE" ]; then
  DATA_ROOT="$DATA_ROOT_OVERRIDE"
fi

if [ -z "$DATA_ROOT" ]; then
  log "ERROR: DATA_ROOT is not set. Configure it in $RUNNERS_BASE/config.sh or pass --data_root." >&2
  exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
  log "ERROR: DATA_ROOT='$DATA_ROOT' does not exist." >&2
  exit 1
fi

# ======== Resolve Dataset List ========
cd "$PROJECT_ROOT" || exit 1

# Pre-resolve total count and index-to-name mapping in a single Python call
declare -A INDEX_TO_NAME
TOTAL_DATASETS=0
while IFS='|' read -r idx name; do
  INDEX_TO_NAME[$idx]="$name"
  TOTAL_DATASETS=$((idx + 1))
done < <(uv run python -c "
from src.rbspaper.data.data_setup import get_all_datasets
datasets = get_all_datasets(form='list')
for i, name in enumerate(datasets):
    print(f'{i}|{name}')
") || exit 1

INDICES=()
while IFS= read -r line; do
  INDICES+=("$line")
done < <(expand_dataset_spec "$DATASET_SPEC" "$TOTAL_DATASETS")

if [ -n "$FRACTION" ]; then
  original_count=${#INDICES[@]}
  INDICES=()
  while IFS= read -r line; do
    INDICES+=("$line")
  done < <(apply_fraction "$FRACTION" "${INDICES[@]}")
  log "Fraction $FRACTION applied: running on ${#INDICES[@]} datasets (of $original_count expanded from '$DATASET_SPEC')"
fi

if [ ${#INDICES[@]} -eq 0 ]; then
  log "ERROR: No datasets to run. Spec='$DATASET_SPEC', Total=$TOTAL_DATASETS" >&2
  exit 1
fi

log "Batch run: experiment='$EXP_ID', datasets=${INDICES[*]}, total=${#INDICES[@]}"

# ======== Sequential Execution Loop ========
total_runs=0
passed=0
failed=0
RESULTS=()

for dataset_index in "${INDICES[@]}"; do
  total_runs=$((total_runs + 1))

  DATASET_NAME="${INDEX_TO_NAME[$dataset_index]}"

  log "=== Run $total_runs/${#INDICES[@]}: $EXP_ID on dataset '$DATASET_NAME' (index $dataset_index) ==="

  CMD=(uv run python runners/py/runner.py
    "--experiment_id" "$EXP_ID"
    "--dataset_name" "$DATASET_NAME"
    "--data_root" "$DATA_ROOT"
    "--output_dir" "$OUTPUT_DIR"
    "--seed" "$SEED"
  )

  if [ "$FORCE" = "true" ]; then
    CMD+=("--force")
  fi
  if [ -n "$MAX_EPOCHS" ]; then
    CMD+=("--max_epochs" "$MAX_EPOCHS")
  fi
  if [ "$DRY_RUN" = "true" ]; then
    CMD+=("--dry_run")
  fi

  "${CMD[@]}"
  run_exit=$?

  if [ "$run_exit" -eq 0 ]; then
    passed=$((passed + 1))
    RESULTS+=("$DATASET_NAME|PASS")
  else
    failed=$((failed + 1))
    RESULTS+=("$DATASET_NAME|FAIL ($run_exit)")
  fi
done

# ======== Aggregate Report ========
log ""
log "========================================"
log "        BATCH RUN SUMMARY"
log "========================================"
log "Experiment: $EXP_ID"
log "Total:      $total_runs"
log "Passed:     $passed"
log "Failed:     $failed"
log "----------------------------------------"
log "Dataset     | Result"
log "----------------------------------------"

for entry in "${RESULTS[@]}"; do
  dataset="${entry%%|*}"
  result="${entry##*|}"
  log "  $dataset  | $result"
done

log "========================================"

if [ "$failed" -gt 0 ]; then
  log "Batch completed with $failed failure(s)."
  exit 1
fi

log "Batch completed — all $total_runs runs passed."
exit 0
