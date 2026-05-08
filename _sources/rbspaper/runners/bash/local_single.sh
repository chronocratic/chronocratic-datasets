#!/usr/bin/env bash
#
# local_single.sh — Run a single experiment on a single dataset locally.
#
# Usage:
#   ./runners/bash/local_single.sh <experiment_id> <dataset_name_or_index> [options]
#
# Options:
#   --seed <N>           Random seed (default: 42)
#   --force              Ignore existing checkpoint state
#   --max_epochs <N>     Override max training epochs
#   --data_root <path>   Override DATA_ROOT from config.sh
#   --output_dir <path>  Override output directory (default: outputs)
#   --dry_run            Assemble config and print summary without running
#

set -uo pipefail

SCRIPT_ENTRY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_ENTRY/runner_config.sh"

# ======== Config Files ========
ensure_config "$RUNNERS_BASE/config.sh" "$RUNNERS_BASE/config.sh.example" "DATA_ROOT"

# ======== Argument Parsing ========
if [ $# -lt 2 ]; then
  echo "Usage: $0 <experiment_id> <dataset_name_or_index> [options]" >&2
  echo "" >&2
  echo "Options:" >&2
  echo "  --seed <N>           Random seed (default: $DEFAULT_SEED)" >&2
  echo "  --force              Ignore existing checkpoint state" >&2
  echo "  --max_epochs <N>     Override max training epochs" >&2
  echo "  --data_root <path>   Override DATA_ROOT from config.sh" >&2
  echo "  --output_dir <path>  Override output directory" >&2
  echo "  --dry_run            Assemble config and print summary without running" >&2
  exit 1
fi

EXP_ID="$1"
DATASET="$2"
shift 2

SEED="$DEFAULT_SEED"
FORCE="false"
MAX_EPOCHS=""
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
DATA_ROOT_OVERRIDE=""
DRY_RUN="false"

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
    *)
      log "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ======== Resolve DATA_ROOT ========
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

# ======== Build Command ========
CMD=(
  "uv" "run" "python" "$PROJECT_ROOT/runners/py/runner.py"
  "--experiment_id" "$EXP_ID"
  "--data_root" "$DATA_ROOT"
  "--output_dir" "$OUTPUT_DIR"
  "--seed" "$SEED"
)

# Dataset: index or name
if echo "$DATASET" | grep -qE '^[0-9]+$'; then
  CMD+=("--dataset_index" "$DATASET")
else
  CMD+=("--dataset_name" "$DATASET")
fi

# Optional overrides
[ "$FORCE" = "true" ] && CMD+=("--force")
[ -n "$MAX_EPOCHS" ] && CMD+=("--max_epochs" "$MAX_EPOCHS")
[ "$DRY_RUN" = "true" ] && CMD+=("--dry_run")

# ======== Execute ========
log "Running experiment '$EXP_ID' on dataset '$DATASET' (seed=$SEED)"
log "Command: ${CMD[*]}"

cd "$PROJECT_ROOT" || exit 1
"${CMD[@]}"
exit_code=$?

if [ "$exit_code" -eq 0 ]; then
  log "Experiment completed successfully."
else
  log "Experiment failed with exit code $exit_code." >&2
fi

exit "$exit_code"
