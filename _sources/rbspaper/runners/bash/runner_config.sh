#!/usr/bin/env bash
#
# runner_config.sh — Shared configuration for all bash runners.
#
# Sourced by every runner script. Validates RBS_PROJECT_ROOT and derives
# PROJECT_ROOT and RUNNERS_BASE from it.
#
# Usage:
#   export RBS_PROJECT_ROOT=/path/to/repo
#   source "$RBS_PROJECT_ROOT/runners/bash/runner_config.sh"
#

# ======== Path Resolution ========
# Requires RBS_PROJECT_ROOT to be set before sourcing. Validates it points to a real project.
: "${RBS_PROJECT_ROOT:?Please set RBS_PROJECT_ROOT environment variable}"

if [ ! -f "$RBS_PROJECT_ROOT/pyproject.toml" ]; then
  echo "[ERROR] runner_config.sh: No pyproject.toml found at '$RBS_PROJECT_ROOT'" >&2
  echo "[ERROR] Please set RBS_PROJECT_ROOT to the correct project directory." >&2
  exit 1
fi

PROJECT_ROOT="$RBS_PROJECT_ROOT"
RUNNERS_BASE="${PROJECT_ROOT}/runners/bash"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ======== Logger ========
log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

# ======== Defaults (overridable via CLI args or config.sh) ========
DEFAULT_SEED="${SEED:-42}"
DEFAULT_NUM_RUNS=1
DEFAULT_OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

# HPC retry settings (overridable in hpc_config.sh)
HPC_RETRY_BACKOFF="${HPC_RETRY_BACKOFF:-10}"
HPC_RETRY_MAX_ATTEMPTS="${HPC_RETRY_MAX_ATTEMPTS:-3}"
HPC_SUBMIT_PAUSE="${HPC_SUBMIT_PAUSE:-0.5}"

# ======== Config Auto-Creation ========
ensure_config() {
  # $1 = config file path, $2 = example file path, $3 = variable name to check
  local config_file="$1"
  local config_example="$2"
  local check_var="$3"

  if [ ! -f "$config_file" ]; then
    cp "$config_example" "$config_file"
    log "Created $config_file from template."
    log "Please set $check_var in $config_file and re-run."
    exit 1
  fi

  # shellcheck source=/dev/null
  source "$config_file"
}

# ======== Registry Helpers ========
get_experiment_list() {
  # Returns newline-separated list of registered experiment IDs
  local ids
  ids=$(uv run python -c "
from experiment_instances.instances import list_experiment_ids
for eid in list_experiment_ids():
    print(eid)
") || {
    echo "[ERROR] Failed to resolve experiment list from registry" >&2
    echo "[ERROR] Ensure 'uv' is available and the environment is set up." >&2
    return 1
  }
  echo "$ids"
}
