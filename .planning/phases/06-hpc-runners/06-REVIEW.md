---
phase: 06-hpc-runners
reviewed: 2026-05-07T00:00:00Z
depth: deep
files_reviewed: 15
files_reviewed_list:
  - runners/bash/runner_config.sh
  - runners/bash/config.sh.example
  - runners/bash/hpc_config.sh.example
  - runners/bash/local_single.sh
  - runners/bash/local_batch.sh
  - runners/bash/hpc_submit.sh
  - runners/bash/hpc_submit_single.sh
  - runners/bash/classification/univariate/run_on.sh
  - runners/bash/classification/univariate/run_all.sh
  - runners/bash/classification/multivariate/run_on.sh
  - runners/bash/classification/multivariate/run_all.sh
  - runners/bash/forecasting/univariate/run_on.sh
  - runners/bash/forecasting/univariate/run_all.sh
  - runners/bash/forecasting/multivariate/run_on.sh
  - runners/bash/forecasting/multivariate/run_all.sh
findings:
  critical: 4
  warning: 5
  info: 4
  total: 13
status: issues_found
---

# Phase 06: Code Review Report

**Reviewed:** 2026-05-07
**Depth:** deep (cross-file analysis, call chain tracing, import graph)
**Files Reviewed:** 15
**Status:** issues_found

## Summary

This review covers the HPC runner bash scripts: a central `runner_config.sh` shared library, two config templates, four core runner scripts (local_single, local_batch, hpc_submit, hpc_submit_single), and eight task/modality wrapper scripts (run_on.sh and run_all.sh across classification/univariate, classification/multivariate, forecasting/univariate, forecasting/multivariate).

The refactor from `../..` path chains to `$PROJECT_ROOT` is sound in principle, but several cross-cutting issues were found: generated SLURM scripts lack error handling, `run_all.sh` wrappers submit ALL registered experiments to specific dataset families without filtering (causing silent crashes on incompatible experiments), `get_experiment_list` suppresses stderr making environment failures undiagnosable, and retry backoff state leaks across seed iterations.

## Critical Issues

### CR-01: Generated SLURM scripts have no error handling — job failures are silent

**File:** `runners/bash/hpc_submit.sh:122-154` (heredoc generation)
**File:** `runners/bash/hpc_submit_single.sh:110-134` (heredoc generation)
**Issue:** The heredocs that generate SLURM scripts produce files with no `set -euo pipefail` or equivalent error handling. If `uv run python` fails (import error, missing dependency, crashed runner), the SLURM job exits silently with no indication of failure. For `hpc_submit.sh`, the embedded Python call that resolves `dataset_name` from `SLURM_ARRAY_TASK_ID` also runs without error guards — a failed resolution produces an empty `dataset_name` that gets passed to the runner, causing cryptic downstream errors.

This is particularly dangerous on HPC where jobs run unattended and diagnosing failures requires log output. The SLURM stdout/stderr files (`${SLURM_ARRAY_TASK_ID}.out`) would contain nothing useful.

**Fix:**
```bash
cat > "$job_file" <<EOL
#!/bin/bash
set -euo pipefail
#SBATCH --qos=${HPC_QOS}
...

dataset_name=\$(set -euo pipefail && uv run python -c "
import sys
from src.rbspaper.data.data_setup import get_datasets_names
datasets = get_datasets_names(sys.argv[1], form='list')
if int(sys.argv[2]) >= len(datasets):
    print(f'Index {sys.argv[2]} out of range for {len(datasets)} datasets', file=sys.stderr)
    sys.exit(1)
print(datasets[int(sys.argv[2])])
" "${FAMILY}" "\$SLURM_ARRAY_TASK_ID") || {
  echo "FATAL: Failed to resolve dataset name for array_id=\$SLURM_ARRAY_TASK_ID" >&2
  exit 1
}

uv run python "\${PROJECT_ROOT}/runners/py/runner.py" \
    --experiment_id="${EXP_ID}" \
    --dataset_name="\$dataset_name" \
    --data_root="${DATA_ROOT}" \
    --output_dir="\${HPC_OUTPUT_ROOT}" \
    --seed=${run_seed}${FORCE_ARG}

EOL
```

### CR-02: `run_all.sh` wrappers submit ALL experiments to specific families — no filtering

**File:** `runners/bash/classification/univariate/run_all.sh:23-56`
**File:** `runners/bash/classification/multivariate/run_all.sh:23-56`
**File:** `runners/bash/forecasting/univariate/run_all.sh:23-56`
**File:** `runners/bash/forecasting/multivariate/run_all.sh:23-56`
**Issue:** All four `run_all.sh` scripts call `get_experiment_list()` to retrieve ALL registered experiments, then iterate over every experiment against every dataset in the specific family (ucr/uea/forecasting). There is no filtering mechanism to exclude experiments incompatible with the dataset family.

For example, `classification/univariate/run_all.sh` will submit forecasting-only experiments to UCR classification datasets, causing Python-side crashes. The `run_on.sh` wrappers then forward to `hpc_submit_single.sh`, which generates and submits SLURM jobs that fail at runtime. Because `run_all.sh` does not check return codes of the `bash .../run_on.sh` calls, these failures are swallowed silently — the final log message says "All classification/univariate experiments submitted" even though many crashed.

With the current registry containing only `ts2vec` and `autotcl` (both classification experiments), this is latent. The moment a forecasting-specific experiment is registered, running `classification/univariate/run_all.sh` will submit broken jobs.

**Fix:** Implement experiment filtering at the wrapper level. Either:
1. Add a family-aware experiment list function: `get_experiment_list(family)` that returns only compatible experiments, or
2. Check the experiment's `downstream_tasks` field in the Python call and filter accordingly:

```bash
# Example: filter experiments by task compatibility
EXPERIMENTS=()
while IFS= read -r line; do
  EXPERIMENTS+=("$line")
done < <(cd "$PROJECT_ROOT" && uv run python -c "
from experiment_instances.instances import list_experiment_ids, EXPERIMENTS_REGISTRY
for eid in list_experiment_ids():
    inst = EXPERIMENTS_REGISTRY[eid]
    if 'classification' in inst.downstream_tasks:
        print(eid)
")
```

### CR-03: `get_experiment_list()` suppresses stderr — environment failures become undiagnosable

**File:** `runners/bash/runner_config.sh:60-64`
**Issue:** The function pipes stderr to `/dev/null` via `2>/dev/null`. This means:
- `uv` not being installed produces no diagnostic — just an empty experiment list
- Broken Python environment, missing `pyproject.toml`, or import errors are swallowed
- The fallback error message "Failed to resolve experiment list from registry" provides no actionable information

The callers (all four `run_all.sh` scripts) then report "No experiments found in registry", which is misleading when the real cause is a broken `uv` installation or corrupted virtual environment.

**Fix:** Remove the `2>/dev/null` redirection or gate it behind a verbose flag:
```bash
get_experiment_list() {
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
```

### CR-04: Retry backoff state leaks across seed iterations in HPC submit scripts

**File:** `runners/bash/hpc_submit.sh:169-186`
**File:** `runners/bash/hpc_submit_single.sh:149-165`
**Issue:** `HPC_RETRY_BACKOFF` is a global variable that is doubled inside the retry loop (line 184 in hpc_submit.sh: `HPC_RETRY_BACKOFF=$((HPC_RETRY_BACKOFF * 2))`). This mutation persists across the outer `for s in $(seq 0 ...)` seed loop. If seed 0 triggers QoS retries, the backoff is doubled, and seed 1 inherits the inflated backoff.

With `HPC_RETRY_MAX_ATTEMPTS=3` and initial `HPC_RETRY_BACKOFF=10`, the worst case after one fully-retried seed: seed 0 backoffs are 10s, 20s (backoff becomes 20 after first retry, then 40 after second). Seed 1 starts with `HPC_RETRY_BACKOFF=40` instead of the configured 10.

**Fix:** Save and restore the original backoff within the seed loop:
```bash
for s in $(seq 0 "$((NUM_RUNS - 1))"); do
  run_seed=$((SEED + s))
  local_backoff="$HPC_RETRY_BACKOFF"  # save original

  # ... heredoc generation ...

  # Retry loop uses local_backoff instead:
  while [ "$attempt" -lt "$HPC_RETRY_MAX_ATTEMPTS" ]; do
    ...
    log "  QoS limit reached. Retrying in \${local_backoff}s..."
    sleep "$local_backoff"
    local_backoff=$((local_backoff * 2))
  done
done
```

## Warnings

### WR-01: SLURM script `--output` uses bare filenames without centralized path

**File:** `runners/bash/hpc_submit.sh:133-134`
**Issue:** `#SBATCH --output=\${SLURM_ARRAY_TASK_ID}.out` and `#SBATCH --error=\${SLURM_ARRAY_TASK_ID}.err` write log files to the job's working directory (where `sbatch` was invoked). For array jobs, this scatters logs across multiple `.out`/`.err` files with generic names. If a user submits from different directories or multiple experiments run concurrently, logs can be overwritten or misplaced.

**Fix:** Include experiment and seed context in the output path:
```bash
#SBATCH --output=${HPC_OUTPUT_ROOT}/logs/%x_%a.out
#SBATCH --error=${HPC_OUTPUT_ROOT}/logs/%x_%a.err
```
Where `%x` is the job name and `%a` is the array task ID (SLURM built-in placeholders).

### WR-02: `local_batch.sh` spawns N Python subprocesses to resolve dataset names

**File:** `runners/bash/local_batch.sh:187-192`
**Issue:** Inside the execution loop, each iteration spawns a new `uv run python` process to resolve the dataset name from its index. For a batch of 85 datasets (full UCR suite), this means 85 separate Python interpreter startups, each loading the full dependency chain. This is unnecessary overhead when the name-to-index mapping could be resolved once upfront.

Additionally, line 155 already resolved `TOTAL_DATASETS` via a separate Python call. The total count and the full name list should be fetched in a single Python invocation.

**Fix:** Pre-resolve the complete index-to-name mapping in one call:
```bash
cd "$PROJECT_ROOT" || exit 1
declare -A INDEX_TO_NAME
while IFS='|' read -r idx name; do
  INDEX_TO_NAME[$idx]="$name"
done < <(uv run python -c "
from src.rbspaper.data.data_setup import get_all_datasets
datasets = get_all_datasets(form='list')
for i, name in enumerate(datasets):
    print(f'{i}|{name}')
") || exit 1
```

### WR-03: `hpc_submit_single.sh` `FORECASTING_MODE_ARG` is subject to shell injection

**File:** `runners/bash/hpc_submit_single.sh:93-96` (argument building) and line 132 (heredoc usage)
**Issue:** `FORECASTING_MODE_ARG` is constructed by direct string concatenation: `" --forecasting_mode ${FORECASTING_MODE}"`. The `FORECASTING_MODE` value comes from CLI argument `--forecasting_mode "$2"`. In the generated SLURM script, this unquoted string is appended to the `uv run python` command. A value containing shell metacharacters (e.g., `univariate; rm -rf /`) would be interpreted by the SLURM job's shell.

In practice, the forecasting wrappers pass hardcoded values (`"univariate"` or `"multivariate"`), so this requires direct user input to exploit. However, the script is documented as accepting arbitrary CLI arguments.

**Fix:** Validate `FORECASTING_MODE` against an allowlist before use:
```bash
case "$FORECASTING_MODE" in
  univariate|multivariate) ;;
  *)
    log "ERROR: --forecasting_mode must be 'univariate' or 'multivariate'. Got: '$FORECASTING_MODE'" >&2
    exit 1 ;;
esac
```

### WR-04: `expand_dataset_spec()` accepts non-integer input silently

**File:** `runners/bash/local_batch.sh:32-58`
**Issue:** The function handles three spec formats: `all`, comma-separated, and dash-separated ranges. However, it does not validate that the extracted values are integers. A spec like `abc-def` would produce `start_val="abc"` and `end_val="def"`, and the comparison `[ "$start_val" -gt "$end_val" ]` would fail with a bash arithmetic error (non-integer). Due to `set -uo pipefail` without `-e`, the error message is printed but execution continues, potentially producing garbage output.

Similarly, a single-value spec like `hello` falls through to the final `echo "$spec"` on line 57, passing the string as if it were a valid index.

**Fix:** Add integer validation:
```bash
if echo "$spec" | grep -q '-'; then
  local start_val="${spec%-*}"
  local end_val="${spec#*-}"
  if ! echo "$start_val" | grep -qE '^[0-9]+$' || ! echo "$end_val" | grep -qE '^[0-9]+$'; then
    echo "ERROR: Range values must be integers. Got: '$spec'" >&2
    return 1
  fi
  ...
fi

# Fallback: validate single value is integer
if ! echo "$spec" | grep -qE '^[0-9]+$'; then
  echo "ERROR: Invalid dataset spec: '$spec'" >&2
  return 1
fi
echo "$spec"
```

### WR-05: `run_all.sh` scripts do not propagate `run_on.sh` failure exit codes

**File:** `runners/bash/classification/univariate/run_all.sh:52-55` (and 3 sibling files)
**Issue:** The nested loop calls `bash .../run_on.sh "$exp_id" "$dataset_name" "$@"` but does not check the return value. If `hpc_submit_single.sh` (called through `run_on.sh`) fails for a specific experiment-dataset pair — due to QoS limits, SLURM errors, or other reasons — the loop continues silently. The final message "All classification/univariate experiments submitted" is misleading when only a subset actually succeeded.

**Fix:** Track failures in the loop:
```bash
total_submitted=0
total_failed=0
for exp_id in "${EXPERIMENTS[@]}"; do
  for dataset_name in "${DATASETS[@]}"; do
    if bash "$PROJECT_ROOT/runners/bash/classification/univariate/run_on.sh" "$exp_id" "$dataset_name" "$@"; then
      total_submitted=$((total_submitted + 1))
    else
      total_failed=$((total_failed + 1))
    fi
  done
done
log "Submitted: $total_submitted, Failed: $total_failed"
[ "$total_failed" -gt 0 ] && exit 1
```

## Info

### IN-01: `get_experiment_list()` docstring says "space-separated" but output is newline-separated

**File:** `runners/bash/runner_config.sh:58`
**Issue:** The inline comment reads `# Returns space-separated list of registered experiment IDs`, but the function prints one ID per line via `echo "$ids"` where `$ids` was collected line-by-line from Python `print()`. All callers correctly use `while IFS= read -r line` to consume the output, so functionality is not affected. Only the comment is stale.

**Fix:** Update the comment to `# Returns newline-separated list of registered experiment IDs`.

### IN-02: `set -e` (errexit) not enabled in any runner script

**File:** All 15 files use `set -uo pipefail` without `-e`.
**Issue:** The scripts rely on explicit `|| exit 1` or `if [ "$ret" -eq 0 ]` patterns for error handling. Without `set -e`, unexpected failures (e.g., `mkdir -p` failing due to permissions, `chmod` errors) are silently ignored. This is a deliberate trade-off for error handling via return codes, but it increases the risk of missed failures in code paths that were not anticipated.

**Fix:** Consider adding `set -e` for strict error propagation, or document explicitly why `-e` is omitted and add sentinel checks at critical operations.

### IN-03: Config example files have unnecessary `#!/usr/bin/env bash` shebangs

**File:** `runners/bash/config.sh.example:1`
**File:** `runners/bash/hpc_config.sh.example:1`
**Issue:** These files are sourced, never executed directly. The shebang is copied into the user's `config.sh` by `ensure_config`, where it serves no purpose since config files contain only variable assignments and comments.

**Fix:** Remove the shebang lines from both `.example` templates.

### IN-04: `DEFAULT_SEED` in `runner_config.sh` captures `SEED` from env at source time

**File:** `runners/bash/runner_config.sh:29`
**Issue:** `DEFAULT_SEED="${SEED:-42}"` is evaluated when `runner_config.sh` is sourced (early). Then `local_single.sh` line 43 sets `SEED="$DEFAULT_SEED"` (already captured), and the `--seed` CLI override modifies `SEED` afterward. Meanwhile, `hpc_submit.sh` line 44 uses `SEED="${SEED:-$DEFAULT_SEED}"`, which checks the current `SEED` env var against the already-captured default. This works correctly in practice but creates a confusing two-layer fallback: env SEED is captured into DEFAULT_SEED at source time, then the runtime SEED also checks env. If someone sets `SEED=99` in the environment and also passes `--seed 42`, the CLI override wins (correct), but the env variable has already polluted `DEFAULT_SEED`.

**Fix:** Simplify to a single source of truth:
```bash
# In runner_config.sh — do NOT capture env SEED
DEFAULT_SEED=42

# In each script — env SEED acts as default, CLI --seed overrides
SEED="${SEED:-$DEFAULT_SEED}"
# Then --seed parsing overrides SEED
```

---

_Reviewed: 2026-05-07_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
