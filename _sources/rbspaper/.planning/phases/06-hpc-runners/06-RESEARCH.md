# Phase 6: HPC Runners - Research

**Researched:** 2026-05-07
**Domain:** SLURM job scheduling, bash scripting for HPC, experiment orchestration
**Confidence:** HIGH

## Summary

This phase creates SLURM-based HPC submission scripts that wrap the existing Python runner (`runners/py/runner.py`), enabling large-scale experiments across 128+ datasets on the Kathleen cluster. The architecture follows a three-tier hierarchy per dataset family, mirroring the AutoTSAugment reference pattern but with significant improvements: family-scoped dynamic indexing (no hardcoded ranges), `uv run` instead of conda, `HPC_OUTPUT_ROOT` as a single source of truth, retained job files for debugging, and exponential backoff for QoS retry.

**Primary recommendation:** Implement three bash scripts (hpc_submit.sh core engine, per-family tier-2 launchers, per-family tier-3 experiment-set runners) plus one config template (hpc_config.sh.example). The core technical challenge is family-scoped index resolution — a small Python one-liner inside the generated SLURM script maps `SLURM_ARRAY_TASK_ID` to a dataset name via `get_datasets_names(family, form='list')`. Everything else is straightforward bash wrapping around the existing runner.

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Script Architecture:**
- **D-01:** SLURM array job — single `sbatch` with `--array=0-N%N`. One task per dataset via `SLURM_ARRAY_TASK_ID`. Matches autotsaugment reference pattern.

**Dataset Resolution — Family-Scoped Indexes:**
- **D-06:** Family-scoped indexes (Option A) — `SLURM_ARRAY_TASK_ID` maps to a position within the specified dataset family, not a global position. The submit script queries the registry at generation time (`get_datasets_names(family, form='list')`) to compute the correct array range. The generated SLURM script resolves `family + array_id` to dataset name via a small Python one-liner, then passes `--dataset_name` to the runner.

**CLI Interface:**
- **D-02:** Config-driven + `--family` flag — `hpc_submit.sh <experiment_id> --family <ucr|uea|forecast,...> [--seed N] [--force]`. Per-category convenience scripts bake `--family` into short wrappers.

**HPC Configuration:**
- **D-03:** Separate `hpc_config.sh` — clean separation between local (`config.sh`) and HPC settings. `hpc_submit.sh` sources both. Auto-copies `hpc_config.sh.example` on first run if missing.
- **D-07:** `HPC_OUTPUT_ROOT` as single source of truth — eliminates repeated path typing across all runners.

**Script Hierarchy — AutoTSAugment Pattern:**
- **D-08:** Three-tier hierarchy per dataset family:
  - Tier 1: `hpc_submit.sh` — core engine (SLURM job generation + submission + QoS retry)
  - Tier 2: `{family}/run_on_{family}.sh` — per-category launcher
  - Tier 3: `{family}/run_all_on_{family}.sh` — experiment-set runner

**Job File Management:**
- **D-04:** Keep job files in `outputs/` — with `--delete_job_files` flag.
- **D-05:** QoS retry loop — 3 attempts with exponential backoff (10s -> 20s -> 40s).

### Claude's Discretion

- Exact layout of the `outputs/` job file directory
- QoS retry backoff timing constants (10s -> 20s -> 40s is the baseline)
- Exact EXPERIMENTS array content per family (planner derives from registry)
- Whether `hpc_submit.sh` supports multi-seed loops or delegates to separate submissions
- Aggregate submission report format (job IDs, task ranges, estimated completion)

### Deferred Ideas (OUT OF SCOPE)

- Non-SLURM scheduler support (PBS, LSF)
- Parallel batch execution with configurable MAX_JOBS at the bash level
- Monitoring/dashboard for job status
- Project-level output registry tracking all runs

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| SLURM job generation | Bash (hpc_submit.sh) | — | SLURM SBATCH directives are bash-native |
| Dataset index resolution | Python one-liner (inside generated SLURM script) | — | Requires `get_datasets_names(family)` from registry |
| Experiment execution | Python (runner.py) | — | Pipeline orchestration, model loading, training |
| Config management | Bash (hpc_config.sh) | — | Cluster-specific settings sourced at bash level |
| QoS retry logic | Bash (hpc_submit.sh) | — | Parses `sbatch` stderr for QoS limit messages |
| Output path construction | Python (runner.py / config.py) | Bash (HPC_OUTPUT_ROOT) | `build_hierarchical_run_name()` in Python; HPC_OUTPUT_ROOT passed as `--output_dir` |
| Tier 2/3 convenience | Bash (run_on_*, run_all_on_*) | — | Thin wrappers around hpc_submit.sh |
| Project root detection | Bash (SCRIPT_DIR traversal) | — | Reused from local_single.sh pattern |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SLURM (sbatch/squeue) | Cluster-managed [VERIFIED: Context7 slurm-25.05] | Job scheduler and array task execution | Kathleen cluster uses SLURM; no alternative |
| Bash 3.2+ | System default | Scripting language | HPC login nodes typically have minimal tooling; bash is universally available |
| uv | Current | Python environment management and execution | Replaces conda; already established in Phase 5 |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `get_datasets_names(family)` | Project function | Family-scoped dataset resolution | Inside generated SLURM script for index-to-name mapping |
| `list_experiment_ids()` | Project function | Experiment validation | Pre-submission check that experiment_id is registered |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| SLURM array job | Per-dataset sbatch calls | Loses array convenience, higher QoS burden |
| uv run | conda activate | Conda is forbidden — project uses uv exclusively |
| Family-scoped index | Global index | Global index causes hardcoded ranges to rot |
| Bash script generation | Python job generator | Adds Python dependency on login node; bash is simpler |

## Architecture Patterns

### System Architecture Diagram

```
User invokes Tier 2/3 script
         |
         v
Tier 2: run_on_ucr.sh ts2vec          Tier 3: run_all_on_ucr.sh
(bakes --family ucr)                  (loops over EXPERIMENTS array)
         |                                    |
         +------------------+-----------------+
                            |
                            v
Tier 1: hpc_submit.sh ts2vec --family ucr [--seed N] [--force]
                            |
            +---------------+---------------+
            |                               |
            v                               v
    Query registry                    Parse arguments
    get_datasets_names('ucr')         Validate experiment_id
    -> ['Coffee','ECG200','FaceFour'] -> family, seed, force
            |                               |
            +---------------+---------------+
                            |
                            v
        Generate SLURM script (hpc_exp_ts2vec_seed_42.sh)
        Contains:
        - #SBATCH directives (partition, time, mem, array)
        - SLURM_ARRAY_TASK_ID -> dataset_name resolution
        - uv run python runners/py/runner.py --dataset_name ...
                            |
                            v
        Submit via sbatch (with QoS retry loop)
            |
            +-- QoS limit? --[10s]--> retry --[20s]--> retry --[40s]--> retry --[abort]
            |
            v (success)
        Retain job file in outputs/
        Print submission report
```

### Recommended Project Structure

```
runners/bash/
├── config.sh.example           # Already exists (Phase 5)
├── config.sh                   # Already exists (gitignored, auto-created)
├── hpc_config.sh.example       # NEW — Template HPC config
├── hpc_config.sh               # NEW — Live HPC config (gitignored)
├── hpc_submit.sh               # NEW — Core engine (Tier 1)
├── local_single.sh             # Already exists (Phase 5)
├── local_batch.sh              # Already exists (Phase 5)
├── ucr/
│   ├── run_on_ucr.sh           # NEW — Tier 2 UCR launcher
│   └── run_all_on_ucr.sh       # NEW — Tier 3 UCR experiment-set
├── uea/
│   ├── run_on_uea.sh           # NEW — Tier 2 UEA launcher
│   └── run_all_on_uea.sh       # NEW — Tier 3 UEA experiment-set
├── forecast/
│   ├── run_on_forecast.sh      # NEW — Tier 2 Forecast launcher
│   └── run_all_on_forecast.sh  # NEW — Tier 3 Forecast experiment-set
└── electricity/
    ├── run_on_electricity.sh   # NEW — Tier 2 Electricity launcher
    └── run_all_on_electricity.sh  # NEW — Tier 3 Electricity experiment-set
```

**Note on family directories:** The CONTEXT.md mentions `{ucr,uea,forecast,...}`. Based on the registry, the families are: `ucr`, `uea`, `ett`, `electricity`, `weather`, `exchange`, `traffic`, `illness`. The tier 2/3 scripts should be created per family. For forecasting families with single datasets (ett, electricity, weather, etc.), the array range is `0-0` (one task). The planner should decide whether to create per-family directories for single-dataset families or group them under a broader `forecast/` directory.

### Pattern 1: Family-Scoped Index Resolution

**What:** Resolve `SLURM_ARRAY_TASK_ID` to a dataset name within a specific family, preventing index drift when the registry changes.

**When to use:** Inside the generated SLURM script, at job execution time.

**Example:**
```bash
# Inside the generated SLURM script (heredoc)
# $family is expanded at generation time; $SLURM_ARRAY_TASK_ID is set by SLURM at runtime
dataset_name=$(uv run python -c "
import sys
from src.rbspaper.data.data_setup import get_datasets_names
family = sys.argv[1]
idx = int(sys.argv[2])
datasets = get_datasets_names(family, form='list')
print(datasets[idx])
" "${family}" "\$SLURM_ARRAY_TASK_ID")

uv run python "${PROJECT_ROOT}/runners/py/runner.py" \
    --experiment_id="${exp_id}" \
    --dataset_name="${dataset_name}" \
    --data_root="${DATA_ROOT}" \
    --output_dir="${HPC_OUTPUT_ROOT}" \
    --seed="${seed}" \
    ${force_flag}
```

**Source:** Pattern derived from D-06 decision + `get_datasets_names()` verified against project code.

### Pattern 2: SLURM Array Job Generation

**What:** Generate a complete SLURM script with heredoc, including SBATCH directives and the execution body.

**When to use:** hpc_submit.sh, during the job generation phase.

**Example:**
```bash
# Compute array range dynamically
total_datasets=$(uv run python -c "
import sys
from src.rbspaper.data.data_setup import get_datasets_names
family = sys.argv[1]
print(len(get_datasets_names(family, form='list')))
" "${family}")

array_range="0-$((total_datasets - 1))"

# Generate job script
cat > "${job_file}" <<EOL
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${log_file}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=\${HPC_CPUS_PER_TASK}
#SBATCH --mem-per-cpu=\${HPC_MEM_PER_CPU}
#SBATCH --partition=\${HPC_PARTITION}
#SBATCH --time=\${HPC_TIME}
#SBATCH --account=\${HPC_ACCOUNT}
#SBATCH --array=${array_range}%\${HPC_MAX_CONCURRENT}

export PYTHONPATH="\${PROJECT_ROOT}:\${PYTHONPATH:-}"

family="${family}"
dataset_name=\$(uv run python -c "
import sys
from src.rbspaper.data.data_setup import get_datasets_names
datasets = get_datasets_names(sys.argv[1], form='list')
print(datasets[int(sys.argv[2])])
" "\${family}" "\$SLURM_ARRAY_TASK_ID")

uv run python "\${PROJECT_ROOT}/runners/py/runner.py" \\
    --experiment_id="${exp_id}" \\
    --dataset_name="\$dataset_name" \\
    --data_root="${DATA_ROOT}" \\
    --output_dir="\${HPC_OUTPUT_ROOT}" \\
    --seed="${seed}" \\
    ${force_flag}
EOL
```

**Source:** Adapted from `_sources/autotsaugment/runners/bash/runner.sh`, improved per D-01, D-06, D-07.

### Pattern 3: QoS Retry Loop

**What:** Retry `sbatch` submission on QoS limits with exponential backoff.

**When to use:** hpc_submit.sh, after job file generation, during submission.

**Example:**
```bash
max_retries=3
backoff=10
submit_success=false

for ((attempt=1; attempt<=max_retries; attempt++)); do
    output=\$(sbatch "\${job_file}" 2>&1)
    ret_value=\$?

    log "\$output"

    if [[ \$ret_value -eq 0 ]]; then
        submit_success=true
        job_id=\$(echo "\$output" | grep -oP 'Submitted batch job \K\d+')
        log "Job submitted successfully (ID: \$job_id)"
        break
    elif [[ "\$output" == *"QOSMaxSubmitJobPerUserLimit"* || "\$output" == *"Job violates accounting/QOS policy"* ]]; then
        log "QoS limit reached, retrying in \${backoff}s (attempt \${attempt}/\${max_retries})..."
        sleep "\${backoff}"
        backoff=\$((backoff * 2))
    else
        log "Failed to submit job for an unexpected reason." >&2
        break
    fi
done

if [[ "\$submit_success" == false ]]; then
    log "ERROR: Failed to submit job after \${max_retries} attempts." >&2
    exit 1
fi
```

**Source:** Pattern adapted from `_sources/autotsaugment/runners/bash/runner.sh` (lines 117-139), improved with exponential backoff per D-05.

### Pattern 4: Config File Auto-Copy

**What:** Copy `.example` template to live config on first run, block if required fields are empty.

**When to use:** hpc_submit.sh startup, before any cluster operations.

**Example:**
```bash
HPC_CONFIG_FILE="\$SCRIPT_DIR/hpc_config.sh"
HPC_CONFIG_EXAMPLE="\$SCRIPT_DIR/hpc_config.sh.example"

if [ ! -f "\$HPC_CONFIG_FILE" ]; then
  cp "\$HPC_CONFIG_EXAMPLE" "\$HPC_CONFIG_FILE"
  log "Created hpc_config.sh from template."
  log "Please set HPC_OUTPUT_ROOT and other variables in \$HPC_CONFIG_FILE and re-run."
  exit 1
fi

# shellcheck source=/dev/null
source "\$HPC_CONFIG_FILE"

# Validate required fields
if [ -z "\${HPC_OUTPUT_ROOT:-}" ]; then
  log "ERROR: HPC_OUTPUT_ROOT is not set in \$HPC_CONFIG_FILE" >&2
  exit 1
fi
```

**Source:** Reused from `local_single.sh` pattern (lines 36-43), adapted for HPC config.

### Anti-Patterns to Avoid

- **Global dataset index in SLURM array:** Do NOT use `--dataset_index = SLURM_ARRAY_TASK_ID` with a hardcoded range like `"0-127"`. Ranges rot silently when the registry changes. Always use family-scoped resolution (D-06).
- **Hardcoded storage paths:** Do NOT repeat `/storage/work/skaf/` across multiple files. Use `HPC_OUTPUT_ROOT` from `hpc_config.sh` (D-07).
- **Conda activation in SLURM script:** Do NOT use `source miniconda/etc/profile.d/conda.sh` or `conda activate`. Use `uv run` directly.
- **30-minute flat QoS retry sleep:** Do NOT use a single long sleep. Use exponential backoff (10s -> 20s -> 40s).
- **Deleting job files by default:** Do NOT delete submitted SLURM scripts. Retain them for debugging (D-04).
- **`set -e` without trap:** Do NOT use `set -e` without a proper ERR trap for cleanup. The existing `set -uo pipefail` is sufficient for the scripts' flow.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| SLURM job scheduling | Custom Python SLURM client | `sbatch` CLI + bash | SLURM provides native CLI; no need for Python wrappers |
| Index resolution | Hardcoded index-to-name mapping in bash | `get_datasets_names(family, form='list')` via Python one-liner | Registry is the source of truth; bash should not duplicate dataset lists |
| Experiment validation | Manual experiment ID checking | `list_experiment_ids()` via Python one-liner | Registry-based validation prevents typos |
| Environment management | conda activate | `uv run` | Project standard; no conda on Kathleen cluster |
| Config hashing | Manual hash computation | `compute_config_hash()` from pipeline state module | Already implemented; produces deterministic short_hash |
| Path construction | Custom path building | `build_hierarchical_run_name()` from config.py | Already implemented; produces `{exp_id}/{hash}/seed_{N}/{dataset}` |

**Key insight:** The bash scripts are thin wrappers around the existing Python infrastructure. Every piece of intelligence (dataset registry, experiment registry, output paths, config hashing) already lives in Python. The bash layer only handles SLURM-specific concerns: SBATCH directives, array job generation, QoS retry logic, and config file sourcing.

## Runtime State Inventory

> Not applicable — this is a greenfield phase (new scripts, no rename/refactor).

## Common Pitfalls

### Pitfall 1: Heredoc Variable Escaping

**What goes wrong:** Inside a `cat > file <<EOL` heredoc, bash expands variables at generation time. Variables that should be expanded at runtime (like `SLURM_ARRAY_TASK_ID`) get expanded to empty strings or wrong values.

**Why it happens:** Unquoted heredoc delimiters (`EOL` vs `'EOL'`) cause all variable expansion. Mixing generation-time and runtime variables in the same heredoc requires careful escaping.

**How to avoid:** Use `\$` to escape runtime variables (e.g., `\$SLURM_ARRAY_TASK_ID`) and leave generation-time variables unescaped (e.g., `"\${family}"`). Always review the heredoc to verify which variables expand when.

**Warning signs:** Generated SLURM script contains empty strings where `\$SLURM_ARRAY_TASK_ID` should appear. Test by generating the script and inspecting it before submitting.

**Source:** Verified against `_sources/autotsaugment/runners/bash/runner.sh` line 93: `dataset_index=\$SLURM_ARRAY_TASK_ID` uses backslash escaping.

### Pitfall 2: Family Not Mapped to Dataset

**What goes wrong:** `get_datasets_names(family)` returns an empty list for an unknown family, resulting in `array_range=0--1` or IndexError at runtime.

**Why it happens:** No validation that the `--family` argument is known before computing the range.

**How to avoid:** Validate family against the known family list before proceeding. Fail with a clear message listing valid families.

**Warning signs:** SLURM job submission error: "Bad array specifier" or Python IndexError inside the generated script.

### Pitfall 3: sbatch Return Code Ambiguity

**What goes wrong:** `sbatch` returns non-zero for different failure modes (QoS limit, parser error, cluster unreachable). Treating all failures as "retry" masks real errors.

**Why it happens:** `sbatch` exit code alone doesn't distinguish QoS limits from other errors.

**How to avoid:** Parse `sbatch` stderr output for QoS-specific strings (`QOSMaxSubmitJobPerUserLimit`, `Job violates accounting/QOS policy`). Only retry on those patterns; fail immediately on other errors.

**Warning signs:** Script retries endlessly on a parser error (e.g., invalid partition name) instead of failing fast.

**Source:** Verified against `_sources/autotsaugment/runners/bash/runner.sh` lines 129-138.

### Pitfall 4: PYTHONPATH Not Set in Generated SLURM Script

**What goes wrong:** The generated SLURM script runs on a compute node where `PYTHONPATH` is not inherited from the login shell.

**Why it happens:** SLURM job scripts run in a clean environment. Variables from the submission shell are not automatically propagated (except exported vars, but they may not persist).

**How to avoid:** Always set `PYTHONPATH` explicitly inside the generated SLURM script: `export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"`.

**Warning signs:** `ModuleNotFoundError: No module named 'src'` or `No module named 'experiment_instances'`.

### Pitfall 5: uv Not Available on Compute Nodes

**What goes wrong:** The generated SLURM script calls `uv run` but `uv` is not in the PATH on the compute node.

**Why it happens:** `uv` is installed in the user's local environment (e.g., via Homebrew or system installer). On the Kathleen cluster, it may need to be loaded via a module or have an absolute path.

**How to avoid:** Verify `uv` availability on the Kathleen cluster. If needed, use an absolute path to `uv` in the generated SLURM script, or add a `module load` directive. Document the requirement in `hpc_config.sh.example`.

**Warning signs:** `uv: command not found` in SLURM job output.

**Confidence:** MEDIUM — depends on Kathleen cluster setup; not verifiable without cluster access.

### Pitfall 6: Single-Dataset Family Produces Degenerate Array

**What goes wrong:** For families with one dataset (e.g., `ett`: just `ETTh1`), the array range is `0-0`. This works but is unusual and may confuse users expecting multiple tasks.

**Why it happens:** `len(get_datasets_names('ett', form='list'))` returns 1, so `total_datasets - 1 = 0`.

**How to avoid:** This is not actually a bug — `--array=0-0` is valid SLURM syntax and produces a single task. However, add a log message noting that only 1 dataset was found for the family.

**Warning signs:** None — this works correctly. Just surprising to users.

## Code Examples

### hpc_config.sh.example Template

```bash
#!/usr/bin/env bash
#
# hpc_config.sh — HPC cluster configuration.
#
# Copy this file to hpc_config.sh and set the variables below.
# Do NOT commit hpc_config.sh (it is gitignored).
#

# Single source of truth for HPC storage path (required).
# All experiment outputs go under this directory.
HPC_OUTPUT_ROOT=""

# SLURM partition (e.g., "Kathleen", "c23ms")
HPC_PARTITION="Kathleen"

# SLURM account
HPC_ACCOUNT=""

# Time limit (SLURM format: D-H:M:S or H:M:S)
HPC_TIME="2-0:0:0"

# Memory per CPU
HPC_MEM_PER_CPU="16G"

# CPUs per task
HPC_CPUS_PER_TASK=4

# Maximum concurrent array tasks
HPC_MAX_CONCURRENT=128

# QoS level
HPC_QOS="medium"

# Delete job files after submission (true/false)
DELETE_JOB_FILES=false
```

### Family Validation Example

```bash
# Validate --family against known families
valid_families="ucr uea ett electricity weather exchange traffic illness"
family_valid=false
for f in \$valid_families; do
    if [[ "\$f" == "\$family" ]]; then
        family_valid=true
        break
    fi
done

if [[ "\$family_valid" == false ]]; then
    log "ERROR: Unknown family '\$family'. Valid families: \$valid_families" >&2
    exit 1
fi
```

### Tier 2 Launcher Example (run_on_ucr.sh)

```bash
#!/usr/bin/env bash

set -uo pipefail

current_script_dir=\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" || exit 1; pwd) || exit 1
parent_dir=\$(dirname "\$current_script_dir")

MAIN_SCRIPT_PATH="\${parent_dir}/hpc_submit.sh"

experiment_id="\${1:-}"

if [ -z "\$experiment_id" ]; then
    echo "Usage: \$0 <experiment_id> [--seed N] [--force]" >&2
    exit 1
fi

shift
bash "\$MAIN_SCRIPT_PATH" "\$experiment_id" --family ucr "\$@"
```

### Tier 3 Experiment-Set Example (run_all_on_ucr.sh)

```bash
#!/usr/bin/env bash

set -uo pipefail

current_script_dir=\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" || exit 1; pwd) || exit 1

RUNNER_PATH="\${current_script_dir}/run_on_ucr.sh"

EXPERIMENTS=(
    "ts2vec"
    "autotcl"
)

for exp_id in "\${EXPERIMENTS[@]}"; do
    bash "\$RUNNER_PATH" "\$exp_id"
done
```

## State of the Art

### SLURM Array Jobs

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-dataset sbatch calls | `--array=0-N%M` single submission | SLURM 14.x+ | One sbatch call manages N tasks; lower scheduler overhead |
| Global index rot | Family-scoped dynamic range | Project improvement | Index ranges self-heal when registry changes |
| Hardcoded SBATCH values | Config-driven SBATCH | Project improvement | Partition, time, memory controlled from hpc_config.sh |

### QoS Handling

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 30-minute flat sleep | Exponential backoff (10s/20s/40s) | Project improvement | Faster recovery from transient QoS limits |
| No QoS detection | Parse stderr for QoS strings | Project improvement | Real failures surface immediately |

### Environment Management

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| conda activate | `uv run` | Project-wide (Phase 5) | Single toolchain; no conda env on cluster |

**Deprecated/outdated:**
- **Global dataset indexing:** AutoTSAugment's approach of `SLURM_ARRAY_TASK_ID` = global position. Ranges hardcoded in bash (`"0-127"` for UCR). Replaced by family-scoped indexing (D-06).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Kathleen cluster has `uv` available in PATH on compute nodes | Common Pitfalls (Pitfall 5), Environment Availability | SLURM jobs fail with "uv: command not found"; requires absolute path or module load workaround |
| A2 | Kathleen cluster uses partition name "Kathleen" (from reference runner.sh line 62) | hpc_config.sh.example template | Jobs submitted to wrong partition; user overrides in hpc_config.sh anyway |
| A3 | Kathleen cluster uses account format from reference (e.g., "rwth1497" from hpc_uni_bash/runner.sh) | hpc_config.sh.example template | User must fill in correct account; template has empty default |
| A4 | `get_datasets_names(family)` works for all families listed in registry.py | Pattern 1, Runtime Resolution | If a family has no implementation, Python one-liner crashes; but registry.py shows families map correctly |
| A5 | Forecaster families (electricity, weather, exchange, traffic, illness) have datamodules implemented | Standard Stack | `data_setup.py` line 283 raises NotImplementedError for non-ett forecasting families — scripts would submit jobs that fail at runtime |

**A5 is significant:** The current `data_setup.py` raises `NotImplementedError` for forecasting families other than `ett`. The tier 2/3 scripts for `electricity`, `weather`, `exchange`, `traffic`, `illness` would generate valid SLURM jobs that fail when `get_datamodule_with_downstream_tasks()` is called. The planner should either: (a) skip generating tier scripts for unsupported families, or (b) add a pre-submission check that queries Python to verify the family is runnable.

## Open Questions (RESOLVED)

1. **Which families get tier 2/3 scripts?**
   - What we know: Registry lists 8 families (ucr, uea, ett, electricity, weather, exchange, traffic, illness). Only ucr, uea, ett have working datamodules (verified by code inspection).
   - What's unclear: Should we generate tier scripts for families that will fail at runtime? Or only for families with complete datamodule support?
   - Recommendation: Generate tier scripts only for `ucr`, `uea`, and `ett` (the families with working datamodules). For forecast families beyond ett, add a comment in hpc_submit.sh explaining that datamodules are not yet implemented.
   -- RESOLVED: Plan 02 generates tier scripts only for ucr, uea, ett per recommendation.

2. **How should the EXPERIMENTS array in tier 3 scripts be populated?**
   - What we know: Current registry has `ts2vec` and `autotcl` (verified by runtime query).
   - What's unclear: Whether the EXPERIMENTS array should be hardcoded or generated dynamically from `list_experiment_ids()`.
   - Recommendation: Hardcode the array. Tier 3 scripts are convenience wrappers; a static list is simple and self-documenting. The array is short (2 experiments) and changes rarely.
   -- RESOLVED: Plan 02 hardcodes EXPERIMENTS=(ts2vec autotcl) per recommendation.

3. **Should hpc_submit.sh validate experiment_id before generating the SLURM script?**
   - What we know: `list_experiment_ids()` returns the registered set. The runner already validates at execution time.
   - What's unclear: Whether early validation (failing before sbatch) is worth the overhead.
   - Recommendation: Yes — add a lightweight validation step. A typo in the experiment ID is caught instantly rather than after SLURM processes the script.
   -- RESOLVED: Plan 01, Task 2 Step 5 validates experiment_id via list_experiment_ids() before job generation.

4. **Multi-seed support — single script or loop in hpc_submit.sh?**
   - What we know: Reference runner loops over seeds (`for ((s=0; s<total_runs; s++))`), generating one array job per seed.
   - What's unclear: Whether to support this or leave it as Claude's discretion.
   - Recommendation: Support a `--num_runs N` flag (defaulting to 1) that generates N array jobs with different seeds. This matches the reference pattern without overcomplicating the interface.
   -- RESOLVED: Plan 01, Task 2 includes --num_runs flag with per-seed loop (Step 7).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| SLURM (sbatch) | Job submission | ✗ (local) | — | N/A — runs on Kathleen cluster |
| uv | Python execution | ✓ | (local only) | Absolute path on cluster |
| Python 3.12 | Registry query | ✓ | Project requirement | — |
| bash 3.2+ | Script execution | ✓ | System default | — |
| Kathleen cluster | Target environment | ✗ (remote) | — | No fallback — HPC-only |

**Missing dependencies with no fallback:**
- SLURM — required for submission. Scripts run on the Kathleen cluster, not locally.
- Kathleen cluster access — cannot test sbatch locally.

**Missing dependencies with fallback:**
- None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.x (project standard) |
| Config file | pyproject.toml `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest test/ -x` |
| Full suite command | `uv run pytest test/ --cov` |

### Phase Requirements -> Test Map

This phase produces bash scripts, which are not testable via pytest directly. Validation is structural and behavioral:

| Validation | Type | Command | Notes |
|------------|------|---------|-------|
| hpc_config.sh.example is valid bash | shellcheck | `shellcheck runners/bash/hpc_config.sh.example` | If shellcheck is available |
| hpc_submit.sh parses correctly | bash -n | `bash -n runners/bash/hpc_submit.sh` | Syntax check |
| Tier 2 scripts call hpc_submit.sh | grep | Verify script references | Manual verification |
| Experiment validation rejects unknown ID | behavioral | `hpc_submit.sh nonexistent --family ucr` | Should exit with error |
| Family validation rejects unknown family | behavioral | `hpc_submit.sh ts2vec --family fake` | Should exit with error |
| Config auto-copy works | behavioral | Remove hpc_config.sh, run hpc_submit.sh | Should copy .example |

**Note:** Bats (Bash Automated Testing System) could be used for comprehensive bash testing, but the project does not currently use it. Without SLURM locally, end-to-end testing of job submission is not possible. Dry-run mode (`--dry_run` or equivalent) should be implemented to test script generation without actual sbatch calls.

### Wave 0 Gaps

- [ ] `hpc_submit.sh` `--dry_run` flag — generates SLURM script but does not call sbatch. Enables local validation of script content.
- [ ] Bash syntax validation — `bash -n` on all new scripts as a pre-commit check.
- [ ] Family support check — pre-submission Python query to verify datamodules exist for the requested family.

None of these require new test files. Validation is behavioral (script execution checks).

## Security Domain

This phase does not introduce security-sensitive functionality. The scripts handle:
- Filesystem paths (HPC_OUTPUT_ROOT, DATA_ROOT) — standard path variables, no injection risk.
- SLURM directives — values from config.sh are trusted (user-controlled).
- Python one-liners — use `sys.argv` for argument passing (not eval of user input).

No ASVS categories apply. Input validation for `--family` and `--experiment_id` prevents operational errors, not security issues.

## Sources

### Primary (HIGH confidence)
- Context7: `/websites/slurm_schedmd` — SLURM job array documentation (`job_array.html`) — array syntax, concurrent limits, `%` separator
- Context7: `/websites/slurm_schedmd_archive_slurm-25_05-latest` — SLURM 25.05 documentation — sbatch options, QoS, environment variables
- Project code: `src/rbspaper/data/data_setup.py` — `get_datasets_names(family, form)` verified by runtime query
- Project code: `src/rbspaper/data/registry.py` — `DATASET_REGISTRY` with 11 datasets across 8 families
- Project code: `experiment_instances/instances.py` — 2 registered experiments (ts2vec, autotcl)
- Project code: `src/rbspaper/pipeline/config.py` — `build_hierarchical_run_name()` producing `{exp_id}/{hash}/seed_{N}/{dataset}`

### Secondary (MEDIUM confidence)
- Reference: `_sources/autotsaugment/runners/bash/runner.sh` — QoS retry pattern, SBATCH directives, heredoc escaping
- Reference: `_sources/autotsaugment/runners/hpc_uni_bash/runner.sh` — Array job generation, per-seed loop
- Reference: `_sources/autotsaugment/runners/bash/ucr/run_experiment_ucr.sh` — Tier 2 pattern
- Reference: `_sources/autotsaugment/runners/bash/ucr/run_experiment_all_ucr.sh` — Tier 3 pattern

### Tertiary (LOW confidence)
- Kathleen cluster partition name "Kathleen" — from reference runner.sh line 62; may differ for user
- Kathleen cluster account — from reference hpc_uni_bash/runner.sh; user-specific
- `uv` availability on Kathleen compute nodes — assumed; not verified (A1)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — SLURM + bash + uv are confirmed; no alternatives exist in scope
- Architecture: HIGH — Three-tier hierarchy is locked by D-08; patterns verified against reference code
- Pitfalls: MEDIUM — Heredoc escaping and QoS detection verified against reference; uv availability on cluster is assumed

**Research date:** 2026-05-07
**Valid until:** 2026-06-06 (30 days — SLURM docs are stable; project registry may change)
