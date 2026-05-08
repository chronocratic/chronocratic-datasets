# Phase 5: Local Test Runners — Research

**Researched:** 2026-05-07
**Domain:** Bash scripting, CLI wrapper patterns, Python logging migration
**Confidence:** HIGH

## Summary

This phase delivers three Bash wrapper scripts plus supporting infrastructure that invoke the existing Python runner (`runners/py/runner.py`) for quick local validation before HPC submission. The core work is straightforward: Bash scripts that detect the project root, set PYTHONPATH, manage a config file, and forward arguments to `uv run python runners/py/runner.py`. Two smaller Python tasks round out the scope: creating `runners/__init__.py` for proper package resolution and converting ~15 `print()` calls to logging in the runner.

**Primary recommendation:** Keep the Bash scripts thin -- they are argument-assembly wrappers, not logic layers. All validation, dataset resolution, and experiment orchestration happen inside the Python runner. The scripts should fail fast on missing prerequisites (config, data_root) and otherwise delegate to `uv run`.

## User Constraints (from CONTEXT.md)

### Locked Decisions

**Entry Point & Resolution:**
- **A-01:** Hybrid approach -- create `runners/__init__.py` so `rbspaper-run` entry point works, AND bash scripts detect project root for PYTHONPATH (resolves Phase 3 UAT gap)
- **A-11:** Config file strategy -- commit `config.sh.example`, gitignore `config.sh`, auto-copy on first run with DATA_ROOT validation

**Script Interface:**
- **A-03:** Hybrid arguments -- positional `exp_id` + `dataset` (name or index), named flags for overrides (`--seed`, `--force`, `--max_epochs`, `--data_root`, `--output_dir`)
- **A-08:** Batch dataset specs -- `0-20` (range), `0,3,7` (list), or `all` (registry)
- **A-09:** Sequential execution -- datasets run one-by-one. `--parallel` flag available but not default
- **A-10:** Dataset fraction -- `--fraction 0.25` samples ~25% of datasets for quick smoke tests

**Output & Logging:**
- **A-07:** Same output structure as HPC -- uses standard `outputs/{experiment_id}/{short_hash}/seed_{seed}/{dataset_name}/`. No separate `local_outputs/`.
- **A-05:** Minimal bash logger -- `[HH:MM:SS]` timestamps to stdout + log file per run
- **A-06:** Convert `runner.py` `print()` calls to `logger.info()`/`logger.warning()`/`logger.error()` -- part of this phase

**Validation:**
- **A-04:** Essential checks -- data_root exists, exit code per run, aggregate pass/fail report for batch. No uv sync or experiment_id pre-validation overhead.

### Claude's Discretion
- Exact format of the aggregate batch report (table vs. list)
- How `--fraction` samples datasets (random seed, deterministic first-N, etc.)
- Log file naming convention for bash-level logs vs. runner-level `pipeline.log`

### Deferred Ideas (OUT OF SCOPE)
- Parallel batch execution with configurable `MAX_JOBS` -- useful for HPC (Phase 6), overkill for local validation
- Project-level output registry tracking all runs -- deferred from Phase 3
- `local_batch.sh` as a job generator that writes per-dataset scripts -- user initially wanted autotsaugment-style, but settled on sequential loop for simplicity

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Project root detection | Bash script | — | Must happen before any Python import |
| Config file management | Bash script | — | Shell-level `.sh` sourcing, not Python |
| Argument assembly | Bash script | Python runner | Bash builds the `uv run` command line |
| Dataset resolution (index to name) | Python runner | — | Runner already has `_resolve_dataset()` |
| Experiment instance lookup | Python runner | — | Runner calls `get_experiment_instance()` |
| Pipeline execution | Python pipeline core | — | `run_experiment_pipeline()` |
| Logging (structured) | Python runner | Bash script | Python uses `logging`; Bash uses `[HH:MM:SS]` prefixes |
| Batch orchestration (loop) | Bash script | — | Sequential loop is a shell concern |
| Aggregate pass/fail report | Bash script | — | Shell collects exit codes from loop iterations |

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| (implicit) | `config.sh.example` template | Bash config pattern; DATA_ROOT from runner's `--data_root` |
| (implicit) | `local_single.sh` wrapper | References autotsaugment wrapper pattern; runner.py arg interface |
| (implicit) | `local_batch.sh` wrapper | Sequential loop + dataset spec parsing in Bash |
| (implicit) | `runners/__init__.py` | Entry point `rbspaper-run = "runners.py.runner:main"` needs package marker |
| (implicit) | Convert `print()` to logging in `runner.py` | 15 print calls identified; `setup_logging()` already exists |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Bash 3.2+ | 3.2.57 [VERIFIED: localhost] | Script runtime | macOS default; scripts must be POSIX-compatible Bash |
| uv | 0.11.2 [VERIFIED: localhost] | Package runner | Project-standard execution tool (`uv run python`) |
| Python `logging` | stdlib | Structured output in runner.py | Already configured via `setup_logging()` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Bash `date` | system | `[HH:MM:SS]` timestamps | Log prefix in bash scripts |
| Python `argparse` | stdlib | CLI parsing | Only in runner.py (already exists) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Bash wrappers | Python CLI (click/typer) | Heavier; loses simple `local_single.sh b1 0` ergonomics |
| `uv run python` | Direct `python` | Requires pre-activated venv; breaks `uv` convention |
| Bash timestamp `[HH:MM:SS]` | `ts` utility | Non-standard tool; adds dependency |

**Installation:**
No new Python packages required. All dependencies are standard library or already installed.

## Architecture Patterns

### System Architecture Diagram

```
User (terminal)
  |
  |  ./runners/bash/local_single.sh ts2vec 0 --seed 123
  v
local_single.sh / local_batch.sh
  |-- Detect project root (dirname chain)
  |-- Set PYTHONPATH
  |-- Source/create config.sh (DATA_ROOT validation)
  |-- Parse positional args + named flags
  |-- [batch only] Expand dataset spec (range/list/all/fraction)
  |-- [batch only] Loop: collect exit codes
  v
uv run python runners/py/runner.py --experiment_id ... --dataset_index ... --data_root ...
  |
  |-- _resolve_dataset() -- index to name
  |-- get_experiment_instance() -- registry lookup
  |-- _build_pipeline_config() -- assemble config
  |-- setup_logging() -- configure logger
  v
run_experiment_pipeline()
  |-- Train model
  |-- Encode representations
  |-- Run attacks
  |-- Evaluate downstream tasks
  |-- Save artifacts
  v
outputs/{experiment_id}/{short_hash}/seed_{seed}/{dataset_name}/
```

### Recommended Project Structure

```
runners/
├── __init__.py              # NEW: package marker for rbspaper-run entry point
├── bash/
│   ├── config.sh.example    # NEW: template (DATA_ROOT placeholder)
│   ├── local_single.sh      # NEW: single experiment runner
│   └── local_batch.sh       # NEW: batch experiment runner
└── py/
    ├── __init__.py          # EXISTING
    └── runner.py            # EXISTING: modified (print -> logging)
```

### Pattern 1: Project Root Detection

**What:** Walk parent directories from the script location to find the project root (containing `pyproject.toml`).

**When to use:** All bash scripts that need to resolve PYTHONPATH or invoke Python modules.

**Example:**
```bash
# From autotsaugment reference: _sources/autotsaugment/runners/hpc_uni_bash/runner.sh
current_script_dir=$(cd "$(dirname "$0")" || exit; pwd)
parent_dir=$(dirname "$current_script_dir")
project_main_dir=$(dirname "$parent_dir")

export PYTHONPATH="${project_main_dir}:${PYTHONPATH}"
```

**Adapted for this project:**
```bash
# runners/bash/local_single.sh
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1; pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." || exit 1; pwd)

# Verify we're at the project root
if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
  echo "[ERROR] Cannot find pyproject.toml in $PROJECT_ROOT" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/experiment_instances:${PYTHONPATH:-}"
```

**Note:** `${BASH_SOURCE[0]}` is preferred over `$0` because it works correctly when the script is sourced or called via `source`. `[VERIFIED: runner.sh reference]`

### Pattern 2: Config File Auto-Creation

**What:** On first run, copy `config.sh.example` to `config.sh` and prompt the user to set DATA_ROOT.

**When to use:** Before any experiment invocation.

**Example:**
```bash
CONFIG_FILE="$SCRIPT_DIR/config.sh"
CONFIG_EXAMPLE="$SCRIPT_DIR/config.sh.example"

if [ ! -f "$CONFIG_FILE" ]; then
  cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"
  echo "[INFO] Created $CONFIG_FILE from template."
  echo "[INFO] Please set DATA_ROOT in $CONFIG_FILE and re-run."
  exit 1
fi

# Source the config
# shellcheck source=/dev/null
source "$CONFIG_FILE"

# Validate DATA_ROOT
if [ -z "$DATA_ROOT" ]; then
  echo "[ERROR] DATA_ROOT is not set in $CONFIG_FILE" >&2
  exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "[ERROR] DATA_ROOT='$DATA_ROOT' does not exist" >&2
  exit 1
fi
```

### Pattern 3: Bash Timestamp Logger

**What:** Prefix stdout/stderr with `[HH:MM:SS]` for traceability.

**When to use:** All bash-level output in local_single.sh and local_batch.sh.

**Example:**
```bash
log() {
  echo "[$(date '+%H:%M:%S')] $*"
}

log "Starting experiment: $EXP_ID on dataset $DATASET"
```

### Pattern 4: Dataset Spec Expansion (Batch)

**What:** Parse `0-20` (range), `0,3,7` (list), or `all` into a list of indices.

**When to use:** In `local_batch.sh` only.

**Example:**
```bash
expand_dataset_spec() {
  local spec="$1"
  local total="$2"  # total datasets from registry

  if [ "$spec" = "all" ]; then
    # Generate 0 to total-1
    seq 0 "$((total - 1))"
    return
  fi

  # Check for range (contains -)
  if echo "$spec" | grep -q '-'; then
    local start="${spec%-*}"
    local end="${spec#*-}"
    seq "$start" "$end"
    return
  fi

  # Check for list (contains ,)
  if echo "$spec" | grep -q ','; then
    echo "$spec" | tr ',' '\n'
    return
  fi

  # Single index
  echo "$spec"
}
```

**Note:** `seq` is available on macOS (BSD) and Linux [VERIFIED: standard POSIX utility]. `tr ',' '\n'` is POSIX-safe.

### Anti-Patterns to Avoid

- **Don't replicate Python logic in Bash.** Dataset validation, experiment lookup, and pipeline assembly are Python concerns. Bash only assembles the command line.
- **Don't use `#!/bin/bash` with Bash 4+ features.** macOS ships Bash 3.2.57. No `declare -A`, no `[[ =~ ]]` with extended regex, no `(( ))` arithmetic in strict mode. Use `$(( ))` for arithmetic (POSIX) and `[ ]` or `[[ ]]` with basic operators only.
- **Don't hardcode paths.** Always resolve project root from `SCRIPT_DIR`. Never assume the user runs from the project root.
- **Don't swallow exit codes.** `uv run python` returns the Python exit code. Capture it (`$?`) for the batch report.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Argument parsing in Bash | Custom getopt loop | `$1`, `$2` positional + shell parameter expansion | A-03 already decided: positional for primary args, named flags forwarded to Python |
| Dataset resolution | Bash copy of registry | `--dataset_index` forwarded to Python runner | Python runner already has `_resolve_dataset()` logic |
| Logging format | Custom timestamp function | `date '+%H:%M:%S'` | Standard, portable, sufficient |
| Parallel execution | Bash job control | Sequential loop (A-09) | Parallel is deferred to Phase 6 HPC |

**Key insight:** Bash scripts are thin wrappers around `uv run python runners/py/runner.py`. The heavy lifting (config assembly, validation, pipeline execution) is already in Python. The scripts should not duplicate that logic.

## Runtime State Inventory

> Not applicable -- this is a greenfield phase (no rename/refactor/migration).

## Common Pitfalls

### Pitfall 1: Bash 3.2 Compatibility on macOS

**What goes wrong:** Script uses `declare -A` (associative array) or `[[ $x =~ regex ]]` -- works on Linux Bash 5, fails silently or crashes on macOS Bash 3.2.

**Why it happens:** macOS ships with Bash 3.2.57 by default. The detected version on this machine is `GNU bash, version 3.2.57(1)-release`.

**How to avoid:** Stick to POSIX features: `$(( ))` arithmetic, `[ ]` test, `seq`, `tr`. No associative arrays. No `(( ))` compound commands.

**Warning signs:** Script runs locally on Linux CI but fails on macOS developer machines.

### Pitfall 2: PYTHONPATH Not Including experiment_instances

**What goes wrong:** `import experiment_instances.instances` fails because `experiment_instances/` is not under `src/`.

**Why it happens:** The package uses `src/` layout for `rbspaper` but `experiment_instances/` is a top-level directory. The pyproject.toml pytest config adds it via `pythonpath = [".", "src", "experiment_instances"]`, but that only applies to pytest.

**How to avoid:** `PYTHONPATH` must include both the project root (for `experiment_instances`) and `src` is not needed because imports use `src.rbspaper.*` absolute paths and the project root is on the path.

**Warning signs:** `ModuleNotFoundError: No module named 'experiment_instances'` when running via `uv run python runners/py/runner.py`.

### Pitfall 3: config.sh Not in .gitignore

**What goes wrong:** User commits their `config.sh` with sensitive paths.

**Why it happens:** `.gitignore` doesn't currently mention `runners/bash/config.sh`. Only `config.sh.example` is committed.

**How to avoid:** Add `runners/bash/config.sh` to `.gitignore` during this phase. The CONTEXT.md (A-11) says "gitignore config.sh" explicitly.

**Warning signs:** `git status` shows `config.sh` as untracked after first run.

### Pitfall 4: Setup_logging Called Too Late for print() Replacement

**What goes wrong:** Converting `print()` to `logger.info()` in `_print_summary()` -- but `setup_logging()` is called after `_print_summary()` in the `main()` flow (line 414 vs line 420).

**Why it happens:** The current flow is: build config -> ensure run_dir -> setup_logging -> _print_summary -> run pipeline. Actually, looking at the code more carefully, `_print_summary()` is called at line 420 and `setup_logging()` at line 414, so logging IS available. But the `--list_experiments` path (line 358-366) and the `__main__` exception handlers (lines 442, 445) are called before `setup_logging()`.

**How to avoid:** For `--list_experiments` and exception handlers, either call `setup_logging()` earlier or use a basicConfig() fallback. The `--list_experiments` path should use `logging.basicConfig()` for minimal setup, or the code should call `setup_logging()` before the `--list_experiments` branch.

**Warning signs:** `logger.info()` calls produce no output when `--list_experiments` is used; or `logging.lastResort` handler produces unformatted output.

### Pitfall 5: BASH_SOURCE vs $0 for Sourced Scripts

**What goes wrong:** Script uses `$0` to find its location, but when invoked via `source local_single.sh`, `$0` is the parent shell, not the script.

**Why it happens:** `$0` reflects the current shell name, not the script file. `${BASH_SOURCE[0]}` reflects the actual script file.

**How to avoid:** Always use `${BASH_SOURCE[0]}` for script directory detection.

**Warning signs:** `SCRIPT_DIR` resolves to the home directory instead of `runners/bash/`.

### Pitfall 6: Fraction Sampling with Small Datasets

**What goes wrong:** `--fraction 0.25` on 11 datasets yields `floor(11 * 0.25) = 2` datasets. User expects 3.

**Why it happens:** Integer truncation in Bash arithmetic.

**How to avoid:** Use `awk` or `python -c` for floating point, or accept truncation and document it. Deterministic first-N is the simplest approach for Claude's discretion.

**Warning signs:** Fraction produces 0 datasets for very small ranges.

## Code Examples

### Config Template (config.sh.example)
```bash
# Local runner configuration
# Copy this file to config.sh and set DATA_ROOT.

# Root directory containing dataset files (REQUIRED)
# Classification:   DATA_ROOT/ucr_classification_univariate/{dataset}/
# Forecasting:       DATA_ROOT/ett/{dataset}.csv
DATA_ROOT=""

# Default output directory (optional, defaults to outputs)
# OUTPUT_DIR="outputs"

# Default seed (optional, defaults to 42)
# SEED=42
```

### local_single.sh -- Argument Forwarding Pattern
```bash
# Build uv run command
CMD=(
  "uv" "run" "python" "$PROJECT_ROOT/runners/py/runner.py"
  "--experiment_id" "$EXP_ID"
  "--data_root" "$DATA_ROOT"
)

# Add dataset arg (name or index)
if [[ "$DATASET" =~ ^[0-9]+$ ]]; then
  CMD+=("--dataset_index" "$DATASET")
else
  CMD+=("--dataset_name" "$DATASET")
fi

# Add optional overrides
[ -n "${SEED:-}" ] && CMD+=("--seed" "$SEED")
[ -n "${MAX_EPOCHS:-}" ] && CMD+=("--max_epochs" "$MAX_EPOCHS")
[ "${FORCE:-false}" = "true" ] && CMD+=("--force")

# Execute
"${CMD[@]}"
exit_code=$?
```

### local_batch.sh -- Aggregate Report Pattern
```bash
# After the loop:
log "========================================"
log "        BATCH RUN SUMMARY"
log "========================================"
log "Total:  $total_runs"
log "Passed: $passed"
log "Failed: $failed"
log "========================================"

if [ "$failed" -gt 0 ]; then
  log "Failed runs:"
  # Print failed dataset names from array
  for entry in "${FAILED_RUNS[@]}"; do
    log "  - $entry"
  done
  exit 1
fi
exit 0
```

### runners/__init__.py -- Minimal Package Marker
```python
"""Entry point package for the rbspaper-run CLI."""
```

### runner.py -- print() to logging() Conversion

**Before (line 335):**
```python
def _print_summary(config: ExperimentPipelineConfig) -> None:
    model_name = getattr(config.model, 'model_name', type(config.model).__name__)
    print(f'Model:        {model_name}')
```

**After:**
```python
def _log_summary(*, config: ExperimentPipelineConfig) -> None:
    logger = logging.getLogger(__name__)
    model_name = getattr(config.model, 'model_name', type(config.model).__name__)
    logger.info('Model:        %s', model_name)
```

**Note:** Use `%s`-style formatting in logging calls (not f-strings) for lazy evaluation -- the string is only formatted if the log level is enabled. This is the standard Python logging best practice. `[VERIFIED: Python logging docs]`

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `conda activate` in scripts | `uv run python` | Project convention | No conda dependency; scripts use `uv run` |
| Hardcoded PYTHONPATH | Dynamic project root detection | autotsaugment reference | Scripts work from any directory |
| `print()` for runner output | `logging` module | Phase 5 (this phase) | Consistent log levels, file output |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `runners/__init__.py` at the top level (not under `py/`) is sufficient for the `rbspaper-run` entry point | Standard Stack | Entry point resolution may require additional pyproject.toml package configuration |
| A2 | Bash `[[ =~ ]]` with simple numeric regex (`^[0-9]+$`) works in Bash 3.2 | Code Examples | Dataset name vs. index detection may need `grep` fallback |
| A3 | `uv run python` works from any subdirectory given correct PYTHONPATH | Architecture Patterns | If `uv` requires `pyproject.toml` in the current directory, project root detection must `cd` first |
| A4 | 11 datasets in registry is small enough that `seq`-based expansion is performant | Pattern 4 | Not an issue at current scale; would matter only with hundreds of datasets |

**Verification needed for A1:** The `rbspaper-run` entry point resolves `runners.py.runner:main`. Currently `runners/` has no `__init__.py`. Tests import `from runners.py.runner import ...` and work because pytest `pythonpath` includes `.`. Adding `runners/__init__.py` makes it a proper package. This should be sufficient.

**Verification needed for A2:** `[[ "$x" =~ ^[0-9]+$ ]]` -- Bash 3.2 supports `[[ =~ ]]` but the regex must be unquoted. This is a standard pattern, LOW risk.

**Verification needed for A3:** `uv run` resolves the project from the current directory by default. If the script runs from outside the project, we may need `--project` flag or `cd` to project root first. This is MEDIUM risk and should be tested.

## Open Questions (RESOLVED)

1. **Should `uv run` use `--project` flag?**
   - What we know: `uv run` auto-detects the project from the current directory by walking up for `pyproject.toml`.
   - What's unclear: Does `uv run python script.py` work when invoked from a directory outside the project root, given PYTHONPATH is set?
   - Recommendation: Use `cd "$PROJECT_ROOT" && uv run ...` or `uv run --project "$PROJECT_ROOT" ...` to be explicit. The `cd` approach is simpler and more portable.
   - RESOLVED: Plans 02/03 use `cd "$PROJECT_ROOT"` approach. Simpler and more portable.

2. **How should `--fraction` sample datasets?**
   - What we know: A-10 specifies `--fraction 0.25` samples ~25% of datasets.
   - What's unclear: Random shuffle (needs a seed) vs. deterministic first-N vs. evenly spaced.
   - Recommendation: Deterministic first-N. Simple, reproducible, no floating-point math issues. Document that `--fraction 0.25` on 11 datasets = first 2 datasets.
   - RESOLVED: Plan 03 implements deterministic first-N sampling.

3. **Where should bash-level log files go?**
   - What we know: A-05 says "log file per run". Runner already writes `pipeline.log` inside the run directory.
   - What's unclear: Bash logs at the wrapper level (timing, config) vs. Python logs (pipeline detail).
   - Recommendation: Bash-level logs go to `runners/bash/logs/local_single_YYYYMMDD_HHMMSS.log` or similar. Keep them separate from `outputs/` to avoid polluting experiment directories. Or simpler: bash logs to stdout only, runner logs to `pipeline.log`. Minimal bash logging avoids duplication.
   - RESOLVED: Plans use stdout-only for bash-level logging (`[HH:MM:SS]` timestamps). Runner handles file logging via `pipeline.log`.

4. **Should `config.sh` be gitignored via a pattern or explicit path?**
   - What we know: A-11 says "gitignore config.sh".
   - What's unclear: Whether to add `runners/bash/config.sh` to `.gitignore` or `runners/bash/config.sh` as a specific entry.
   - Recommendation: Explicit path: `runners/bash/config.sh`. More specific, less likely to conflict with future config files.
   - RESOLVED: Plan 02 Task 1 adds explicit `runners/bash/config.sh` to `.gitignore`.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Bash | All scripts | Y | 3.2.57 | — |
| uv | Script execution | Y | 0.11.2 | — |
| Python 3.12 | Runner | Y | (project env) | — |
| GNU date / BSD date | Timestamp logging | Y | (macOS) | — |
| seq | Batch range expansion | Y | (BSD/macos) | — |
| tr | Comma-to-newline | Y | (BSD/macos) | — |
| awk | Fraction calculation | Y | (BSD/macos) | — |

All dependencies available. No blocking items.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (via `uv run pytest`) |
| Config file | `pyproject.toml` inline (`[tool.pytest.ini_options]`) |
| Quick run command | `uv run pytest test/test_runner_cli_args.py test/test_runner_logging.py -x` |
| Full suite command | `uv run pytest test/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| __init__.py | `from runners.py.runner import main` works as proper package import | unit | `uv run pytest test/test_runner_logging.py -x` | Yes (existing) |
| print -> logging | `_print_summary` uses `logger.info()` instead of `print()` | unit | `uv run pytest test/test_runner_logging.py -x` | Yes (existing) |
| config.sh.example | Template file exists with DATA_ROOT placeholder | manual | N/A | Wave 0 |
| local_single.sh | Script exits 0 with valid args, exits 1 with invalid | shell | `bash runners/bash/local_single.sh ts2vec 0 --dry_run` | Wave 0 |
| local_batch.sh | Batch loop runs all datasets, produces summary | shell | `bash runners/bash/local_batch.sh ts2vec all --dry_run` | Wave 0 |
| fraction sampling | `--fraction 0.25` selects subset of datasets | shell | Manual: inspect expanded indices | Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest test/test_runner_cli_args.py test/test_runner_logging.py -x` (fast, targeted)
- **Per wave merge:** `uv run pytest test/ -x` (full suite)
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] Shell script tests -- no existing infrastructure for testing Bash scripts. Consider adding a minimal bash-test helper or manual verification checklist.
- [ ] `test_runner_logging.py` currently imports `from runners.py.runner import setup_logging`. After adding `runners/__init__.py`, verify existing imports still work.
- [ ] Print-to-logging conversion will break any tests that capture `stdout` from `main()`. `test_runner_cli_args.py` tests use `SystemExit` assertions (not stdout capture), so they are safe. Check `test_runner_logging.py` -- it tests `setup_logging()` only, not `print()` output.

## Security Domain

Not applicable -- this phase has no authentication, session management, or cryptography concerns. Bash scripts are local-only and invoke trusted Python code. No network calls.

## Sources

### Primary (HIGH confidence)
- `runners/py/runner.py` [VERIFIED: codebase] -- Current runner with 15 print() calls, setup_logging() already present
- `experiment_instances/instances.py` [VERIFIED: codebase] -- Registry with 2 experiments (ts2vec, autotcl)
- `src/rbspaper/data/data_setup.py` [VERIFIED: codebase] -- `get_all_datasets()` returns 11 dataset names
- `src/rbspaper/data/registry.py` [VERIFIED: codebase] -- 11 DatasetMetadata entries
- `pyproject.toml` [VERIFIED: codebase] -- Entry point `rbspaper-run = "runners.py.runner:main"`, pytest pythonpath config
- `.planning/codebase/CONVENTIONS.md` [VERIFIED: codebase] -- Error handling, logging, function design patterns
- `.planning/codebase/STRUCTURE.md` [VERIFIED: codebase] -- Directory layout, entry points

### Secondary (MEDIUM confidence)
- `_sources/autotsaugment/runners/hpc_uni_bash/runner.sh` [VERIFIED: codebase] -- Reference: project root detection, PYTHONPATH, batch loop
- `_sources/autotsaugment/runners_foundation_models/run_encoder_training_wrapper.sh` [VERIFIED: codebase] -- Reference: positional args, logging pattern

### Tertiary (LOW confidence)
- Bash 3.2 `[[ =~ ]]` regex support [ASSUMED] -- Standard feature, but should verify on macOS
- `uv run` behavior outside project root [ASSUMED] -- May require `--project` flag or `cd`

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all tools verified on localhost (uv 0.11.2, Bash 3.2.57, Python 3.12)
- Architecture: HIGH -- patterns derived from existing runner.py + autotsaugment references
- Pitfalls: MEDIUM -- Bash 3.2 compatibility and uv project detection are assumptions pending execution

**Research date:** 2026-05-07
**Valid until:** 2026-06-07 (stable domain -- bash scripting and logging patterns)
