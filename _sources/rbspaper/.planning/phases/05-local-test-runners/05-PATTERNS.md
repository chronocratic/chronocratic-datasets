# Phase 5: Local Test Runners - Pattern Map

**Mapped:** 2026-05-07
**Files analyzed:** 6 (4 new, 2 modified)
**Analogs found:** 5 / 6

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `runners/__init__.py` | config (package marker) | — | `experiment_instances/__init__.py` | exact |
| `runners/bash/config.sh.example` | config | file-I/O | `_sources/autotsaugment/runners/hpc_uni_bash/runner.sh` (config section) | partial |
| `runners/bash/local_single.sh` | config (script) | request-response | `_sources/autotsaugment/runners_foundation_models/run_encoder_training_wrapper.sh` | role-match |
| `runners/bash/local_batch.sh` | config (script) | batch | `_sources/autotsaugment/runners/hpc_uni_bash/runner.sh` | role-match |
| `runners/py/runner.py` | controller (CLI) | request-response | (self — modify existing) | n/a |
| `.gitignore` | config | file-I/O | (self — append entry) | n/a |

## Pattern Assignments

### `runners/__init__.py` (config, package marker)

**Analog:** `experiment_instances/__init__.py`

**Purpose:** Make `runners/` a proper Python package so `rbspaper-run = "runners.py.runner:main"` entry point in `pyproject.toml` line 30 resolves correctly.

**Imports pattern** — `experiment_instances/__init__.py` lines 1-17:
```python
"""Experiment instances and helpers for robustness benchmarking."""

from experiment_instances.data_utils import build_dataset_task_profile
from experiment_instances.instances import (
    ExperimentInstance,
    EXPERIMENTS_REGISTRY,
    get_experiment_instance,
    list_experiment_ids,
)

__all__ = [
    'EXPERIMENTS_REGISTRY',
    'ExperimentInstance',
    'build_dataset_task_profile',
    'get_experiment_instance',
    'list_experiment_ids',
]
```

**Adaptation for `runners/__init__.py`:** This file needs to be minimal -- a simple docstring-only package marker. The `rbspaper-run` entry point resolves `runners.py.runner:main`, meaning Python needs `runners/` to be a package. The `runners/py/__init__.py` already exists (empty file). Following the convention from `experiment_instances/__init__.py`, the `runners/__init__.py` should have a module docstring. It does NOT need `__all__` or re-exports since it is not a barrel file -- the entry point imports directly from `runners.py.runner`.

**Recommended content:**
```python
"""Entry point package for the rbspaper-run CLI."""
```

---

### `runners/bash/config.sh.example` (config, template)

**Analog:** No direct analog (greenfield config template). Pattern derived from RESEARCH.md code examples and runner.py's `--data_root` argument.

**Pattern from `runner.py`** lines 90-92 — the `--data_root` argument shows what config.sh must provide:
```python
parser.add_argument(
    '--data_root', type=Path, default=None, help='Root directory containing dataset files.'
)
```

**Pattern from `runner.py`** lines 99-102 — the `--output_dir` default:
```python
parser.add_argument(
    '--output_dir',
    type=Path,
    default=Path('outputs'),
    help='Directory for experiment outputs (default: outputs).',
)
```

**Pattern from `runner.py`** lines 106-107 — the `--seed` default:
```python
parser.add_argument(
    '--seed', type=int, default=42, help='Random seed for reproducibility (default: 42).'
)
```

**Recommended content** (sourced from RESEARCH.md config template):
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

**Convention:** The `.example` suffix follows standard practice. The actual `config.sh` is gitignored (see `.gitignore` modification below).

---

### `runners/bash/local_single.sh` (script, request-response)

**Analog:** `_sources/autotsaugment/runners_foundation_models/run_encoder_training_wrapper.sh`

**Project root detection** — `run_encoder_training_wrapper.sh` lines 1-7:
```bash
current_script_dir=$(cd "$(dirname "$0")" || exit; pwd)
parent_dir=$(dirname "$current_script_dir")

foundation_model_dir="${parent_dir}/foundation_models/test"

# add the foundation model directory and the main project directory to the PYTHONPATH
export PYTHONPATH="${foundation_model_dir}:${parent_dir}:${PYTHONPATH}"
```

**Adaptation for this project:**
```bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1; pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." || exit 1; pwd)

# Verify we're at the project root
if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
  echo "[ERROR] Cannot find pyproject.toml in $PROJECT_ROOT" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
```

**Note:** `${BASH_SOURCE[0]}` preferred over `$0` per RESEARCH.md Pitfall 5. PYTHONPATH includes project root (which makes `experiment_instances` importable).

**Positional args + defaults** — `run_encoder_training_wrapper.sh` lines 23-35:
```bash
dataset_index="${1:-0}"
dataset_name="${uea_datasets[${dataset_index}]}"

loss_augmentation_sampling_method="${2:-"randsampling"}"

epochs=${3:-10}

default_encoder_save_folder="${foundation_model_dir}/encoders/saved_models/${loss_augmentation_sampling_method}/${dataset_name}"
encoder_save_folder="${4:-"${default_encoder_save_folder}"}"
```

**Adaptation:** Use `${1:-}` for experiment_id (required, no default), `${2:-}` for dataset (required, no default). Named flags `--seed`, `--force`, `--max_epochs`, `--data_root`, `--output_dir` forwarded to Python runner.

**Config file auto-creation** — pattern from RESEARCH.md:
```bash
CONFIG_FILE="$SCRIPT_DIR/config.sh"
CONFIG_EXAMPLE="$SCRIPT_DIR/config.sh.example"

if [ ! -f "$CONFIG_FILE" ]; then
  cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"
  echo "[INFO] Created $CONFIG_FILE from template."
  echo "[INFO] Please set DATA_ROOT in $CONFIG_FILE and re-run."
  exit 1
fi

source "$CONFIG_FILE"
```

**Bash timestamp logger** — pattern from RESEARCH.md:
```bash
log() {
  echo "[$(date '+%H:%M:%S')] $*"
}
```

**Argument forwarding to `uv run`** — pattern from RESEARCH.md:
```bash
CMD=(
  "uv" "run" "python" "$PROJECT_ROOT/runners/py/runner.py"
  "--experiment_id" "$EXP_ID"
  "--data_root" "$DATA_ROOT"
)

if [[ "$DATASET" =~ ^[0-9]+$ ]]; then
  CMD+=("--dataset_index" "$DATASET")
else
  CMD+=("--dataset_name" "$DATASET")
fi

[ -n "${SEED:-}" ] && CMD+=("--seed" "$SEED")
[ -n "${MAX_EPOCHS:-}" ] && CMD+=("--max_epochs" "$MAX_EPOCHS")
[ "${FORCE:-false}" = "true" ] && CMD+=("--force")

"${CMD[@]}"
exit_code=$?
```

---

### `runners/bash/local_batch.sh` (script, batch)

**Analog:** `_sources/autotsaugment/runners/hpc_uni_bash/runner.sh`

**Project root detection** — `runner.sh` lines 10-15:
```bash
current_script_dir=$(cd "$(dirname "$0")" || exit; pwd)
parent_dir=$(dirname "$current_script_dir")
project_main_dir=$(dirname "$parent_dir")

export PYTHONPATH="${project_main_dir}:${PYTHONPATH}"
```

**Loop pattern** — `runner.sh` lines 47-99:
```bash
for ((s=0; s<total_runs; s++)); do
  experiments_seed=$((s*2+1))
  # ... generate job file, submit
done
```

**Adaptation for local batch** — sequential loop with exit code collection:
```bash
total_runs=0
passed=0
failed=0
FAILED_RUNS=()

for dataset_index in "$@"; do
  total_runs=$((total_runs + 1))
  log "Run $total_runs: $EXP_ID on dataset index $dataset_index"

  # Forward to runner via uv run
  cd "$PROJECT_ROOT" || exit 1
  uv run python runners/py/runner.py \
    --experiment_id "$EXP_ID" \
    --dataset_index "$dataset_index" \
    --data_root "$DATA_ROOT" \
    # ... other overrides
  run_exit=$?

  if [ "$run_exit" -eq 0 ]; then
    passed=$((passed + 1))
  else
    failed=$((failed + 1))
    FAILED_RUNS+=("dataset_index=$dataset_index (exit=$run_exit)")
  fi
done
```

**Dataset spec expansion** — from RESEARCH.md Pattern 4:
```bash
expand_dataset_spec() {
  local spec="$1"
  local total="$2"

  if [ "$spec" = "all" ]; then
    seq 0 "$((total - 1))"
    return
  fi

  if echo "$spec" | grep -q '-'; then
    local start="${spec%-*}"
    local end="${spec#*-}"
    seq "$start" "$end"
    return
  fi

  if echo "$spec" | grep -q ','; then
    echo "$spec" | tr ',' '\n'
    return
  fi

  echo "$spec"
}
```

**Aggregate report** — from RESEARCH.md:
```bash
log "========================================"
log "        BATCH RUN SUMMARY"
log "========================================"
log "Total:  $total_runs"
log "Passed: $passed"
log "Failed: $failed"
log "========================================"

if [ "$failed" -gt 0 ]; then
  log "Failed runs:"
  for entry in "${FAILED_RUNS[@]}"; do
    log "  - $entry"
  done
  exit 1
fi
exit 0
```

**Fraction sampling** — from RESEARCH.md, deterministic first-N:
```bash
apply_fraction() {
  local indices=("$@")
  local fraction="$1"
  shift
  local count=${#indices[@]}
  # Use python for float arithmetic (POSIX-safe)
  local target
  target=$(python -c "import math; print(max(1, math.floor($count * $fraction)))")
  printf '%s\n' "${indices[@]}" | head -n "$target"
}
```

---

### `runners/py/runner.py` (controller, CLI — modify existing)

**Analog:** Self-modification. Print-to-logging conversion.

**Existing `setup_logging()`** lines 167-197:
```python
def setup_logging(*, log_dir: Path, log_level: int = logging.INFO) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if root.handlers:
        return

    file_handler = logging.FileHandler(log_dir / 'pipeline.log')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    )
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    root.addHandler(stream_handler)
```

**Print calls to convert** — identified at the following locations:

| Line | Current Code | New Code |
|------|-------------|----------|
| 335 | `print(f'Model:        {model_name}')` | `logger.info('Model:        %s', model_name)` |
| 336 | `print(f'Input dims:   {config.data.data_module.n_features}')` | `logger.info('Input dims:   %s', config.data.data_module.n_features)` |
| 337 | `print(f'Seq length:   {config.data.data_module.sequence_len}')` | `logger.info('Seq length:   %s', config.data.data_module.sequence_len)` |
| 338 | `print(f'Attacks:      {attack_names}')` | `logger.info('Attacks:      %s', attack_names)` |
| 339 | `print(f'Tasks:        {task_names}')` | `logger.info('Tasks:        %s', task_names)` |
| 340 | `print(f'Max epochs:   {config.training.trainer_kwargs.get("max_epochs")}')` | `logger.info('Max epochs:   %s', config.training.trainer_kwargs.get('max_epochs'))` |
| 341 | `print(f'Output dir:   {config.artifacts.run_dir}')` | `logger.info('Output dir:   %s', config.artifacts.run_dir)` |
| 342 | `print(f'Seed:         {config.seed}')` | `logger.info('Seed:         %s', config.seed)` |
| 358 | `print('Available experiments:')` | `logger.info('Available experiments:')` |
| 366 | `print(f'  - {exp_id} ({summary})')` | `logger.info('  - %s (%s)', exp_id, summary)` |
| 420 | `print('\nRunning experiment...')` | `logger.info('Running experiment...')` |
| 429 | `print(f'Resuming from checkpoint: {state_path}')` | `logger.info('Resuming from checkpoint: %s', state_path)` |
| 435 | `print(f'\nExperiment complete. Results saved to: {results.run_dir}')` | `logger.info('Experiment complete. Results saved to: %s', results.run_dir)` |
| 442 | `print('\nInterrupted by user.')` | `logging.basicConfig(level=logging.INFO); logging.getLogger(__name__).info('Interrupted by user.')` |
| 445 | `print(f'\nError: {e}', file=sys.stderr)` | `logging.basicConfig(level=logging.WARNING); logging.getLogger(__name__).warning('Error: %s', e)` |

**Key patterns:**
- Use `%s`-style formatting (not f-strings) for lazy evaluation -- standard Python logging best practice per RESEARCH.md.
- `_print_summary()` function renamed to `_log_summary()` (private convention maintained).
- For lines 442 and 445 (exception handlers outside `main()`), `setup_logging()` may not have been called yet. Use `logging.basicConfig(level=logging.INFO)` as fallback for those cases.
- For `--list_experiments` path (lines 358, 366), `setup_logging()` is NOT called. Either call `basicConfig` early or use a dedicated minimal logger setup.

**Pitfall warning (RESEARCH.md Pitfall 4):** `--list_experiments` path exits before `setup_logging()` is called. A `logging.basicConfig(level=logging.INFO)` call is needed at the top of the `--list_experiments` branch, or `setup_logging()` must be called earlier in the flow.

**Existing test pattern** — `test_runner_logging.py` lines 14-30 (shows how to reset logger between tests):
```python
def test_creates_file_handler(self, tmp_path: Path) -> None:
    root = logging.getLogger()
    root.handlers.clear()

    from runners.py.runner import setup_logging

    log_dir = tmp_path / 'runs' / 'test_run'
    setup_logging(log_dir=log_dir)
    # ...
```

**Test impact:** After the print-to-logging conversion:
- `test_runner_cli_args.py` -- NO impact. Uses `SystemExit` assertions, not stdout capture.
- `test_runner_logging.py` -- NO impact. Tests `setup_logging()` only, not print output.
- Any tests that capture stdout from `_print_summary()` (renamed to `_log_summary()`) will need updating to use log capture fixtures.

---

### `.gitignore` (config — modify existing)

**Analog:** Self-modification. Append one line.

**Current structure** — lines 1-232 (standard Python gitignore from GitHub template, plus project-specific entries at the end):
```
# Extra Files
uv_lock.sh
_sources/
```

**Required addition:**
```
# Local runner config (gitignored; template is config.sh.example)
runners/bash/config.sh
```

---

## Shared Patterns

### Bash Script Convention
**Source:** `.claude/hooks/gsd-validate-commit.sh` + `_sources/autotsaugment/runners/hpc_uni_bash/runner.sh`
**Apply to:** `local_single.sh`, `local_batch.sh`

- Shebang: `#!/usr/bin/env bash`
- Script directory: `${BASH_SOURCE[0]}` (not `$0`)
- Project root: resolved from `SCRIPT_DIR/../..` (two levels up from `runners/bash/`)
- Always `cd` to `PROJECT_ROOT` before running `uv run`
- Exit on errors: `set -euo pipefail` not used; explicit error checks preferred (hooks pattern from `gsd-validate-commit.sh`)

### Config File Auto-Creation Pattern
**Source:** RESEARCH.md Pattern 2
**Apply to:** Both `local_single.sh` and `local_batch.sh`

- Copy `.example` to actual on first run
- Source the config file after creation
- Validate `DATA_ROOT` is set and exists as a directory
- Exit with helpful message if validation fails

### Timestamp Logger Pattern
**Source:** RESEARCH.md Pattern 3
**Apply to:** Both `local_single.sh` and `local_batch.sh`

```bash
log() {
  echo "[$(date '+%H:%M:%S')] $*"
}
```

### Python Logging Pattern
**Source:** `runner.py` lines 167-197 (`setup_logging()`)
**Apply to:** All print-to-logging conversions in `runner.py`

- Use `logger = logging.getLogger(__name__)` module-level pattern
- `%s`-style format strings (not f-strings) for lazy evaluation
- `logger.info()` for informational output (replacing most `print()`)
- `logger.warning()` for resume messages
- `logging.basicConfig()` fallback for pre-setup_logging paths

### `uv run` Execution Pattern
**Source:** `runner.py` docstring lines 8-12 + project convention
**Apply to:** Both bash scripts

- Always use `uv run python` (not direct `python`)
- `cd` to PROJECT_ROOT before invoking `uv run`
- Set PYTHONPATH to include project root for `experiment_instances` module

### Bash 3.2 Compatibility
**Source:** RESEARCH.md Anti-Patterns
**Apply to:** Both bash scripts

- No `declare -A` (associative arrays)
- No `(( ))` compound arithmetic -- use `$(( ))` instead
- `[[ =~ ]]` with simple numeric regex is acceptable
- `seq` for range expansion (POSIX/BSD compatible)
- `tr ',' '\n'` for comma-separated lists

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `runners/bash/config.sh.example` | config (template) | file-I/O | No `.example` config files exist in the codebase yet. Pattern derived from runner.py argument defaults. |

## Metadata

**Analog search scope:** `runners/`, `experiment_instances/`, `_sources/`, `.claude/hooks/`, `test/`, `.gitignore`
**Files scanned:** 10
**Pattern extraction date:** 2026-05-07
