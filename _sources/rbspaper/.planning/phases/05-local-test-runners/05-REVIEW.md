---
phase: 05-local-test-runners
reviewed: 2026-05-07T10:00:00Z
depth: deep
files_reviewed: 6
files_reviewed_list:
  - runners/__init__.py
  - runners/py/runner.py
  - runners/bash/config.sh.example
  - runners/bash/local_single.sh
  - runners/bash/local_batch.sh
  - .gitignore
findings:
  critical: 3
  warning: 4
  info: 2
  total: 9
status: issues_found
---

# Phase 05: Deep Code Review Report

**Reviewed:** 2026-05-07T10:00:00Z
**Depth:** deep
**Files Reviewed:** 6

## Summary

This phase introduces a bash-based local test runner infrastructure comprising two shell scripts
(`local_single.sh`, `local_batch.sh`) wrapping a Python CLI runner (`runner.py`), plus a
configurable `config.sh.example` template. The prior review round produced 10 findings
(CR-01, CR-02, WR-01 through WR-05, IN-01), all of which have been addressed in the submitted
fix commits.

However, this deep review surfaces three critical issues that prevent the code from functioning
correctly on the target platform (macOS default Bash 3.2) and introduce a code injection
vulnerability. Four warnings and two informational items round out the findings.

The most severe issue is that `local_batch.sh` uses `mapfile`, a Bash 4.0+ feature, which causes
the script to crash immediately on macOS (verified: `mapfile not found`). Additionally, the
`--fraction` CLI argument is interpolated directly into `python -c` strings, enabling arbitrary
Python code execution through crafted input.

## Critical Issues

### CR-01: `mapfile` not available in Bash 3.2 — local_batch.sh crashes on macOS

**File:** `runners/bash/local_batch.sh:227,232`
**Issue:** The script uses `mapfile -t INDICES < <(expand_dataset_spec ...)` on two lines.
`mapfile` (aka `readarray`) was introduced in Bash 4.0. macOS ships Bash 3.2.57 by default.
Verified on this system: `mapfile not found`. The script will crash with a "command not found"
error on any unmodified macOS machine, rendering the batch runner unusable on the primary
development platform.

This affects lines 227 and 232:
```bash
mapfile -t INDICES < <(expand_dataset_spec "$DATASET_SPEC" "$TOTAL_DATASETS")
# ...
mapfile -t INDICES < <(apply_fraction "$FRACTION" "${INDICES[@]}")
```

**Fix:** Replace `mapfile` with a POSIX-compatible `while read` loop:

```bash
# Line 227 replacement
INDICES=()
while IFS= read -r line; do
  INDICES+=("$line")
done < <(expand_dataset_spec "$DATASET_SPEC" "$TOTAL_DATASETS")

# Line 232 replacement
INDICES=()
while IFS= read -r line; do
  INDICES+=("$line")
done < <(apply_fraction "$FRACTION" "${INDICES[@]}")
```

Note: process substitution `<(...)` IS available in Bash 3.2; only `mapfile` is the problem.

### CR-02: Python code injection via `--fraction` argument in local_batch.sh

**File:** `runners/bash/local_batch.sh:99,215`
**Issue:** The `$FRACTION` variable (user-controlled CLI input) is interpolated directly into
`python -c` strings without sanitization. An attacker can inject and execute arbitrary Python
code.

Affected lines:

Line 99 (`apply_fraction`):
```bash
target=$(uv run python -c "import math; print(max(1, math.floor($count * $fraction)))")
```

Line 215 (validation):
```bash
valid=$(uv run python -c "print('yes' if 0 <= $FRACTION <= 1 else 'no')")
```

The validation itself is vulnerable — it does NOT prevent injection because the crafted payload
evaluates before the comparison. For example:

```bash
--fraction "__import__('os').system('echo pwned') or 0.5"
```

This would:
1. Execute `os.system('echo pwned')` (returns 0 on success)
2. Evaluate `0 <= 0 or 0.5 <= 1`, which is `True`
3. Pass validation AND execute the injected code

Both the validation and `apply_fraction` are vulnerable because the fix for WR-03
(which replaced `python -c` with `uv run python -c`) did NOT address the injection surface —
it only changed the invocation method.

**Fix:** Pass user input via `sys.argv` instead of string interpolation:

```bash
# apply_fraction (line 99):
target=$(uv run python -c "
import math, sys
count = int(sys.argv[1])
fraction = float(sys.argv[2])
print(max(1, math.floor(count * fraction)))
" "$count" "$fraction")

# Validation (line 215):
valid=$(uv run python -c "
import sys
try:
    val = float(sys.argv[1])
    print('yes' if 0 <= val <= 1 else 'no')
except ValueError:
    print('no')
" "$FRACTION")
```

This approach ensures user input is treated as data, never as code. The `float()` wrapper in the
validation also rejects non-numeric input safely.

### CR-03: Missing argument count guards in `--data_root` and `--output_dir` case arms

**File:** `runners/bash/local_batch.sh:169,177` and `runners/bash/local_single.sh:102,110`
**Issue:** The prior fix for CR-02 added argument count guards (`[ $# -lt 2 ]`) for `--seed`,
`--max_epochs`, and `--fraction`, but the guards were NOT added for `--data_root` and
`--output_dir` in either script.

In `local_batch.sh`:
```bash
--data_root)
  # MISSING: if [ $# -lt 2 ]; then ... fi
  DATA_ROOT_OVERRIDE="$2"
  shift 2
  ;;
--output_dir)
  # MISSING: if [ $# -lt 2 ]; then ... fi
  OUTPUT_DIR="$2"
  shift
  ;;
```

If a user runs `./local_batch.sh ts2vec all --data_root` (missing the path argument), `$2` is
unset. With `set -u` active, this triggers "unbound variable" and the script crashes with an
unclear error message instead of a helpful usage hint.

**Fix:** Add the same guard pattern used for `--seed`:

```bash
--data_root)
  if [ $# -lt 2 ]; then
    log "ERROR: --data_root requires a value" >&2
    exit 1
  fi
  DATA_ROOT_OVERRIDE="$2"
  shift 2
  ;;
--output_dir)
  if [ $# -lt 2 ]; then
    log "ERROR: --output_dir requires a value" >&2
    exit 1
  fi
  OUTPUT_DIR="$2"
  shift
  ;;
```

Apply identically to both `local_single.sh` (lines 97-111) and `local_batch.sh` (lines 163-177).

## Warnings

### WR-01: Missing `set -e` — silent command failures in bash scripts

**File:** `runners/bash/local_single.sh:17` and `runners/bash/local_batch.sh:23`
**Issue:** Both scripts use `set -uo pipefail` but omit `-e` (exit on error). While the scripts
do manually check exit codes for critical operations (`"${CMD[@]}"`, `cd`), several intermediate
operations lack error handling:

- `cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"` (lines 40, 46) — if the copy fails (permissions,
  read-only filesystem), the script proceeds to `source "$CONFIG_FILE"`, which would source a
  potentially empty or partial file.
- `source "$CONFIG_FILE"` (lines 47, 53) — if config.sh is malformed, bash may exit or
  continue with partially initialized variables, depending on the error.

**Fix:** Either add `set -e` or wrap critical operations with explicit error checks:

```bash
if ! cp "$CONFIG_EXAMPLE" "$CONFIG_FILE"; then
  log "ERROR: Failed to copy config template to $CONFIG_FILE" >&2
  exit 1
fi
```

### WR-02: `$dataset_index` interpolated into `python -c` in batch loop

**File:** `runners/bash/local_batch.sh:267-271`
**Issue:** The dataset index resolution uses string interpolation:

```bash
DATASET_NAME=$(uv run python -c "
from src.rbspaper.data.data_setup import get_all_datasets
datasets = get_all_datasets(form='list')
print(datasets[$dataset_index])
")
```

While `$dataset_index` originates from `expand_dataset_spec` and is validated with `grep -qE
'^[0-9]+$'` on line 243, the validation only checks for digits — it does NOT enforce a specific
range before the Python code executes. If the `grep` validation were ever bypassed or weakened,
this would become an injection surface. More importantly, it is inconsistent with the secure
pattern recommended for CR-02's fix.

**Fix:** Use `sys.argv` for the index value:

```bash
DATASET_NAME=$(uv run python -c "
import sys
from src.rbspaper.data.data_setup import get_all_datasets
datasets = get_all_datasets(form='list')
print(datasets[int(sys.argv[1])])
" "$dataset_index")
```

### WR-03: Redundant directory creation in `_build_pipeline_config`

**File:** `runners/py/runner.py:276`
**Issue:** The function creates `run_dir.parent` explicitly:

```python
run_dir = output_dir / run_name
run_dir.parent.mkdir(parents=True, exist_ok=True)
```

However, this is unnecessary. Line 416 later creates the full `run_dir` with
`parents=True`:

```python
config.artifacts.run_dir.mkdir(parents=True, exist_ok=True)
```

The `parents=True` flag on line 416 handles all intermediate directory creation. Line 276 creates
a subdirectory prematurely during config assembly, before logging is set up. If the config
assembly fails after this point but before line 416, an empty directory structure is left behind.

**Fix:** Remove line 276 entirely. The directory creation on line 416 is sufficient.

### WR-04: `extra_params or None` silently drops empty dict

**File:** `runners/py/runner.py:227`
**Issue:** The code passes `extra_params or None` to `get_datamodule_with_downstream_tasks`:

```python
extra_params: dict[str, str | None] = {}
if forecasting_mode:
    extra_params['forecasting_mode'] = forecasting_mode
if batch_size_override:
    extra_params['batch_size'] = str(batch_size_override)

datamodule_result = get_datamodule_with_downstream_tasks(
    ...
    extra_params=extra_params or None,
)
```

When both `forecasting_mode` and `batch_size_override` are unset, `extra_params` is an empty
dict `{}`. The expression `extra_params or None` evaluates to `None` (empty dict is falsy).
This means an empty dict is NEVER passed to the callee — it always receives `None` instead.

Looking at the callee (data_setup.py line 211), the parameter type is
`dict[str, Any] | None` and line 233 does `extra_params = extra_params or {}`. So `None` and
`{}` are functionally equivalent at the callee. The current behavior is correct but the
`or None` idiom is misleading — it signals "sometimes I pass None, sometimes I pass a dict"
when the intent is clearly "pass nothing if empty."

**Fix:** Remove the `or None` for clarity, since the callee handles empty dicts identically:

```python
extra_params=extra_params if extra_params else None,
```

Or simply pass the dict directly (the callee's `or {}` is defensive):

```python
extra_params=extra_params,
```

## Informational

### IN-01: Inconsistent `_resolve_dataset` error reporting pattern

**File:** `runners/py/runner.py:308-326`
**Issue:** The function creates new `ArgumentParser` instances solely to call `.error()`:

```python
parser = argparse.ArgumentParser(description='Run a robustness benchmark experiment.')
parser.error('one of --dataset_name or --dataset_index is required')
```

This pattern (repeated on lines 309, 314, 321) works but is suboptimal. Creating a fresh parser
means the error message does not include the standard argparse footer ("try -h for help"). A
cleaner approach is to reuse the original parser from `_parse_args`, or define a dedicated
`_error(msg: str) -> NoReturn` helper that prints to stderr and calls `sys.exit(2)`.

### IN-02: Logging format inconsistency between bash and Python layers

**File:** `runners/bash/local_single.sh:31-33`, `runners/bash/local_batch.sh:37-39`,
`runners/py/runner.py:188-196`
**Issue:** The bash `log()` function uses `[$(date '+%H:%M:%S')]` (time-only), while the Python
logger uses `%(asctime)s` which includes full date-time by default. When a batch run produces
output from both layers, the timestamps have different granularity and format. This is purely
cosmetic but can make log analysis slightly harder.

Example:
```
[14:32:01] Running experiment 'ts2vec' on dataset 'Coffee' (seed=42)
[14:32:01] Command: uv run python /path/to/runner.py ...
2026-05-07 14:32:01,234 [INFO] runners.py.runner: Model: ts2vec
```

Consider standardizing on a shared format (e.g., `YYYY-MM-DD HH:MM:SS`) for unified log files.

## Cross-File Analysis

### Import graph (runner.py)

```
runners/py/runner.py
├── experiment_instances.data_utils.build_dataset_task_profile
├── experiment_instances.instances.{ExperimentInstance, EXPERIMENTS_REGISTRY,
│   get_experiment_instance, list_experiment_ids}
├── src.rbspaper.attacks.enums.AttackFamily
├── src.rbspaper.data.data_setup.{get_all_datasets,
│   get_datamodule_with_downstream_tasks}
├── src.rbspaper.pipeline.config.{build_hierarchical_run_name, DataConfig,
│   DatasetTaskProfile, DownstreamTaskConfig, ExperimentPipelineConfig,
│   PipelineArtifactConfig, RepresentationAnalysisConfig,
│   RepresentationEncodingConfig, TrainingConfig}
├── src.rbspaper.pipeline.core.run_experiment_pipeline
├── src.rbspaper.pipeline.setup.model.build_model_from_parameters
└── src.rbspaper.pipeline.state.{compute_config_hash, load_pipeline_state,
    STATE_FILENAME}
```

All imports are verified against the codebase. No broken imports detected.

### Error propagation chain

```
local_single.sh / local_batch.sh
  └── "uv run python runners/py/runner.py ..."
       └── main()
            ├── _resolve_dataset() → parser.error() → sys.exit(2)
            ├── get_experiment_instance() → KeyError
            ├── _build_pipeline_config() → various (KeyError, AttributeError)
            └── run_experiment_pipeline() → various
       └── except KeyboardInterrupt → sys.exit(130)
       └── except Exception → sys.exit(1)
```

Exit code propagation is correct: `local_single.sh` captures `$?` from the runner and passes it
through. `local_batch.sh` captures per-run exit codes and reports them in the summary.

### Attack family filtering (redundant but correct)

The `--attack_family` flag triggers filtering at two levels:
1. `get_experiment_instance()` in `instances.py` returns a deepcopy with only the specified family.
2. `_build_pipeline_config()` checks `attack_family` again and selects from
   `experiment_instance.attack_families.get(attack_family, ())`.

Since the instance is already filtered at step 1, step 2 is redundant. However, it is not
incorrect — it acts as a defensive double-check. If `get_experiment_instance` ever changes to
return the unfiltered instance, the pipeline config would still be correct.

## Prior Fix Verification

All findings from the previous review round have been verified as fixed:

- **CR-01 (previous):** `local_batch.sh` now resolves indices to names using `--dataset_name`
  (line 277). Verified.
- **CR-02 (previous):** `--seed`, `--max_epochs`, and `--fraction` have argument count guards.
  Partially fixed — `--data_root` and `--output_dir` are missing guards (see CR-03 above).
- **WR-02:** Reverse range validation added in `expand_dataset_spec` (line 75). Verified.
- **WR-03:** `apply_fraction` uses `uv run python -c` (line 99). Invocation fixed; injection
  vulnerability remains (see CR-02 above).
- **WR-04:** `--fraction` validation added (lines 214-220). Logic fixed; injection via the
  validation itself remains (see CR-02 above).
- **WR-05:** `--list_experiments` uses `logging.basicConfig` with consistent format
  (lines 358-361). Verified.
- **IN-01:** `from __future__ import annotations` removed. Verified — not present in runner.py.

---

_Reviewed: 2026-05-07T10:00:00Z_
_Reviewer: gsd-code-review (deep)_
_Depth: deep_
