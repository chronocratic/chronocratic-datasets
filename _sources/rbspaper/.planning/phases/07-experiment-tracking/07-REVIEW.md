---
phase: 07-experiment-tracking
reviewed: 2026-05-08T12:00:00Z
depth: deep
files_reviewed: 8
files_reviewed_list:
  - pyproject.toml
  - ruff.toml
  - runners/py/runner.py
  - src/rbspaper/pipeline/config.py
  - src/rbspaper/pipeline/core.py
  - src/rbspaper/pipeline/loggers.py
  - test/test_logger_factory.py
  - test/test_runner_cli_args.py
findings:
  critical: 3
  warning: 5
  info: 4
  total: 12
status: issues_found
---

# Phase 07: Code Review Report

**Reviewed:** 2026-05-08T12:00:00Z
**Depth:** deep
**Files Reviewed:** 8
**Status:** issues_found

## Summary

This phase introduces W&B + TensorBoard dual logging, step timing instrumentation, HPC auto-detection via `SLURM_JOB_ID`, and `--tracking_mode` CLI support across 8 files (2 new, 6 modified). The logger factory pattern in `loggers.py` is well-structured with appropriate lazy imports and graceful fallbacks. However, deep review surfaced 3 critical defects: a `wandb.Table` comparison that is entirely empty due to key mismatches (rendering the W&B table feature non-functional), a missing `WandbLogger` TYPE_CHECKING import causing type checker failures, and an untyped parameter in `runner.py` violating project conventions. Additionally, the `_log_results_to_wandb` function will crash on network errors when W&B is in online mode, with no error handling around the `run.log()` call.

## Critical Issues

### CR-01: `wandb.Table` comparison columns reference wrong flattened keys -- table is always empty

**File:** `src/rbspaper/pipeline/loggers.py:194-222`
**Issue:** The `wandb.Table` in `_log_results_to_wandb` references flat keys like `classification_clean_accuracy`, `geometry_centroid_margin`, `shift_mean_l2`, etc. However, the `results_summary` dict passed from `core.py` has a deeply nested structure:

```
results_summary_data = {
    'model_name': ...,
    'downstream_metrics': {
        'classification': [
            {'scope': 'clean', 'metrics': {'accuracy': 0.85}},
            ...
        ]
    },
    'analysis': {
        'classification': {
            'clean_geometry': {'centroid_margin': 3.2},
            ...
        }
    }
}
```

After `_flatten_dict`, the actual keys are:
- `downstream_metrics_classification_0_metrics_accuracy` (not `classification_clean_accuracy`)
- `analysis_classification_clean_geometry_centroid_margin` (not `geometry_centroid_margin`)
- `analysis_classification_attacked_shift_mean_l2` (not `shift_mean_l2`)

Every `.get()` call in the table row returns `None` except `model_name` and `timing_total`. The comparison table logged to W&B is effectively a mostly-empty row. This makes the entire `wandb.Table` feature (D-06) non-functional.

**Fix:** Either (a) restructure `results_summary_data` to have flat keys that match the table columns, or (b) update the table column lookups to reference the actual flattened keys. Option (a) is cleaner:

```python
# In core.py, build a flat results_summary for W&B:
flat_metrics: dict[str, Any] = {}
for task_name, task_metrics in downstream_metrics.items():
    for case in task_metrics:
        if case['scope'] == 'clean':
            for k, v in case['metrics'].items():
                flat_metrics[f'{task_name}_clean_{k}'] = v

flat_analysis: dict[str, Any] = {}
for task_name, task_analysis in analysis_results.items():
    for metric_name, metric_data in task_analysis.items():
        if isinstance(metric_data, dict):
            for k, v in metric_data.items():
                flat_analysis[f'{task_name}_{metric_name}_{k}'] = v

results_summary_data = {
    'model_name': results.model_name,
    'checkpoint_path': str(checkpoint_path) if checkpoint_path else None,
    **flat_metrics,
    **flat_analysis,
    'total_seconds': timing.get('total', 0.0),
}
```

Then update the table to reference these keys:
```python
flat_results.get('classification_clean_accuracy'),
flat_results.get('classification_clean_f1'),
flat_results.get('forecasting_clean_mae'),
flat_results.get('classification_clean_geometry_centroid_margin'),
flat_results.get('classification_attacked_shift_mean_l2'),
```

### CR-02: Missing `WandbLogger` in TYPE_CHECKING block causes type checker errors

**File:** `src/rbspaper/pipeline/loggers.py:12-16`
**Issue:** The `TYPE_CHECKING` block imports `pl` but does NOT import `WandbLogger` or `TensorBoardLogger`. The function signatures on lines 23, 80, and 139 reference `WandbLogger` and `TensorBoardLogger` in type annotations. While `from __future__ import annotations` prevents runtime crashes (annotations are strings), any static type checker (mypy, pyright, pyright-based tools) will report `WandbLogger` and `TensorBoardLogger` as undefined names.

The current TYPE_CHECKING block:
```python
if TYPE_CHECKING:
    from pathlib import Path
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import WandbLogger  # <-- MISSING
```

Meanwhile, `TensorBoardLogger` is imported at runtime (line 10), which is unnecessary -- it could also be moved under TYPE_CHECKING since it's only used in annotations. However, `TensorBoardLogger` IS instantiated at runtime on line 53, so it must remain a runtime import.

**Fix:** Add `WandbLogger` to the TYPE_CHECKING block:

```python
if TYPE_CHECKING:
    from pathlib import Path

    import lightning.pytorch as pl
    from lightning.pytorch.loggers import WandbLogger
```

Note: `TensorBoardLogger` must stay as a runtime import since it's instantiated in `create_loggers`.

### CR-03: `_build_pipeline_config` uses untyped `loggers: tuple = ()` parameter

**File:** `runners/py/runner.py:223`
**Issue:** The `loggers` parameter is typed as bare `tuple` with default `()`. The project CLAUDE.md mandates: "Type hints should be used for all functions, including return types; it is not optional to omit them." Bare `tuple` provides zero type safety -- it accepts any tuple of any element type. The `runner.py` file does NOT have `from __future__ import annotations`, so forward references would fail.

The ruff.toml `ANN401` ignore applies to `runners/py/*.py`, but `ANN401` only covers `Any` types, not bare generics like `tuple`. This is a different rule entirely.

**Fix:** Use a string annotation to avoid import issues:

```python
loggers: tuple['pl.loggers.LightningLogger', ...] = (),
```

Or more practically, since `pl` is not imported in runner.py and we do not want to add that import:

```python
loggers: tuple[object, ...] = (),
```

Or the cleanest approach -- import the Logger type:

```python
from lightning.pytorch.loggers import Logger

def _build_pipeline_config(
    ...
    loggers: tuple[Logger, ...] = (),
) -> ExperimentPipelineConfig:
```

## Warnings

### WR-01: `_log_results_to_wandb` crashes on W&B network errors with no handling

**File:** `src/rbspaper/pipeline/loggers.py:186-222`
**Issue:** When W&B is in `online` mode and there is a network disruption (common in research environments), `run.log()` at line 191 and `run.log({'comparison': table})` at line 222 will raise exceptions from the W&B SDK. These exceptions propagate through `core.py` line 347-350, crashing the entire pipeline AFTER all computation and disk persistence has completed. The results are saved to disk but the process exits with a non-zero code, which can break batch job orchestration (SLURM arrays, Makefiles, etc.).

**Fix:** Wrap the W&B logging in a try/except with a warning log:

```python
def _log_results_to_wandb(...) -> None:
    wandb_logger = _find_wandb_logger(loggers=loggers)
    if wandb_logger is None:
        return

    try:
        import wandb  # noqa: PLC0415
        run = wandb_logger.experiment
        flat_results = _flatten_dict(d=results_summary)
        flat_timing = {f'timing_{k}': v for k, v in timing.items()}
        run.log({**flat_results, **flat_timing})
        # ... table logging ...
    except Exception:
        logger.warning('Failed to log results to W&B. Results persisted to disk.', exc_info=True)
```

### WR-02: `timing['total']` does not represent actual wall-clock elapsed time

**File:** `src/rbspaper/pipeline/core.py:344`
**Issue:** `timing['total']` is computed as `sum(timing.values())`, which sums individual step timings. However, step boundaries overlap slightly (each `step_start = time.perf_counter()` reset happens after the previous timing capture, but there are operations between steps -- e.g., `_collect_partition_tensors` on line 180 -- that are not attributed to any timing key). More importantly, the `timing` dict also contains a `task_loop` key that wraps the entire for-loop, but the per-task steps inside (encoding, attacks, evaluate) are not individually timed. The sum includes `task_loop` which is already an aggregate, so there is double-counting if any sub-timings existed.

The leftover `step_start = time.perf_counter()` on lines 315-316 after the analysis step is dead code -- it initializes a timing capture that is never completed.

**Fix:** Remove the dead `step_start` initialization after analysis. For accurate total wall-clock time, use a single pair of `time.perf_counter()` calls at the top and bottom of `run_experiment_pipeline`:

```python
pipeline_start = time.perf_counter()
# ... pipeline execution ...
timing['total'] = time.perf_counter() - pipeline_start
```

### WR-03: `_log_config_to_wandb` has no error handling for W&B config update failures

**File:** `src/rbspaper/pipeline/loggers.py:150-154`
**Issue:** Line 154 calls `wandb_logger.experiment.config.update(config_data)`. If W&B is in online mode and the network drops at pipeline start, this call will raise an exception. Unlike `_log_results_to_wandb` (which has the same issue per WR-01), this happens early in the pipeline before any computation, potentially aborting the entire run due to a transient network issue when TensorBoard-only logging would have been sufficient.

**Fix:** Add the same try/except pattern as recommended for WR-01:

```python
def _log_config_to_wandb(...) -> None:
    wandb_logger = _find_wandb_logger(loggers=loggers)
    if wandb_logger is None:
        return
    try:
        wandb_logger.experiment.config.update(config_data)
    except Exception:
        logger.warning('Failed to update W&B config. Continuing without W&B config logging.', exc_info=True)
```

### WR-04: No test coverage for HPC auto-detection logic

**File:** `test/test_runner_cli_args.py` (missing tests)
**Issue:** The runner's HPC auto-detection (`os.environ.get('SLURM_JOB_ID')` on `runner.py:403-405`) is a key requirement (D-03) but has zero test coverage. The test file covers `--tracking_mode` CLI parsing and default values, but does not test the auto-detection branch where `args.tracking_mode is None` and the code checks for `SLURM_JOB_ID`.

This is tested indirectly only if someone runs the full `main()` with the env var set, which the existing tests do not do (they test early-exit paths like `--list_experiments` or validation errors).

**Fix:** Add tests for the auto-detection path:

```python
def test_hpc_detection_defaults_to_offline(self, monkeypatch) -> None:
    """When SLURM_JOB_ID is set and --tracking_mode is None, mode should be offline."""
    # This requires mocking main() or testing a helper function
    # Consider extracting _resolve_tracking_mode() for testability
    ...

def test_local_detection_defaults_to_online(self, monkeypatch) -> None:
    """When SLURM_JOB_ID is absent and --tracking_mode is None, mode should be online."""
    ...
```

### WR-05: `create_loggers` uses runtime `TensorBoardLogger` import that could fail

**File:** `src/rbspaper/pipeline/loggers.py:10`
**Issue:** `TensorBoardLogger` is imported at runtime (line 10), unlike `WandbLogger` which is lazily imported. If `tensorboard` is not installed (e.g., `uv sync` without `--extra tracking`), importing `loggers.py` itself will crash with `ImportError`. The `tracking` dependency group contains both `wandb` AND `tensorboard`, so if a user installs one they install both. However, the module does not guard against the case where neither is installed, and the crash happens at import time -- before any function is called.

This differs from W&B, which has the try/except pattern. If tensorboard is missing, the entire module fails to load.

**Fix:** Either (a) add `tensorboard` as a core dependency (since it's the local fallback with no network requirements), or (b) wrap the TensorBoardLogger import in a try/except similar to W&B:

```python
try:
    from lightning.pytorch.loggers import TensorBoardLogger
except ImportError:
    TensorBoardLogger = None  # type: ignore[misc, assignment]
```

Then gate the TensorBoardLogger creation:
```python
if TensorBoardLogger is not None:
    loggers.append(TensorBoardLogger(save_dir=str(run_dir), name='tensorboard'))
```

## Info

### IN-01: Corrupted HTML entities in module and function docstrings

**File:** `src/rbspaper/pipeline/loggers.py:1, 141, 143, 148`
**Issue:** The docstrings contain `&amp;` instead of `&`. Line 1 reads `"Factory and helpers for W&amp;B + TensorBoard experiment tracking loggers."` and similar in lines 141, 143, 148. This is malformed documentation -- likely introduced during XML/HTML processing of the plan files. While this does not affect code execution, it produces incorrect docstrings visible in help(), IDE tooltips, and generated documentation.

**Fix:** Replace all `&amp;` with `&` in the file:
- Line 1: `W&B` (not `W&amp;B`)
- Line 141: `W&B` (not `W&amp;B`)
- Line 143: `W&B` (not `W&amp;B`)
- Line 148: `W&B` (not `W&amp;B`)

### IN-02: Unused `step_start` initialization after analysis step

**File:** `src/rbspaper/pipeline/core.py:315-316`
**Issue:** After recording `timing['analysis']`, the code does:
```python
step_start = time.perf_counter()
```
This `step_start` is never used -- there is no subsequent `timing[...] = time.perf_counter() - step_start`. It is dead code, likely a leftover from a timing pattern that was planned but not implemented for the final cleanup/persistence steps.

**Fix:** Remove the unused `step_start = time.perf_counter()` line.

### IN-03: `_log_results_to_wandb` imports `wandb` redundantly

**File:** `src/rbspaper/pipeline/loggers.py:184`
**Issue:** The function does `import wandb` on line 184, even though reaching this point requires that `WandbLogger` was successfully created during `create_loggers`, which already required `wandb` to be importable. Lightning's `WandbLogger.__init__` calls `wandb.init()`, so `wandb` is guaranteed to be in `sys.modules` if a `WandbLogger` exists. The lazy import is unnecessary overhead.

However, this is not harmful -- `import wandb` is a no-op if already in `sys.modules`. It does serve as documentation of the dependency. Consider removing it for cleanliness or keeping it as an explicit contract.

### IN-04: Missing test for `_flatten_dict` with `None` values and non-serializable types

**File:** `test/test_logger_factory.py`
**Issue:** The `_flatten_dict` tests cover flat dicts, nested dicts, lists with primitives, lists with dicts, empty dicts, and custom separators. They do NOT cover:
- `None` values in leaf nodes (common in results: `checkpoint_path` can be `None`)
- Non-dict, non-list, non-primitive values (e.g., Path objects, Enum values, numpy types)

The actual `results_summary_data` passed to `_log_results_to_wandb` includes `None` values (`checkpoint_path`), so `_flatten_dict` will encounter them. Currently, `None` would be stored as-is in the flat dict, and `run.log()` would need to handle it.

**Fix:** Add tests:
```python
def test_none_values_preserved(self) -> None:
    result = _flatten_dict(d={'a': None, 'b': {'c': None}})
    assert result == {'a': None, 'b_c': None}

def test_mixed_types(self) -> None:
    result = _flatten_dict(d={'path': '/some/path', 'count': 42, 'flag': True})
    assert result == {'path': '/some/path', 'count': 42, 'flag': True}
```

## Cross-File Analysis

### Import Graph

```
runner.py
  -> config.py (ExperimentPipelineConfig, etc.)
  -> core.py (run_experiment_pipeline)
  -> loggers.py (create_loggers)

core.py
  -> config.py (PartitionRepresentations, etc.)
  -> loggers.py (_log_config_to_wandb, _log_results_to_wandb)

loggers.py
  -> lightning.pytorch.loggers (TensorBoardLogger at runtime, WandbLogger lazy)
  -> wandb (lazy, inside _log_results_to_wandb)

config.py
  -> lightning.pytorch (TYPE_CHECKING only)
  -> lightning.pytorch.loggers.Logger (TYPE_CHECKING only)
```

No circular imports detected. The import chain is acyclic: runner -> config, core, loggers; core -> config, loggers; loggers has no internal dependencies on config or core.

However, core.py imports private functions from loggers.py (`_log_config_to_wandb`, `_log_results_to_wandb`), which are not in `__all__`. This is a cross-module dependency on internal APIs. If loggers.py refactors these functions, core.py breaks. Consider making them public (remove leading underscore) or move them into a shared module.

### Call Chain Trace: `create_loggers` -> Trainer

1. `runner.py:main()` calls `create_loggers(run_dir=..., run_name=..., tracking_mode=..., persist_artifacts=True)`
2. `create_loggers` returns `tuple[WandbLogger | TensorBoardLogger, ...]`
3. Tuple passed to `_build_pipeline_config(..., loggers=loggers)`
4. `_build_pipeline_config` creates `ExperimentPipelineConfig(..., loggers=loggers)`
5. `run_experiment_pipeline(config=...)` receives config with loggers
6. `_train_model(config=...)` checks `if config.loggers:` and sets `trainer_kwargs['loggers'] = list(config.loggers)`
7. `pl.Trainer(**trainer_kwargs)` receives loggers

Chain is correct. The conditional wiring (`if config.loggers:`) properly avoids Pitfall 3 (empty list triggering default TensorBoardLogger).

### Call Chain Trace: `_log_config_to_wandb`

1. `run_experiment_pipeline` -> `_prepare_run_directory` -> `_write_experiment_config`
2. `_write_experiment_config` builds `config_data` dict
3. Calls `_log_config_to_wandb(config_data=config_data, loggers=config.loggers)`
4. `_find_wandb_logger` locates WandbLogger by type name
5. `wandb_logger.experiment.config.update(config_data)` writes to W&B

Chain is correct.

### Call Chain Trace: `_log_results_to_wandb`

1. `run_experiment_pipeline` builds `results_summary_data` dict
2. Calls `_log_results_to_wandb(results_summary=..., timing=..., loggers=config.loggers)`
3. `_find_wandb_logger` locates WandbLogger
4. `import wandb` (lazy)
5. `run.log(flat_results + flat_timing)`
6. `run.log({'comparison': wandb.Table(..., data=[...])})`

Chain is correct but the table data is empty due to CR-01.

### Type Safety Summary

| Location | Declared Type | Actual Type | Compatible? |
|----------|--------------|-------------|-------------|
| `config.py:289` | `tuple[Logger, ...]` | `tuple[WandbLogger\|TensorBoardLogger, ...]` | Yes (subtypes) |
| `runner.py:223` | `tuple` | `tuple[WandbLogger\|TensorBoardLogger, ...]` | Bare tuple -- no enforcement |
| `loggers.py:23` | `tuple[WandbLogger\|TensorBoardLogger, ...]` | Same | Yes |
| `loggers.py:80` | `tuple[pl.loggers.LightningLogger, ...]` | Same as above | Yes (Logger == LightningLogger) |
| `loggers.py:139` | `tuple[pl.loggers.LightningLogger, ...]` | Same | Yes |

---

_Reviewed: 2026-05-08T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
