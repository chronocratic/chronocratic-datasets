---
phase: 07-experiment-tracking
fixed_at: "2026-05-08T12:30:00Z"
review_path: .planning/phases/07-experiment-tracking/07-REVIEW.md
iteration: 2
findings_in_scope: 12
fixed: 4
skipped: 8
status: partial
---

# Phase 07: Code Review Fix Report

**Fixed at:** 2026-05-08T12:30:00Z
**Source review:** .planning/phases/07-experiment-tracking/07-REVIEW.md
**Iteration:** 2

**Summary:**
- Findings in scope: 12
- Fixed: 4
- Skipped: 8

**Note:** This is iteration 2. A prior fix attempt (iteration 1) was reverted via commit 134331a due to incorrect timing fixes. This iteration addresses the findings that remained broken after the revert.

## Fixed Issues

### CR-01: `wandb.Table` comparison columns reference wrong flattened keys -- table is always empty

**Files modified:** `src/rbspaper/pipeline/core.py`
**Commit:** 939e672
**Applied fix:** Computed `timing['wall_clock']` from `pipeline_start` BEFORE building `results_summary_data`. The `total_seconds` field was always 0.0 because `wall_clock` had not been set at the time of dict construction. Now `total_seconds` correctly references the computed wall-clock elapsed time. The flat metric keys (`classification_clean_accuracy`, `geometry_centroid_margin`, etc.) were already being generated correctly from the `flat_metrics` and `flat_analysis` dicts in the prior iteration 1 fix.

### WR-02: `timing['total']` does not represent actual wall-clock elapsed time

**Files modified:** `src/rbspaper/pipeline/core.py`
**Commit:** 939e672
**Applied fix:** Reordered timing computation: `wall_clock` (true elapsed) is calculated before `results_summary_data`. The `running_compute_time` field now sums only individual step timings, excluding `wall_clock` and itself to prevent double-counting. This addresses the reviewer's concern about `total` being an inaccurate aggregate.

### IN-02: Unused `step_start` initialization after analysis step

**Files modified:** `src/rbspaper/pipeline/core.py`
**Commit:** 939e672
**Applied fix:** Removed dead `step_start = time.perf_counter()` line that appeared after the analysis timing capture. This was leftover code with no corresponding completion.

### WR-05: `create_loggers` uses runtime `TensorBoardLogger` import that could fail

**Files modified:** `src/rbspaper/pipeline/loggers.py`
**Commit:** c2b7195
**Applied fix:** Wrapped `TensorBoardLogger` instantiation in try/except `ModuleNotFoundError` with warning log. Extended `WandbLogger` error handling to catch both `ImportError` and `ModuleNotFoundError` (Lightning's `__init__` raises ModuleNotFoundError when the underlying package is missing, even though the class import succeeds). Both loggers now fail gracefully.

## Skipped Issues

### CR-02: Missing `WandbLogger` in TYPE_CHECKING block

**File:** `src/rbspaper/pipeline/loggers.py:12-16`
**Reason:** Already fixed. `WandbLogger` is present in the TYPE_CHECKING block (line 16 of current source).

### CR-03: `_build_pipeline_config` uses untyped `loggers: tuple = ()`

**File:** `runners/py/runner.py:243`
**Reason:** Already fixed. Parameter is typed as `tuple[Logger, ...]` with `Logger` imported from `lightning.pytorch.loggers`.

### WR-01: `_log_results_to_wandb` crashes on W&B network errors

**File:** `src/rbspaper/pipeline/loggers.py`
**Reason:** Already fixed. Function has try/except wrapping all W&B calls.

### WR-03: `_log_config_to_wandb` has no error handling

**File:** `src/rbspaper/pipeline/loggers.py`
**Reason:** Already fixed. Function has try/except wrapping `experiment.config.update()`.

### WR-04: No test coverage for HPC auto-detection logic

**File:** `test/test_runner_cli_args.py`
**Reason:** Already fixed. `TestHpcAutoDetection` class exists with 4 tests covering SLURM_JOB_ID detection.

### IN-01: Corrupted HTML entities in docstrings

**File:** `src/rbspaper/pipeline/loggers.py`
**Reason:** Already fixed. Module docstring reads `W&B` (not `W&amp;B`).

### IN-03: `_log_results_to_wandb` imports `wandb` redundantly

**File:** `src/rbspaper/pipeline/loggers.py:195`
**Reason:** Not harmful. The `import wandb` is needed for `wandb.Table` usage and serves as documentation of the dependency. Keeping it as-is per reviewer guidance.

### IN-04: Missing test for `_flatten_dict` with `None` values

**File:** `test/test_logger_factory.py`
**Reason:** Already fixed. Tests `test_none_values_preserved` and `test_mixed_types` exist.

---

_Fixed: 2026-05-08T12:30:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
