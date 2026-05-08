# Phase 7: Experiment Tracking - Research

**Researched:** 2026-05-08
**Domain:** W&B + TensorBoard integration with PyTorch Lightning, experiment metadata logging, HPC offline tracking
**Confidence:** HIGH

## Summary

This phase integrates W&B and TensorBoard as dual experiment loggers into the existing PyTorch Lightning pipeline. The core integration point is `pl.Trainer(loggers=[WandbLogger(...), TensorBoardLogger(...)])` at line 368 in `core.py`. W&B provides cloud-based run comparison, hyperparameter filtering, and tabular metrics via `wandb.Table`. TensorBoard provides zero-config local fallback — events files always survive on disk regardless of network state.

The hybrid logging strategy (D-04) auto-logs `results_summary.json` contents via a single `wandb.log()` call at pipeline end, making the tracking code resilient to future metric additions. Manual logging is reserved for step-level timing (`time.perf_counter()`). Training curves are handled automatically by Lightning's `self.log()` integration with the Trainer's loggers.

On HPC (detected via `SLURM_JOB_ID`), W&B runs in `mode="offline"`, writing locally and syncing later via `wandb sync`. Both `wandb` (0.26.1) and `tensorboard` (2.20.0) are new dependencies added as a `tracking` dependency group in `pyproject.toml`.

**Primary recommendation:** Add a `loggers` field to `ExperimentPipelineConfig`, build loggers in the runner before config assembly, and pass them to `pl.Trainer`. A thin `_log_to_tracking()` helper in `core.py` handles W&B-specific calls (`wandb.config.update`, `wandb.log`, `wandb.Table`) after `_persist_artifacts`.

## User Constraints (from CONTEXT.md)

### Locked Decisions

**D-01:** Dual loggers: W&B + TensorBoard. W&B provides rich cloud UI for run comparison, filtering, and tabular metrics. TensorBoard provides zero-config local fallback — events files always survive on disk regardless of network state. Both wired into Lightning via `loggers=[WandbLogger(...), TensorBoardLogger(...)]`.

**D-02:** New `loggers` field on `ExperimentPipelineConfig` (type: `list[pl.loggers.LightningLogger]` or `tuple[pl.loggers.LightningLogger, ...]`). Runner creates logger instances before building config. Core.py passes them to `pl.Trainer(loggers=config.loggers, **config.training.trainer_kwargs)`. If `config.loggers` is empty, Trainer should be created without the `loggers` arg.

**D-03:** Smart default: if `SLURM_JOB_ID` env var is set -> `mode="offline"`, else -> `mode="online"`. User override via `--tracking_mode offline|online` flag on all runners.

**D-04:** Auto-log the `results_summary.json` dict at pipeline end via one `wandb.log(flattened(results_summary))` call. Manual logging only for step-level timing (`time.perf_counter()` per step). Training curves handled automatically by Lightning's logger integration.

**D-05:** `wandb.config.update(experiment_config.json)` at pipeline start — all hyperparams, model params, seed, dataset, attacks, tasks, output dir.

**D-06 (run naming):** W&B run name uses same hierarchy as the output dir: `{experiment_name}_{dataset}_{seed}`. Project name: `rbspaper`.

**D-06 (table):** `wandb.Table` — populated from the same auto-logged results_summary. One row per run. Powers W&B interactive tables.

**D-07:** Do NOT log model checkpoints as W&B artifacts — they're already on disk, upload cost is high.

### Why Dual Loggers

128+ datasets x 3 models x multiple seeds produces hundreds of runs. W&B is the only free-tier tool that lets you filter, sort, and compare across this scale. TensorBoard on disk is local insurance if W&B sync never happens.

### Why Hybrid Logging

Hybrid approach means the logging code is resilient to pipeline changes. Adding a new analysis metric = it appears in results_summary = it logs to W&B automatically. Only operational data (timing) needs explicit code changes.

### Claude's Discretion

- Exact structure of the logger factory function and where it lives (new file vs runner.py)
- Whether to gate W&B logging behind a `persist_artifacts` check or make it independent
- Specific key names in the `wandb.log()` dict — should be flat (e.g., `classification_clean_accuracy` not nested) for table compatibility
- How to handle W&B import when not installed (graceful fallback to no-op logger)
- Whether TensorBoardLogger `save_dir` should be `run_dir` or a subfolder
- Exact columns included in the `wandb.Table`

### Deferred Ideas (OUT OF SCOPE)

- Model checkpoint upload as W&B artifacts — deferred to reduce upload cost on HPC
- Real-time monitoring dashboard — future phase if experiment failure detection becomes important
- Comet ML / Neptune integration — not needed given W&B feature set
- MLflow self-hosted tracking — would require infra setup on HPC
- `wandb.Sweep` for hyperparameter search — Phase 7 is about tracking, not optimization

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| (implicit) | Integrate W&B + TensorBoard as dual loggers | Standard Stack, Architecture Patterns |
| (implicit) | HPC offline mode via SLURM_JOB_ID detection | W&B offline docs, Environment Availability |
| (implicit) | Log hyperparams, metrics, timing, wandb.Table | Code Examples section |
| (implicit) | Graceful fallback if wandb not installed | Don't Hand-Roll, Common Pitfalls |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Logger factory (create WandbLogger, TensorBoardLogger) | Runner (CLI layer) | — | Runner owns which tools are active; pipeline owns how |
| `loggers` field on ExperimentPipelineConfig | Config (data layer) | — | Frozen dataclass carries logger instances to core |
| `pl.Trainer(loggers=...)` wiring | Core (orchestration) | — | Core passes loggers to Trainer for training curves |
| `wandb.config.update()` at pipeline start | Core (orchestration) | — | Core has access to full experiment config dict |
| `wandb.log()` for results_summary | Core (orchestration) | — | Core produces results_summary at pipeline end |
| Step-level timing instrumentation | Core (orchestration) | — | Core controls step boundaries (train, encode, attack, etc.) |
| `wandb.Table` for run comparison | Core (orchestration) | — | Core has all metrics needed to build table rows |
| `--tracking_mode` CLI flag | Runner (CLI layer) | — | Runner parses args; determines online vs offline |
| TensorBoard events file | Local disk | — | Always writes locally; no network dependency |
| W&B offline sync (`wandb sync`) | Manual (post-HPC) | — | User runs `wandb sync` after HPC job completes |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `wandb` | 0.26.1 [VERIFIED: pip index] | Cloud experiment tracking, metrics, config, tables | Free tier handles 128+ datasets x 3 models x seeds; interactive comparison UI |
| `tensorboard` | 2.20.0 [VERIFIED: pip index] | Local events-file logging, zero-config fallback | Survives on disk regardless of network state; Lightning default logger |
| `lightning` | 2.5.5 [VERIFIED: pyproject.toml] | Provides `WandbLogger` and `TensorBoardLogger` | Already pinned dependency; native logger integration with `pl.Trainer` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pandas` | >=2.2.0 [VERIFIED: pyproject.toml] | `wandb.Table(dataframe=df)` for run comparison | Building wandb.Table from results_summary |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| WandbLogger | CSVLogger | No cloud UI, no filtering across hundreds of runs |
| TensorBoardLogger | No logger (Trainer default) | Default IS TensorBoardLogger — explicit is better for `save_dir` control |
| wandb | MLflow | MLflow requires self-hosted infra; W&B free tier sufficient |

**Installation:**
```bash
# Add as new dependency group in pyproject.toml:
# [dependency-groups]
# tracking = ["wandb>=0.18.0", "tensorboard>=2.17.0"]
```

**Version verification:**
- `wandb` 0.26.1 — latest as of 2026-05-08 [VERIFIED: pip index]
- `tensorboard` 2.20.0 — latest as of 2026-05-08 [VERIFIED: pip index]
- `lightning` 2.5.5 — pinned in pyproject.toml [VERIFIED: pyproject.toml]

## Architecture Patterns

### System Architecture Diagram

```
CLI Runner (runner.py)
  |
  |  -- parses --tracking_mode, detects SLURM_JOB_ID
  |  -- calls create_loggers() [Factory Method pattern]
  |  -- builds ExperimentPipelineConfig(loggers=[WandbLogger, TensorBoardLogger])
  |
  v
Pipeline Core (core.py)
  |
  |  -- _prepare_run_directory()
  |  -- _write_experiment_config() --> wandb.config.update()  [if loggers]
  |  -- time.perf_counter() around each step boundary
  |
  |  +-- pl.Trainer(loggers=config.loggers) --+
  |       |                                    |
  |       v                                    v
  |  WandbLogger.experiment             TensorBoardLogger.experiment
  |    (auto-logs self.log() from       (auto-logs self.log() from
  |     LightningModule hooks)          LightningModule hooks)
  |                                    |
  |  v                                 v
  |  Training curves (auto)         Training curves (auto)
  |
  |  -- Steps: train, shared_attacks, encoding, attacks, evaluate, analysis
  |  -- Each step: timing recorded (timing_train_seconds, etc.)
  |
  v
_Persist Artifacts (results_summary.json)
  |
  +-- _log_to_tracking() [if WandbLogger present]
  |     |-- wandb.log(flattened(results_summary))
  |     |-- wandb.log({"timing/step_name": duration})
  |     |-- wandb.log({"comparison": wandb.Table(...)})
  |
  v
Results on disk + W&B cloud (or offline local files)
```

Entry point: `runner.py` parses args, creates loggers, builds config.
Processing: `core.py` orchestrates train -> encode -> attack -> evaluate -> analyze.
Logger wiring: `pl.Trainer(loggers=...)` captures training curves automatically.
Manual logging: `_log_to_tracking()` captures results, timing, tables at pipeline end.
External deps: W&B cloud (online) or local files (offline); TensorBoard always local.

### Recommended Project Structure

```
src/rbspaper/pipeline/
├── loggers.py              # NEW: create_loggers() factory, _log_to_tracking() helper
├── core.py                 # MODIFIED: pass loggers to Trainer, call _log_to_tracking, timing
├── config.py               # MODIFIED: add loggers field to ExperimentPipelineConfig
└── ...

runners/py/
├── runner.py               # MODIFIED: --tracking_mode flag, call create_loggers()
└── ...

test/
├── test_logger_factory.py  # NEW: factory creates correct loggers per mode
└── ...
```

### Pattern 1: Logger Factory (Factory Method)

**What:** A pure function that creates logger instances based on the tracking mode and run context. Lives in a new `loggers.py` module.

**When to use:** Called by `runner.py` before building `ExperimentPipelineConfig`.

**Example:**
```python
# Source: Context7 /lightning-ai/pytorch-lightning verified API
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

def create_loggers(
    *,
    run_dir: Path,
    run_name: str,
    tracking_mode: str = "online",
    persist_artifacts: bool = True,
) -> tuple[WandbLogger | TensorBoardLogger, ...]:
    """Create W&B and TensorBoard logger instances.

    Args:
        run_dir: Output directory for this run.
        run_name: Hierarchical run name for W&B identification.
        tracking_mode: "online", "offline", or "disabled".
        persist_artifacts: If False, skip W&B logger entirely.

    Returns:
        Tuple of Logger instances (empty if tracking disabled).
    """
    if not persist_artifacts:
        return ()

    loggers: list[WandbLogger | TensorBoardLogger] = []

    # TensorBoard: always writes locally
    tb_logger = TensorBoardLogger(
        save_dir=str(run_dir),
        name="tensorboard",
    )
    loggers.append(tb_logger)

    # W&B: gate behind import + mode check
    if tracking_mode != "disabled":
        wandb_logger = WandbLogger(
            project="rbspaper",
            name=run_name,
            mode=tracking_mode,
            log_model=False,  # D-07: no checkpoint upload
            save_dir=str(run_dir),
        )
        loggers.append(wandb_logger)

    return tuple(loggers)
```

### Pattern 2: W&B Config + Results Logging

**What:** After `_write_experiment_config()` and before pipeline steps, update `wandb.config`. After `_persist_artifacts()`, call `wandb.log()` with flattened results and timing.

**When to use:** Hooked into existing `_write_experiment_config()` and `_persist_artifacts()` call sites.

**Example:**
```python
# Source: Context7 /wandb/wandb verified API
def _log_config_to_wandb(
    *,
    config_data: dict[str, Any],
    loggers: tuple[WandbLogger | TensorBoardLogger, ...],
) -> None:
    """Log experiment config to W&B wandb.config.

    Accesses the raw wandb.Run object via logger.experiment.
    """
    wandb_logger = _find_wandb_logger(loggers=loggers)
    if wandb_logger is None:
        return

    # logger.experiment is the wandb.Run object
    wandb_logger.experiment.config.update(config_data)


def _log_results_to_wandb(
    *,
    results_summary: dict[str, Any],
    timing: dict[str, float],
    loggers: tuple[WandbLogger | TensorBoardLogger, ...],
) -> None:
    """Log flattened results, timing, and comparison table to W&B."""
    wandb_logger = _find_wandb_logger(loggers=loggers)
    if wandb_logger is None:
        return

    run = wandb_logger.experiment

    # Flatten nested results (e.g., downstream_metrics["classification"][0]["accuracy"])
    flat_results = _flatten_dict(d=results_summary, separator="_", prefix="")
    flat_timing = {f"timing_{k}": v for k, v in timing.items()}
    run.log({**flat_results, **flat_timing})

    # Build comparison table
    table = wandb.Table(
        columns=[
            "experiment", "dataset", "seed", "model_name",
            "classification_clean_accuracy", "classification_clean_f1",
            "forecasting_clean_mae", "geometry_centroid_margin",
            "shift_mean_l2", "total_seconds",
        ]
    )
    table.add_data(
        flat_results.get("experiment", ""),
        flat_results.get("dataset", ""),
        flat_results.get("seed", ""),
        flat_results.get("model_name", ""),
        flat_results.get("downstream_metrics_classification_0_accuracy"),
        flat_results.get("downstream_metrics_classification_0_f1"),
        flat_results.get("downstream_metrics_forecasting_0_mae_loss"),
        flat_results.get("analysis_classification_clean_geometry_centroid_margin"),
        flat_results.get("analysis_classification_attacked_shift_mean_l2"),
        timing.get("total", 0.0),
    )
    run.log({"comparison": table})


def _find_wandb_logger(
    loggers: tuple[WandbLogger | TensorBoardLogger, ...],
) -> WandbLogger | None:
    """Find the WandbLogger in the logger tuple, if present."""
    for logger in loggers:
        if type(logger).__name__ == "WandbLogger":
            return logger
    return None
```

### Pattern 3: Step Timing Decorator

**What:** Wrap each step's timing with `time.perf_counter()` around the existing `logger.info('Step: ...')` hooks. Accumulate in a dict for logging at the end.

**When to use:** At each step boundary (train, shared_attacks, encoding, attacks, evaluate, analysis).

**Example:**
```python
import time

# In run_experiment_pipeline():
timing: dict[str, float] = {}
step_start = time.perf_counter()

# ... train step ...
timing['train'] = time.perf_counter() - step_start

step_start = time.perf_counter()
# ... shared_attacks step ...
timing['shared_attacks'] = time.perf_counter() - step_start

# ... (repeat for each step) ...
timing['total'] = sum(timing.values())
```

### Anti-Patterns to Avoid

- **Importing wandb at module level:** Causes ImportError when wandb is not installed. Always use lazy import inside functions or try/except blocks.
- **Calling `wandb.init()` directly when using Lightning:** Lightning's `WandbLogger` already calls `wandb.init()`. Direct `wandb.init()` creates a second run and causes conflicts.
- **Passing `logger=None` to Trainer:** Use `logger=False` to disable logging entirely, or omit the `loggers` arg. Passing `None` or an empty list triggers Lightning's default TensorBoardLogger.
- **Logging nested dicts to `run.log()`:** W&B flattens keys using `/` separators. Use flat keys for table compatibility.
- **`log_model=True` by accident:** Checkpoint upload costs bandwidth on HPC. Always explicitly set `log_model=False` per D-07.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Experiment tracking UI | Custom dashboard/web app | W&B cloud UI | Rich filtering, comparison, tabular views for 128+ datasets x 3 models |
| Local metrics logging | Custom file-based metrics store | TensorBoardLogger | Industry standard, `tensorboard --logdir` viewer, Lightning native |
| Timing instrumentation | Custom profiler | `time.perf_counter()` | Built-in, wall-clock, highest resolution available |
| Hyperparameter storage | Custom config DB | `wandb.config.update()` | Built into W&B, filterable in UI |
| Run comparison tables | Custom CSV/HTML | `wandb.Table` | Interactive, filterable, versioned |
| Offline-then-sync | Manual file transfer | `wandb sync` | Native W&B CLI command for offline runs |

**Key insight:** The experiment tracking ecosystem (W&B + TensorBoard) is mature. Hand-rolling any component introduces subtle bugs (metric alignment, timing precision, file format compatibility) without meaningful benefits. Use the standard stack and focus effort on integration quality.

## Common Pitfalls

### Pitfall 1: WandbLogger Creates Second Run
**What goes wrong:** Calling `wandb.init()` manually AND using `WandbLogger` creates two separate W&B runs. Metrics go to the manual init, not the logger's run.
**Why it happens:** `WandbLogger.__init__()` calls `wandb.init()` internally. The project already has manual `wandb.init()` patterns in other codebases.
**How to avoid:** Always access the wandb.Run object via `logger.experiment`, never call `wandb.init()` directly.
**Warning signs:** Two runs appear in W&B dashboard for one pipeline execution.

### Pitfall 2: Offline Mode Without Wandb Sync
**What goes wrong:** HPC runs complete with `mode="offline"`, data is written locally, but never synced to W&B cloud. Results are inaccessible from the dashboard.
**Why it happens:** `mode="offline"` only writes to disk. Syncing is a separate manual step (`wandb sync <dir>`).
**How to avoid:** Document the sync step clearly in HPC runner README. Optionally add a post-job sync script.
**Warning signs:** W&B dashboard shows no new runs after HPC jobs complete.

### Pitfall 3: Empty Loggers List Triggers Default TensorBoard
**What goes wrong:** Passing `loggers=[]` to `pl.Trainer` causes Lightning to use its default TensorBoardLogger at `lightning_logs/`.
**Why it happens:** Lightning's Trainer treats empty list or `logger=True` the same way — it creates a default logger.
**How to avoid:** If no loggers are configured, omit the `loggers` arg entirely or pass `logger=False`.
**Warning signs:** Unexpected `lightning_logs/` directory appearing alongside intentional TensorBoard output.

### Pitfall 4: Frozen Dataclass with Mutable Logger List
**What goes wrong:** `ExperimentPipelineConfig` is `frozen=True`. A mutable `list` of loggers could be modified after config creation.
**Why it happens:** Frozen dataclasses prevent attribute reassignment, but the list object itself is still mutable.
**How to avoid:** Use `tuple[WandbLogger | TensorBoardLogger, ...]` for the `loggers` field. Tuples are immutable.
**Warning signs:** Logger list accidentally modified by code that expects config to be immutable.

### Pitfall 5: Nested Dict Keys Broken for Tables
**What goes wrong:** Logging nested dicts to `run.log()` produces keys like `downstream_metrics/classification/0/accuracy`. These are hard to reference in `wandb.Table`.
**Why it happens:** W&B converts `.` in keys to `/` for grouping. Nested structures flatten with `/` separators.
**How to avoid:** Flatten manually before logging. Use `_` as separator for table-compatible keys.
**Warning signs:** Table columns reference keys that don't match the logged metric paths.

### Pitfall 6: TensorBoardLogger save_dir Collision
**What goes wrong:** Multiple runs writing TensorBoard events to the same directory cause garbled data.
**Why it happens:** TensorBoardLogger defaults to `save_dir="."` and appends version subdirs. Without explicit `save_dir`, runs collide.
**How to avoid:** Set `save_dir=str(run_dir)` and `name="tensorboard"` so each run gets its own isolated events file.
**Warning signs:** TensorBoard UI shows mixed metrics from different runs.

## Code Examples

### Access wandb.Run from WandbLogger
```python
# Source: Context7 /lightning-ai/pytorch-lightning verified
# In any LightningModule method (not __init__):
wandb_logger = self.logger  # or self.logger.experiment

# Direct wandb API calls:
wandb_logger.log_image(key="tsne", images=[wandb.Image(tsne_plot)])
wandb_logger.experiment.config.update({"extra_key": "value"})
```

### Flatten Nested Dict for W&B Logging
```python
# Source: Standard pattern, [VERIFIED: wandb docs convention]
def _flatten_dict(
    *, d: dict[str, Any], separator: str = "_", prefix: str = ""
) -> dict[str, Any]:
    """Recursively flatten a nested dictionary.

    Converts {'a': {'b': 1}} to {'a_b': 1}.
    Handles lists by using index as key: {'a': [1, 2]} to {'a_0': 1, 'a_1': 2}.
    """
    items: dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, dict):
            items.update(_flatten_dict(d=value, separator=separator, prefix=full_key))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    items.update(
                        _flatten_dict(d=item, separator=separator, prefix=f"{full_key}_{i}")
                    )
                else:
                    items[f"{full_key}_{i}"] = item
        else:
            items[full_key] = value
    return items
```

### WandbLogger with Offline Mode
```python
# Source: Context7 /wandb/wandb verified + /lightning-ai/pytorch-lightning verified
from lightning.pytorch.loggers import WandbLogger

# HPC mode: writes locally, sync later with `wandb sync`
wandb_logger = WandbLogger(
    project="rbspaper",
    name="ts2vec_a1b2c3d4_seed_42_Coffee",
    mode="offline",
    log_model=False,
    save_dir=str(Path("/path/to/run_dir")),
)
```

### TensorBoardLogger Per-Run Isolation
```python
# Source: Context7 /lightning-ai/pytorch-lightning verified
from lightning.pytorch.loggers import TensorBoardLogger

tb_logger = TensorBoardLogger(
    save_dir=str(Path("/path/to/run_dir")),
    name="tensorboard",
)
# Writes to: /path/to/run_dir/tensorboard/events.out.tfevents.*
```

### Trainer with Multiple Loggers
```python
# Source: Context7 /lightning-ai/pytorch-lightning verified
import lightning.pytorch as pl

trainer = pl.Trainer(
    loggers=[tb_logger, wandb_logger],
    **config.training.trainer_kwargs,
)
```

### wandb.Table for Run Comparison
```python
# Source: Context7 /wandb/wandb verified
# Build table from flattened results
table = wandb.Table(
    columns=["experiment", "dataset", "seed", "accuracy", "f1", "centroid_margin", "total_seconds"]
)
table.add_data("ts2vec", "Coffee", 42, 0.87, 0.85, 3.2, 125.4)
run.log({"comparison": table})
```

## Runtime State Inventory

> Not applicable — this is a greenfield integration phase, not a rename/refactor/migration.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — new feature, no existing state to migrate | — |
| Live service config | None — W&B cloud account configured by user, not by code | — |
| OS-registered state | None | — |
| Secrets/env vars | `WANDB_API_KEY` — user must set before first W&B sync | User responsibility, documented |
| Build artifacts | None | — |

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `wandb.init()` + manual logging | `WandbLogger` (Lightning wrapper) | Lightning 2.x | Cleaner integration, no double-init |
| CSV/JSON metrics only | TensorBoard events files | Ongoing | Visualizable with `tensorboard` CLI |
| Manual sync via scp | `wandb sync <dir>` | W&B 0.12+ | Native offline support for HPC |
| `log_model=True` default | `log_model=False` explicit | Ongoing | Prevents accidental checkpoint upload |

**Deprecated/outdated:**
- `WANDBLogger` vs `WandbLogger`: Context.md notes the class is `WandbLogger` (not `WANDBLogger`). Verified via Context7.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `WandbLogger(mode="offline")` is equivalent to `wandb.init(mode="offline")` — data written locally, synced later via `wandb sync` | Standard Stack, Pattern 1 | If incorrect, offline mode may not work; HPC runs silently drop data |
| A2 | `wandb` >= 0.18.0 is compatible with `lightning` 2.5.5 via `WandbLogger` | Standard Stack | Version mismatch causes import or runtime errors |
| A3 | `tensorboard` >= 2.17.0 is compatible with `lightning` 2.5.5 via `TensorBoardLogger` | Standard Stack | Version mismatch causes import or runtime errors |
| A4 | `logger.experiment` on `WandbLogger` returns a `wandb.Run` object | Pattern 2, Code Examples | If API changed, direct wandb calls fail |
| A5 | `WandbLogger(log_model=False)` disables all artifact upload behavior | Pattern 1 | If False doesn't fully disable, checkpoints may still upload |
| A6 | `SLURM_JOB_ID` is reliably set on Kathleen HPC nodes and absent locally | Environment Availability | If unreliable, mode detection fails |

**Verification needed before planning:** A2 and A3 — confirm `wandb` 0.26.1 and `tensorboard` 2.20.0 work with `lightning` 2.5.5's logger classes.

## Open Questions

1. **TensorBoardLogger `name` param — subfolder or flat?**
   - What we know: `save_dir=str(run_dir), name="tensorboard"` writes to `run_dir/tensorboard/events.out.tfevents.*`
   - What's unclear: Whether `name=None` writes flat to `run_dir/` (simpler for `tensorboard --logdir`) vs creating a `version=0` subdir
   - Recommendation: Use `name="tensorboard"` for explicit isolation; user can adjust during review

2. **W&B entity (team/workspace) configuration?**
   - What we know: `wandb.init()` accepts `entity` parameter for team workspace
   - What's unclear: Whether the project has a W&B team account or only personal accounts
   - Recommendation: Omit `entity` from factory; W&B defaults to user's personal workspace. Entity can be set via `WANDB_ENTITY` env var if needed.

3. **How to handle W&B auth on HPC (first sync)?**
   - What we know: `wandb login` is needed before `wandb sync`
   - What's unclear: Whether the Kathleen cluster has persistent W&B credentials
   - Recommendation: Document `wandb login` step in HPC runner README; do not bake credentials into code

4. **Should the `tracking` dependency group be in `[project.optional-dependencies]` or `[dependency-groups]`?**
   - What we know: Current pyproject.toml uses `[dependency-groups]` for dev, attacks, notebooks
   - What's unclear: Whether `uv` treats dependency-groups vs optional-dependencies differently for runtime availability
   - Recommendation: Use `[dependency-groups]` for consistency with existing pattern; group name `tracking`

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `wandb` (Python) | W&B logging, tables, config | Not checked — new dep | 0.26.1 latest | Pipeline works without W&B (TensorBoard only) |
| `tensorboard` (Python) | Local events-file logging | Not checked — new dep | 2.20.0 latest | Lightning default TensorBoardLogger (if `tensorboard` is installed as Lightning dep) |
| `wandb` (CLI) | `wandb sync` post-HPC | Not checked — comes with Python pkg | N/A | Manual file transfer |
| `WANDB_API_KEY` (env) | W&B auth for online mode | User responsibility | N/A | `mode="offline"` skips auth |
| `SLURM_JOB_ID` (env) | HPC detection | Kathleen cluster only | N/A | Local runs default to online |

**Missing dependencies with no fallback:**
- None. Both `wandb` and `tensorboard` are new deps added by this phase. Pipeline works with TensorBoard only if `wandb` import fails.

**Missing dependencies with fallback:**
- `wandb` (Python): If import fails, skip W&B logger, keep TensorBoard. Documented in Don't Hand-Roll.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest >= 8.2` (from `[dependency-groups]` dev) |
| Config file | None — pytest config in `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `uv run pytest test/test_logger_factory.py -x` |
| Full suite command | `uv run pytest test/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| D-01 | Dual loggers created | unit | `pytest test/test_logger_factory.py::test_dual_loggers -x` | Gap — Wave 0 |
| D-02 | `loggers` field on config | unit | `pytest test/test_logger_factory.py::test_config_loggers_field -x` | Gap — Wave 0 |
| D-03 | HPC detection (SLURM_JOB_ID) | unit | `pytest test/test_logger_factory.py::test_hpc_mode_detection -x` | Gap — Wave 0 |
| D-04 | Results auto-logged to wandb | unit (mocked) | `pytest test/test_logger_factory.py::test_results_logging -x` | Gap — Wave 0 |
| D-05 | Config logged to wandb.config | unit (mocked) | `pytest test/test_logger_factory.py::test_config_logging -x` | Gap — Wave 0 |
| D-07 | log_model=False | unit | `pytest test/test_logger_factory.py::test_no_model_logging -x` | Gap — Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest test/test_logger_factory.py -x`
- **Per wave merge:** `uv run pytest test/ -x`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `test/test_logger_factory.py` — covers D-01 through D-07
  - Factory creates correct loggers per mode (online/offline/disabled)
  - Config accepts tuple of loggers
  - HPC detection via SLURM_JOB_ID
  - log_model=False verified
  - Results logging mocked wandb.Run
  - Config logging mocked wandb.config
  - Graceful fallback when wandb import fails

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | `tracking_mode` flag validated against enum values (online/offline/disabled) |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| WANDB_API_KEY in source code | Information Disclosure | Env var only; never hardcoded. Document in HPC setup |
| wandb.Table leaking sensitive data | Information Disclosure | Table columns are metrics only; no raw data logged |

## Sources

### Primary (HIGH confidence)
- Context7 `/wandb/wandb` — wandb.init mode parameter, wandb.log, wandb.Table, wandb.Settings offline, wandb.config update [VERIFIED: Context7]
- Context7 `/lightning-ai/pytorch-lightning` — WandbLogger, TensorBoardLogger, Trainer loggers, logger.experiment, log_hyperparams [VERIFIED: Context7]
- `pip index versions wandb` — latest 0.26.1 [VERIFIED: pip registry]
- `pip index versions tensorboard` — latest 2.20.0 [VERIFIED: pip registry]
- `pyproject.toml` — lightning 2.5.5 pinned [VERIFIED: codebase]

### Secondary (MEDIUM confidence)
- Existing `core.py` code patterns — frozen dataclass, keyword args, step boundaries [VERIFIED: codebase]
- Existing `config.py` code patterns — TYPE_CHECKING imports, tuple defaults [VERIFIED: codebase]
- Existing `runner.py` code patterns — argparse, setup_logging, config building [VERIFIED: codebase]

### Tertiary (LOW confidence)
- Assumption A2/A3: wandb 0.26.1 + tensorboard 2.20.0 compatibility with lightning 2.5.5 loggers — not tested yet
- Assumption A6: SLURM_JOB_ID reliability on Kathleen cluster — based on Phase 6 CONTEXT.md, not verified on cluster

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions verified via pip index, Lightning 2.5.5 confirmed in pyproject.toml, Context7 verified WandbLogger/TensorBoardLogger APIs
- Architecture: HIGH — integration points identified from reading existing code (core.py, config.py, runner.py), patterns verified from Context7 docs
- Pitfalls: HIGH — based on well-documented Lightning logger behavior and W&B SDK patterns from official sources

**Research date:** 2026-05-08
**Valid until:** 2026-06-08 (30 days — stable APIs, versions verified current)
