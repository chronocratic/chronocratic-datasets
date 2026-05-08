# Phase 7: Experiment Tracking - Context

**Gathered:** 2026-05-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Integrate W&B and TensorBoard as dual experiment loggers into the existing pipeline, so that all runs (local and HPC) are logged with hyperparameters, metrics, artifacts, and analysis results — queryable via W&B's cloud UI and TensorBoard's local viewer.

**In scope:**
- Add `wandb` and `tensorboard` as project dependencies
- New `loggers` field on `ExperimentPipelineConfig` — accepts list of Lightning `Logger` instances
- Runner-level logger factory: creates W&B + TensorBoard loggers and passes them to pipeline config
- Smart mode detection: HPC (`SLURM_JOB_ID` env var set) defaults to W&B `mode="offline"`, local defaults to `mode="online"`, overridable via `--tracking_mode` CLI flag
- Log to W&B: full config (`wandb.config`), downstream metrics (`wandb.log`), analysis metrics (geometry, shift, separability), step-level timing, and a `wandb.Table` for run comparison
- TensorBoard: local events file per run — zero-config fallback, always writes
- Pipeline core wires `loggers` into `pl.Trainer(loggers=..., **trainer_kwargs)`
- All runs logged without breaking existing local/HPC runner interfaces
- No changes to the runner bash scripts beyond the new `--tracking_mode` flag

**Out of scope:**
- Model checkpoint upload to W&B artifacts
- Non-W&B cloud services (Comet ML, Neptune, MLflow)
- Real-time monitoring / alerting dashboard
- UI for the runners themselves (CLI only)

</domain>

<decisions>
## Implementation Decisions

### Tracking Tool Selection
- **D-01:** Dual loggers: W&B + TensorBoard. W&B provides rich cloud UI for run comparison, filtering, and tabular metrics. TensorBoard provides zero-config local fallback — events files always survive on disk regardless of network state. Both wired into Lightning via `loggers=[WANDBLogger(...), TensorBoardLogger(...)]`.
- **Why:** 128+ datasets × 3 models × multiple seeds produces hundreds of runs. W&B is the only free-tier tool that lets you filter, sort, and compare across this scale. TensorBoard on disk is local insurance if W&B sync never happens.

### Logger Wiring
- **D-02:** New `loggers` field on `ExperimentPipelineConfig` (type: `list[pl.loggers.LightningLogger]` or `tuple[pl.loggers.LightningLogger, ...]`). Runner creates logger instances before building config. Core.py passes them to `pl.Trainer(loggers=config.loggers, **config.training.trainer_kwargs)`.
- **Why:** Cleanest separation — runner owns *which* tools, pipeline owns *how* to use them. Keeps the frozen dataclass pattern consistent. Avoids nesting loggers inside `trainer_kwargs` dict.
- **Note:** If `config.loggers` is empty (no loggers provided), Trainer should be created without the `loggers` arg to maintain backward compatibility.

### W&B Offline/Online Mode
- **D-03:** Smart default: if `SLURM_JOB_ID` env var is set → `mode="offline"`, else → `mode="online"`. User override via `--tracking_mode offline|online` flag on all runners.
- **Why:** HPC nodes may have intermittent network. Local runs benefit from live dashboard feedback. User can always override.

### Logging Scope — Hybrid (Auto + Manual)
- **D-04:** Auto-log the `results_summary.json` dict at pipeline end via one `wandb.log(flattened(results_summary))` call. This dict already contains downstream_metrics, analysis results, model name. Any new metric added to the pipeline automatically appears in W&B without code changes. Manual logging only for step-level timing (`time.perf_counter()` per step) — these are operational, not results. Training curves are handled automatically by Lightning's logger integration.
- **D-05:** `wandb.config.update(experiment_config.json)` at pipeline start — all hyperparams, model params, seed, dataset, attacks, tasks, output dir. Makes runs filterable.
- **D-06:** `wandb.Table` — populated from the same auto-logged results_summary. One row per run. Powers W&B interactive tables.
- **D-07:** Do NOT log model checkpoints as W&B artifacts — they're already on disk, upload cost is high (hundreds of runs × tens of MB each).
- **Why:** Hybrid approach means the logging code is resilient to pipeline changes. Adding a new analysis metric = it appears in results_summary = it logs to W&B automatically. Only operational data (timing) needs explicit code changes.

### Run Naming Convention
- **D-06:** W&B run name uses same hierarchy as the output dir: `{experiment_name}_{dataset}_{seed}`. Project name: `rbspaper`.
- **Why:** Consistent naming between W&B UI and local filesystem. Makes it easy to correlate a W&B run with its output directory.

### Claude's Discretion
- Exact structure of the logger factory function and where it lives (new file vs runner.py)
- Whether to gate W&B logging behind a `persist_artifacts` check or make it independent
- Specific key names in the `wandb.log()` dict — should be flat (e.g., `classification_clean_accuracy` not nested) for table compatibility
- How to handle W&B import when not installed (graceful fallback to no-op logger)
- Whether TensorBoardLogger `save_dir` should be `run_dir` or a subfolder
- Exact columns included in the `wandb.Table`

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Pipeline Integration Points
- `src/rbspaper/pipeline/core.py` — Main pipeline function. `pl.Trainer(**config.training.trainer_kwargs)` at line 368 needs `loggers=` arg. Step boundaries (lines 154-300) are where timing should be added. `_persist_artifacts` (line 312) and `_write_experiment_config` (line 334) are the logging hooks.
- `src/rbspaper/pipeline/config.py` — `ExperimentPipelineConfig` dataclass (line 275) needs new `loggers` field. `TrainingConfig` (line 142) has `trainer_kwargs`. All dataclasses are `frozen=True`.
- `runners/py/runner.py` — CLI entry point. Parses args (line ~335-370), builds pipeline config (line ~380-420), calls `run_experiment_pipeline`. Add `--tracking_mode` flag and logger factory here.

### Experiment Results and Metrics
- `src/rbspaper/pipeline/core.py` — `ExperimentPipelineResults` returned by pipeline. Contains `downstream_metrics` (dict of task → list of metric dicts) and `analysis` (dict of geometry/shift/separability results). `results_summary.json` is the aggregation of these.
- `src/rbspaper/evaluation/evaluation.py` — `evaluate()` function. Returns metric dict per task (accuracy, F1 for classification; MAE, RMSE, MAPE for forecasting).
- `src/rbspaper/pipeline/analysis.py` — `compute_geometry_metrics()`, `compute_shift_metrics()`, `compute_linear_separability()`, `compute_low_dim_artifacts()`. Returns flat dicts of float values.

### Pipeline State and Timing
- `src/rbspaper/pipeline/core.py` — `PipelineStateBuilder` marks steps complete. Steps: train, shared_attacks, encoding, attacks, evaluate, analysis. Each step logged with `logger.info('Step: ...')` — these lines are the hooks for `time.perf_counter()`.

### W&B and Lightning Integration
- Lightning `WANDBLogger` — native integration via `from pytorch_lightning.loggers import WANDBLogger`. Accepts `project`, `mode`, `save_dir`, `log_model`, `name` kwargs. Passes all kwargs to `wandb.init()`.
- Lightning `TensorBoardLogger` — `from pytorch_lightning.loggers import TensorBoardLogger`. Accepts `save_dir`, `name`, `version`, `log_graph`.
- `pl.Trainer(loggers=[...])` — accepts list of Logger instances. Each logger receives `log_hyperparams()`, `log_metrics()`, `log_hparams_and_metrics()` calls automatically.

### Prior Phase Context
- `.planning/phases/06-hpc-runners/06-CONTEXT.md` — Phase 6 HPC decisions. SLURM array jobs on Kathleen cluster, `SLURM_JOB_ID` available for HPC detection. Runner scripts use `uv run`.
- `.planning/phases/03-pipeline-hardening/03-CONTEXT.md` — Phase 3 pipeline decisions. Frozen dataclass config pattern, `persist_artifacts` flag, step-level checkpointing.

### Architecture Reference
- `.planning/codebase/ARCHITECTURE.md` — Full component map. Pipeline core orchestrates train → encode → attack → evaluate → analyze. Results persisted as JSON/NPZ to `run_dir`.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `logger = logging.getLogger(__name__)` in `core.py` (line 64) — structured step logging already exists. Add `time.perf_counter()` around each step block for timing.
- `_write_experiment_config()` (core.py line 334) — already serializes config to dict and writes JSON. Can call `wandb.config.update()` with the same dict.
- `_persist_artifacts()` (core.py line 312) — writes results_summary.json. Same data is what `wandb.log()` needs.
- `ExperimentPipelineResults` — downstream_metrics and analysis fields hold all post-training results.

### Established Patterns
- All config dataclasses are `frozen=True` — new `loggers` field must be immutable.
- `TYPE_CHECKING` import guard for `pl.LightningModule` in config.py — use same pattern for `pl.loggers.LightningLogger`.
- Pipeline core only imports `pl` as `import lightning.pytorch as pl` — loggers accessed via `pl.loggers.WandbLogger` (note: Lightning class name is `WandbLogger`, not `WANDBLogger`).
- Runner uses `setup_logging()` for structured logging — add logger creation there or alongside it.
- `uv` for dependency management — add `wandb` to `pyproject.toml` dependencies.

### Integration Points
- `core.py:368` — `pl.Trainer(**config.training.trainer_kwargs)` → change to `pl.Trainer(loggers=config.loggers or None, **config.training.trainer_kwargs)`. Filter out `None`.
- `runner.py` — After resolving experiment instance and before building pipeline config, add logger factory: `loggers = create_loggers(tracking_mode, run_dir)`.
- `pipeline/config.py` — Add `loggers: tuple[pl.loggers.LightningLogger, ...] = field(default_factory=tuple)` to `ExperimentPipelineConfig`.
- `_write_experiment_config()` — After writing JSON, call `_log_to_tracking(config=config)` for W&B config/log.

### Dependency Considerations
- `wandb` is not currently a project dependency — must be added to `pyproject.toml`.
- `tensorboard` is not currently a project dependency — must be added to `pyproject.toml`.
- Both should handle import gracefully if not installed — W&B logging should be a soft dependency (pipeline works without it).

</code_context>

<specifics>
## Specific Ideas

- W&B project name: `rbspaper` — matches package name
- Run name format: `{experiment_name}_{dataset}_{seed}` — correlates with output directory structure
- `wandb.log()` keys should be flat for table compatibility: `classification_clean_accuracy`, `classification_fgsm_accuracy`, `forecasting_clean_mae`, `geometry_centroid_margin`, `shift_mean_l2`, `timing_train_seconds`, etc.
- TensorBoard `save_dir` should be `run_dir` — events file lives alongside other artifacts
- Gate W&B logging behind same `persist_artifacts` flag as disk writes — if user doesn't want artifacts, they don't want cloud logs
- Smart mode detection checks `os.environ.get("SLURM_JOB_ID")` — set on Kathleen HPC nodes, absent locally
- Lightning's `WandbLogger(mode="offline")` is equivalent to `wandb.init(mode="offline")` — no manual sync needed
- `wandb.Table` columns: experiment, dataset, seed, model_params (as JSON string), task, representation, accuracy/F1/MAE, centroid_margin, mean_l2_shift, total_seconds

</specifics>

<deferred>
## Deferred Ideas

- Model checkpoint upload as W&B artifacts — deferred to reduce upload cost on HPC. Can be added per-run if needed.
- Real-time monitoring dashboard — future phase if experiment failure detection becomes important.
- Comet ML / Neptune integration — not needed given W&B feature set. Can be added if W&B free tier limits are hit.
- MLflow self-hosted tracking — would require infra setup on HPC. Not needed with W&B offline mode.
- `wandb.Sweep` for hyperparameter search — Phase 7 is about tracking, not optimization. Future phase if needed.

</deferred>

---

*Phase: 07-Experiment Tracking*
*Context gathered: 2026-05-08*
