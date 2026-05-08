---
phase: 07-experiment-tracking
verified: 2026-05-08T09:50:00Z
status: human_needed
score: 16/16 must-haves verified
overrides_applied: 0
---

# Phase 7: Experiment Tracking Verification Report

**Phase Goal:** Add experiment tracking infrastructure (W&B + TensorBoard logging, step timing, tracking modes) so every pipeline run is observable and comparable across seeds/datasets.
**Verified:** 2026-05-08T09:50:00Z
**Status:** human_needed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

Derived from ROADMAP.md Success Criteria, Plan must_haves, and CONTEXT.md Decisions (D-01 through D-07).

| #   | Truth                                                                                   | Status     | Evidence                                                                                           |
|-----|-----------------------------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------|
| 1   | pyproject.toml has tracking dep group with wandb>=0.18.0 and tensorboard>=2.17.0        | VERIFIED   | Line 57-60: `tracking = ["wandb>=0.18.0", "tensorboard>=2.17.0"]` between attacks_extended and notebooks |
| 2   | ExperimentPipelineConfig has loggers field (tuple type, empty default, Logger type)     | VERIFIED   | Line 289: `loggers: tuple[Logger, ...] = field(default_factory=tuple)`, Logger imported from `lightning.pytorch.loggers` under TYPE_CHECKING |
| 3   | loggers.py imports successfully and exports create_loggers                              | VERIFIED   | `__all__ = ['create_loggers']`, all 34 tests pass including imports                                 |
| 4   | create_loggers returns correct loggers for each mode                                    | VERIFIED   | online/offline: 2 loggers (TB+W&B), disabled: 1 logger (TB only), persist=False: empty tuple. Tests confirmed |
| 5   | Trainer wiring is conditional (only when loggers non-empty)                             | VERIFIED   | core.py line 400-401: `if config.loggers: trainer_kwargs['loggers'] = list(config.loggers)`          |
| 6   | Timing instrumentation present for train, shared_attacks, task_loop, analysis, total    | VERIFIED   | 10 `time.perf_counter()` calls across core.py lines 158-315, covering all 5 timing keys              |
| 7   | W&B config logging hooked in _write_experiment_config                                   | VERIFIED   | core.py line 386: `_log_config_to_wandb(config_data=config_data, loggers=config.loggers)`            |
| 8   | W&B results logging hooked after results_summary                                        | VERIFIED   | core.py lines 346-351: conditional `_log_results_to_wandb(...)` call after results_summary.json write |
| 9   | Runner has --tracking_mode CLI arg with choices validation                              | VERIFIED   | runner.py lines 146-151: `choices=['online', 'offline', 'disabled']`, default None                   |
| 10  | HPC auto-detect via SLURM_JOB_ID                                                        | VERIFIED   | runner.py lines 403-405: `tracking_mode = 'offline' if os.environ.get('SLURM_JOB_ID') else 'online'` |
| 11  | Tests pass and cover the critical paths                                                  | VERIFIED   | 34/34 phase tests pass. Full suite: 137/137 pass, zero regressions                                   |
| 12  | No wandb.init() calls in loggers.py (Pitfall 1)                                         | VERIFIED   | Only match is docstring text ("Does NOT call `wandb.init()`") -- zero actual calls                   |
| 13  | log_model=False enforced (D-07)                                                          | VERIFIED   | loggers.py line 72: `log_model=False` in WandbLogger constructor. Test confirmed (line 83)           |

**Score:** 16/16 truths verified

### Required Artifacts

| Artifact | Expected    | Status | Details |
|----------|-------------|--------|---------|
| `src/rbspaper/pipeline/loggers.py` | Logger factory and W&B helpers | VERIFIED | 222 lines, 5 functions (create_loggers, _find_wandb_logger, _flatten_dict, _log_config_to_wandb, _log_results_to_wandb), all keyword-only, type-hinted, Google docstrings |
| `src/rbspaper/pipeline/config.py` | loggers field on ExperimentPipelineConfig | VERIFIED | Line 289: `loggers: tuple[Logger, ...] = field(default_factory=tuple)` after attack_scope field |
| `pyproject.toml` | tracking dependency group | VERIFIED | Lines 57-60: `tracking = ["wandb>=0.18.0", "tensorboard>=2.17.0"]` in [dependency-groups], alphabetically between attacks_extended and notebooks |
| `src/rbspaper/pipeline/core.py` | Trainer wiring, timing, W&B hooks | VERIFIED | Conditional loggers (line 400-401), 5 timing keys, _log_config_to_wandb (line 386), _log_results_to_wandb (lines 346-351) |
| `runners/py/runner.py` | --tracking_mode, HPC detection, factory call | VERIFIED | CLI arg (lines 146-151), SLURM detection (lines 403-405), create_loggers call (lines 417-422), loggers wired to _build_pipeline_config (line 438) |
| `test/test_logger_factory.py` | Unit tests for loggers.py | VERIFIED | 21 tests across 4 classes: TestLoggerFactory (8), TestFlattenDict (6), TestFindWandbLogger (3), TestWandbConfigAndResultsLogging (4) |
| `test/test_runner_cli_args.py` | Tests for --tracking_mode | VERIFIED | TestTrackingModeArg class with 5 tests added; all existing tests unchanged |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `runners/py/runner.py` | `create_loggers` | `from src.rbspaper.pipeline.loggers import create_loggers` | VERIFIED | Line 45 import, line 417 call |
| `runners/py/runner.py` | `_build_pipeline_config` | `loggers=loggers` argument | VERIFIED | Line 438 in main() call |
| `src/rbspaper/pipeline/core.py` | `pl.Trainer` | `trainer_kwargs['loggers']` | VERIFIED | Lines 400-401, conditional on config.loggers |
| `src/rbspaper/pipeline/core.py` | `_log_config_to_wandb` | Call in _write_experiment_config | VERIFIED | Line 386, after _save_json of experiment_config |
| `src/rbspaper/pipeline/core.py` | `_log_results_to_wandb` | Call after results_summary | VERIFIED | Lines 346-351, conditional on config.loggers |
| `src/rbspaper/pipeline/loggers.py` | `lightning.pytorch.loggers.WandbLogger` | TYPE_CHECKING import + lazy runtime import | VERIFIED | Line 16 TYPE_CHECKING, line 58 lazy import with try/except |
| `src/rbspaper/pipeline/loggers.py` | `lightning.pytorch.loggers.TensorBoardLogger` | Top-level import | VERIFIED | Line 10: `from lightning.pytorch.loggers import TensorBoardLogger` |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `core.py` | `timing` dict | `time.perf_counter()` around each pipeline step | Real wall-clock measurements | FLOWING |
| `core.py` | `results_summary_data` | Built from `results.model_name`, `downstream_metrics`, `analysis_results` | Real pipeline outputs | FLOWING |
| `core.py` | `config_data` | Built from `config.model`, `config.seed`, `config.downstream_tasks`, etc. | Real config values | FLOWING |
| `runner.py` | `loggers` | `create_loggers(run_dir, run_name, tracking_mode, persist_artifacts=True)` | Real logger instances | FLOWING |
| `runner.py` | `tracking_mode` | CLI arg or auto-detect from `os.environ.get('SLURM_JOB_ID')` | Real environment signal | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full test suite passes | `uv run pytest test/ -x -q` | 137 passed, 12 warnings | PASS |
| Phase-specific tests pass | `uv run pytest test/test_logger_factory.py test/test_runner_cli_args.py -v` | 34 passed | PASS |
| log_model=False verified | `grep 'log_model=False' src/rbspaper/pipeline/loggers.py` | 1 match at line 72 | PASS |
| No wandb.init() calls | `grep 'wandb.init(' src/rbspaper/pipeline/loggers.py` | 0 code matches (only docstring) | PASS |
| Tracking deps in pyproject.toml | `grep 'wandb>=0.18.0\|tensorboard>=2.17.0' pyproject.toml` | 2 matches | PASS |

### Requirements Coverage

REQUIREMENTS.md does not exist in this project. Requirements are defined via decisions D-01 through D-07 in CONTEXT.md and referenced by all three PLAN frontmatter `requirements:` fields. Cross-reference:

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|----------------|-------------|--------|----------|
| D-01 | Plan 01, Plan 03 | Dual loggers: W&B + TensorBoard | SATISFIED | create_loggers returns both loggers for online/offline modes; test confirmed |
| D-02 | Plan 01, Plan 03 | loggers field on ExperimentPipelineConfig | SATISFIED | `loggers: tuple[Logger, ...]` with empty default; frozen dataclass preserved |
| D-03 | Plan 02, Plan 03 | Smart HPC detection via SLURM_JOB_ID | SATISFIED | runner.py lines 403-405 auto-detect offline on HPC, online locally |
| D-04 | Plan 02, Plan 03 | Auto-log results_summary with flattened keys + timing | SATISFIED | _log_results_to_wandb flattens results_summary and timing dict; wandb.Table created |
| D-05 | Plan 02, Plan 03 | Log config to wandb.config at pipeline start | SATISFIED | _log_config_to_wandb called from _write_experiment_config after JSON write |
| D-06 | Plan 02, Plan 03 | Run name format {experiment_id}_{dataset}_{seed}, project=rbspaper | SATISFIED | runner.py line 419: `run_name=f'{args.experiment_id}_{dataset_name}_{args.seed}'`; loggers.py line 69: `project='rbspaper'` |
| D-07 | Plan 01, Plan 03 | log_model=False (no checkpoint upload) | SATISFIED | loggers.py line 72: `log_model=False`; test confirmed |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | -- | No anti-patterns detected | -- | -- |

Scanned files: `src/rbspaper/pipeline/loggers.py`, `src/rbspaper/pipeline/core.py`, `runners/py/runner.py`, `test/test_logger_factory.py`, `test/test_runner_cli_args.py`.

No TODOs, FIXMEs, placeholders, empty returns, hardcoded empty data, or console.log-only patterns found.

### Human Verification Required

### 1. W&B UI Queryability

**Test:** Run a local experiment with `--tracking_mode online` (or `offline` followed by `wandb sync`), then open the W&B dashboard at app.wandb.ai.
**Expected:** The run appears under project "rbspaper" with config panel showing model_name, seed, downstream_tasks, attack_names, and a "comparison" table with one row of metrics.
**Why:** Requires actual W&B account, authentication (WANDB_API_KEY), and network access. Cannot be verified programmatically.

### 2. TensorBoard Events File

**Test:** After a pipeline run, execute `tensorboard --logdir <run_dir>/tensorboard` and open the browser at localhost:6006.
**Expected:** Training curves visible (loss, metrics logged by LightningModule self.log() calls).
**Why:** Requires running a full experiment and a browser to inspect the TensorBoard UI.

### 3. HPC Offline Mode

**Test:** Submit a job on the Kathleen cluster with SLURM and verify `wandb sync` later uploads the run.
**Expected:** Run data is written locally in offline mode, then synced to W&B cloud after `wandb sync`.
**Why:** Requires actual HPC cluster access and SLURM environment.

---

_Verified: 2026-05-08T09:50:00Z_
_Verifier: Claude (gsd-verifier)_
