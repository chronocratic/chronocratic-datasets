# Phase 3: Pipeline Hardening - Research

**Researched:** 2026-05-06
**Domain:** Pipeline checkpointing, resume logic, structured logging, HPC-resilient file I/O
**Confidence:** HIGH

## Summary

Phase 3 transforms `run_experiment_pipeline()` from a linear, non-recoverable function into a resumable, checkpointed pipeline with deterministic output paths. The core changes are: (1) a per-task checkpoint state file tracking which steps completed for which downstream tasks, (2) resume logic that reads the checkpoint and skips completed work, (3) a new hierarchical output path replacing the flat `{experiment_id}_{dataset_name}` pattern, (4) hash-based run identity for drift detection, and (5) runner improvements including `--dataset_index`, `--force`, and structured logging.

**Primary recommendation:** Implement checkpointing as a thin layer inside `run_experiment_pipeline()` using a dedicated `PipelineState` dataclass and atomic JSON writes. Use tenacity (v9.1.4) for retry-with-backoff on individual step functions. Keep the approach minimal -- no external state store, no database, no message queue. The state file is a hidden JSON file in the run directory.

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Per-task granularity -- state tracks completion per (step, task_name) pair. E.g., `{"encoding": ["classification"], "attacks": ["classification"]}`. More precise recovery with minimal complexity cost.
- **D-02:** State file at `run_dir/.pipeline_state.json` -- self-contained, hidden file, travels with the run. Same output folder holds weights, logs, and state (for scratch/work on HPC).
- **D-03:** Atomic state writes -- write to `.pipeline_state.json.tmp` then `os.rename()` for POSIX atomicity. Prevents corrupted state on crash.
- **D-04:** Retry with backoff -- on step failure, retry N times with exponential backoff before marking as failed. Handles transient GPU memory fragmentation.
- **D-05:** Hierarchical output: `output_dir/{experiment_id}/{short_hash}/seed_{seed}/{dataset_name}/` -- grouped by experiment, then seed, then dataset. Deterministic and human-scannable.
- **D-06:** Short hash appendix -- 8-char SHA-256 of serialized model params catches parameter drift while staying readable.
- **D-07:** `experiment_config.json` written at run start (model params, attack params, seed, dataset, trainer kwargs). `results_summary.json` written at end with metrics. Crash-safe config.
- **D-08:** Add `--dataset_index` for HPC array jobs, keep `--dataset_name` for local testing. Mutually exclusive.
- **D-09:** Structured logging with Python `logging` module + INFO-level step transitions. tqdm for inner loops. Log files saved in output dir, not runner dir.
- **D-10:** Automatic resume when `.pipeline_state.json` exists. `--force` flag to override and start fresh. HPC-friendly default.

### Claude's Discretion
- Exact structure of `.pipeline_state.json` (timestamps, config hash for integrity)
- Number of retries and backoff interval (suggest: 3 retries, 30s base)
- Whether to log to both stdout and file, or file only with stdout summary

### Deferred Ideas (OUT OF SCOPE)
- Project-level state registry (`output_dir/.pipeline_registry.json`) -- useful for cross-run queries but not needed now
- Full SHA-256 hash folder names -- overkill when experiment registry already guarantees ID uniqueness
- JSON-structured logging -- machine-parseable but noisy for research code

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REQ-03 | Step-level checkpoint with per-task granularity | `PipelineState` dataclass, atomic JSON writes |
| REQ-04 | Resume logic with automatic detection | State file read before each step, `--force` override |
| REQ-05 | Hierarchical deterministic output structure | Hash-based naming (D-05, D-06), `run_dir` extension |
| REQ-06 | Experiment metadata (config + summary JSON) | `experiment_config.json` at start, `results_summary.json` at end |
| REQ-07 | Runner improvements (--dataset_index, logging, etc.) | `argparse` extension, `logging` module, file handlers |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Pipeline checkpoint state | Pipeline core (`core.py`) | — | State lives with the pipeline loop; runner just triggers |
| Hash-based output paths | Config (`config.py`) | Runner (`runner.py`) | `PipelineArtifactConfig` computes paths; runner passes args |
| Retry with backoff | Pipeline core (`core.py`) | — | Wraps step functions inside the pipeline |
| Runner CLI args | Runner (`runner.py`) | — | Entry point for all user interaction |
| Structured logging setup | Runner (`runner.py`) | Pipeline core (`core.py`) | Runner configures handlers; core logs via `logging` module |
| State file I/O | New module (`state.py`) | — | Dedicated module for read/write/atomic persist |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| tenacity | 9.1.4 | Retry with exponential backoff | Industry-standard Python retry library; decorator-based; `before_sleep` callbacks for logging; `RetryError` for graceful failure |
| Python `logging` | stdlib | Structured logging | Built-in; no new dependency; `RotatingFileHandler` for HPC scratch; `logging.getLogger()` for module-level loggers |
| Python `hashlib` | stdlib | SHA-256 hash computation | Built-in; deterministic; used by reference runner (`_sources/autotsaugment/runners/py/runner.py:76`) |
| Python `json` | stdlib | State file serialization | Built-in; already used via `_save_json()` in `core.py:826` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tqdm | 4.67.3 | Progress bars | Already a project dependency; use for inner loops per D-09 |
| pytest | 9.0.3 | Testing resume flow | Already configured; `tmp_path` fixture for state file tests |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| tenacity | Hand-rolled retry loop | tenacity provides `before_sleep`, `RetryError`, and composable stop/wait strategies; hand-rolling adds ~30 lines per step |
| tenacity | `retrying` library | `retrying` is unmaintained (last release 2020); tenacity v9.1.4 is actively maintained [VERIFIED: PyPI, uploaded 2026-02-07] |
| `logging` module | `structlog` / `loguru` | Overkill for research code; `logging` is stdlib and already used across `src/rbspaper/` |

**Installation:**
```bash
uv add tenacity
```

**Version verification:** tenacity 9.1.4 confirmed via PyPI [VERIFIED: PyPI API, 2026-02-07].

## Architecture Patterns

### System Architecture Diagram

```
runner.py (CLI entry point)
    |
    |-- Parse args: --experiment_id, --dataset_index/--dataset_name, --seed, --force
    |-- Setup logging: FileHandler (run_dir) + StreamHandler (stdout)
    |-- Resolve dataset name from index (if --dataset_index)
    |-- Compute hierarchical run_dir: {exp_id}/{hash8}/seed_{seed}/{dataset}/
    |-- Check .pipeline_state.json exists AND --force not set
    |
    v
run_experiment_pipeline(config)
    |
    |-- _prepare_run_directory() -> creates run_dir, writes experiment_config.json
    |-- init_pipeline_state() -> creates/reset .pipeline_state.json
    |
    |-- STEP 1: _train_model() [retried]
    |       |-- Check state: if 'train' in completed, skip
    |       |-- Train model, save checkpoint
    |       |-- Mark 'train' complete in state
    |
    |-- Collect partition tensors
    |-- Generate shared attacked inputs (if SHARED_INPUT scope) [retried]
    |
    |-- FOR EACH downstream task:
    |       |-- STEP 2: Extract clean reps [retried]
    |       |       |-- Check state: if ('encoding', task) completed, skip
    |       |       |-- Encode train/val/test
    |       |       |-- Mark ('encoding', task) complete
    |       |
    |       |-- STEP 3: Build attacked reps [retried]
    |       |       |-- Check state: if ('attacks', task) completed, skip
    |       |       |-- Generate/encode attacked inputs
    |       |       |-- Mark ('attacks', task) complete
    |       |
    |       |-- STEP 4: Evaluate downstream [retried]
    |               |-- Check state: if ('evaluate', task) completed, skip
    |               |-- Run evaluation metrics
    |               |-- Mark ('evaluate', task) complete
    |
    |-- STEP 5: Analysis [retried]
    |       |-- Check state: if 'analysis' completed, skip
    |       |-- Run representation analysis
    |       |-- Mark 'analysis' complete
    |
    |-- Write results_summary.json
    |
    v
ExperimentPipelineResults
```

File-to-implementation mapping:
- Logging setup, args parsing, dataset resolution: `runner.py`
- State management (read/write/check): New `src/rbspaper/pipeline/state.py`
- Pipeline loop with resume gates: `core.py` -- `run_experiment_pipeline()`
- Hash computation: `runner.py` or new helper in `state.py`
- Config serialization: Reuse `_save_json()` from `core.py:826`

### Recommended Project Structure

```
src/rbspaper/pipeline/
├── core.py          # Modified: resume gates, retry wrappers, step logging
├── config.py        # Modified: new run_dir hierarchy property, state config
├── state.py         # NEW: PipelineState dataclass, atomic read/write
├── analysis.py      # Unchanged
├── setup/           # Unchanged
└── __init__.py      # Unchanged

runners/py/
└── runner.py        # Modified: --dataset_index, --force, logging, hash paths

test/
└── test_pipeline_state.py   # NEW: checkpoint, resume, atomic write tests
```

### Pattern 1: Atomic State Write

Write to `.tmp` then `os.rename()` for POSIX atomicity. On crash during `json.dump()`, the `.tmp` file is corrupted but the real state file is untouched. On restart, the old valid state is read.

```python
# Source: Standard POSIX pattern
import json
import os
from pathlib import Path

def _atomic_write_json(path: Path, data: dict[str, object]) -> None:
    tmp_path = Path(str(path) + '.tmp')
    with tmp_path.open(mode='w', encoding='utf-8') as fh:
        json.dump(obj=data, fp=fh, indent=2, default=_json_default)
        fh.flush()
        os.fsync(fh.fileno())
    os.rename(src=tmp_path, dst=path)
```

`os.rename()` is atomic on the same filesystem (POSIX guarantee). The `fsync()` flushes to disk before rename. On NFS/Lustre (HPC scratch), `os.rename()` remains atomic as long as source and destination are on the same mount.

### Pattern 2: PipelineState Dataclass

Follow the project's frozen dataclass convention:

```python
@dataclass(frozen=True)
class PipelineState:
    """Immutable snapshot of pipeline step completion."""

    completed: dict[str, list[str]]  # step -> [task_names]
    config_hash: str                  # 8-char SHA-256 for integrity
    started_at: str                   # ISO timestamp
    last_updated: str                 # ISO timestamp

    def is_step_complete(self, *, step: str, task_name: str | None = None) -> bool:
        """Check if a step (optionally for a specific task) is already done."""
        completed_tasks = self.completed.get(step, [])
        if task_name is None:
            return step in self.completed
        return task_name in completed_tasks
```

Mutable builder pattern for constructing state before making it frozen:

```python
class _PipelineStateBuilder:
    """Mutable builder for PipelineState snapshots."""

    def __init__(self, *, config_hash: str) -> None:
        self._completed: dict[str, list[str]] = {}
        self._config_hash = config_hash
        self._started_at = datetime.now(timezone.utc).isoformat()

    def mark_complete(self, *, step: str, task_name: str | None = None) -> None:
        if step not in self._completed:
            self._completed[step] = []
        if task_name and task_name not in self._completed[step]:
            self._completed[step].append(task_name)
        if task_name is None:
            self._completed[step] = []  # marker: no task granularity

    def build(self) -> PipelineState:
        return PipelineState(
            completed=dict(self._completed),
            config_hash=self._config_hash,
            started_at=self._started_at,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )
```

### Pattern 3: Retry with Tenacity

Wrap each step function with tenacity retry. Use `before_sleep` for logging retry attempts.

```python
# Source: Context7 /jd/tenacity
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)
import logging

logger = logging.getLogger(__name__)

train_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=15, min=15, max=120),
    retry=retry_if_exception_type((RuntimeError, MemoryError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

@train_retry
def _train_model_with_retry(...) -> Path | None:
    ...
```

Wait times: 15s, 30s, 60s (capped at 120s). Retries on `RuntimeError` (OOM, CUDA errors) and `MemoryError`. After 3 attempts, `RetryError` propagates up.

### Pattern 4: Hash-Based Run Identity

Adapted from the reference runner (`_sources/autotsaugment/runners/py/runner.py:64-95`). The reference hashes model params AND augmentation params separately. For this project, hash model params only (aug params are inside model params already).

```python
import hashlib
import json

def _compute_config_hash(*, model_params: dict[str, object], seed: int) -> str:
    """Compute 8-char SHA-256 of model params for drift detection."""
    payload = json.dumps({'params': model_params, 'seed': seed}, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:8]
```

This produces a deterministic 8-char hex string. Same params + same seed = same hash. Different params = different hash (with negligible collision probability: 2^-32).

### Pattern 5: Structured Logging Setup

Configure both file and stream handlers. File handler goes in run_dir (HPC scratch-friendly). Stream handler uses `logging.StreamHandler()` (stdout).

```python
import logging
from pathlib import Path

def setup_logging(*, log_dir: Path, log_level: int = logging.INFO) -> None:
    """Configure root logger with file + stream handlers."""
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(log_level)

    # File handler -- goes in run_dir
    file_handler = logging.FileHandler(log_dir / 'pipeline.log')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    root.addHandler(file_handler)

    # Stream handler -- stdout summary
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s'
    ))
    root.addHandler(stream_handler)
```

Module-level loggers in `core.py`:
```python
logger = logging.getLogger(__name__)
logger.info('Starting step: train')
# ... train logic ...
logger.info('Step complete: train')
```

### Anti-Patterns to Avoid

- **String-dispatched steps:** Do not use `if step_name == 'train': train()` -- the current codebase already moved away from this pattern (Phase 2 mixin refactor). Keep step logic as direct function calls.
- **Global state mutations:** Do not use module-level mutable dicts for state. Use the frozen `PipelineState` dataclass pattern, consistent with the project's frozen config convention.
- **Blocking stdout during long steps:** Do not print raw progress to stdout during encoding/attacks -- tqdm already handles that. The logger should only log step transitions, not per-sample progress.
- **State file outside run_dir:** Per D-02, all outputs (state, logs, weights) live in the output folder. Do not write state to a separate "global" location.
- **Non-atomic JSON writes:** Writing directly to `.pipeline_state.json` risks corruption on crash. Always use `.tmp` + `os.rename()`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Retry logic | Manual `while` loop with `time.sleep()` | tenacity (`@retry`) | Handles `RetryError`, composable stop/wait/retry conditions, `before_sleep` logging |
| JSON serialization | Custom serializer | Existing `_save_json()` + `_json_default()` in `core.py:52-66` | Already handles Path, numpy arrays, Enums |
| Hash computation | Rolling own hash | `hashlib.sha256` (stdlib) | Standard, fast, deterministic |
| Log file rotation | Manual file size checks | `logging.FileHandler` (or `RotatingFileHandler`) | Built-in rotation, thread-safe |
| Dataset index resolution | Manual index mapping | `get_all_datasets()` + `list_dataset_names()` from `data_setup.py` | Registry-backed, consistent |
| Checkpoint file locking | `fcntl` / manual locks | `.tmp` + `os.rename()` | Simpler, POSIX atomic, no lock management |

**Key insight:** The pipeline's complexity comes from orchestration (resume gates, step ordering), not from low-level utilities. Delegate the plumbing to tenacity, stdlib logging, and the existing `_save_json()` helper.

## Common Pitfalls

### Pitfall 1: Stale State After Partial Crash
**What goes wrong:** Pipeline crashes mid-step (e.g., during encoding). The state file marks the step as complete because `mark_complete()` ran before the crash, but the actual output (encoded representations) was never persisted.
**Why it happens:** `mark_complete()` is called before the step function returns. If the crash happens between marking and actual completion, the state is inconsistent with reality.
**How to avoid:** Mark completion AFTER the step produces verifiable output. For encoding: mark complete only after representations are computed (not just after the encoding function returns). For training: mark complete only after `canonical_checkpoint` exists on disk.
**Warning signs:** Resume skips a step but the expected output file is missing.

### Pitfall 2: NFS Atomicity Assumptions
**What goes wrong:** `os.rename()` is assumed atomic, but on some NFS configurations, cross-filesystem renames are not atomic.
**Why it happens:** `.tmp` and `.json` must be on the same filesystem. If the run_dir spans a mount boundary, rename may fail or be non-atomic.
**How to avoid:** Ensure `.tmp` and `.json` are on the same path (they are, since both are `run_dir/.pipeline_state.json*`). Add a fallback: if `os.rename()` raises, fall back to `shutil.move()`.
**Warning signs:** `OSError: Invalid cross-device link` during state write.

### Pitfall 3: tqdm and Logging Interference
**What goes wrong:** tqdm progress bars get corrupted when logging writes to the same stdout.
**Why it happens:** Both tqdm and logging's `StreamHandler` write to stdout simultaneously, causing interleaved output.
**How to avoid:** Use `tqdm.write()` instead of `print()` for messages during progress bars. For the logger, use `logging.FileHandler` as the primary handler and keep `StreamHandler` at WARNING level only (step transitions are INFO, visible in the log file but not on stdout).
**Warning signs:** Garbled progress bar output, missing log lines.

### Pitfall 4: Frozen Config Immutability vs Hash Input
**What goes wrong:** `ExperimentPipelineConfig` is frozen (immutable), but computing the hash requires extracting model params. The model itself is a `LightningModule`, not a dict.
**Why it happens:** The config holds the live model instance, not a serializable params dict. The hash must come from the params dict used to build the model, not the model itself.
**How to avoid:** Pass the `model_params` dict (from `ExperimentInstance.model_params`) to the runner separately, before building the model. Compute the hash from the params dict + seed in the runner, before calling `run_experiment_pipeline()`.
**Warning signs:** Serialization errors when trying to hash the model directly.

### Pitfall 5: Resume Without Config Consistency Check
**What goes wrong:** State file exists from a previous run, but the config has changed (e.g., different epsilon). Pipeline resumes and skips steps based on stale state.
**Why it happens:** Resume logic only checks if the state file exists and if steps are marked complete, without verifying the config matches.
**How to avoid:** Include a `config_hash` field in the state file. On resume, recompute the hash and compare. If it differs, log a warning and either fail or reset state.
**Warning signs:** Resume produces incorrect results because the config changed between runs.

## Code Examples

### Resume Gate in Pipeline Loop

```python
def run_experiment_pipeline(*, config: ExperimentPipelineConfig, force: bool = False) -> ExperimentPipelineResults:
    # ... preflight ...

    run_dir = _prepare_run_directory(config=config)
    state_path = run_dir / '.pipeline_state.json'

    # Resume or fresh start
    if force:
        _atomic_write_json(state_path, {})  # reset
        state = None
    elif state_path.exists():
        state = _load_pipeline_state(state_path)
        logger.info(f'Resuming from checkpoint: {list(state.completed.keys())} steps complete')
    else:
        state = None

    # Step 1: Train
    if not _is_complete(state=state, step='train'):
        checkpoint = _train_model_with_retry(...)
        _mark_complete(state_path, step='train', config_hash=config_hash)

    # ... per-task loop ...
    for task_config in config.downstream_tasks:
        task_name = task_config.task_name

        if not _is_complete(state=state, step='encoding', task_name=task_name):
            clean_bundle = _extract_clean_representations(...)
            _mark_complete(state_path, step='encoding', task_name=task_name, config_hash=config_hash)

        # ... attacks, evaluate, analysis ...
```

### Hash-Based Path Construction (Runner Side)

```python
# In runner.py, before building the pipeline config
from dataclasses import asdict
import hashlib
import json

model_param_dict = asdict(experiment_instance.model_params)
model_param_dict['input_dims'] = input_dims  # resolved at runtime
payload = json.dumps({'params': model_param_dict, 'seed': seed}, sort_keys=True)
short_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()[:8]

# New hierarchical path: {exp_id}/{hash8}/seed_{seed}/{dataset_name}/
run_name = f'{args.experiment_id}/{short_hash}/seed_{seed}/{dataset_name}'
```

### Dataset Index Resolution (Runner Side)

```python
# In runner.py main()
if args.dataset_index is not None and args.dataset_name is not None:
    parser.error('--dataset_index and --dataset_name are mutually exclusive')

if args.dataset_index is not None:
    all_datasets = get_all_datasets(form='list')
    if args.dataset_index >= len(all_datasets):
        parser.error(f'dataset_index {args.dataset_index} out of range (max {len(all_datasets) - 1})')
    dataset_name = all_datasets[args.dataset_index]
else:
    dataset_name = args.dataset_name
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Flat run names (`exp_dataset`) | Hierarchical (`{exp}/{hash}/seed_{s}/{ds}`) | This phase | Deterministic, hash-verifiable paths |
| No checkpoint | Per-task JSON state | This phase | Resume from any step |
| Manual retry (none) | tenacity `@retry` decorator | This phase | Automatic GPU OOM recovery |
| `print()` for progress | `logging` module + file handler | This phase | Searchable logs in run_dir |
| `--dataset_name` only | `--dataset_index` (HPC) + `--dataset_name` (local) | This phase | SLURM array job compatible |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | tenacity v9.1.4 is API-stable (decorator syntax from docs matches current) | Standard Stack, Pattern 3 | Low -- tenacity has stable API since v8; v9 is a minor bump |
| A2 | `os.rename()` is atomic on the target HPC filesystem (NFS/Lustre same-mount) | Pitfall 2 | Medium -- if not atomic, state file could be corrupted on crash |
| A3 | Logging to both file and stdout is acceptable (stream handler at WARNING, file at INFO) | Pattern 5, Pitfall 3 | Low -- user discretion per D-09 |
| A4 | 8-char SHA-256 is sufficient for collision resistance (2^-32) | Pattern 4 | Low -- for ~128 datasets x ~7 experiments x few seeds, collision probability is negligible |
| A5 | `tenacity` can be added as a dev dependency without conflict | Standard Stack | Low -- tenacity has minimal deps (typing-extensions only) |

## Open Questions

1. **Retry count and backoff interval**
   - What we know: D-04 specifies retry with backoff; Claude's Discretion suggests "3 retries, 30s base"
   - What's unclear: Whether 30s base is appropriate for the target HPC (GPU OOM recovery may need less)
   - Recommendation: 3 retries, 15s base (15s, 30s, 60s caps at 120s). User can confirm in discuss phase.

2. **Logging to stdout vs file only**
   - What we know: D-09 says "structured logging + log files in output dir"
   - What's unclear: Whether stdout should also get INFO-level logs (visible during local runs) or only WARNING+
   - Recommendation: File gets INFO+, stdout gets WARNING+ (step transitions visible in log, tqdm unobstructed on terminal)

3. **Config hash scope**
   - What we know: D-06 specifies 8-char SHA-256 of serialized model params
   - What's unclear: Whether the hash should include attack params too (or only model params)
   - Recommendation: Hash model params + seed (not attack params) -- attack params vary per experiment instance, and the experiment_id already captures that identity

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | All code | N/A | 3.12.13 | — |
| tenacity | Retry logic | Not installed | 9.1.4 (PyPI) | None -- must add to dev deps |
| pytest | Testing | Installed | 9.0.3 | — |
| PyTorch | Training | Installed | 2.x (MPS) | — |
| Lightning | Training | Installed | 2.5.5 | — |
| tqdm | Progress bars | Installed (via ART) | 4.67.3 | — |

**Missing dependencies with no fallback:**
- None

**Missing dependencies with fallback:**
- tenacity -- must be added to `pyproject.toml` dev deps (or a new group). Hand-rolled retry is possible but discouraged (see Don't Hand-Roll table).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` -- testpaths = `["test"]` |
| Quick run command | `uv run pytest test/test_pipeline_state.py -x` |
| Full suite command | `uv run pytest test/ -x` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REQ-03 | State tracks per (step, task_name) completion | unit | `uv run pytest test/test_pipeline_state.py::test_state_mark_complete -x` | ❌ Wave 0 |
| REQ-04 | Resume skips completed steps | unit (monkeypatch) | `uv run pytest test/test_pipeline_state.py::test_resume_skips_complete -x` | ❌ Wave 0 |
| REQ-04 | `--force` overrides resume | integration | `uv run pytest test/test_pipeline_state.py::test_force_resets_state -x` | ❌ Wave 0 |
| REQ-05 | Hash-based path is deterministic | unit | `uv run pytest test/test_pipeline_state.py::test_config_hash_deterministic -x` | ❌ Wave 0 |
| REQ-05 | Hash changes with different params | unit | `uv run pytest test/test_pipeline_state.py::test_config_hash_differs -x` | ❌ Wave 0 |
| REQ-06 | `experiment_config.json` written at start | unit (tmp_path) | `uv run pytest test/test_pipeline_state.py::test_config_json_written -x` | ❌ Wave 0 |
| REQ-06 | `results_summary.json` written at end | unit (tmp_path) | Already covered by `test_pipeline_smoke_run` | ✅ |
| REQ-07 | `--dataset_index` resolves correctly | unit | `uv run pytest test/test_pipeline_state.py::test_dataset_index_resolution -x` | ❌ Wave 0 |
| REQ-07 | `--dataset_index` and `--dataset_name` mutually exclusive | unit | `uv run pytest test/test_pipeline_state.py::test_mutually_exclusive_args -x` | ❌ Wave 0 |
| REQ-07 | Logging setup creates file handler | unit | `uv run pytest test/test_pipeline_state.py::test_logging_setup -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest test/test_pipeline_state.py -x`
- **Per wave merge:** `uv run pytest test/ -x`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `test/test_pipeline_state.py` -- covers REQ-03, REQ-04, REQ-05, REQ-06, REQ-07
  - Tests for `PipelineState` dataclass (mark_complete, is_step_complete)
  - Tests for atomic write (tmp file cleanup on error, rename success)
  - Tests for resume logic (state loaded, steps skipped)
  - Tests for `--force` flag (state reset)
  - Tests for config hash computation (deterministic, differs on change)
  - Tests for dataset index resolution
  - Tests for mutually exclusive args
  - Tests for logging setup
- [ ] Framework install: `uv add tenacity` -- tenacity not yet in project deps

## Security Domain

This phase does not introduce authentication, session management, or cryptographic requirements. The hash is a non-security checksum (8-char SHA-256 prefix for drift detection, not integrity verification). No ASVS categories apply beyond:

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | `argparse` type validation + range checks on `--dataset_index` |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Path traversal via `--dataset_name` | Tampering | `argparse` accepts raw string; validate against registry via `get_dataset_metadata()` which raises `KeyError` for unknown names |
| Integer overflow on `--dataset_index` | Repudiation | `argparse type=int` + bounds check against `len(all_datasets)` |

## Sources

### Primary (HIGH confidence)
- Context7 `/jd/tenacity` - tenacity retry decorator, stop_after_attempt, wait_exponential, before_sleep_log, RetryError [VERIFIED: Context7]
- PyPI tenacity 9.1.4 - latest version, uploaded 2026-02-07 [VERIFIED: PyPI API]
- `src/rbspaper/pipeline/core.py` - existing `_save_json()`, `_json_default()`, `run_experiment_pipeline()` flow [VERIFIED: codebase]
- `src/rbspaper/pipeline/config.py` - frozen dataclass patterns, `PipelineArtifactConfig.run_dir` [VERIFIED: codebase]

### Secondary (MEDIUM confidence)
- `_sources/autotsaugment/runners/py/runner.py` - reference hash-based naming pattern (line 64-95), dataset index lookup (line 260) [VERIFIED: codebase]
- `_sources/autotsaugment/runners/bash/runner.sh` - SLURM array job pattern, QoS retry loop [VERIFIED: codebase]
- Python stdlib `logging` module - `FileHandler`, `StreamHandler`, `RotatingFileHandler` [CITED: docs.python.org/3/library/logging.html]
- Python stdlib `hashlib` - SHA-256 [CITED: docs.python.org/3/library/hashlib.html]

### Tertiary (LOW confidence)
- POSIX `os.rename()` atomicity on NFS/Lustre - depends on specific mount configuration [ASSUMED: standard NFS behavior, verify on target HPC]

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - tenacity verified via PyPI + Context7; stdlib modules are stable
- Architecture: HIGH - pipeline flow is well-understood from existing `core.py`; integration points are clear
- Pitfalls: MEDIUM - NFS atomicity and tqdm/logging interference are environment-dependent; assumptions A2 and A3 noted

**Research date:** 2026-05-06
**Valid until:** 2026-06-06 (stable domain -- Python stdlib, tenacity API stable since v8)
