# Phase 7: Experiment Tracking - Pattern Map

**Mapped:** 2026-05-08
**Files analyzed:** 6 (2 new, 4 modified)
**Analogs found:** 5 / 6

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/rbspaper/pipeline/loggers.py` (new) | utility | request-response | `src/rbspaper/pipeline/setup/model.py` | role-match |
| `src/rbspaper/pipeline/config.py` (modify) | config | CRUD | `src/rbspaper/pipeline/config.py` (self) | exact |
| `src/rbspaper/pipeline/core.py` (modify) | service | CRUD | `src/rbspaper/pipeline/core.py` (self) | exact |
| `runners/py/runner.py` (modify) | controller | request-response | `runners/py/runner.py` (self) | exact |
| `pyproject.toml` (modify) | config | file I/O | `pyproject.toml` (self) | exact |
| `test/test_logger_factory.py` (new) | test | request-response | `test/test_runner_cli_args.py` | role-match |

## Pattern Assignments

### `src/rbspaper/pipeline/loggers.py` (new, utility, request-response)

**Analog:** `src/rbspaper/pipeline/setup/model.py`

This is a new module. The closest analog is `model.py`, which is a small factory module in the same `pipeline/setup/` package that demonstrates the project's patterns for pure-function factories, type-narrowing, and `__all__` exports.

**Module structure and docstring** (model.py lines 1-3):
```python
"""Factory for building pipeline models from parameter dataclasses."""

__all__ = ['build_model_from_parameters']
```
Apply: Start with a one-line module docstring and `__all__` listing public functions. For loggers.py, `__all__` will include `create_loggers`, `_log_config_to_wandb`, `_log_results_to_wandb`, `_flatten_dict`, `_find_wandb_logger`.

**Imports pattern** (model.py lines 5-17):
```python
from dataclasses import asdict

import lightning.pytorch as pl

from src.rbspaper.configs.models import (
    AutoTCLModelParameters,
    CoSTModelParameters,
    ModelParameters,
    TS2VecModelParameters,
)
```
Apply: Use `import lightning.pytorch as pl` (not `from lightning.pytorch.loggers import ...`). Import logger classes via `pl.loggers.WandbLogger` and `pl.loggers.TensorBoardLogger` through `TYPE_CHECKING` guard, consistent with how `config.py` imports `pl.LightningModule`.

**Factory function pattern** (model.py lines 20-30):
```python
def build_model_from_parameters(*, parameters: ModelParameters) -> pl.LightningModule:
    """Build a Lightning model from typed model parameters."""
    if isinstance(parameters, TS2VecModelParameters):
        return TS2Vec(**asdict(parameters))
    if isinstance(parameters, AutoTCLModelParameters):
        return AutoTCL(**asdict(parameters))
    if isinstance(parameters, CoSTModelParameters):
        return CoST(**asdict(parameters))

    message = f'Unsupported model parameters type: {type(parameters)}'
    raise ValueError(message)
```
Apply: `create_loggers` uses keyword-only arguments (`*,`), returns a typed tuple, and has early-exit guard clauses. The function builds and returns `tuple[WandbLogger | TensorBoardLogger, ...]`.

**Key patterns for loggers.py from RESEARCH.md:**
- TensorBoard always created (zero-config local fallback)
- W&B gated behind `tracking_mode != "disabled"` and lazy import
- `log_model=False` explicitly set
- `save_dir=str(run_dir)` for both loggers
- Return empty tuple `()` when `persist_artifacts=False`

**Additional utility functions (no direct analog in model.py):**

The `_flatten_dict` helper follows the same functional pattern as analysis.py's utilities (pure functions, type hints, keyword-only args). See analysis.py lines 24-41 for the project's utility function style:
```python
def _ensure_2d(*, features: np.ndarray) -> np.ndarray:
    if features.ndim <= TWO_DIMS:
        return features
    return features.reshape(features.shape[0], -1)
```
Apply: `_flatten_dict`, `_find_wandb_logger`, `_log_config_to_wandb`, `_log_results_to_wandb` all use `*,` keyword-only, return type hints, and Google-style docstrings.

**TYPE_CHECKING pattern for logger types** (config.py lines 8-13):
```python
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    import lightning.pytorch as pl
```
Apply: loggers.py should import `WandbLogger` and `TensorBoardLogger` under `TYPE_CHECKING` to avoid runtime import of wandb when not installed.

---

### `src/rbspaper/pipeline/config.py` (modify, config, CRUD)

**Analog:** `src/rbspaper/pipeline/config.py` (self-modify)

**frozen dataclass pattern** (lines 51-68):
```python
@dataclass(frozen=True)
class AttackScopeConfig:
    """Controls the scope relationship between attacks and downstream task execution."""
    scope: AttackScopePolicy = AttackScopePolicy.TASK_CONDITIONED
    anchor_task: TimeSeriesDownstreamTask | None = None

    def __post_init__(self) -> None:
        if self.scope == AttackScopePolicy.SHARED_INPUT and self.anchor_task is None:
            message = 'anchor_task is required when attack_scope is SHARED_INPUT.'
            raise ValueError(message)
```
Apply: The `loggers` field on `ExperimentPipelineConfig` uses `tuple` (immutable) with `default_factory=tuple`. No `__post_init__` needed.

**ExperimentPipelineConfig modification** (lines 274-287):
```python
@dataclass(frozen=True)
class ExperimentPipelineConfig:
    """Top-level pipeline configuration."""
    model: pl.LightningModule
    data: DataConfig
    training: TrainingConfig
    encoding: RepresentationEncodingConfig
    attacks: tuple[AttackRunConfig, ...]
    downstream_tasks: tuple[DownstreamTaskConfig, ...]
    analysis: RepresentationAnalysisConfig
    artifacts: PipelineArtifactConfig
    seed: int = 42
    attack_scope: AttackScopeConfig = field(default_factory=AttackScopeConfig)
```
Apply: Add new field after `attack_scope`:
```python
    loggers: tuple[pl.loggers.LightningLogger, ...] = field(default_factory=tuple)
```
The TYPE_CHECKING block already imports `pl`, so `pl.loggers.LightningLogger` is available at type-check time. The import section needs updating.

**TYPE_CHECKING update** (lines 8-18):
```python
if TYPE_CHECKING:
    from pathlib import Path
    import lightning.pytorch as pl
    import numpy as np
    from src.rbspaper.attacks.config import AttackExecutionContext, AttackParameters
    from src.rbspaper.data.modules.abstract import BaseTimeSeriesDataModule
    from src.rbspaper.enums.general import TimeSeriesDownstreamTask
```
Apply: No change needed — `pl` is already imported under TYPE_CHECKING. `pl.loggers.LightningLogger` is accessible as a string annotation since `from __future__ import annotations` is used (line 3).

---

### `src/rbspaper/pipeline/core.py` (modify, service, CRUD)

**Analog:** `src/rbspaper/pipeline/core.py` (self-modify)

**Trainer instantiation** (lines 358-379):
```python
def _train_model(
    *, config: ExperimentPipelineConfig, run_dir: Path, data_module: pl.LightningDataModule
) -> Path | None:
    checkpoints_dir = run_dir / 'checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    canonical_checkpoint = checkpoints_dir / config.training.checkpoint_filename

    if config.training.reuse_trained_checkpoint and canonical_checkpoint.exists():
        return canonical_checkpoint

    trainer = pl.Trainer(**config.training.trainer_kwargs)
    trainer.fit(
        model=config.model, datamodule=data_module, ckpt_path=config.training.resume_from_checkpoint
    )
```
Apply: Line 368 (`trainer = pl.Trainer(**config.training.trainer_kwargs)`) must be modified to conditionally include `loggers`:
```python
    trainer_kwargs = dict(config.training.trainer_kwargs)
    if config.loggers:
        trainer_kwargs['loggers'] = list(config.loggers)
    trainer = pl.Trainer(**trainer_kwargs)
```
CRITICAL: Do NOT pass `loggers=[]` — per RESEARCH.md Pitfall 3, an empty list triggers Lightning's default TensorBoardLogger. Only add the `loggers` key when the tuple is non-empty.

**Imports pattern** (lines 1-62):
```python
from __future__ import annotations

from dataclasses import asdict, fields
from enum import Enum
import json
import logging
from pathlib import Path
import shutil
from typing import Any, cast, TYPE_CHECKING

import lightning.pytorch as pl
```
Apply: Add import for the new loggers module:
```python
from src.rbspaper.pipeline.loggers import (
    _log_config_to_wandb,
    _log_results_to_wandb,
)
```
This follows the existing pattern of intra-package imports (lines 26-52).

**Step timing pattern** (lines 151-164):
```python
    # Train step
    state = builder.build()
    if not state.is_step_complete(step='train'):
        logger.info('Step: train')
        try:
            checkpoint_path = retry_step(_train_model)(
                config=config, run_dir=run_dir, data_module=data_module
            )
        except RetryError:
            logger.exception('Step failed after retries: train')
            raise
        builder.mark_complete(step='train')
        save_pipeline_state(state=builder.build(), path=state_path)
        logger.info('Step complete: train')
```
Apply: Wrap each step block with `time.perf_counter()`:
```python
import time

# At top of run_experiment_pipeline, after state setup:
timing: dict[str, float] = {}
step_start = time.perf_counter()

# ... train step ...
timing['train'] = time.perf_counter() - step_start

# (repeat for each step)
```

**_write_experiment_config hook** (lines 334-355):
```python
def _write_experiment_config(*, config: ExperimentPipelineConfig, run_dir: Path) -> None:
    model_name = getattr(config.model, 'model_name', type(config.model).__name__)
    config_data = {
        'model_name': model_name,
        'seed': config.seed,
        'downstream_tasks': [t.task_name for t in config.downstream_tasks],
        'attack_names': [a.name for a in config.attacks],
        'trainer_kwargs': config.training.trainer_kwargs,
        'attack_scope': config.attack_scope.scope.value,
        'encoding_batch_size': config.encoding.batch_size,
    }
    _save_json(file_path=run_dir / 'experiment_config.json', data=config_data)
```
Apply: After `_save_json`, add:
```python
    _log_config_to_wandb(config_data=config_data, loggers=config.loggers)
```

**_persist_artifacts hook** (lines 311-321):
```python
    if config.artifacts.persist_artifacts:
        _save_json(
            file_path=run_dir / 'results_summary.json',
            data={
                'run_dir': run_dir,
                'model_name': results.model_name,
                'checkpoint_path': checkpoint_path,
                'downstream_metrics': downstream_metrics,
                'analysis': analysis_results,
            },
        )
```
Apply: After the `_save_json` call, add:
```python
        _log_results_to_wandb(
            results_summary=results_summary_data,
            timing=timing,
            loggers=config.loggers,
        )
```

**_prepare_run_directory modification** (lines 326-331):
```python
def _prepare_run_directory(*, config: ExperimentPipelineConfig) -> Path:
    run_dir = config.artifacts.run_dir
    if config.artifacts.persist_artifacts:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_experiment_config(config=config, run_dir=run_dir)
    return run_dir
```
Apply: No structural change — `_write_experiment_config` already calls `_log_config_to_wandb` internally.

---

### `runners/py/runner.py` (modify, controller, request-response)

**Analog:** `runners/py/runner.py` (self-modify)

**Argument parsing pattern** (lines 71-143):
```python
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run a robustness benchmark experiment.')
    parser.add_argument(
        '--experiment_id', type=str, default=None, help='Registered experiment ID (e.g. ts2vec).'
    )
    # ... more arguments ...
    parser.add_argument(
        '--attack_family',
        type=str,
        default=None,
        choices=['white_box', 'black_box'],
        help='Filter attacks by family (white_box, black_box). Default: all families.',
    )
    return parser.parse_args(argv)
```
Apply: Add `--tracking_mode` argument before `return parser.parse_args(argv)`:
```python
    parser.add_argument(
        '--tracking_mode',
        type=str,
        default=None,
        choices=['online', 'offline', 'disabled'],
        help='Experiment tracking mode. Default: auto-detect (offline on HPC, online locally).',
    )
```
This follows the exact same pattern as `--attack_family` (lines 136-142): `type=str`, `default=None`, `choices=[...]`, `help='...'`.

**Imports** (lines 15-45):
```python
import argparse
import copy
from dataclasses import asdict
from enum import Enum
import logging
from pathlib import Path
import sys

from experiment_instances.data_utils import build_dataset_task_profile
# ...
from src.rbspaper.pipeline.core import run_experiment_pipeline
from src.rbspaper.pipeline.setup.model import build_model_from_parameters
from src.rbspaper.pipeline.state import compute_config_hash, load_pipeline_state, STATE_FILENAME
```
Apply: Add import after existing `src.rbspaper.pipeline` imports:
```python
from src.rbspaper.pipeline.loggers import create_loggers
```

**HPC detection and logger factory** (main function, lines 345-440):
```python
def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    # ... resolve experiment, build run_name ...

    # Assemble pipeline config
    config = _build_pipeline_config(
        experiment_instance=experiment,
        # ... other args ...
    )
```
Apply: Between resolving the experiment and building the pipeline config, add HPC detection + logger creation:
```python
    # Resolve tracking mode (D-03: auto-detect HPC via SLURM_JOB_ID)
    tracking_mode = args.tracking_mode
    if tracking_mode is None:
        tracking_mode = 'offline' if os.environ.get('SLURM_JOB_ID') else 'online'

    # Create loggers (D-01, D-02)
    loggers = create_loggers(
        run_dir=output_dir / run_name,
        run_name=f'{args.experiment_id}_{dataset_name}_{args.seed}',
        tracking_mode=tracking_mode,
        persist_artifacts=True,
    )
```
Note: `os` needs to be imported if not already present. Check: runner.py does not currently import `os`. Add `import os` to the imports block.

**_build_pipeline_config modification** (lines 200-289):
```python
def _build_pipeline_config(
    *,
    experiment_instance: ExperimentInstance,
    # ... existing params ...
    attack_family: AttackFamily | None = None,
) -> ExperimentPipelineConfig:
```
Apply: Add `loggers` parameter and pass it to `ExperimentPipelineConfig`:
```python
def _build_pipeline_config(
    *,
    # ... existing params ...
    attack_family: AttackFamily | None = None,
    loggers: tuple = (),
) -> ExperimentPipelineConfig:
```
And in the `ExperimentPipelineConfig(...)` constructor call (line 278-288):
```python
    return ExperimentPipelineConfig(
        model=model,
        # ... existing fields ...
        attack_scope=experiment_instance.attack_scope,
        loggers=loggers,
    )
```

---

### `pyproject.toml` (modify, config, file I/O)

**Analog:** `pyproject.toml` (self-modify)

**Dependency groups pattern** (lines 39-60):
```toml
[dependency-groups]
dev = [
    "pytest>=8.2",
    "pytest-cov>=5.0",
    "ruff>=0.15.9",
    "ty>=0.0.28",
]

attacks = [
    "adversarial-robustness-toolbox~=1.20",
    "torchattacks~=3.5",
    "foolbox~=3.3",
]

attacks_extended = [
    "cleverhans~=4.0",
]

notebooks = [
    "notebook>=7.3",
    "jupyterlab>=4.3",
]
```
Apply: Add a new `tracking` group following the same format:
```toml
tracking = [
    "wandb>=0.18.0",
    "tensorboard>=2.17.0",
]
```
Insert between `attacks_extended` and `notebooks` groups to maintain alphabetical ordering.

**Conflicts declaration** (lines 62-68):
```toml
[tool.uv]
default-groups = ["dev"]
conflicts = [
    [
        { group = "attacks" },
        { group = "notebooks" },
    ],
]
```
Apply: No change needed. The `tracking` group has no known conflicts with other groups.

---

### `test/test_logger_factory.py` (new, test, request-response)

**Analog:** `test/test_runner_cli_args.py`

**Test imports pattern** (test_runner_cli_args.py lines 1-7):
```python
from __future__ import annotations

import pytest

from runners.py.runner import _parse_args, main
```
Apply: Import from the new loggers module:
```python
from src.rbspaper.pipeline.loggers import create_loggers, _flatten_dict, _find_wandb_logger
```

**Test function pattern** (test_runner_cli_args.py lines 13-26):
```python
class TestDatasetIndexArg:
    """Verify --dataset_index is accepted by the parser."""

    def test_dataset_index_accepted(self) -> None:
        """--dataset_index 0 should parse without error."""
        args = _parse_args(
            ['--dataset_index', '0', '--experiment_id', 'ts2vec', '--data_root', '/fake/data']
        )
        assert args.dataset_index == 0
```
Apply: Use class-based organization with descriptive docstrings, type hints on all methods, keyword-only arguments to functions under test.

**Test pattern for main validation** (test_runner_cli_args.py lines 54-76):
```python
class TestMainValidation:
    """Verify main() enforces mutual exclusivity and dataset resolution."""

    def test_mutually_exclusive_args_raise(self) -> None:
        """--dataset_index and --dataset_name together must raise SystemExit."""
        with pytest.raises(SystemExit):
            main([...])
```
Apply: Use `pytest.raises` for validation tests, `tmp_path` fixture for file-system-dependent tests.

**Pipeline test pattern** (test_pipeline_core.py lines 62-108):
```python
class _TinyLightningModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=4, out_features=2)
        self.model_name = 'TinyModel'
    # ... forward, training_step, etc.
```
Apply: For logger factory tests that need mocking wandb, use `monkeypatch.setattr` or `unittest.mock.patch` to avoid actual W&B API calls. The test_pipeline_core.py pattern of monkeypatching is the project standard.

## Shared Patterns

### Keyword-Only Function Arguments
**Source:** All files in `src/rbspaper/pipeline/` consistently use `*,` to enforce keyword-only arguments.
**Apply to:** All new functions in `loggers.py` and all parameter additions to existing functions.
```python
# From model.py line 20
def build_model_from_parameters(*, parameters: ModelParameters) -> pl.LightningModule:

# From state.py line 148
def compute_config_hash(*, model_params: dict[str, object], seed: int) -> str:

# From analysis.py line 24
def _ensure_2d(*, features: np.ndarray) -> np.ndarray:
```

### Type Hints on All Functions
**Source:** Every function across the pipeline modules has explicit parameter and return type hints.
**Apply to:** All new functions in `loggers.py` and all tests in `test_logger_factory.py`.

### Google-Style Docstrings
**Source:** config.py line 53-58, state.py line 22-33, model.py line 20-21.
**Apply to:** All new functions.
```python
def compute_config_hash(*, model_params: dict[str, object], seed: int) -> str:
    """Compute an 8-character SHA-256 prefix for config drift detection.

    Args:
        model_params: Serializable model parameter dictionary.
        seed: Random seed used for the experiment run.

    Returns:
        8-character lowercase hexadecimal string.
    ```

### Frozen Dataclass with Tuple Defaults
**Source:** config.py lines 274-287 — `attacks: tuple[AttackRunConfig, ...]` and `downstream_tasks: tuple[DownstreamTaskConfig, ...]`.
**Apply to:** New `loggers` field: `loggers: tuple[pl.loggers.LightningLogger, ...] = field(default_factory=tuple)`.

### Logging with `logging.getLogger(__name__)`
**Source:** core.py line 64, runner.py line 47.
**Apply to:** loggers.py should use `logger = logging.getLogger(__name__)` for any internal logging.

### Snake_case Naming
**Source:** All Python files follow snake_case for functions/variables and PascalCase for classes.
**Apply to:** All new code.

### Future Annotations
**Source:** config.py line 3, core.py line 3, state.py line 3, analysis.py line 3.
**Apply to:** `loggers.py` should start with `from __future__ import annotations` for consistency.

### Private Helper Functions with Underscore Prefix
**Source:** core.py has `_train_model`, `_write_experiment_config`, `_json_default`, etc. state.py has `_PipelineStateBuilder`, `_json_default`, `_atomic_write_json`.
**Apply to:** In loggers.py, `_find_wandb_logger`, `_flatten_dict`, `_log_config_to_wandb`, `_log_results_to_wandb` are private helpers. Only `create_loggers` is public (listed in `__all__`).

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| W&B-specific wandb.Run access pattern | utility | event-driven | No existing code interacts with external tracking SDKs; use RESEARCH.md Pattern 2 code examples directly |

## Metadata

**Analog search scope:** `src/rbspaper/pipeline/`, `runners/py/`, `test/`, `pyproject.toml`
**Files scanned:** 10 (model.py, config.py, core.py, runner.py, state.py, analysis.py, pyproject.toml, test_pipeline_core.py, test_runner_cli_args.py, test_runner_logging.py)
**Pattern extraction date:** 2026-05-08
