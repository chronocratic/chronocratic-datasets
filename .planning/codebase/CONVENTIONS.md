# Coding Conventions

**Analysis Date:** 2026-05-05

## Naming Patterns

**Files:**
- `snake_case` for all Python modules. Examples: `pipeline/core.py`, `attacks/functional.py`, `data/registry.py`
- `__init__.py` used as barrel files that re-export public API with explicit `__all__` lists
- Private helper functions prefixed with underscore: `_train_model()`, `_json_default()`, `_run_attack_for_payload()`

**Functions:**
- `snake_case` for all functions and methods
- All public functions have full type hints including return types
- All function calls use **keyword arguments only** (no positional args). This is a strict project convention. Example:
  ```python
  results = run_experiment_pipeline(config=config)
  encode_data(data=partition_tensors['train']['inputs'], model=model, batch_size=batch_size)
  ```
- Private helper functions use leading underscore prefix: `_build_attack_kwargs_from_parameters()`

**Variables:**
- `snake_case` for all variables
- Module-level constants use `UPPER_SNAKE_CASE`: `MIN_LABELED_BATCH_LENGTH = 2`, `MIN_NDIMS_TO_MERGE = 3`
- Type aliases use `PascalCase`: `AttackKwargValue = float | int | str | bool | Tensor | None`

**Classes:**
- `PascalCase` for all classes
- Abstract base classes named with `Base` prefix: `BaseTimeSeriesDataModule`, `BaseClassificationTimeSeriesDataModule`
- Dataclasses use descriptive names: `AttackExecutionContext`, `FgsmAttackParameters`, `ExperimentPipelineConfig`
- Enums inherit from `StrEnum` (Python 3.11+): `AttackBackend(StrEnum)`, `TimeSeriesDownstreamTask(StrEnum)`

**Types:**
- Union types use the `|` operator (PEP 604): `int | None`, `Tensor | tuple[Tensor, AttackExecutionMetadata]`
- `TYPE_CHECKING` guard used for imports that would cause circular imports or are only needed for type annotations
- `from __future__ import annotations` used in most files (36 of ~90 source files) to enable forward references
- `from typing import override` used for method overrides on abstract classes

## Code Style

**Formatting:**
- Tool: `ruff` (version >= 0.15.9)
- Config: `ruff.toml`
- Line length: 100 characters
- Quote style: single quotes (`'string'`)
- Indent: spaces (4 spaces)
- Target version: Python 3.12 (`py312`)
- Trailing commas: skipped (`skip-magic-trailing-comma = true`)
- Import sorting: `isort` via ruff with `combine-as-imports = true`, `force-sort-within-sections = true`, `order-by-type = false`

**Linting:**
- Tool: `ruff` with `select = ["ALL"]`
- Key ignores:
  - `D107` — Missing `__init__` docstring when class docstring exists
  - `D212` — In favor of D213 (multi-line docstring starts on second line)
  - `Q000` — Double quotes allowed
  - `D100` — Missing module docstring allowed
  - `INP001` — Implicit namespace packages allowed
  - `COM812` — Trailing comma enforcement skipped
  - `D101` — Missing class docstring allowed
  - `RET504` — Unnecessary variable assignment in return allowed (readability in AI code)
  - `PLR0913` — Too many function arguments allowed (readability in AI code)
- Per-file ignores:
  - `__init__.py`: `F401` (unused imports — barrel re-exports)
  - `notebooks/**/*.ipynb`: `D`, `E402`, `T201`
  - `tests/**/*.py`: `D` (docstrings), `PLR2004` (magic numbers), `S101` (assertions)
- Type checking: `ty` (version >= 0.0.28) from Astral

**Docstring Convention:**
- Google-style docstrings enforced by ruff (`convention = "google"`)
- All public modules, functions, classes, and methods have docstrings
- Format:
  ```python
  def run_experiment_pipeline(*, config: ExperimentPipelineConfig) -> ExperimentPipelineResults:
      """Execute the full robust representation experiment pipeline.

      Args:
          config: Pipeline configuration object.

      Returns:
          Structured experiment results with artifacts and metrics.
      """
  ```
- Class docstrings include Args section for `__init__` parameters:
  ```python
  class BaseTimeSeriesDataModule(pl.LightningDataModule, ABC):
      """Shared base for all time series LightningDataModules.

      Handles batch size, scaling, and dataloader construction.

      Args:
          batch_size: Batch size for dataloaders.
          seq_len: Sequence length (for forecasting).
          ...
      """
  ```

## Import Organization

**Order (isort via ruff):**
1. `from __future__ import annotations` (when present — always first)
2. Standard library imports
3. Third-party imports (PyTorch, Lightning, numpy, etc.)
4. Application imports (`from src.rbspaper...`)
5. `if TYPE_CHECKING:` block imports (at top level, after regular imports)

**Path Style:**
- All internal imports use absolute paths via `src.rbspaper.` prefix. Example:
  ```python
  from src.rbspaper.attacks.config import AttackExecutionContext
  from src.rbspaper.pipeline.core import run_experiment_pipeline
  ```
- No relative imports within the `src/rbspaper` package
- `experiment_instances` package is a separate namespace (not under `src/`), imported directly:
  ```python
  from experiment_instances.instances import get_experiment_instance
  ```
- Some data modules use short package imports (e.g., `from rbspaper.data.utils import ...`) — this is an older pattern; the convention is `src.rbspaper.` prefix

**Barrel Files (`__init__.py`):**
- Every subpackage has an `__init__.py` that re-exports the public API
- `__all__` lists are explicitly defined and alphabetically sorted
- Example (`src/rbspaper/attacks/__init__.py`):
  ```python
  from src.rbspaper.attacks.functional import (
      autoattack,
      bim_attack,
      cw_attack,
      ...
  )

  __all__ = [
      'AttackBackend',
      'AttackExecutionContext',
      ...
      'execute_attack',
      'fgsm_attack',
      ...
  ]
  ```

## Error Handling

**Strategy:** Explicit `ValueError`/`TypeError` with descriptive message strings assigned to a variable before raising.

**Patterns:**
- Prefer assigning error message to variable, then raising:
  ```python
  message = 'Pipeline requires at least one downstream task configuration.'
  raise ValueError(message)
  ```
- Or shorter form with `msg`:
  ```python
  msg = 'Unsupported batch type: {type(batch)}'
  raise TypeError(msg)
  ```
- Validation happens early (preflight pattern):
  ```python
  def _preflight_pipeline_config(*, config: ExperimentPipelineConfig) -> None:
      # Validate all constraints before any side effects
  ```
- `__post_init__` validation on frozen dataclasses:
  ```python
  @dataclass(frozen=True)
  class AttackScopeConfig:
      scope: AttackScopePolicy = AttackScopePolicy.TASK_CONDITIONED
      anchor_task: TimeSeriesDownstreamTask | None = None

      def __post_init__(self) -> None:
          if self.scope == AttackScopePolicy.SHARED_INPUT and self.anchor_task is None:
              message = 'anchor_task is required when attack_scope is SHARED_INPUT.'
              raise ValueError(message)
  ```
- Type checking with `isinstance` guards:
  ```python
  if not isinstance(attacked_result, tuple):
      message = 'Attack execution must return (attacked_inputs, metadata) when return_metadata=True.'
      raise TypeError(message)
  ```

**No try/except for control flow** — errors are raised at validation boundaries and propagated. The runner (`runners/py/runner.py`) is the top-level handler:
```python
try:
    main()
except KeyboardInterrupt:
    print('\nInterrupted by user.')
    sys.exit(130)
except Exception as e:
    print(f'\nError: {e}', file=sys.stderr)
    sys.exit(1)
```

## Logging

**Framework:** Standard library `logging` module

**Patterns:**
- Logger created per module: `logger = logging.getLogger(name=__name__)`
- Used in evaluation module for reproducibility tracking:
  ```python
  logger.info(f'Setting evaluation seed to {evaluation_seed} to ensure reproducibility')
  ```
- Used for warnings about data size limits:
  ```python
  logger.info(message)  # Training data size limitation notice
  ```
- `logger.error()` used before raising errors

**Note:** The `logging` approach is minimal. Most user-facing output uses `print()` in the runner.

## Comments

**When to Comment:**
- Module-level docstrings describe the module purpose: `"""Core orchestration logic for robust time-series experiments."""`
- Section comments use `# ======== Header =========` for visual separation in larger files
- Inline comments are sparing; code is expected to be self-documenting through naming
- `# noqa` directives used selectively for ruff rule suppression:
  - `# ruff: noqa: S101` — Allow assertions in tests
  - `# ruff: noqa: D103, S101, PLR2004` — Test file header suppressions
  - `# noqa: ANN001` — Suppress missing type annotation on batch parameter

## Function Design

**Size:** Functions are typically 10-50 lines. Pipeline core has larger functions (~40 lines each) that are internal helpers with single responsibility.

**Parameters:** Keyword-only arguments enforced by `*` in signature:
```python
def run_experiment_pipeline(*, config: ExperimentPipelineConfig) -> ExperimentPipelineResults:
def _extract_clean_representations(
    *,
    partition_tensors: dict[str, dict[str, Tensor]],
    model: pl.LightningModule,
    task_name: str,
    batch_size: int,
    num_workers: int,
) -> TaskRepresentationBundle:
```

**Return Values:**
- Single return values for simple functions
- Tuples for multiple returns: `tuple[Tensor, AttackExecutionMetadata]`
- `None` return for validation/setup functions: `def _preflight_pipeline_config(...) -> None`

## Module Design

**Exports:**
- `__all__` defined in all `__init__.py` barrel files and in key modules (`evaluation.py`, `models/config.py`)
- Re-exports consolidate public API at package level

**Configuration Objects:**
- Frozen dataclasses (`@dataclass(frozen=True)`) for immutable config: `AttackScopeConfig`, `ExperimentPipelineConfig`
- Mutable dataclasses for runtime results: `ExperimentPipelineResults`, `AttackExecutionMetadata`
- ABCs (`from abc import ABC, abstractmethod`) for base classes: `AttackParameters(ABC)`, `ModelParameters(ABC)`

**Design Patterns Used:**
- **Factory Method:** `_fgsm_attack_run()`, `_ts2vec_params()` helper factories in experiment instances
- **Strategy:** `AttackScopePolicy` enum drives different execution paths in `_build_attacked_reps_for_task()`
- **Registry:** `EXPERIMENTS_REGISTRY` dict, `ATTACK_THREAT_MODEL` dict, `SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK` dict
- **Adapter:** `AttackParameters` subclasses adapt different attack configs to common interface via `attack_method` property
- **Template Method:** `BaseTimeSeriesDataModule` defines shared datamodule infrastructure; subclasses implement abstract hooks

## Dataclass Patterns

**Frozen (immutable) configs:**
```python
@dataclass(frozen=True)
class AttackScopeConfig:
    scope: AttackScopePolicy = AttackScopePolicy.TASK_CONDITIONED
    anchor_task: TimeSeriesDownstreamTask | None = None
```

**Mutable with defaults:**
```python
@dataclass
class AttackExecutionMetadata:
    attack: AttackMethod
    backend: AttackBackend
    success_rate: float | None
    mean_l2: float
    mean_linf: float
    extras: dict[str, float | int | str | bool] = field(default_factory=dict)
```

**Abstract base with concrete subclasses:**
```python
@dataclass
class AttackParameters(ABC):
    backend: AttackBackend
    clip_min: float | None = None
    clip_max: float | None = None

    @property
    @abstractmethod
    def attack_method(self) -> AttackMethod:
        """Return canonical attack method identifier."""

@dataclass
class FgsmAttackParameters(AttackParameters):
    epsilon: float = 8.0 / 255.0

    @property
    @override
    def attack_method(self) -> AttackMethod:
        return AttackMethod.FGSM
```

---

*Convention analysis: 2026-05-05*
