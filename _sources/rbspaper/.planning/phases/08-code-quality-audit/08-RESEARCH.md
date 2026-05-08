# Phase 08: Code Quality Audit - Research

**Researched:** 2026-05-08
**Domain:** Python type contract cleanup (ty static checker), linting (ruff), dead code removal
**Confidence:** HIGH

## Summary

This phase removes all fixable `ty: ignore` suppressions, `noqa` directives, and `ruff.toml` per-file-ignores from source code. The goal is to fix underlying type contracts rather than suppress diagnostics. Only genuine `ty` limitations should remain, documented with `# why:` comments.

**Current state:** 18 `ty` diagnostics across `src/rbspaper/` (11 errors + 7 warnings) and 5 `ruff` errors. Most `ty` errors cascade from the lazy `__getattr__` import in `ts2vec/__init__.py`, which causes `TS2Vec` to resolve as `object` at type-checking time, breaking the `_ModelType` union, all `isinstance` checks, and all parameter annotations that reference it.

**Primary recommendation:** Remove the lazy import from `ts2vec/__init__.py` (D-01) as the first fix -- it resolves 7 `ty` errors in `encoding.py`, 1 in `setup/model.py`, and turns 3 of the 4 `ty: ignore` comments in `core.py` into unused-ignore warnings. Runtime verification confirms no circular dependency exists. The remaining fixes are independent single-file changes.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Remove `__getattr__` lazy import from `ts2vec/__init__.py` and use direct import from `model.py`. The circular dependency that motivated it no longer exists -- verified at runtime. Fixes 9 `ty` errors in `encoding.py` (unsupported-operator, invalid-type-form, invalid-argument-type, invalid-method-override).
- **D-02:** Widen `DataConfig.data_module` from `BaseTimeSeriesDataModule` to `pl.LightningDataModule`. Removes 5 ty ignores.
- **D-03:** Align `attack_kwargs` value type from `dict[str, object]` to `dict[str, AttackKwargValue]` in `_backend.py`. Removes 1 ty ignore in `functional.py`.
- **D-04:** Use TypedDict for pipeline state serialization in `state.py`. Defines exact JSON keys and types, removing 4 `ty: ignore` comments on dict access.
- **D-05:** Define Protocol for model parameters in `runner.py`. Covers `set_sequence_length` and `max_train_length` as optional attributes. Removes 4 `ty: ignore` comments on `hasattr`-guarded access.
- **D-06:** Remove entire `src/rbspaper/adapters/` package (attack_adapter, model_adapter, task_adapter). Zero external imports since creation. Clean `ruff.toml` per-file-ignore entry.
- **D-07:** Fix all fixable noqa in source code: extract `BINARY_CLASSIFICATION_THRESHOLD = 2` constant (PLR2004), add `# why:` comments to structural ignores (N812, PLC0415, SLF001, S311, F401), fix UP017 datetime calls.
- **D-08:** Remove `ruff.toml` per-file-ignores for `core.py` (C901, PLR0912, PLR0915). Add scoped `noqa` on specific complex functions instead.
- **D-09:** Fix `CoST._get_slice()` return annotation from `-> None` to `-> slice | None`. Trivial one-line change.
- **D-10:** Widen `AugmentationMethod.augment()` abstract return type to include `tuple[torch.Tensor, torch.Tensor, int]` for `CropShiftAugmentation`. Fixes Liskov violation.
- **D-11:** Fix `TimeSeriesDataset.__getitem__(self, item)` to `__getitem__(self, index: int)` to match `torch.utils.data.Dataset`. Removes 1 `ty: ignore`. (Already committed.)
- **D-12:** Extract `isinstance` check in `strategies.py` to a helper function. Isolates the narrowing issue. Keeps 1 documented `ty: ignore` as genuine limitation. (Already done; ignore retained with `# why:` comment.)
- **D-13:** Remove 3 unused `ty: ignore` in `pipeline/core.py` (lines 537, 544, 671). Ty reports them as `unused-ignore-comment`.

### Claude's Discretion

- Exact TypedDict structure for `load_pipeline_state`
- Protocol naming and placement (new file vs. existing module)
- Helper function signature for the isinstance narrowing
- Specific noqa `# why:` comment text for each structural ignore

### Deferred Ideas (OUT OF SCOPE)

- **Mixin architectural refactor** -- Split `EncodingFunctionalityMixin` into separate pooling-based and concatenation-based contracts. CoST's (trend, seasonality) tuple encoding does not fit the pooling-based interface. Requires changing `_evaluate_with_feature_concatenation` signature. Future phase (Phase 9).
- **Narrow `__getitem__` return type** -- Current fix only aligns the parameter. Narrowing `Any` to concrete sample types per mode is a larger refactor involving transform chain typing. Future improvement.
- **Augmentation -> encoders dependency** -- `CropShiftAugmentation` imports from `ts2vec/utils.py`. While the cycle is gone, the augmentation layer depending on model-internal utils is fragile. Extract shared types to `models/types.py` to prevent future cycles.
- **`autotcl/model.py` `call-non-callable` errors** -- Lines 278, 325: `self._averaged_encoder.update_parameters()` fails because `_averaged_encoder` is typed as `Tensor | Module`. Requires fixing the `_averaged_encoder` attribute type on the mixin. Deferred.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Type contracts (ty) | Source code (`src/`) | — | Static analysis; no runtime component |
| Lint rules (ruff) | Source code (`src/`) | Config (`ruff.toml`) | Rules live in config, fixes in source |
| Dead code removal | Source code (`src/rbspaper/adapters/`) | Config (`ruff.toml`) | Remove files and their config entries |
| Runner type safety | `runners/py/runner.py` | Protocol def (`src/`) | Protocol bridges runner and model params |
| Pipeline state typing | `src/rbspaper/pipeline/state.py` | — | Self-contained serialization module |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `ty` | 0.0.29 `[VERIFIED: uv run ty --version]` | Static type checker | Project uses `ty` (not mypy) for type checking. All fixes target `ty` diagnostics. |
| `ruff` | Latest `[ASSUMED: project default]` | Linter + formatter | Project uses `ruff` for all linting. Config in `ruff.toml`. |
| `pytest` | Current `[VERIFIED: 143 tests collected]` | Test runner | All 143 existing tests must pass after changes. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `typing.TypedDict` | 3.12 stdlib | Explicit dict key types | D-04: state serialization |
| `typing.Protocol` | 3.12 stdlib | Structural typing | D-05: runner model params |
| `torch.nn.Module` | Current | Base type for model attrs | D-01: widening model parameter |
| `lightning.pytorch.LightningDataModule` | Current | Base type for data modules | D-02: widening data_module type |
| `lightning.pytorch.loggers.Logger` | Current | Base class for loggers | loggers.py: replacing deprecated `LightningLogger` |

**Installation:** No new packages needed. All tools are already available via `uv`.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Protocol for runner params | `typing.cast` + assert | Protocol is cleaner, more precise; cast is a runtime hack |
| TypedDict for state | `dict` + `dict.get()` with defaults | TypedDict gives static guarantee; dict.get() hides missing keys |
| Direct import (D-01) | TYPE_CHECKING guard | Direct import is simpler and verified working; TYPE_CHECKING adds complexity |

## Architecture Patterns

### Pattern: Cascade Resolution

The `ts2vec/__init__.py` lazy import is a **root-cause fix**: removing it resolves errors in 5 downstream files (`encoding.py`, `setup/model.py`, `core.py`, and indirectly any file importing `_ModelType`). The planner should sequence D-01 first.

### Pattern: Type Widening at Boundaries

Several fixes widen types at API boundaries rather than narrowing them:
- `encoding.py`: `_ModelType = TS2Vec | AutoTCL | CoST` -> `pl.LightningModule`
- `config.py`: `BaseTimeSeriesDataModule` -> `pl.LightningDataModule`
- `strategies.py`: abstract return type widened for `CropShiftAugmentation`

This is correct because these are **dispatch points** -- the function receives the base type, uses `isinstance()` to narrow, and dispatches. The static checker can see the narrowing, so widening the accepted type eliminates the error without losing safety.

### Pattern: Structural Typing for Dynamic Access

The `runner.py` hasattr-guarded attribute access (e.g., `hasattr(model_params, 'set_sequence_length')`) is a structural typing problem. A Protocol with optional attributes captures the intent precisely: the code only accesses attributes that exist.

### Anti-Patterns to Avoid

- **Lazy `__getattr__` imports for type resolution:** These cause the imported name to resolve as `object` at type-checking time. Use direct imports or `TYPE_CHECKING` guards instead.
- **Per-file ruff ignores for complexity:** Complexity warnings (C901, PLR0912, PLR0915) should be scoped to specific functions via inline `noqa`, not suppressed for an entire file.
- **`dict[str, object]` for typed values:** When values have a known set of acceptable types, define a union type alias (`AttackKwargValue`) and use it consistently across the call chain.

## Current Diagnostic Inventory

### Ty Diagnostics (18 total)

**Errors that cascade from D-01 (lazy import removal):**
| File | Line | Rule | Current | After D-01 |
|------|------|------|---------|------------|
| `encoding.py` | 12 | unsupported-operator | `_ModelType = TS2Vec \| AutoTCL \| CoST` | Resolved (TS2Vec is proper class) |
| `encoding.py` | 33 | invalid-argument-type | `isinstance(model, TS2Vec)` | Resolved |
| `encoding.py` | 101, 110, 118 | invalid-type-form | `model: TS2Vec` param | Resolved |
| `setup/model.py` | 23 | call-non-callable | `TS2Vec(**asdict(params))` | Resolved |
| `core.py` | 537, 544, 671 | unused-ignore-comment | `model=model, # ty: ignore` | Becomes unused (remove) |

**Errors fixed by D-09 (CoST return type):**
| File | Line | Rule | Fix |
|------|------|------|-----|
| `cost/model.py` | 350 | invalid-method-override | Change `-> None` to `-> slice \| None` |

**Errors fixed by widening model type in encoding.py:**
| File | Line | Rule | Fix |
|------|------|------|-----|
| `encoding.py` | 12 | (post D-01) | Widen `_ModelType` to `pl.LightningModule` |
| `core.py` | 530 | (post D-01) | Keep 1 `ty: ignore` until `_ModelType` widened; then remove |

**Warnings fixed by loggers.py rename:**
| File | Line | Rule | Fix |
|------|------|------|-----|
| `loggers.py` | 86, 145, 172 | possibly-missing-submodule | Replace `pl.loggers.LightningLogger` with `pl.loggers.Logger` |

**Errors deferred (Phase 9 / genuine limitation):**
| File | Line | Rule | Status |
|------|------|------|--------|
| `autotcl/model.py` | 278, 325 | call-non-callable | Deferred; `_averaged_encoder` typed `Tensor | Module` by ty bug; add `ty: ignore[call-non-callable] # why: ty cannot resolve AveragedModel from mixin attr` |
| `ts2vec/model.py` | 157 | call-non-callable | Same root cause as above; add `ty: ignore[call-non-callable] # why: ...` |

### Ty: ignore Directives (19 total)

| File | Lines | Rule | Action |
|------|-------|------|--------|
| `core.py` | 530 | invalid-argument-type | Remove after D-01 + widen `_ModelType` |
| `core.py` | 537 | invalid-argument-type | Remove (unused-ignore) |
| `core.py` | 544 | invalid-argument-type | Remove (unused-ignore) |
| `core.py` | 671 | invalid-argument-type | Remove (unused-ignore) |
| `state.py` | 138-141 | invalid-argument-type | Remove via D-04 (TypedDict) |
| `augmentation/strategies.py` | 93 | invalid-method-override | Remove via D-10 (widen abstract return) |
| `data/datasets/strategies.py` | 120 | invalid-assignment | Keep with `# why:` (genuine limitation) |
| `attacks/functional.py` | 152 | invalid-argument-type | Remove via D-03 (align dict value type) |
| `attacks/batch.py` | 98 | invalid-argument-type | Remove via widening `attack_dataset` param |
| `runner.py` | 271 | call-non-callable | Remove via D-05 (Protocol) |
| `runner.py` | 275 | invalid-assignment | Remove via D-05 (Protocol) |
| `runner.py` | 277 | invalid-argument-type | Remove via D-05 (Protocol) |
| `runner.py` | 349 | invalid-assignment | Remove via `cast` or `typing.override` |
| `runner.py` | 418 | invalid-argument-type | Remove via widening `compute_config_hash` param |
| `autotcl/model.py` | 278, 325 | call-non-callable | Add `ty: ignore` with `# why:` (deferred) |
| `ts2vec/model.py` | 157 | call-non-callable | Add `ty: ignore` with `# why:` (deferred) |

### Ruff Errors (5 total)

| File | Line | Rule | Fix |
|------|------|------|-----|
| `augmentation/__init__.py` | 9 | PLC0415 | Replace `__getattr__` with direct import |
| `ts2vec/__init__.py` | 9 | PLC0415 | Replace `__getattr__` with direct import |
| `cost/model.py` | 21 | TC001 | Move `CoSTAugmentationMode` into TYPE_CHECKING block |
| `loggers.py` | 162, 234 | BLE001 | Narrow `except Exception` to specific types |

### Noqa Inventory (structural, keep with why comments)

| File | Line | Rule | Reason |
|------|------|------|--------|
| `pipeline/core.py` | 50 | F401 | Intentional barrel re-export |
| `pipeline/core.py` | 101 | C901, PLR0912, PLR0915 | Complex orchestrator function; move from per-file-ignore to inline noqa |
| `models/utils.py` | 19 | N812 | `torch.nn.functional` convention |
| `models/losses.py` | 22 | N812 | `torch.nn.functional` convention |
| `models/autotcl/model.py` | 7 | N812 | `torch.nn.functional` convention |
| `models/abstract/encoding_functionality_mixin.py` | 10 | N812 | `torch.nn.functional` convention |
| `models/layers/convolutions/dilated.py` | 9 | N812 | `torch.nn.functional` convention |
| `models/layers/convolutions/same_pad.py` | 9 | N812 | `torch.nn.functional` convention |
| `models/cost/model.py` | 12 | N812 | `torch.nn.functional` convention |
| `evaluation/classification.py` | 20 | PLR2004 | Magic number 2 for binary classification (extract to constant per D-07) |
| `attacks/_backend.py` | 77, 103 | PLC0415 | Lazy ART import (structural; unavoidable) |
| `models/layers/general.py` | 88 | SLF001 | PyTorch internal access |
| `pipeline/state.py` | 73, 101 | UP017 | Already clean -- `datetime.now(timezone.utc)` is current best practice |
| `pipeline/loggers.py` | 64, 195 | PLC0415 | Lazy wandb import (structural; unavoidable) |

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Type narrowing for hasattr | `assert` / `cast` chains | `typing.Protocol` | Protocol provides static guarantee; assert is a runtime hack |
| Dict key typing for state | Manual `dict.get()` checks | `typing.TypedDict` | TypedDict gives compile-time key existence guarantee |
| Lazy import for type resolution | `__getattr__` patterns | Direct import or `TYPE_CHECKING` guard | `__getattr__` resolves as `object` to type checkers |

**Key insight:** `ty` is a structural type checker. It cannot reason about `__getattr__` dynamic dispatch. Every lazy import pattern causes the imported name to resolve as `object`, breaking unions, isinstance checks, parameter annotations, and callability. The fix is always: direct import, TYPE_CHECKING guard, or accepting that the type will be `object` and widening at the call site.

## Common Pitfalls

### Pitfall 1: Removing lazy import without verifying runtime
**What goes wrong:** Replacing `__getattr__` with direct import causes `ImportError` at module load time due to circular dependencies.
**Why it happens:** The lazy import was added to solve a real circular dependency that may still exist.
**How to avoid:** Test the import chain at runtime before committing the change. Verified: `from src.rbspaper.models.ts2vec.model import TS2Vec` works without cycle. Also verify `augmentation/__init__.py` change: the import chain `augmentation.factories -> augmentation.strategies -> ts2vec.utils` is a lazy import inside a method, not at module level.
**Warning signs:** `ImportError` or `AttributeError` on any `import src.rbspaper.*` statement.

### Pitfall 2: Removing `ty: ignore` before fixing the root cause
**What goes wrong:** A `ty: ignore` comment is removed but the underlying type error remains, causing `ty check` to fail.
**Why it happens:** Some ignores are upstream of others. For example, the `core.py:530` ignore depends on `encoding.py` having a proper `_ModelType`.
**How to avoid:** Follow the dependency order: D-01 first (encoding.py type fix), then downstream consumers (core.py, setup/model.py). Run `ty check` after each batch of changes.
**Warning signs:** New `ty` errors appearing where `ty: ignore` was removed.

### Pitfall 3: UP017 false positives
**What goes wrong:** `datetime.now()` is flagged by UP017 as "should use `datetime.now(UTC)` instead of `datetime.now(timezone.utc)`". However, `datetime.now(timezone.utc)` IS the correct modern form, and `datetime.UTC` (the UP017 suggestion) was introduced in Python 3.11. The `noqa: UP017` comments on lines 73 and 101 of `state.py` are stale -- `ruff check --select UP017` passes cleanly on those lines.
**Why it happens:** The noqa was added when the code used `datetime.utcnow()` and hasn't been cleaned up after the fix.
**How to avoid:** Run `ruff check --select UP017` on each file after removing the noqa. If it passes, the noqa was already unnecessary.
**Warning signs:** `UP017` appears in `ruff.toml` ignores or `noqa` comments but `ruff check --select UP017` passes.

### Pitfall 4: BLE001 over-narrowing
**What goes wrong:** `except Exception` is flagged by BLE001. Narrowing to a specific exception type is the fix, but W&B's `experiment.config.update()` and `run.log()` can raise various exceptions (network errors, auth errors, etc.).
**Why it happens:** The code is intentionally broad to handle any W&B failure gracefully.
**How to avoid:** Use `except (OSError, RuntimeError, AttributeError)` to cover the most common failure modes, or add `# noqa: BLE001 # why: W&B can fail with various exception types; graceful fallback is intentional` if broad catch is needed.
**Warning signs:** Tests for W&B error handling fail after narrowing the except clause.

### Pitfall 5: `LightningLogger` is deprecated
**What goes wrong:** `pl.loggers.LightningLogger` does not exist at runtime; the class was renamed to `pl.loggers.Logger`. `ty` reports `possibly-missing-submodule` because it cannot resolve the nested attribute.
**Why it happens:** Lightning renamed the base logger class. The code was written against an older version or a documentation example.
**How to avoid:** Replace all occurrences of `pl.loggers.LightningLogger` with `pl.loggers.Logger`. This is the actual runtime class name.
**Warning signs:** `AttributeError: module 'lightning.pytorch.loggers' has no attribute 'LightningLogger'` at runtime.

## Code Examples

### D-01: Direct import in `__init__.py`

```python
# Before (ts2vec/__init__.py):
__all__ = ['TS2Vec']

def __getattr__(name: str) -> object:
    """Lazy import to avoid circular dependency with augmentation factories."""
    if name == 'TS2Vec':
        from src.rbspaper.models.ts2vec.model import TS2Vec
        return TS2Vec
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

# After:
__all__ = ['TS2Vec']

from src.rbspaper.models.ts2vec.model import TS2Vec  # Verified no circular dep
```

Same pattern for `augmentation/__init__.py`:

```python
# Before:
__all__ = ['AugmentationMethod']

def __getattr__(name: str) -> object:
    """Lazy import to avoid circular dependency with strategies module."""
    if name == 'AugmentationMethod':
        from src.rbspaper.models.augmentation.strategies import AugmentationMethod
        return AugmentationMethod
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)

# After:
__all__ = ['AugmentationMethod']

from src.rbspaper.models.augmentation.strategies import AugmentationMethod
```

### D-01 continued: Widen `_ModelType` in `encoding.py`

After the direct import, `TS2Vec` is a proper class. But the `_ModelType` union is still too narrow for callers passing `pl.LightningModule`. Widen it:

```python
# Before:
from src.rbspaper.models.ts2vec import TS2Vec
from src.rbspaper.models.autotcl import AutoTCL
from src.rbspaper.models.cost import CoST

_ModelType = TS2Vec | AutoTCL | CoST

# After:
import lightning.pytorch as pl

# Widen to pl.LightningModule to accept any trained model from the pipeline.
# isinstance() checks inside dispatch to concrete types.
_ModelType = pl.LightningModule
```

### D-04: TypedDict for state serialization

```python
# Before (state.py):
def from_dict(*, data: dict[str, object]) -> PipelineState:
    return PipelineState(
        completed=data['completed'],  # ty: ignore[invalid-argument-type]
        config_hash=data['config_hash'],  # ty: ignore[invalid-argument-type]
        started_at=data['started_at'],  # ty: ignore[invalid-argument-type]
        last_updated=data['last_updated'],  # ty: ignore[invalid-argument-type]
    )

# After:
class PipelineStateDict(TypedDict):
    completed: dict[str, list[str]]
    config_hash: str
    started_at: str
    last_updated: str

def from_dict(*, data: PipelineStateDict) -> PipelineState:
    return PipelineState(
        completed=data['completed'],
        config_hash=data['config_hash'],
        started_at=data['started_at'],
        last_updated=data['last_updated'],
    )
```

### D-05: Protocol for runner model params

```python
# In runner.py (or a shared types module):
class ModelParamsWithSequenceLength(Protocol):
    def set_sequence_length(self, length: int) -> None: ...

class ModelParamsWithMaxTrainLength(Protocol):
    max_train_length: int

# Usage in runner.py:
if isinstance(model_params, ModelParamsWithSequenceLength):
    model_params.set_sequence_length(sequence_len)
if isinstance(model_params, ModelParamsWithMaxTrainLength):
    model_params.max_train_length = max(sequence_len, model_params.max_train_length)
```

Alternative: use `typing.cast` after `hasattr` check (less precise but simpler):

```python
if hasattr(model_params, 'set_sequence_length'):
    cast('ModelParamsWithSequenceLength', model_params).set_sequence_length(sequence_len)
```

### D-09: CoST `_get_slice` return type

```python
# Before (cost/model.py:350):
@override
def _get_slice(self, _sliding_padding: int, _sliding_length: int) -> None:
    return None

# After:
@override
def _get_slice(self, _sliding_padding: int, _sliding_length: int) -> slice | None:
    return None
```

### D-10: Widen augmentation abstract return type

```python
# Before (augmentation/strategies.py):
@abstractmethod
def augment(self, data: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:

# After (include the triple return of CropShiftAugmentation):
@abstractmethod
def augment(self, data: torch.Tensor, **kwargs) -> torch.Tensor | tuple[torch.Tensor, ...]:
```

Note: `CropShiftAugmentation.augment()` also takes `temporal_unit: int = 0` as an extra keyword argument. The abstract method needs to accept `**kwargs` to allow subclasses to extend the signature, or the concrete override needs to match. The cleanest fix is to widen the abstract return type AND make the abstract method accept `**kwargs`:

```python
class AugmentationMethod(ABC):
    @abstractmethod
    def augment(self, data: torch.Tensor, **kwargs) -> torch.Tensor | tuple[torch.Tensor, ...]:
```

### Deferred: AveragedModel `ty: ignore` with why comment

```python
# autotcl/model.py:278
self._averaged_encoder.update_parameters(self._encoder)  # ty: ignore[call-non-callable] # why: ty infers _averaged_encoder as Tensor|Module due to mixin attr; AveragedModel.update_parameters exists at runtime

# ts2vec/model.py:157
self._averaged_encoder.update_parameters(self._encoder)  # ty: ignore[call-non-callable] # why: ty infers _averaged_encoder as Tensor|Module due to mixin attr; AveragedModel.update_parameters exists at runtime
```

### D-03: Align attack_kwargs type

```python
# Before (_backend.py:131):
def run_torchattacks(
    *,
    ...
    attack_kwargs: dict[str, object],  # Too broad
) -> Tensor:

# After:
def run_torchattacks(
    *,
    ...
    attack_kwargs: dict[str, AttackKwargValue],  # Matches functional.py
) -> Tensor:
```

Also need to import `AttackKwargValue` or define it at the module level.

### D-08: Move per-file-ignore to inline noqa

```python
# Before (ruff.toml):
"src/rbspaper/pipeline/core.py" = ["C901", "PLR0912"]

# After (core.py:101):
def run_experiment_pipeline(  # noqa: C901, PLR0912, PLR0915
```

Remove the entire `"src/rbspaper/pipeline/core.py"` line from `ruff.toml`.

### D-07: Extract binary classification threshold constant

```python
# Before (evaluation/classification.py:20):
average_strategy = 'binary' if num_classes == 2 else 'macro'  # noqa: PLR2004

# After:
BINARY_CLASSIFICATION_THRESHOLD = 2

average_strategy = 'binary' if num_classes == BINARY_CLASSIFICATION_THRESHOLD else 'macro'
```

### BLE001 fix: Narrow except clause

```python
# Before (loggers.py:162):
except Exception:
    logger.warning(...)

# After:
except (OSError, RuntimeError):
    logger.warning(...)
```

### `pl.loggers.LightningLogger` -> `pl.loggers.Logger`

```python
# Before (loggers.py:86):
def _find_wandb_logger(*, loggers: tuple[pl.loggers.LightningLogger, ...]) -> WandbLogger | None:

# After:
def _find_wandb_logger(*, loggers: tuple[pl.loggers.Logger, ...]) -> WandbLogger | None:
```

Apply to all 3 occurrences (lines 86, 145, 172).

## Dependency Ordering

The fixes must be applied in waves to respect cascading type dependencies.

**Wave 0 -- Root cause fixes (foundational, must come first):**
1. D-01: Remove lazy `__getattr__` from `ts2vec/__init__.py` and `augmentation/__init__.py`
2. D-01 continued: Widen `_ModelType` in `encoding.py` to `pl.LightningModule`
3. Fix: Replace `pl.loggers.LightningLogger` with `pl.loggers.Logger` in `loggers.py` (3 lines)

These resolve the cascade: 7 `ty` errors in `encoding.py`, 1 in `setup/model.py`, 4 unused-ignore in `core.py`, and 3 possibly-missing-submodule warnings.

**Wave 1 -- Independent single-file fixes:**
4. D-09: Fix `CoST._get_slice()` return type (1 line)
5. D-10: Widen `AugmentationMethod.augment()` abstract return type
6. D-04: Add `TypedDict` to `state.py`
7. D-03: Align `attack_kwargs` type in `_backend.py`
8. Fix `batch.py`: Widen `attack_dataset` dataloader parameter
9. D-05: Add Protocol to `runner.py` (or import from shared types)
10. Fix `runner.py:349`: Handle `get_all_datasets` return type narrowing
11. Fix `runner.py:418`: Widen `compute_config_hash` parameter type
12. Deferred: Add `ty: ignore` with `# why:` to `autotcl/model.py:278,325` and `ts2vec/model.py:157`
13. D-13: Remove unused `ty: ignore` from `core.py:537,544,671`
14. D-13 continued: Remove `ty: ignore` from `core.py:530` (now valid after Wave 0)

**Wave 2 -- Ruff cleanup:**
15. D-07: Extract `BINARY_CLASSIFICATION_THRESHOLD` constant
16. D-07: Remove stale `noqa: UP017` from `state.py` (already passes)
17. D-07: Add `# why:` to all structural noqa comments
18. TC001: Move `CoSTAugmentationMode` into TYPE_CHECKING block
19. BLE001: Narrow `except Exception` in `loggers.py`
20. D-08: Remove `core.py` per-file-ignore from `ruff.toml`; add inline noqa

**Wave 3 -- Dead code and final verification:**
21. D-06: Remove `src/rbspaper/adapters/` package
22. D-06: Remove adapters per-file-ignore from `ruff.toml`
23. Run `ty check` -- verify only 3 documented limitations remain
24. Run `ruff check` -- verify clean
25. Run `pytest` -- verify all 143 tests pass

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `ty` (Python type checker) | `ty` 0.0.29 (Apr 2026) | Current | Fast, structural type checking. Known limitation: cannot reason about `__getattr__` dynamic dispatch or mixin attribute type resolution through MRO. |
| `datetime.utcnow()` (deprecated) | `datetime.now(timezone.utc)` | Python 3.12 | UP017 rule. Already fixed in codebase; stale noqa remains. |
| `pl.loggers.LightningLogger` | `pl.loggers.Logger` | Lightning 2.x | Class was renamed. Old name no longer exists at runtime. |
| `__getattr__` lazy import | Direct import or TYPE_CHECKING | Ongoing | `__getattr__` resolves as `object` to static checkers. Direct imports preferred when no cycle exists. |

**Deprecated/outdated:**
- `LightningLogger`: Renamed to `Logger` in Lightning 2.x. All 3 occurrences in `loggers.py` need updating.
- `datetime.utcnow()`: Deprecated since Python 3.12. Codebase already uses `datetime.now(timezone.utc)`; noqa comments are stale.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `ty` 0.0.29 infers `_averaged_encoder` as `Tensor | Module` due to a limitation in resolving mixin attribute types through MRO with `AveragedModel` | Deferred fixes | Fix may not be as simple as adding ignore; might need mixin restructuring |
| A2 | `ruff check --select UP017` passes cleanly on `state.py` lines 73 and 101, meaning the `noqa: UP017` comments are stale | D-07 noqa cleanup | Low -- verified by running the check |
| A3 | No code outside the project imports from `augmentation/__init__.py` at the package level (only direct submodule imports exist) | D-01 augmentation lazy import | Low -- verified by grep; removing the `__getattr__` is safe |
| A4 | `AveragedModel` is always used (not a raw `Tensor`) when `_averaged_encoder` is accessed in training steps | Deferred fixes | Medium -- if a code path sets `_averaged_encoder` to a Tensor, the ignore masks a real bug |
| A5 | The `BLE001` narrow `except (OSError, RuntimeError)` covers all W&B failure modes | Code Examples | Low -- W&B failures are network/auth related; these types are sufficient |

## Open Questions (RESOLVED)

1. **Should the `_averaged_encoder` issue be treated as a `ty` bug and reported?**
   - What we know: `ty` infers `_averaged_encoder` as `Tensor | Module` despite the mixin declaring it as `nn.Module` and `AveragedModel` being a subclass of `nn.Module`. The `Tensor` variant appears to come from stub confusion.
   - What's unclear: Whether this is a known `ty` limitation with `torch.optim.swa_utils.AveragedModel` specifically, or a general MRO resolution bug.
   - Recommendation: Document as a genuine limitation with `ty: ignore` and `# why:` comment. File a `ty` issue if time permits, but do not block the phase.
   - **RESOLVED:** Document as genuine ty limitation with `ty: ignore[call-non-callable]` + `# why:` comment. Plan 08-08 Task 3 applies the suppression.

2. **Should `AttackKwargValue` be imported from `functional.py` into `_backend.py`, or defined independently?**
   - What we know: `functional.py` defines `AttackKwargValue = float | int | str | bool | Tensor | None`. `_backend.py` uses `dict[str, object]`.
   - What's unclear: Whether `_backend.py` should depend on `functional.py` (creates a reverse import direction) or whether the type alias should be extracted to a shared module (e.g., `attacks/types.py`).
   - Recommendation: Import from `functional.py` using a `TYPE_CHECKING` guard to avoid runtime circular imports. This is minimal and clean.
   - **RESOLVED:** Import using `TYPE_CHECKING` guard. Plan 08-05 Task 1 uses this approach.

3. **Where should the `runner.py` Protocol be defined?**
   - What we know: CONTEXT.md says "new file vs. existing module" is at Claude's discretion.
   - What's unclear: Whether a dedicated `types.py` in `runners/py/` or reusing `pipeline/config.py` is preferable.
   - Recommendation: Define inline in `runner.py` since the Protocol is only used there. A dedicated types file adds indirection for a single-use abstraction.
   - **RESOLVED:** Define inline in `runner.py`. Plan 08-07 Task 1 follows this approach.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | All fixes | Verified | 3.12 | — |
| uv | Test execution | Verified | Current | — |
| ty | Type checking | Verified | 0.0.29 | — |
| ruff | Linting | Verified | Current | — |
| pytest | Test validation | Verified | 143 tests | — |
| PyTorch | Import verification | Verified | Current | — |
| Lightning | Import verification | Verified | Current | — |

**Missing dependencies:** None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | None detected (uses pytest defaults) |
| Quick run command | `uv run pytest -x -q` |
| Full suite command | `uv run pytest -x` |
| Test count | 143 tests |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| D-01 | Direct imports work without circular dep | Import smoke | `uv run python -c "from src.rbspaper.models.encoding import encode_data"` | Manual |
| D-02 | `DataConfig` accepts `pl.LightningDataModule` | Unit | `uv run pytest test/test_pipeline_state.py -x` | Existing |
| D-04 | State serialization round-trips | Unit | `uv run pytest test/test_pipeline_state.py -x` | Existing |
| D-05 | Runner type-checks cleanly | Static | `uv run ty check runners/py/runner.py` | Manual |
| General | All existing tests pass | Integration | `uv run pytest -x` | Existing |

### Sampling Rate
- **Per task commit:** `uv run ty check src/rbspaper runners/py/runner.py --quiet` + `uv run ruff check src/rbspaper runners/py/runner.py`
- **Per wave merge:** `uv run pytest -x` (full suite)
- **Phase gate:** `uv run ty check src/rbspaper runners/py/runner.py` clean (only documented limitations), `uv run ruff check src/rbspaper runners/py/runner.py` clean, all 143 tests pass

### Wave 0 Gaps
- None. Existing test infrastructure covers all phase requirements. The phase is static analysis cleanup, not behavioral change.

## Security Domain

Not applicable. This phase involves type contract fixes and linting cleanup. No authentication, session management, access control, or cryptographic operations are touched.

## Sources

### Primary (HIGH confidence)
- `ty` 0.0.29 output: All 18 diagnostics verified by running `uv run ty check src/rbspaper runners/py/runner.py`
- `ruff` output: All 5 errors verified by running `uv run ruff check src/rbspaper runners/py/runner.py`
- Import chain verification: Runtime test confirmed `from src.rbspaper.models.ts2vec.model import TS2Vec` works without circular dependency
- `pl.loggers.LightningLogger` confirmed absent: `hasattr(pl.loggers, 'LightningLogger')` returns `False`; `hasattr(pl.loggers, 'Logger')` returns `True`
- `AveragedModel` confirmed subclass of `nn.Module`: `issubclass(AveragedModel, nn.Module)` returns `True`

### Secondary (MEDIUM confidence)
- `ty` MRO resolution limitation for `_averaged_encoder`: Inferred from error message "Attempted to call union type `Tensor | Module`" despite mixin declaring `nn.Module`. Root cause not confirmed in `ty` issue tracker.

### Tertiary (LOW confidence)
- UP017 noqa staleness on `state.py`: Verified by running `uv run ruff check --select UP017` which passes cleanly, suggesting the code was already updated and noqa is residual.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all tools verified present and working
- Architecture: HIGH -- cascade dependency chain traced through imports and runtime verification
- Pitfalls: HIGH -- each pitfall verified against actual codebase state (import chains, ruff rules, ty diagnostics)

**Research date:** 2026-05-08
**Valid until:** 2026-06-08 (stable tools; ty 0.0.29 as of April 2026)
