# Phase 08: Code Quality Audit - Context

**Gathered:** 2026-05-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove all fixable `ty: ignore` suppressions, `noqa` directives, and `ruff.toml` per-file-ignores from source code (`src/`, `runners/`). Fix the underlying type contracts and code issues rather than suppressing. Document only genuine `ty` limitations with `# why:` comments.

**In scope:**
- `src/rbspaper/**/` — all Python source files
- `runners/py/runner.py` — CLI runner with dynamic model_params
- `ruff.toml` — per-file-ignores cleanup
- Dead code removal (`adapters/` package)

**Out of scope:**
- Test files (`test/`) — broader per-file-ignores remain
- New feature additions or architectural refactors (mixin redesign deferred)
- `_sources/autotsaugment/` — reference code, not in scope

</domain>

<decisions>
## Implementation Decisions

### TS2Vec Lazy Import (`ts2vec/__init__.py`)
- **D-01:** Remove `__getattr__` lazy import and use direct import from `model.py`. The circular dependency that motivated it no longer exists — verified at runtime. Fixes 9 `ty` errors in `encoding.py` (unsupported-operator, invalid-type-form, invalid-argument-type, invalid-method-override).
- **Deferred:** Restructure `augmentation/` → `encoders/` dependency to prevent future cycles. Document risk in CONTEXT.md for future phases.

### Type Contract Widening
- **D-02:** Widen `DataConfig.data_module` from `BaseTimeSeriesDataModule` to `pl.LightningDataModule`. Removes 5 ty ignores.
- **D-03:** Align `attack_kwargs` value type from `dict[str, object]` to `dict[str, AttackKwargValue]` in `_backend.py`. Removes 1 ty ignore in `functional.py`.

### State Serialization
- **D-04:** Use TypedDict for pipeline state serialization in `state.py`. Defines exact JSON keys and types, removing 4 `ty: ignore` comments on dict access.

### Runner Dynamic Model Params
- **D-05:** Define Protocol for model parameters in `runner.py`. Covers `set_sequence_length` and `max_train_length` as optional attributes. Removes 4 `ty: ignore` comments on `hasattr`-guarded access.

### Dead Code Removal
- **D-06:** Remove entire `src/rbspaper/adapters/` package (attack_adapter, model_adapter, task_adapter). Zero external imports since creation. Clean `ruff.toml` per-file-ignore entry.

### Noqa Cleanup
- **D-07:** Fix all fixable noqa in source code: extract `BINARY_CLASSIFICATION_THRESHOLD = 2` constant (PLR2004), add `# why:` comments to structural ignores (N812, PLC0415, SLF001, S311, F401), fix UP017 datetime calls.
- **D-08:** Remove `ruff.toml` per-file-ignores for `core.py` (C901, PLR0912, PLR0915). Add scoped `noqa` on specific complex functions instead.

### CoST Mixin Override
- **D-09:** Fix `CoST._get_slice()` return annotation from `-> None` to `-> slice | None`. Trivial one-line change.
- **Deferred:** Proper mixin refactor to separate pooling-based encoding (TS2Vec, AutoTCL) from concatenation-based encoding (CoST). Dead parameters (`slicing`, `encoding_window`, `mask`) in `_evaluate_with_feature_concatenation` should be removed. New phase.

### Augmentation Signature
- **D-10:** Widen `AugmentationMethod.augment()` abstract return type to include `tuple[torch.Tensor, torch.Tensor, int]` for `CropShiftAugmentation`. Fixes Liskov violation.

### Dataset `__getitem__` Override
- **D-11:** Fix `TimeSeriesDataset.__getitem__(self, item)` to `__getitem__(self, index: int)` to match `torch.utils.data.Dataset`. Removes 1 `ty: ignore`.
- **Deferred:** Narrow return type from `Any` to concrete sample types per mode (WITHOUT_LABELS, WITH_LABELS, FORECASTING). Future improvement.

### Strategies `isinstance` Narrowing
- **D-12:** Extract `isinstance` check in `strategies.py` to a helper function. Isolates the narrowing issue. Keeps 1 documented `ty: ignore` as genuine limitation.

### Genuine Ty Limitations (updated)
- **4 documented limitations** (confirmed by research):
  1. `isinstance` narrowing on `np.ndarray | list[np.ndarray]` unions (`strategies.py`)
  2. `call-non-callable` on `_averaged_encoder.update_parameters()` — `ty` infers `Tensor | Module` due to MRO with `AveragedModel` (`autotcl/model.py`, 2 occurrences)
  3. `call-non-callable` on `_averaged_encoder.update_parameters()` — same MRO issue (`ts2vec/model.py`, 1 occurrence)
  4. `__getitem__` generic override is now fixed — removed from limitation list (D-11 committed)
- `hasattr` narrowing (`runner.py`) — covered by D-05 (Protocol approach).

### Cleanup
- **D-13:** Remove 3 unused `ty: ignore` in `pipeline/core.py` (lines 537, 544, 671). Ty reports them as `unused-ignore-comment`.

### Claude's Discretion
- Exact TypedDict structure for `load_pipeline_state`
- Protocol naming and placement (new file vs. existing module)
- Helper function signature for the isinstance narrowing
- Specific noqa `# why:` comment text for each structural ignore

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Files with Ty Ignores
- `src/rbspaper/models/encoding.py` — 9 ty errors: unsupported-operator, invalid-type-form, isinstance, invalid-argument-type
- `src/rbspaper/pipeline/core.py` — 4 unused ty: ignore[invalid-argument-type] (line 530 still needed until encoding.py fixed), 3 unused (537, 544, 671)
- `src/rbspaper/pipeline/state.py` — 4 ty: ignore[invalid-argument-type] on dict access, 2 noqa: UP017
- `src/rbspaper/models/augmentation/strategies.py` — 1 ty: ignore[invalid-method-override], 1 noqa: PLC0415, 2 noqa: S311
- `src/rbspaper/data/datasets/strategies.py` — 1 ty: ignore[invalid-assignment] (isinstance narrowing)
- `src/rbspaper/data/datasets/abstract.py` — 1 ty: ignore[invalid-method-override] (now fixed, D-11)
- `src/rbspaper/attacks/functional.py` — 1 ty: ignore[invalid-argument-type] (attack_kwargs type)
- `src/rbspaper/attacks/batch.py` — 1 ty: ignore[invalid-argument-type] (dataloader)
- `runners/py/runner.py` — 4 ty: ignore (call-non-callable, invalid-assignment, invalid-argument-type, invalid-assignment)
- `src/rbspaper/models/cost/model.py` — 1 ty: ignore[invalid-method-override] (_get_slice)
- `src/rbspaper/models/ts2vec/__init__.py` — __getattr__ lazy import workaround (removing per D-01)
- `src/rbspaper/pipeline/loggers.py` — 3 possibly-missing-submodule warnings (pl.loggers)

### Type Definitions & Config
- `src/rbspaper/pipeline/config.py` — `DataConfig.data_module` type (line 271)
- `src/rbspaper/attacks/functional.py` — `AttackKwargValue` type alias
- `src/rbspaper/attacks/_backend.py` — `attack_kwargs: dict[str, object]` (line 131)
- `ruff.toml` — per-file-ignores for core.py (C901, PLR0912) and adapters (ANN401, TC001)

### Dead Code
- `src/rbspaper/adapters/__init__.py` — exports AttackAdapter, ModelAdapter, TaskAdapter
- `src/rbspaper/adapters/attack_adapter.py` — imports from model_adapter.py
- `src/rbspaper/adapters/model_adapter.py` — unused adapter
- `src/rbspaper/adapters/task_adapter.py` — unused adapter

### Noqa Directives to Review
- `src/rbspaper/pipeline/core.py:50` — noqa: F401 (intentional barrel re-export)
- `src/rbspaper/pipeline/core.py:101` — noqa: PLR0915 (long function)
- `src/rbspaper/models/utils.py:19` — noqa: N812 (torch.nn.functional convention)
- `src/rbspaper/models/losses.py:22` — noqa: N812
- `src/rbspaper/models/autotcl/model.py:7` — noqa: N812
- `src/rbspaper/models/abstract/encoding_functionality_mixin.py:10` — noqa: N812
- `src/rbspaper/models/layers/convolutions/dilated.py:9` — noqa: N812
- `src/rbspaper/models/layers/convolutions/same_pad.py:9` — noqa: N812
- `src/rbspaper/models/cost/model.py:12` — noqa: N812
- `src/rbspaper/evaluation/classification.py:20` — noqa: PLR2004 (magic number == 2)
- `src/rbspaper/attacks/_backend.py:77,103` — noqa: PLC0415 (lazy imports)
- `src/rbspaper/models/layers/general.py:88` — noqa: SLF001 (PyTorch internals)
- `src/rbspaper/pipeline/state.py:73,101` — noqa: UP017 (datetime)
- `src/rbspaper/pipeline/loggers.py:64,195` — noqa: PLC0415 (lazy wandb import)

### Reference Code
- `_sources/autotsaugment/src/models/evaluation/encoding.py` — original encoding.py (used `model: Any`)
- `_sources/autotsaugment/src/models/abstract/encoding_functionality_mixin.py` — original string-dispatch mixin

</canonical_refs>

<code_context>
## Existing Code Insights

### Established Patterns
- Frozen dataclass configs (`frozen=True`) for all pipeline config
- Keyword-only args for all function calls (strict project convention)
- `TYPE_CHECKING` guard for circular import prevention
- `from __future__ import annotations` where forward refs needed
- Google-style docstrings enforced by ruff

### Integration Points
- `encoding.py` imports TS2Vec, AutoTCL, CoST from model packages — type alias fix propagates
- `pipeline/core.py` uses `encode_data()` — widening model type there is consistent
- `runner.py` uses `build_model_from_parameters()` — Protocol applies to the params chain

### Current Ty Diagnostic Count
- 19 total diagnostics (2 errors + 4 warnings + 13 ignores)
- Goal: only 2 documented limitation ignores remain

</code_context>

<specifics>
## Specific Ideas

- User verified `ts2vec/__init__.py` lazy import is no longer needed — direct import works
- User wants Protocol for model_params instead of cast/assert (cleaner, more precise)
- User wants TypedDict for state serialization (explicit key types)
- User wants helper function for isinstance narrowing in strategies.py (isolates issue)
- `__getitem__` parameter fixed from `item` to `index: int` — ty: ignore removed successfully
- User wants all noqa reviewed and fixed where possible
- Only 4 genuine ty limitations: isinstance narrowing (strategies.py) + 3 AveragedModel call-non-callable

</specifics>

<deferred>
## Deferred Ideas

- **Mixin architectural refactor** — Split `EncodingFunctionalityMixin` into separate pooling-based and concatenation-based contracts. CoST shouldn't inherit methods with dead parameters. Requires changing `_evaluate_with_feature_concatenation` signature and how `_get_slice` dispatches. Future phase.
- **Narrow `__getitem__` return type** — Current fix only aligns the parameter. Narrowing `Any` to concrete sample types per mode is a larger refactor involving transform chain typing. Future improvement.
- **Augmentation → encoders dependency** — `CropShiftAugmentation` imports from `ts2vec/utils.py`. While the cycle is gone, the augmentation layer depending on model-internal utils is fragile. Extract shared types to `models/types.py` to prevent future cycles.
- **`autotcl/model.py` `call-non-callable` errors** — Lines 278, 325: `self._averaged_encoder.update_parameters()` fails because `_averaged_encoder` is typed as `Tensor | Module`. Requires fixing the `_averaged_encoder` attribute type on the mixin.

---

*Phase: 08-Code Quality Audit*
*Context gathered: 2026-05-08*

</deferred>
