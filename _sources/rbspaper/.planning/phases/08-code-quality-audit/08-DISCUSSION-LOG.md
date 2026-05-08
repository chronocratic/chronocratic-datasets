# Phase 08: Code Quality Audit - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-08
**Phase:** 08-code-quality-audit
**Areas discussed:** State serialization, TS2Vec lazy import, unused ty ignores, runner model params, dead adapters, noqa scope, CoST mixin, augmentation signature, dataset __getitem__, isinstance narrowing, attack_kwargs, DataConfig type, genuine ty limitations

---

## State Serialization (`state.py` dict access, 4 ignores)

| Option | Description | Selected |
|--------|-------------|----------|
| TypedDict for state | Define TypedDict matching JSON keys, precise types | ✓ |
| Cast (trust me bro) | Cast dict values at load time, minimal change | |

**User's choice:** TypedDict for state
**Notes:** TypedDict provides explicit key types and removes ignores cleanly.

## TS2Vec Lazy Import (`ts2vec/__init__.py`, 9 errors)

| Option | Description | Selected |
|--------|-------------|----------|
| Remove, use direct import | Replace __getattr__, fixes 9 ty errors | ✓ |
| Keep as-is | Leave lazy import and 9 ignores | |

**User's choice:** Remove, use direct import
**Notes:** User asked for deeper explanation. Circular import verified gone at runtime. Lazy import is leftover workaround. Also defer restructuring augmentation→encoders dependency to future phase.

## Unused Ty Ignores (`pipeline/core.py`, 3 warnings)

| Option | Description | Selected |
|--------|-------------|----------|
| Remove all 3 | Clean up dead noise | ✓ |
| Leave them | Harmless, removed naturally later | |

**User's choice:** Remove all 3

## Runner Dynamic Model Params (`runner.py`, 4 ignores)

| Option | Description | Selected |
|--------|-------------|----------|
| Protocol for model params | Define Protocol with expected attrs, proper types | ✓ |
| Runtime assertions + cast | isinstance checks, minimal architecture change | |
| Keep as ty limitation | Document hasattr narrowing gap | |

**User's choice:** Protocol for model params
**Notes:** More refactoring but cleanest long-term solution.

## Dead Adapters Package (`src/rbspaper/adapters/`)

| Option | Description | Selected |
|--------|-------------|----------|
| Remove entire directory | Zero risk, nothing imports it | ✓ |
| Keep for now | Might be needed later | |

**User's choice:** Remove entire directory

## Noqa Cleanup Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Fix all fixable noqa | PLR2004, F401, UP017, S311, N812 — review every one | ✓ |
| Fix only clear wins | PLR2004, F401. Leave rest documented | |
| Skip noqa cleanup | Focus only on ty ignores | |

**User's choice:** Fix all fixable noqa
**Notes:** Also remove ruff.toml per-file-ignores for core.py, add scoped noqa instead.

## CoST Mixin Design (`_get_slice`, `_evaluate_with_feature_concatenation`)

| Option | Description | Selected |
|--------|-------------|----------|
| Defer to new phase | Fix annotation now, proper refactor later | ✓ |
| Include in Phase 8 | Expand scope to mixin refactor | |
| Just fix ty errors | Minimal, document design debt | |

**User's choice:** Defer to new phase (selected preview)
**Notes:** User recognized architectural issue — CoST forced into mixin contract it doesn't fit. Original AutoTSAugment used string dispatch. Strategy refactor was correct direction but didn't separate encoding paths. Split into PoolingEncoderMixin + ConcatEncoderMixin in future phase.

## Augmentation Signature (`CropShiftAugmentation.augment`)

| Option | Description | Selected |
|--------|-------------|----------|
| Widen abstract return type | Include 3-tuple in abstract return | ✓ |
| Return tuple of tensors only | Pack crop_length separately, refactor | |
| Keep as ty limitation | Tuple covariance gap | |

**User's choice:** Widen abstract return type

## DataConfig Type (`pipeline/config.py`)

| Option | Description | Selected |
|--------|-------------|----------|
| Widen to LightningDataModule | Remove 5 ignores, matches broader usage | ✓ |
| Keep as BaseTimeSeriesDataModule | More precise type | |

**User's choice:** Widen to LightningDataModule

## Attack Kwargs Type (`functional.py` vs `_backend.py`)

| Option | Description | Selected |
|--------|-------------|----------|
| Align both to AttackKwargValue | Tighter contract, removes 1 ignore | ✓ |
| Align both to object | Simpler but less precise | |

**User's choice:** Align both to AttackKwargValue

## Dataset `__getitem__` Override (`abstract.py`)

| Option | Description | Selected |
|--------|-------------|----------|
| Fix signature to match Dataset | Change `item` to `index: int`, narrow return from Any | ✓ |
| Use cast on override | Satisfies ty without behavior change | |
| Keep as documented ignore | Known ty limitation | |

**User's choice:** Fix signature to match Dataset (just `index: int` for now, defer narrowing return type)
**Notes:** User wanted trade-off analysis first. After review, agreed minimal fix (`item` → `index: int`) is enough for Phase 8. Narrowing `Any` return type deferred to future.

## Strategies `isinstance` Narrowing (`strategies.py`)

| Option | Description | Selected |
|--------|-------------|----------|
| Extract to helper function | Isolates narrowing issue, cleaner | ✓ |
| Per-subclass override | Each subclass knows exact data type | |
| Cast after isinstance | Loses type safety | |
| Keep as documented ignore | Known ty gap | |

**User's choice:** Extract to helper function
**Notes:** User initially wanted to fix differently from "genuine limitation". After trade-off analysis showing if/else doesn't help, agreed on helper function + 1 documented ignore.

## CoST `_get_slice()` Return Type

| Option | Description | Selected |
|--------|-------------|----------|
| Fix annotation to `slice | None` | One-line change, correct return | ✓ |

**User's choice:** Fix annotation (implicit, discussed as part of mixin deferral)

## Genuine Ty Limitations

**Updated count:** 2 (down from 3 in ROADMAP)
1. `isinstance` narrowing on `np.ndarray | list[np.ndarray]` unions (`strategies.py`)
2. `__getitem__` generic override — FIXED, removed from list

---

## Claude's Discretion

- Exact TypedDict structure for `load_pipeline_state`
- Protocol naming and placement for model_params
- Helper function signature for isinstance narrowing
- Specific noqa `# why:` comment text for structural ignores
- How to handle the `_compute_sliding_representations` dead params in CoST mixin deferred refactor

## Deferred Ideas

- **Mixin architectural refactor** — Split `EncodingFunctionalityMixin` into PoolingEncoderMixin + ConcatEncoderMixin. CoST shouldn't inherit methods with dead parameters. Future phase.
- **Narrow `__getitem__` return type** — From `Any` to concrete sample types per mode. Future improvement.
- **Augmentation → encoders dependency** — Extract shared types to `models/types.py` to prevent future cycles.
- **`autotcl/model.py` `call-non-callable` errors** — `_averaged_encoder` typed as `Tensor | Module`. Fix attribute type on mixin.
