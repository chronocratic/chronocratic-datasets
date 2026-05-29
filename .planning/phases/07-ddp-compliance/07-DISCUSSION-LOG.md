# Phase 07: DDP Compliance - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-29
**Phase:** 07-ddp-compliance
**Areas discussed:** _full_data type split, cache directory, cache key, prepare_data_per_node, scaler persistence, prepare_dimensions replacement

---

## `_full_data` Type Split Design

| Option | Description | Selected |
|--------|-------------|----------|
| Option A — both DataFrame | Wrap scaler output back into DataFrame. Index preserved. Slower concat. | |
| Option B — both ndarray + cached index | `_full_data_raw`, `_time_index`, `_full_data_scaled`. Fast ops. No isinstance branches. | ✓ |
| Option C — type guard | Runtime assertion at setup() entry. Cheap, doesn't fix root cause. | |

**User's choice:** Option B
**Notes:** Idempotency falls out for free — raw immutable, scaled rebuilt each setup().

## Cache Directory Location

| Option | Description | Selected |
|--------|-------------|----------|
| Per-dataset dir | `<dataset_dir>/.tsdatasets_cache/`. Co-located. Pollutes data dirs. | |
| Centralized | `~/.cache/tsdatasets/`. Clean. Customizable via `cache_dir` param. | ✓ |

**User's choice:** Centralized, with `cache_dir: Path | None = None` override for HPC users
**Notes:** Some HPC setups have faster local scratch/work folders. Customizable default.

## Cache Key Derivation

| Option | Description | Selected |
|--------|-------------|----------|
| Full param hash | SHA256 of param JSON. Short, opaque. | |
| Human-readable | `<dataset>_<seqLen>_<mode>_<scaling>`. Debuggable, fragile. | |
| Hybrid | 8-char hash + readable suffix. Underscore-separated. | ✓ |

**User's choice:** Hybrid with `_` separator (not `.`) — glob-safe, grep-friendly.
**Notes:** Example: `a3f8e1c2_ETTm1_seq128_univ_minmax.cache`

## `prepare_data_per_node` Default

| Option | Description | Selected |
|--------|-------------|----------|
| `True` (Lightning default) | Each node writes own cache. No cross-node dependency. | ✓ |
| `False` (override) | Global rank-0 only. Efficient on shared FS. Breaks on local FS. | |

**User's choice:** `True` with `False` documented as HPC optimization.

## Scaler Persistence Format

| Option | Description | Selected |
|--------|-------------|----------|
| `joblib` | sklearn standard. Extra import (transitive dep). | |
| `numpy` raw arrays | Portable. Fragile across sklearn versions. | |
| `torch.save` | Native to project. Pickle backend. `weights_only=False`. | ✓ |

**User's choice:** `torch.save` / `torch.load`
**Notes:** Already a hard dependency (PyTorch/Lightning package). sklearn scalers pickle trivially. No extra import.

## `prepare_dimensions()` — Replace or Fallback

| Option | Description | Selected |
|--------|-------------|----------|
| Replace in place | `metadata.json` only. No dual-path. | ✓ |
| Keep fallback | Try metadata, fall back to in-memory. DDP-unsafe path. | |

**User's choice:** Replace in place. Schema version in `metadata.json` for future cache invalidation.
**Notes:** New package — no stale-cache users. Missing/wrong-version metadata → clear error.

---

## Claude's Discretion
- `utils/cache.py` module structure
- Cache-key param encoding granularity
- Classification cache pattern details

## Deferred Ideas
- Cache clear utility
- Shared-FS rank-0 coordination (document only)
- Cache hit/miss logging
