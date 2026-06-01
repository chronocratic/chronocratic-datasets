# Phase 07: DDP Compliance + `_full_data` Split - Context

**Gathered:** 2026-05-29
**Status:** Ready for planning

## Phase Boundary

Make the package safe under Lightning Distributed Data Parallel (multi-GPU) and fix `_full_data` type drift. Two coupled refactors: (1) DDP-safe cache layer so `prepare_data()` is I/O-only and `setup()` reads per-rank state from disk, (2) split `_full_data` into typed, immutable attrs to eliminate `isinstance` branches.

## Implementation Decisions

### `_full_data` Type Split (D-01)
- **Option B — both ndarray + cached index.** `_full_data_raw: np.ndarray | None` (set in `setup()`, never mutated), `_time_index: pd.DatetimeIndex | None` (cached right after read), `_full_data_scaled: np.ndarray | None` (rebuilt from raw each `setup()` call). Fast numeric ops. No `isinstance` branches downstream. Idempotency falls out: raw is immutable, scaled is rebuilt.

### Cache Directory (D-02)
- Centralized (`~/.cache/tsdatasets/<dataset_name>/<cache_key>`). `cache_dir: Path | None = None` param on `BaseTimeSeriesDataModule.__init__`. `None` → default. Custom path → HPC users with faster local storage.

### Cache Key (D-03)
- Hybrid: `<8-char-sha256>_<dataset>_<key-params>.cache`. Underscore-separated. E.g., `a3f8e1c2_ETTm1_seq128_univ_minmax.cache`. Hash for uniqueness, readable suffix for debugging.

### Cache Format (D-04)
- Data: `<cache_key>.npz` via `numpy.savez_compressed`. Metadata: sibling `metadata.json` with schema version (`version: 1`). Scale params: `<cache_key>_data_scaler.pt`, `<cache_key>_ts_scaler.pt` via `torch.save(..., pickle_protocol=5)`. Atomic write: tmpfile + `os.replace`.

### `prepare_data_per_node` Default (D-05)
- `True` (Lightning default). Each node writes own cache. No cross-node dependency. Document `prepare_data_per_node = False` as HPC/shared-FS optimization — class attr set before `prepare_data()` runs.

### Scaler Persistence (D-06)
- `torch.save` / `torch.load(weights_only=False)`. Sklearn scalers are trivially pickle-able. Already in project. Zero extra import.

### `prepare_dimensions()` — Replace Interim (D-07)
- Replace in place. Read `metadata.json` only. No in-memory fallback. If metadata missing or version mismatch → raise clear error. Schema version (`version: 1`) handles future cache invalidation.

### `metadata.json` Schema (D-08)
```json
{
  "version": 1,
  "dataset_name": "ETTm1",
  "n_features": 7,
  "seq_len": 128,
  "splits": {"train": [0, 8640], "valid": [8640, 11520], "test": [11520, 14400]},
  "has_datetime_index": true,
  "data_scaling_method": "MINMAX",
  "data_scaling_range": [0, 1]
}
```

### Claude's Discretion
- How to structure the new `utils/cache.py` module (key derivation, atomic write, cache-dir resolution).
- Exact cache-key param encoding — minimal set that distinguishes layouts vs. full constructor dump.
- How to handle classification modules — they load data in `prepare_data()` too; same cache pattern applies for ARFF data.

## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Planning Documents
- `.planning/ROADMAP.md` §Phase 7: DDP Compliance — goal, success criteria
- `.planning/REQUIREMENTS.md` — MOD-01 (module loads from file paths), MOD-05 (read-only properties)
- `.planning/PROJECT.md` — v1 scope, package structure, Lightning priority

### Phase Context Files
- `.planning/phases/06-lightning-lifecycle/06-CONTEXT.md` — Phase 6 decisions D1-D4 (setup signature, scaler cache types, finalize hook, prepare_dimensions interim). Phase 7 replaces interim with cache layer.
- `.planning/phases/08-forecasting-mode-wiring/08-CONTEXT.md` — Phase 8 depends on phase 7 completion; both touch forecasting dataloader sites.

### Source Code — Base Modules (all touched by this phase)
- `src/tscollection/datasets/modules/_base/base.py` — `BaseTimeSeriesDataModule`: add `cache_dir` param, `prepare_data_per_node` attr, rewrite `prepare_data()` flow
- `src/tscollection/datasets/modules/_base/forecasting.py` — `BaseForecastingTimeSeriesDataModule`: split `_full_data` into `_full_data_raw`, `_time_index`, `_full_data_scaled`; rewrite `setup()` to read from cache
- `src/tscollection/datasets/modules/_base/classification.py` — `BaseClassificationTimeSeriesDataModule`: cache data in `prepare_data()`, read in `setup()`

### Source Code — Concrete Modules
- `src/tscollection/datasets/modules/ett.py` — `_do_prepare_data()` writes cache; `_transform_data()`, `_set_data_slices()` adapted to new attr names
- `src/tscollection/datasets/modules/weather.py` — same pattern
- `src/tscollection/datasets/modules/electricity.py` — same pattern
- `src/tscollection/datasets/modules/ucr.py` — classification cache pattern
- `src/tscollection/datasets/modules/uea.py` — classification cache pattern

### New Files
- `src/tscollection/datasets/utils/cache.py` — `build_cache_key()`, `resolve_cache_dir()`, `atomic_save_npz()`, `atomic_save_metadata()`, `load_metadata()`
- `src/tscollection/datasets/utils/__init__.py` — export cache helpers

### Prior Phase DDP Analysis
- `.planning/phases/07-ddp-compliance/lightning-ddp-compliance.md` — original 8-step DDP compliance plan (cache layout, key schema, metadata versioning, scaler persistence, DDP smoke test). Source of this phase's scope.

## Existing Code Insights

### Reusable Assets
- `BaseTimeSeriesDataModule` idempotency sentinel (`_prepare_data_called`) and stage tracking (`_setup_completed_stages`) — phase 6 foundation
- `create_data_scaler()` from `utils/scaling.py` — factory for sklearn scalers
- `TIME_FEATURE_COUNT` from `utils/features.py` — exported constant
- Existing `_finalize_prepare_data()` hook pattern — cache-read dispatch fits here

### Established Patterns
- Template method: `prepare_data()` wrapper → `_do_prepare_data()` (abstract) → `_finalize_prepare_data()` (hook)
- All concrete modules use `_full_data` as `pd.DataFrame | np.ndarray | None` class attr — needs renaming
- Forecasting `_transform_data()` does `isinstance(self._full_data, pd.DataFrame)` branch — eliminated with Option B
- Dataloaders use `torch.from_numpy(self._train_data_samples).to(torch.float32)` — no change needed

### Integration Points
- `prepare_data()` → now writes to disk (cache files + metadata.json)
- `setup(stage)` → reads from disk, builds per-rank state
- `_compute_dimensions()` → reads `metadata.json` instead of `_full_data`
- All 5 concrete modules: `_do_prepare_data()` body unchanged (still reads CSV/ARFF), but data is cached instead of stored on `self`

## Deferred Ideas

- **Cache invalidation UI** — `clear_cache()` method or CLI command. Future convenience.
- **Shared-FS coordination** — `prepare_data_per_node = False` with rank-0 barrier. Document, don't implement.
- **Cache statistics** — hit/miss logging. Future observability.
- **Backward compat** — reading old cache format when version changes. Handle via schema version check.

---

*Phase: 07-DDP Compliance*
*Context gathered: 2026-05-29*
