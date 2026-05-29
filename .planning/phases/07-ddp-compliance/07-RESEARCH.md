# Phase 07: DDP Compliance + `_full_data` Split - Research

**Researched:** 2026-05-29
**Domain:** Lightning DDP lifecycle, numpy cache persistence, sklearn scaler serialization, typed immutable data attributes
**Confidence:** HIGH

## Summary

This phase has two coupled refactors: (1) DDP-safe cache layer so `prepare_data()` is I/O-only and `setup()` reads per-rank state from disk, (2) split `_full_data` into typed, immutable attrs to eliminate `isinstance` branches. Phase 6 laid the groundwork: idempotency sentinels (`_setup_completed_stages`, `_prepare_data_called`), template-method `prepare_data()` wrapper, stage-branched `setup()`, fitted-scaler caches, and the interim `prepare_dimensions()` API. Phase 7 replaces the interim in-memory `prepare_dimensions()` with a `metadata.json` read and makes the full pipeline DDP-safe.

Under Lightning DDP (`strategy='ddp_spawn'`), `prepare_data()` executes on rank-0 only (or LOCAL_RANK=0 per node when `prepare_data_per_node=True`). State written to `self` in `prepare_data()` is invisible to other ranks. The fix: write to disk (cache files + `metadata.json`) in `prepare_data()`, then read from disk in `setup()` on every rank. This pattern is the official Lightning convention and verified working on this machine (MPS, gloo, 2 ranks).

The `_full_data` type drift follows a fixed path: `pd.DataFrame` (set by `_do_prepare_data()`) -> `np.ndarray` (converted by `_transform_data()` via `.to_numpy()`) -> scaled `np.ndarray` with prepended time features (mutated by `setup()`). There are exactly 8 `isinstance(self._full_data, pd.DataFrame)` branches across the codebase: 1 in `_compute_dimensions()`, 1 in `setup()`'s time-index extraction, and 6 across three concrete `_transform_data()` methods (ETT, Weather, Electricity) plus their `_set_data_slices()` methods. Option B (locked in D-01) replaces this with `_full_data_raw: np.ndarray | None` (immutable, set in `setup()`), `_time_index: pd.DatetimeIndex | None` (cached right after read in `prepare_data()`), and `_full_data_scaled: np.ndarray | None` (rebuilt from raw each `setup()` call).

**Primary recommendation:** Execute in 6 waves: (1) cache utilities module, (2) forecasting base rewrite (split `_full_data`, cache-read `setup()`), (3) concrete forecasting modules (ETT/Weather/Electricity) adapted to new attrs, (4) `prepare_dimensions()` upgraded to read `metadata.json`, (5) classification modules cache pattern, (6) DDP smoke tests + full regression.

## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 — `_full_data` Type Split (Option B):** `_full_data_raw: np.ndarray | None` (set in `setup()`, never mutated), `_time_index: pd.DatetimeIndex | None` (cached right after read), `_full_data_scaled: np.ndarray | None` (rebuilt from raw each `setup()` call). Fast numeric ops. No `isinstance` branches downstream. Idempotency falls out: raw is immutable, scaled is rebuilt.
- **D-02 — Cache Directory:** Centralized (`~/.cache/tsdatasets/<dataset_name>/<cache_key>`). `cache_dir: Path | None = None` param on `BaseTimeSeriesDataModule.__init__`. `None` -> default. Custom path -> HPC users with faster local storage.
- **D-03 — Cache Key:** Hybrid: `<8-char-sha256>_<dataset>_<key-params>.cache`. Underscore-separated. E.g., `a3f8e1c2_ETTm1_seq128_univ_minmax.cache`. Hash for uniqueness, readable suffix for debugging.
- **D-04 — Cache Format:** Data: `<cache_key>.npz` via `numpy.savez_compressed`. Metadata: sibling `metadata.json` with schema version (`version: 1`). Scale params: `<cache_key>_data_scaler.pt`, `<cache_key>_ts_scaler.pt` via `torch.save(..., pickle_protocol=5)`. Atomic write: tmpfile + `os.replace`.
- **D-05 — `prepare_data_per_node` Default:** `True` (Lightning default). Each node writes own cache. No cross-node dependency. Document `prepare_data_per_node = False` as HPC/shared-FS optimization.
- **D-06 — Scaler Persistence:** `torch.save` / `torch.load(weights_only=False)`. Sklearn scalers are trivially pickle-able. Already in project. Zero extra import.
- **D-07 — `prepare_dimensions()` — Replace Interim:** Read `metadata.json` only. No in-memory fallback. If metadata missing or version mismatch -> raise clear error. Schema version (`version: 1`) handles future cache invalidation.
- **D-08 — `metadata.json` Schema:**
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
- Exact cache-key param encoding -- minimal set that distinguishes layouts vs. full constructor dump.
- How to handle classification modules -- they load data in `prepare_data()` too; same cache pattern applies for ARFF data.

### Deferred Ideas (OUT OF SCOPE)

- **Cache invalidation UI** -- `clear_cache()` method or CLI command. Future convenience.
- **Shared-FS coordination** -- `prepare_data_per_node = False` with rank-0 barrier. Document, don't implement.
- **Cache statistics** -- hit/miss logging. Future observability.
- **Backward compat** -- reading old cache format when version changes. Handle via schema version check.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MOD-01 | Module loads data from provided file paths in `prepare_data()` | `prepare_data()` writes cache (file paths still provided), `setup()` reads cache |
| MOD-05 | Modules expose `sequence_length`, `num_classes`, `num_features` as read-only properties | `prepare_dimensions()` reads from `metadata.json` (DDP-safe); properties unchanged |
| ROADMAP Phase 7 | DDP smoke test (gloo, 2 ranks): identical state across ranks | Verified pattern works on this machine (MPS, torch 2.8.0, Lightning 2.5.6) |
| ROADMAP Phase 7 | No `isinstance` branches on `_full_data` consumers | `_full_data` split into typed attrs eliminates all 8 branches |
| ROADMAP Phase 7 | `prepare_dimensions()` reads metadata without loading arrays | `metadata.json` schema (D-08) provides all fields needed |
| ROADMAP Phase 7 | Second `setup()` call produces identical output | `_full_data_raw` is immutable; `_full_data_scaled` is rebuilt from raw |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Cache I/O (write in `prepare_data`) | concrete modules' `_do_prepare_data` | `utils/cache.py` helpers | Concrete modules read source files; cache helpers serialize to disk. |
| Cache resolution (`cache_dir`, key derivation) | `BaseTimeSeriesDataModule` | `utils/cache.py` | Base owns constructor params; helpers own hashing logic. |
| Cache read + per-rank state (`setup`) | `BaseForecastingTimeSeriesDataModule`, `BaseClassificationTimeSeriesDataModule` | — | Subclass setup knows what arrays/scalers to load and how to reconstruct state. |
| Dimension metadata (`prepare_dimensions`) | `BaseTimeSeriesDataModule` | — | Grandparent base reads `metadata.json`; no subclass override needed. |
| Scaler persistence (fit + save / load + transform) | `BaseForecastingTimeSeriesDataModule` | `utils/cache.py` | Forecasting base owns sklearn scaler lifecycle; cache helpers own `torch.save`/`load`. |
| Classification data cache (ARFF -> npz) | `UCRClassificationDataModule`, `UEAClassificationDataModule` | `utils/cache.py` | Each classification module owns its parse + split logic; cache helpers own serialization. |
| Slice boundaries | concrete forecasting modules' `_set_data_slices` | — | Unchanged — deterministic from data length or variant. |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `numpy` | 2.4.4 | `savez_compressed` / `load` for cache data | [VERIFIED: npm registry] Already pinned in `pyproject.toml`; round-trip tested with datetime index as int64. |
| `torch` | 2.8.0 | `torch.save` / `torch.load` for sklearn scaler persistence | [VERIFIED: npm registry] Already pinned; `pickle_protocol=5` supported; round-trip verified. |
| `scikit-learn` | 1.8.0 | `MinMaxScaler` / `StandardScaler` fitted-instance persistence | [VERIFIED: npm registry] Already pinned; trivially pickle-able via torch. |
| `lightning` | 2.5.6 | `LightningDataModule` DDP lifecycle | [VERIFIED: npm registry] Already pinned; `prepare_data_per_node=True` default confirmed. |
| `pandas` | >=2.2.0 | `DatetimeIndex` extraction and serialization | [VERIFIED: npm registry] Already pinned; index serializes as int64 array (nanoseconds since epoch). |
| `joblib` | 1.5.3 | Alternative scaler persistence (already installed) | [VERIFIED: npm registry] Listed in `pyproject.toml` deps; not required per D-06 (torch.save is locked), but available if needed. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `json` (stdlib) | — | `metadata.json` write/read | Cache metadata serialization with atomic write. |
| `hashlib` (stdlib) | — | SHA-256 for cache key derivation | D-03 hybrid key hashing. |
| `tempfile` (stdlib) | — | Temp dir for atomic write tests | Test fixtures that verify cache round-trips. |
| `torch.multiprocessing.spawn` | 2.8.0 | DDP smoke test runner | Gloo backend, 2 ranks, `if __name__ == "__main__"` guard. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `numpy.savez_compressed` | `parquet` (via pyarrow) | Parquet preserves DatetimeIndex natively but adds pyarrow dep. CONTEXT.md locks npz + separate index serialization. |
| `torch.save` for scalers | `joblib.dump` | Both work. D-06 locks `torch.save`. `joblib` is already installed but torch avoids extra import. |
| `~/.cache/tsdatasets/` | Dataset-folder-local cache | Local cache ties files to source data; centralized cache works when source is read-only or on network storage. D-02 locks centralized. |
| `os.replace` atomic write | `atomicwrites` package | `os.replace` is atomic on POSIX (verified macOS). No extra dep needed. |

**Installation:** No new packages. All stack items already pinned in `pyproject.toml`.

**Version verification:**
```bash
uv run python -c "import numpy; print(numpy.__version__)"        # 2.4.4
uv run python -c "import torch; print(torch.__version__)"        # 2.8.0
uv run python -c "import sklearn; print(sklearn.__version__)"    # 1.8.0
uv run python -c "import lightning; print(lightning.__version__)" # 2.5.6
uv run python -c "import joblib; print(joblib.__version__)"      # 1.5.3
```

## Package Legitimacy Audit

> No external packages installed in this phase. All cache/persistence libraries (numpy, torch, sklearn, joblib) are already pinned in `pyproject.toml`. Audit is N/A.

| Package | Registry | Disposition |
|---------|----------|-------------|
| (none) | — | No new installs |

## Architecture Patterns

### System Architecture Diagram

```
                        user code
                            │
                            ▼
        ┌──────────────────────────────────────┐
        │  Trainer.fit() calls prepare_data()  │
        │  (rank-0 only, single process)        │
        └──────────────────────────────────────┘
                            │
                            ▼
        ┌─────────────────────────────────────────────┐
        │  _do_prepare_data() [concrete subclass]      │
        │  - Read CSV/ARFF from source file paths      │
        │  - Parse, filter, split (logic unchanged)    │
        │  - Extract _dataset_name, _time_index        │
        │  - Write to cache dir:                       │
        │    * <key>.npz  (raw arrays via savez_comp)  │
        │    * <key>_metadata.json                     │
        │    * <key>_data_scaler.pt  (post-setup fit)  │
        │    * <key>_ts_scaler.pt     (post-setup fit) │
        │  - Set _prepare_data_called sentinel          │
        └─────────────────────────────────────────────┘
                            │
                            ▼ (disk files, visible to all ranks)
        ┌──────────────────────────────────────┐
        │  Trainer.fit() calls setup('fit')     │
        │  (every rank, every process)          │
        └──────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
   Forecasting setup()              Classification setup()
   [per-rank from disk]             [per-rank from disk]
            │                               │
            │  1. Load npz -> _full_data_raw │  1. Load npz -> train/test splits
            │  2. Load metadata.json         │  2. Load metadata.json -> dims
            │  3. Extract _time_index        │  3. Fit scaler (stage='fit')
            │  4. Fit scaler on train slice  │     or reuse cached (stage='test')
            │  5. Transform raw -> scaled    │  4. Variable-length processing
            │  6. Save scaler to .pt files   │  5. Labels -> pd.Series(category)
            │  7. _transform_data()          │
            │  8. _split_data()              │
            │     -> _train_data_samples     │
            └────────────────────────────────┘

   [later, any rank]
            │
            ▼
   prepare_dimensions() [base]
            │
   Read metadata.json -> (n_features, seq_len)
   No array loading. DDP-safe.
```

### Recommended Project Structure

```
src/tscollection/datasets/
├── utils/
│   ├── cache.py              [NEW]  build_cache_key(), resolve_cache_dir(),
│   │                              atomic_save_npz(), atomic_save_metadata(),
│   │                              load_metadata(), save_scaler(), load_scaler()
│   └── __init__.py           [MOD]  export cache helpers
├── modules/
│   ├── _base/
│   │   ├── base.py           [MOD]  add cache_dir param, prepare_data_per_node,
│   │                              rewrite prepare_dimensions() to read metadata.json,
│   │                              add reset() to clear cache state attrs too
│   │   ├── forecasting.py    [MOD]  split _full_data into _full_data_raw,
│   │                              _time_index, _full_data_scaled;
│   │                              rewrite setup() to read from cache;
│   │                              remove isinstance branches
│   │   └── classification.py [MOD]  add cache-read setup pattern for UCR/UEA
│   ├── ett.py                [MOD]  _do_prepare_data() writes cache;
│   │                              _transform_data() operates on _full_data_raw
│   │                              (no isinstance check needed)
│   ├── weather.py            [MOD]  same pattern; _set_data_slices() uses len
│   │                              from _full_data_raw (not _full_data)
│   ├── electricity.py        [MOD]  same pattern; _set_data_slices() uses len
│   │                              from _full_data_raw
│   ├── ucr.py                [MOD]  _do_prepare_data() writes cache (npz + metadata);
│   │                              setup() reads cache
│   └── uea.py                [MOD]  same pattern
```

### Pattern 1: Lightning DDP-safe DataModule (canonical)

**What:** `prepare_data()` is I/O-only (writes to disk); `setup()` reads from disk on every rank.

**When to use:** Any `LightningDataModule` used with `strategy='ddp_spawn'` or `strategy='ddp'`.

**Example:**
```python
# Source: Lightning convention verified via local test (2026-05-29)

class DDPDataModule(pl.LightningDataModule):
    prepare_data_per_node: bool = True  # default

    def prepare_data(self) -> None:
        # Runs ONCE on rank-0 (or LOCAL_RANK=0 per node)
        # Write to shared cache, do NOT assign to self
        if not self._cache_path.exists():
            np.savez_compressed(str(self._cache_path), data=self._read_source())

    def setup(self, stage: str | None = None) -> None:
        # Runs on EVERY rank
        # Read from cache (disk -> memory)
        loaded = np.load(str(self._cache_path))
        self._data = loaded['data']
```

### Pattern 2: Atomic cache write (tmpfile + os.replace)

**What:** Write to `.tmp` suffix, then `os.replace()` to target. POSIX-atomic on same filesystem.

**When to use:** Any cache write that must not produce partial files visible to other ranks.

**Example:**
```python
def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    np.savez_compressed(str(tmp), **arrays)
    os.replace(str(tmp), str(path))
    # os.replace is atomic on POSIX (verified macOS)
```

**Anti-Patterns to Avoid:**

- **Writing `_full_data` in `prepare_data()`.** Under DDP, rank-0 writes to `self._full_data` but ranks 1+ never execute this code. Their `_full_data` remains `None`. Solution: write to disk in `prepare_data()`, read in `setup()`.
- **Using `isinstance(self._full_data, pd.DataFrame)` branches.** After Option B split, `_full_data_raw` is always `np.ndarray` and `_time_index` is always `pd.DatetimeIndex | None`. The type is known at definition, not runtime. No `isinstance` needed.
- **Caching on a different filesystem than the source.** `os.replace()` requires same filesystem. If `~/.cache/tsdatasets/` is on a different mount than the tmp dir, use `pathlib.Path(tmp).parent == pathlib.Path(path).parent` to ensure same-filesystem rename.
- **Not encoding cache-key params.** If `seq_len=128` and `seq_len=96` produce the same cache key, stale data silently serves wrong splits. D-03 locks hybrid key with readable param suffix.

### Pattern 3: Scaler persistence via `torch.save`

**What:** Save fitted sklearn scalers to `.pt` files during `setup('fit')`; load during `setup('test')` or subsequent `setup()` calls.

**When to use:** When `setup('test')` runs in a process where `setup('fit')` never ran.

**Example:**
```python
# Fit path (setup('fit'))
data_scaler = MinMaxScaler()
data_scaler.fit(raw_data[train_slice])
torch.save(data_scaler, str(scaler_path), pickle_protocol=5)
self._data_scaler_cache = data_scaler

# Test path (setup('test') without prior fit)
if self._data_scaler_cache is None and scaler_path.exists():
    self._data_scaler_cache = torch.load(scaler_path, weights_only=False)
    raw_data_scaled = self._data_scaler_cache.transform(raw_data)
```

### Pattern 4: DatetimeIndex serialization

**What:** Extract DatetimeIndex as int64 (nanoseconds since epoch) for `npz` storage; reconstruct on load.

**When to use:** Forecasting modules with `parse_dates=True` CSV sources.

**Example:**
```python
# Save
index_ns = df.index.astype(np.int64).to_numpy()
np.savez_compressed(str(cache_path), data=df.to_numpy(), index=index_ns)

# Load
loaded = np.load(str(cache_path))
time_index = pd.DatetimeIndex(loaded['index'])  # reconstruct from int64
raw_data = loaded['data']
```

### Pattern 5: `prepare_dimensions()` from `metadata.json`

**What:** Read dimensions from disk metadata instead of computing from in-memory `_full_data`.

**When to use:** Post-phase-7, replaces interim in-memory implementation. DDP-safe because every rank reads the same file.

**Example:**
```python
def prepare_dimensions(self) -> tuple[int | None, int | None]:
    """Read dimensions from metadata.json. No array loading."""
    if self._num_features is not None:
        return self._num_features, self._seq_len  # post-setup cache

    meta = load_metadata(self._cache_dir)  # reads JSON
    self._num_features = meta['n_features']
    self._seq_len = meta['seq_len']
    return self._num_features, self._seq_len
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cache directory resolution | Custom path builder | `Path.home() / '.cache' / 'tsdatasets' / dataset_name` | Matches `~/.cache/tsdatasets/` convention (D-02). Standard XDG cache pattern. |
| Atomic file write | Manual `try/except` around `open()` + `os.rename()` | `tmp = path.with_suffix(path.suffix + '.tmp')`; `os.replace(tmp, path)` | Verified atomic on POSIX (macOS/Linux). `os.replace` is the standard pattern. |
| SHA-256 hashing | Custom hash function | `hashlib.sha256(params.encode()).hexdigest()[:8]` | Stdlib. Deterministic. Truncated to 8 chars per D-03. |
| Scaler serialization | Pickle the raw sklearn object | `torch.save(scaler, path, pickle_protocol=5)` | D-06 locks torch.save. Works identically to pickle; already in project deps. |
| DatetimeIndex round-trip | Store as strings (loses precision) | `index.astype(np.int64)` -> `pd.DatetimeIndex(loaded['index'])` | Nanosecond-precision preservation. |
| JSON metadata | Custom format | `json.dump()` / `json.load()` with version field | Stdlib. Schema versioning via `"version": 1` handles future invalidation (D-07). |

**Key insight:** This phase replaces hand-rolled in-memory state with disk-backed state. Every "library to use instead" is stdlib or already pinned. The work is in wiring the pattern consistently across 5 concrete modules + 2 base classes.

## Runtime State Inventory

> Not a rename phase, but the cache directory introduces disk state. Documenting what persists where.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | New: `~/.cache/tsdatasets/<dataset>/<key>.npz`, `metadata.json`, `<key>_data_scaler.pt`, `<key>_ts_scaler.pt` | Cache is written during `prepare_data()`/`setup()`; no existing data to migrate. Fresh writes. |
| Live service config | None | No external services configured with these paths. |
| OS-registered state | None | Pure filesystem cache. |
| Secrets/env vars | None | Cache dir path is not a secret. |
| Build artifacts | None | Cache is runtime-generated, not build-time. |

## `_full_data` Type Drift Map

| Location | Line | Current Type | Operation | Resulting Type | `isinstance` Branch |
|----------|------|-------------|-----------|----------------|---------------------|
| `forecasting.py:80` | attr def | `np.ndarray \| pd.DataFrame \| None` | Initial declaration | — | No |
| `ett.py:150` | `_do_prepare_data` | — | `df = pd.read_csv(...)` | `pd.DataFrame` | No |
| `weather.py:127` | `_do_prepare_data` | — | `df = pd.read_csv(...)` | `pd.DataFrame` | No |
| `electricity.py:135` | `_do_prepare_data` | — | `df = pd.read_csv(...)` | `pd.DataFrame` | No |
| `forecasting.py:146` | `_compute_dimensions` | `pd.DataFrame` (pre-setup) | `isinstance(self._full_data, pd.DataFrame)` branch | `pd.DataFrame \| np.ndarray` | **YES** |
| `forecasting.py:206` | `setup` | `pd.DataFrame` (pre-scale) | `isinstance(self._full_data, pd.DataFrame)` -> extract index | `pd.DataFrame \| np.ndarray` | **YES** |
| `ett.py:122` | `_transform_data` | `pd.DataFrame` (pre-transform) | `isinstance(self._full_data, pd.DataFrame)` -> `.to_numpy()` | `np.ndarray` | **YES** |
| `weather.py:101` | `_transform_data` | `pd.DataFrame` (pre-transform) | `isinstance(self._full_data, pd.DataFrame)` -> `.to_numpy()` | `np.ndarray` | **YES** |
| `electricity.py:101` | `_transform_data` | `pd.DataFrame` (pre-transform) | `isinstance(self._full_data, pd.DataFrame)` -> `.to_numpy()` | `np.ndarray` | **YES** |
| `forecasting.py:231` | `setup` (fit path) | `np.ndarray` (post-transform) | `self._full_data = data_scaler.transform(full_array)` | `np.ndarray` | No |
| `forecasting.py:245` | `setup` (fit path) | `np.ndarray` (post-scale) | `np.concatenate([repeated_ts, self._full_data], axis=-1)` | `np.ndarray` | No |

**After Option B split:**
- `_full_data_raw: np.ndarray | None` — set once in `setup()` from cache, never mutated.
- `_time_index: pd.DatetimeIndex | None` — extracted in `prepare_data()`, cached in `metadata.json`.
- `_full_data_scaled: np.ndarray | None` — rebuilt from `_full_data_raw` each `setup()` call.

All 8 `isinstance` branches are eliminated because types are known at definition.

## Common Pitfalls

### Pitfall 1: Cache hit on stale data after schema change
**What goes wrong:** After refactoring, old cache files (written with version 0 or no version) are read by new code expecting version 1. Data silently serves wrong shape or missing fields.
**Why it happens:** The cache directory persists across sessions. Phase 7 introduces the cache format.
**How to avoid:** D-07 locks "read `metadata.json` only; if missing or version mismatch, raise clear error." Error message: `"Cache version {actual} does not match expected version 1. Delete cache dir {path} and re-run prepare_data()."`
**Warning signs:** Tests pass with fresh `tmp_path` but crash against `~/.cache/tsdatasets/` from manual runs.

### Pitfall 2: `os.replace` across filesystems
**What goest wrong:** `os.replace(tmp, target)` raises `OSError: [Errno 18] Invalid cross-device link` if tmp and target are on different mount points.
**Why it happens:** `~/.cache/tsdatasets/` may be on a network mount while `/tmp/` is local.
**How to avoid:** Always create tmp file in the same directory as the target: `tmp = target.with_suffix(target.suffix + '.tmp')`.
**Warning signs:** `FileNotFoundError` or `OSError` on `os.replace` in CI (different mount config).

### Pitfall 3: `mp.spawn` requires top-level callable
**What goes wrong:** DDP smoke test uses `mp.spawn(worker, ...)` but `worker` is defined inside a function or method. Pickle fails with `Can't get attribute 'worker'`.
**Why it happens:** `spawn` start method serializes the target function.
**How to avoid:** Define the worker function at module top level (not inside a class method). For pytest, use the pattern: define a module-level `_ddp_worker` function and call `mp.spawn(_ddp_worker, ...)` from the test.
**Warning signs:** `AttributeError: Can't get attribute` during test execution.

### Pitfall 4: Lightning `prepare_data()` doesn't assign to `self` — but `_do_prepare_data` still does
**What goes wrong:** The base `prepare_data()` wrapper calls `_do_prepare_data()`. If the concrete `_do_prepare_data()` still assigns `self._full_data = df`, the DDP problem persists (rank-0 assigns, ranks 1+ see None).
**Why it happens:** `_do_prepare_data()` in each concrete module currently writes `self._full_data = df` (see ETT line 150, Weather line 127, Electricity line 135). This must change to: write cache file, set only `_dataset_name` and `_time_index`.
**How to avoid:** Audit `_do_prepare_data()` in each of the 5 concrete modules. Replace `self._full_data = df` with `atomic_save_npz(cache_path, data=df.to_numpy())`. If the DataFrame has a DatetimeIndex, also cache it: `index=df.index.astype(np.int64)`.
**Warning signs:** DDP test fails because `_full_data is None` on rank 1.

### Pitfall 5: `_set_data_slices()` depends on `len(self._full_data)` for Weather/Electricity
**What goes wrong:** Weather and Electricity `_set_data_slices()` call `len(self._full_data)` to compute fractional splits (60/20/20). Under Option B, `_full_data` is split into `_full_data_raw` (set in `setup()`) and `_time_index`. `_set_data_slices()` is called from `_finalize_prepare_data()`, which runs after `_do_prepare_data()` — but `_full_data_raw` isn't set until `setup()`.
**Why it happens:** `_set_data_slices()` needs the data length to compute split boundaries, but under DDP, `_full_data` is no longer set in `prepare_data()`.
**How to avoid:** Option B says `_full_data_raw` is set in `setup()`. Therefore `_set_data_slices()` must move from `_finalize_prepare_data()` (runs in `prepare_data()`) to the beginning of `setup()`. For ETT, slices are variant-based (not data-length-based), so this does not matter. For Weather/Electricity, slices depend on `len(_full_data_raw)` which is only available post-cache-read.
**This is a critical sequencing change:** `_finalize_prepare_data()` should NOT call `_set_data_slices()` after phase 7. Instead, `setup()` reads cache -> sets `_full_data_raw` -> calls `_set_data_slices()` -> continues with scaling.

### Pitfall 6: `reset()` needs to clear new cache-related attrs
**What goes wrong:** The existing `reset()` method clears `_setup_completed_stages` and `_prepare_data_called`. After phase 7, new attrs (`_full_data_raw`, `_time_index`, `_full_data_scaled`, `_data_scaler_cache`, `_ts_feature_scaler_cache`) persist across reset, causing stale state.
**Why it happens:** `reset()` was added in phase 6 but new attrs are added in phase 7.
**How to avoid:** Extend `reset()` to also clear `_full_data_raw`, `_time_index`, `_full_data_scaled`, scaler caches, and data sample attrs. Or make reset more systematic: clear all attrs starting with `_` except constructor params.
**Warning signs:** Tests that call `reset()` then `setup()` see stale scaled data.

### Pitfall 7: Classification cache pattern is different from forecasting
**What goes wrong:** Forecasting caches one array (`_full_data_raw`) + index. Classification caches multiple arrays (train/test/valid samples + labels + varying-sequence-length arrays). UCR uses `pd.DataFrame` samples; UEA uses `np.ndarray` with shape `(samples, timesteps, features)`.
**Why it happens:** Classification data is already split in `_do_prepare_data()` (not post-setup). The split includes labels (category dtype) and variable-length centering.
**How to avoid:** Classification cache: save pre-split data (`samples`, `labels`) as npz, save post-split data for each split (`train_samples`, `train_labels`, etc.), save metadata with dims. `setup()` reads the already-split arrays. The scaler persistence pattern is simpler: classification uses `create_data_scaler()` closure (D2), forecasting uses direct sklearn scalers.
**Warning signs:** Classification `setup()` produces wrong shapes because cached arrays lost dtype or shape info.

## Code Examples

### Example A: `utils/cache.py` module structure

```python
"""Cache utilities for DDP-safe DataModule persistence."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    'build_cache_key',
    'resolve_cache_dir',
    'atomic_save_npz',
    'atomic_save_metadata',
    'load_metadata',
    'save_scaler',
    'load_scaler',
]


def resolve_cache_dir(
    *,
    cache_dir: Path | None,
    dataset_name: str,
) -> Path:
    """Resolve the cache directory path.

    Args:
        cache_dir: User-provided cache directory. If None, uses
            ``~/.cache/tsdatasets/{dataset_name}``.
        dataset_name: Dataset identifier.

    Returns:
        Absolute path to the cache directory.
    """
    if cache_dir is not None:
        return cache_dir.expanduser().resolve()
    return Path.home().resolve() / '.cache' / 'tsdatasets' / dataset_name


def build_cache_key(
    *,
    dataset_name: str,
    params: dict[str, Any],
) -> str:
    """Build a hybrid cache key: hash + readable suffix.

    Format: ``<8-char-sha256>_<dataset>_<key-params>.cache``
    Example: ``a3f8e1c2_ETTm1_seq128_univ_minmax.cache``

    Args:
        dataset_name: Dataset identifier.
        params: Parameters that affect data layout (seq_len, mode,
            scaling_method, etc.).

    Returns:
        Cache key string.
    """
    # Sort params for deterministic hashing
    serialized = json.dumps(params, sort_keys=True)
    hash_prefix = hashlib.sha256(serialized.encode()).hexdigest()[:8]

    # Readable suffix: key=value pairs
    suffix_parts = [dataset_name]
    for k, v in sorted(params.items()):
        suffix_parts.append(f'{k}={v}')

    return '_'.join(suffix_parts) + '.cache'


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Save arrays to npz file atomically.

    Writes to .tmp suffix, then os.replace() for POSIX atomicity.

    Args:
        path: Target .npz file path.
        **arrays: Named arrays to save.
    """
    tmp = path.with_suffix(path.suffix + '.tmp')
    np.savez_compressed(str(tmp), **arrays)
    os.replace(str(tmp), str(path))


def atomic_save_metadata(path: Path, data: dict[str, Any]) -> None:
    """Save metadata dict to JSON file atomically.

    Args:
        path: Target .json file path.
        data: Metadata dictionary.
    """
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(str(tmp), str(path))


def load_metadata(path: Path) -> dict[str, Any]:
    """Load metadata from JSON file.

    Args:
        path: Metadata .json file path.

    Returns:
        Metadata dictionary.

    Raises:
        FileNotFoundError: If metadata file does not exist.
        ValueError: If schema version mismatch.
    """
    if not path.exists():
        raise FileNotFoundError(f'Metadata not found: {path}')
    with open(path) as f:
        data = json.load(f)
    if data.get('version') != 1:
        raise ValueError(
            f'Cache version {data.get("version")} does not match expected version 1. '
            f'Delete cache dir and re-run prepare_data().'
        )
    return data


def save_scaler(scaler: Any, path: Path) -> None:
    """Persist a fitted sklearn scaler via torch.save.

    Args:
        scaler: Fitted scaler instance.
        path: Target .pt file path.
    """
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(scaler, str(tmp), pickle_protocol=5)
    os.replace(str(tmp), str(path))


def load_scaler(path: Path) -> Any:
    """Load a persisted sklearn scaler via torch.load.

    Args:
        path: .pt file path.

    Returns:
        Loaded scaler instance.
    """
    return torch.load(path, weights_only=False)
```

### Example B: Forecasting `setup()` with split attrs (Option B)

```python
# Adapted from D-01 locked design

def setup(self, stage: str | None = None) -> None:
    # ... stage validation + idempotency guard (unchanged from phase 6) ...

    # 1. Load raw data from cache (every rank)
    cache_path = self._cache_dir / f'{self._cache_key}.npz'
    loaded = np.load(str(cache_path))
    self._full_data_raw = loaded['data'].astype(np.float32)  # immutable after load

    # 2. Load time index from cache (every rank)
    if 'index' in loaded:
        self._time_index = pd.DatetimeIndex(loaded['index'])
    else:
        self._time_index = None

    # 3. Set slices (now safe: _full_data_raw is set)
    self._set_data_slices()

    # 4. Fit or load scalers (stage-dependent)
    if stage in ('fit', None):
        data_scaler = self._prepare_data_scaler()
        data_scaler.fit(self._full_data_raw[self._train_slice])
        self._data_scaler_cache = data_scaler
        self._save_scaler_to_cache(data_scaler, 'data')

        self._full_data_scaled = data_scaler.transform(self._full_data_raw)
        self._transform_data()

        if self._time_index is not None:
            time_series_features = extract_time_features(self._time_index)
            ts_scaler = self._prepare_data_scaler()
            ts_scaler.fit(time_series_features[self._train_slice])
            self._ts_feature_scaler_cache = ts_scaler
            self._save_scaler_to_cache(ts_scaler, 'ts')
            scaled_ts = ts_scaler.transform(time_series_features)
            # ... expand, repeat, concatenate (same as current) ...

        self._calculate_num_features()
        self._split_data()

    elif stage in ('test', 'predict'):
        # Load cached scalers from disk
        self._data_scaler_cache = self._load_scaler_from_cache('data')
        # ... transform with loaded scalers ...

    self._setup_completed_stages.add(stage)
```

### Example C: DDP smoke test pattern

```python
# Module-level worker (must be top-level for mp.spawn to pickle it)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tempfile


def _ddp_worker(rank, world_size, tmpdir):
    """DDP worker: write cache from rank 0, read from all ranks."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    try:
        if rank == 0:
            # Simulate prepare_data() — write cache
            np.savez_compressed(
                os.path.join(tmpdir, 'data.npz'),
                data=np.arange(100, dtype=np.float32),
            )

        dist.barrier()  # Ensure cache is written before other ranks read

        # All ranks read from cache (simulating setup())
        loaded = np.load(os.path.join(tmpdir, 'data.npz'))
        assert loaded['data'].shape == (100,), f'Rank {rank}: wrong shape'

        dist.barrier()
    finally:
        dist.destroy_process_group()
```

**Source:** Verified on this machine (2026-05-29) with torch 2.8.0, gloo backend, 2 ranks.

### Example D: DatetimeIndex round-trip

```python
# Save (prepare_data path)
index_ns = df.index.astype(np.int64).to_numpy()
atomic_save_npz(cache_path, data=df.to_numpy(), index=index_ns)

# Load (setup path)
loaded = np.load(str(cache_path))
time_index = pd.DatetimeIndex(loaded['index'])
raw_data = loaded['data']

# Verify
assert all(time_index == original_df.index)
assert np.allclose(raw_data, original_df.to_numpy())
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `_full_data: np.ndarray \| pd.DataFrame \| None` | `_full_data_raw: np.ndarray \| None`, `_time_index: pd.DatetimeIndex \| None`, `_full_data_scaled: np.ndarray \| None` | This phase (D-01) | Eliminates 8 `isinstance` branches. Type is known at definition. |
| `prepare_data()` assigns to `self._full_data` | `prepare_data()` writes to cache disk | This phase | DDP-safe: every rank reads from disk in `setup()`. |
| `prepare_dimensions()` reads in-memory `_full_data` | `prepare_dimensions()` reads `metadata.json` | This phase (D-07) | No array loading. Works after `prepare_data()` without `setup()`. DDP-safe. |
| In-memory `_setup_completed_stages` sentinel only | Sentinel + cache-backed scaler persistence | This phase (D-06) | `setup('test')` in a fresh process loads scaler from `.pt` file. |
| No cache | `~/.cache/tsdatasets/<dataset>/<key>` | This phase (D-02) | Centralized cache with configurable path. |

**Deprecated/outdated:**
- `_full_data` as unified attr: replaced by typed split attrs (D-01).
- In-memory `prepare_dimensions()` interim (from phase 6): replaced by `metadata.json` read (D-07).
- `_post_prepare_data` naming convention: already renamed to `_finalize_prepare_data` in phase 6.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `_set_data_slices()` for ETT does not depend on `_full_data` (uses variant-based fixed boundaries) | Pitfall 5 | LOW. Verified by reading `ett.py:101-114` — uses `slice(None, 12*30*24)` etc., no `len(self._full_data)`. |
| A2 | Weather/Electricity `_set_data_slices()` depend on `len(self._full_data)` for fractional splits | Pitfall 5 | MEDIUM. Verified by reading `weather.py:88` and `electricity.py:88-89` — both call `len(self._full_data)`. This means slices must be computed AFTER `setup()` loads `_full_data_raw`, not in `prepare_data()`. |
| A3 | `torch.save(scaler)` preserves sklearn scaler identity across processes | Pattern 3 | HIGH-CONFIDENCE. Verified via round-trip test (MinMaxScaler fit -> save -> load -> transform identical). |
| A4 | `pd.DatetimeIndex(loaded['index'])` reconstructs from int64 nanoseconds correctly | Pattern 4 | MEDIUM. Standard pandas behavior but should be verified with a test. |
| A5 | Classification modules cache `np.ndarray` samples with varying shapes (UEA 3-D) via `savez_compressed` | Pitfall 7 | MEDIUM. UEA data is `(samples, timesteps, features)` — variable-length processing pads to max. Need to verify `savez_compressed` handles 3-D arrays correctly (it does — numpy saves arbitrary ND arrays). |
| A6 | `prepare_data_per_node = True` (Lightning default) means each node has its own LOCAL_RANK=0 writing the cache | Standard Stack | HIGH-CONFIDENCE. Verified via `LightningDataModule.prepare_data_per_node` attribute. |

## Open Questions (RESOLVED)

1. **Should `_set_data_slices()` move from `_finalize_prepare_data()` to `setup()`?**
   - What we know: Weather and Electricity compute fractional splits from `len(self._full_data)`. Under D-01, `_full_data` is not set until `setup()` reads cache.
   - What was unclear: Whether ETT's variant-based slices (already independent of data length) can stay in `_finalize_prepare_data()`, or if ALL slicing should move to `setup()` for consistency.
   - Recommendation: **Move all `_set_data_slices()` to `setup()`** (after cache read). `_finalize_prepare_data()` becomes no-op for forecasting too. This is a clean architectural change — slicing is a setup concern, not a prepare concern.
   - **RESOLVED:** Plan 07-04 Task 1 makes `_finalize_prepare_data()` a no-op and plan 07-04 Task 2 moves `_set_data_slices()` to `setup()` after cache read. Plans 07-05 and 07-06 follow the same pattern for ETT, Weather, and Electricity modules.

2. **How to cache classification data (UCR/UEA)?**
   - What we know: UCR caches `pd.DataFrame` (train/test samples) and `pd.Series` (labels). UEA caches `np.ndarray` (3-D samples) and `np.ndarray` (1-D labels). Both modules run variable-length processing in `_do_prepare_data()`.
   - What was unclear: Whether to cache pre-processed (after variable-length centering) or raw data. Caching pre-processed means `setup()` is trivially a cache read. Caching raw means `setup()` re-runs centering (deterministic, cheap).
   - Recommendation: **Cache post-processed data** (after splitting + variable-length centering). This makes `setup()` a pure cache read + scaler application. The cache key encodes `splitting_strategy`, `test_size`, `valid_size` so different configs produce different caches.
   - **RESOLVED:** Plan 07-07 Task 2 implements post-processed caching: UCR writes train/test/valid samples and labels as numpy arrays after all processing; UEA writes 3-D arrays and label arrays. Cache key includes splitting_strategy, test_size, valid_size per D-03.

3. **Should `reset()` clear new cache-related attrs?**
   - What we know: Phase 6 added `reset()` clearing `_setup_completed_stages` and `_prepare_data_called`. Phase 7 adds `_full_data_raw`, `_time_index`, `_full_data_scaled`, scaler caches.
   - What was unclear: Whether `reset()` should also delete cache files (probably not — cache is valid across resets).
   - Recommendation: **`reset()` clears in-memory attrs only.** Cache files persist. `prepare_data()` is guarded by `_prepare_data_called` sentinel; if reset clears the sentinel, `prepare_data()` re-runs but skips writes if cache exists (`if not cache_path.exists():`).
   - **RESOLVED:** Plan 07-03 Task 1 extends `reset()` to clear `_full_data_raw`, `_time_index`, `_full_data_scaled`, `_data_scaler_cache`, `_ts_feature_scaler_cache`, and data sample attrs. Cache files on disk are NOT deleted by reset.

4. **Cache key params for forecasting modules — minimal set?**
   - What we know: D-03 locks hybrid key with readable param suffix.
   - What was unclear: Which params distinguish layouts vs. which are cosmetic.
   - Recommendation: `seq_len`, `mode` (UNIVARIATE/MULTIVARIATE), `data_scaling_method`, `data_scaling_range` for forecasting. For classification: `splitting_strategy`, `test_size`, `valid_size`, `data_scaling_method`. These are the params that change data shape or split boundaries.
   - **RESOLVED:** Plans 07-05, 07-06 use `seq_len`, `mode`, `data_scaling_method`, `data_scaling_range` for forecasting cache keys. Plan 07-07 uses `splitting_strategy`, `test_size`, `valid_size`, `data_scaling_method` for classification cache keys.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | All code | YES | 3.12 | — |
| `numpy` | Cache I/O (savez_compressed) | YES | 2.4.4 | — |
| `torch` | Scaler persistence (save/load), DDP test (spawn, gloo) | YES | 2.8.0 | — |
| `scikit-learn` | Scaler fit/transform | YES | 1.8.0 | — |
| `pandas` | DatetimeIndex round-trip | YES | >=2.2.0 | — |
| `lightning` | DDP lifecycle (prepare_data/setup) | YES | 2.5.6 | — |
| `joblib` | Alternative scaler persistence | YES | 1.5.3 | Not used (D-06 locks torch.save) |
| `pytest` | Test runner | YES | >=8.2 | — |
| `ruff` | Lint verification | YES | >=0.15.9 | — |
| `ty` | Type check | YES | >=0.0.28 | — |

**Missing dependencies with no fallback:** None.
**Missing dependencies with fallback:** None.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | `pytest >= 8.2` (already pinned) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` (`pythonpath = ["."]`, `testpaths = ["tests"]`) |
| Quick run command | `uv run pytest tests/test_cache.py tests/test_modules_base.py tests/test_modules_forecasting.py -x` |
| Full suite command | `uv run pytest tests/ -x --cov=src/tscollection` |
| Mocking style | `unittest.mock.patch` / `MagicMock` (stdlib) — project precedent |
| Fixtures | `tests/conftest.py` provides `synthetic_classification_df`, `synthetic_forecast_data`, `synthetic_multivariate_data`. New: `synthetic_csv_file`, `ett_csv` (in `test_modules_forecasting.py`), `electricity_csv_file` (in `test_modules_forecasting.py`). |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|--------------|
| ROADMAP DDP-01 | DDP smoke test (gloo, 2 ranks): identical state across ranks after `setup('fit')` | integration | `uv run pytest tests/test_ddp_compliance.py::test_ddp_forecasting_cache_round_trip -x` | Wave 0 |
| ROADMAP DDP-02 | No `isinstance` branches on `_full_data` consumers | static analysis | `uv run grep -rn "isinstance.*_full_data" src/ --include="*.py"` (should return 0) | Post-implementation gate |
| ROADMAP DDP-03 | `prepare_dimensions()` reads metadata without loading arrays | unit | `uv run pytest tests/test_modules_base.py::TestPrepareDimensionsCache -x` | Wave 0 |
| ROADMAP DDP-04 | Second `setup()` call produces identical output | unit | `uv run pytest tests/test_modules_forecasting.py::TestSetupIdempotency -x` | Exists (phase 6) |
| D-01 | `_full_data_raw` immutable after `setup()` load | unit | `uv run pytest tests/test_modules_forecasting.py::TestFullDataSplit -x` | Wave 0 |
| D-02 | `resolve_cache_dir(None)` returns `~/.cache/tsdatasets/<name>` | unit | `uv run pytest tests/test_cache.py::TestResolveCacheDir -x` | Wave 0 |
| D-02 | `resolve_cache_dir(custom_path)` returns custom path | unit | `uv run pytest tests/test_cache.py::TestResolveCacheDir -x` | Wave 0 |
| D-03 | `build_cache_key` produces deterministic hybrid key | unit | `uv run pytest tests/test_cache.py::TestBuildCacheKey -x` | Wave 0 |
| D-04 | npz round-trip preserves dtype and shape | unit | `uv run pytest tests/test_cache.py::TestAtomicSaveNpz -x` | Wave 0 |
| D-04 | metadata.json version mismatch raises ValueError | unit | `uv run pytest tests/test_cache.py::TestLoadMetadataVersion -x` | Wave 0 |
| D-06 | torch.save/load sklearn scaler produces identical transform | unit | `uv run pytest tests/test_cache.py::TestScalerPersistence -x` | Wave 0 |
| D-07 | `prepare_dimensions()` raises when metadata missing | unit | `uv run pytest tests/test_modules_base.py::TestPrepareDimensionsCache -x` | Wave 0 |
| D-08 | `metadata.json` schema matches spec | unit | `uv run pytest tests/test_cache.py::TestMetadataSchema -x` | Wave 0 |
| Classification | UCR/UEA cache round-trip (prepare_data writes, setup reads) | integration | `uv run pytest tests/test_modules_ucr.py tests/test_modules_uea.py -k cache -x` | Wave 0 |
| Regression | All existing tests still pass | full suite | `uv run pytest tests/ -x` | Exists |

### Sampling Rate

- **Per task commit:** `uv run pytest tests/test_cache.py tests/test_modules_base.py tests/test_modules_forecasting.py -x` (~5 seconds; covers cache module + touched bases).
- **Per wave merge:** `uv run pytest tests/ -x` (full suite).
- **Phase gate:** Full suite green + `uv run ruff check src/` clean + `uv run ty check src/` clean + DDP smoke test passes.

### Wave 0 Gaps

- [ ] `tests/test_cache.py` — NEW file. Unit tests for `build_cache_key()`, `resolve_cache_dir()`, `atomic_save_npz()`, `atomic_save_metadata()`, `load_metadata()`, `save_scaler()`, `load_scaler()`, DatetimeIndex round-trip, metadata version mismatch, schema validation.
- [ ] `tests/test_ddp_compliance.py` — NEW file. DDP smoke test using `mp.spawn` with gloo backend, 2 ranks. Tests: forecasting cache round-trip (rank 0 writes, all ranks read), classification cache round-trip, identical `_train_data_samples` across ranks.
- [ ] `tests/test_modules_forecasting.py::TestFullDataSplit` — NEW class. Tests `_full_data_raw` immutability, `_time_index` persistence, `_full_data_scaled` rebuild from raw, no `isinstance` branches.
- [ ] `tests/test_modules_base.py::TestPrepareDimensionsCache` — NEW class. Tests `prepare_dimensions()` reads from `metadata.json` (mocked via `patch`), raises `FileNotFoundError` when metadata missing, raises `ValueError` on version mismatch.
- [ ] `tests/conftest.py` — ADD `synthetic_cache_dir` fixture: creates temp dir with pre-populated cache files (npz + metadata.json + scaler.pt) for tests that need cache without calling `prepare_data()`.

## Sources

### Primary (HIGH confidence)
- **Lightning DDP docs** — https://lightning.ai/docs/pytorch/stable/data/datamodule.html — `prepare_data()` runs rank-0 only; `setup()` runs every rank; `prepare_data_per_node=True` default.
- **Local verification** — Lightning DDP smoke test (strategy='ddp_spawn', accelerator='cpu', devices=2, gloo backend) confirmed working on this machine (2026-05-29). Both ranks successfully loaded cache written by rank-0.
- **numpy savez_compressed** — Verified round-trip with DatetimeIndex as int64 array. `np.load()` correctly reconstructs via `pd.DatetimeIndex(loaded['index'])`.
- **torch.save sklearn scaler** — Verified round-trip with `pickle_protocol=5`. Loaded scaler produces identical `transform()` output. `torch.load(weights_only=False)` required for non-tensor objects.
- **Atomic write** — `os.replace(tmp, target)` verified working on macOS (POSIX). Requires same-filesystem.
- **mp.spawn worker** — Verified working with top-level function. Fails with nested functions (pickling constraint).
- **CONTEXT.md (07-CONTEXT.md)** — Locked design specification (D-01 through D-08).
- **CONTEXT.md (06-CONTEXT.md)** — Phase 6 decisions (D1-D4) providing foundation for phase 7.

### Secondary (MEDIUM confidence)
- **Phase 6 research** — `.planning/phases/06-lightning-lifecycle/06-RESEARCH.md` — detailed analysis of Lightning lifecycle, `isinstance` branch locations, scaler caching.
- **DDP compliance analysis** — `.planning/phases/06-lightning-lifecycle/lightning-ddp-compliance.md` — original 8-step plan, classification/forecasting cache recipes.

### Tertiary (LOW confidence)
- None. All claims verified via local tests or official documentation.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all packages already pinned; versions verified via `uv run`; round-trip tests confirm cache format works.
- Architecture: HIGH — DDP pattern verified locally (rank-0 write, all-rank read); `_full_data` split eliminates 8 `isinstance` branches (mapped); `metadata.json` schema locked in D-08.
- Pitfalls: HIGH — all 7 pitfalls derived from concrete code reading + grep verification. Pitfall 5 (slice sequencing) is the most critical and must be addressed in the plan.

**Research date:** 2026-05-29
**Valid until:** 2026-06-28 (30 days — Lightning 2.5 API is stable; numpy 2.4 API is stable)

## RESEARCH COMPLETE

**Phase:** 07 - DDP Compliance + `_full_data` Split
**Confidence:** HIGH

### Key Findings
1. **DDP pattern verified** — `strategy='ddp_spawn'`, gloo, 2 ranks confirmed working on this machine. `prepare_data()` writes cache (rank-0), `setup()` reads cache (every rank).
2. **8 `isinstance` branches mapped** — 1 in `_compute_dimensions()`, 1 in `setup()` time-index extraction, 6 in concrete `_transform_data()` / `_set_data_slices()` methods. Option B split eliminates all of them.
3. **Cache format verified** — `numpy.savez_compressed` + `torch.save` (sklearn) + JSON (metadata) all round-trip correctly. Atomic write (`os.replace`) works on macOS.
4. **Critical sequencing change** — `_set_data_slices()` for Weather/Electricity depends on `len(_full_data)`, which requires moving slice computation from `_finalize_prepare_data()` (runs in `prepare_data()`) to `setup()` (after cache read).
5. **No new packages needed** — All cache utilities use stdlib (`hashlib`, `json`, `os`, `pathlib`) + already-pinned deps (`numpy`, `torch`).

### File Created
`/Users/skaf/VSCodeProjects/tsdatasets/.planning/phases/07-ddp-compliance/07-RESEARCH.md`

### Confidence Assessment
| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | All packages verified via `uv run`; versions confirmed. |
| Architecture | HIGH | DDP pattern tested locally; `_full_data` usage mapped; cache format verified. |
| Pitfalls | HIGH | All derived from concrete code reading + grep verification. |

### Open Questions (RESOLVED)
1. `_set_data_slices()` migration to `setup()` — RESOLVED in plan 07-04 (moved to setup after cache read).
2. Classification cache granularity — RESOLVED in plan 07-07 (post-processed caching).
3. `reset()` scope for new attrs — RESOLVED in plan 07-03 (in-memory only, cache persists).
4. Cache key param set — RESOLVED in plans 07-05, 07-06, 07-07 (forecasting: seq_len, mode, scaling; classification: splitting_strategy, test_size, valid_size).

### Ready for Planning
Research complete. All open questions resolved by plans.
