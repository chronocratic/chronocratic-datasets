---
phase: 07-ddp-compliance
plan: 01
subsystem: cache
tags: [numpy, torch, sklearn, atomic-io, hash, json]
requires:
  - phase: 06-lightning-lifecycle
    provides: Base module structure, idempotency sentinels, _finalize_prepare_data hook
provides:
  - Cache utility module (build_cache_key, resolve_cache_dir, atomic_save_npz, atomic_save_metadata, load_metadata, save_scaler, load_scaler)
  - CACHE_SCHEMA_VERSION constant
  - Full test coverage for all cache functions
affects: [07-ddp-compliance]
tech-stack:
  added: []
  patterns:
    - "Atomic file write via temp + Path.replace() (POSIX atomicity)"
    - "Hybrid cache key: 8-char SHA-256 prefix + readable param suffix"
    - "DatetimeIndex serialization via as_unit('ns').view(np.int64)"
key-files:
  created:
    - src/tscollection/datasets/utils/cache.py
    - tests/test_cache.py
  modified:
    - src/tscollection/datasets/utils/__init__.py
key-decisions:
  - "Use Path.replace() instead of os.replace() per ruff PTH105"
  - "Use Path.open() instead of open() per ruff PTH123"
  - "np.savez_compressed appends .npz; use temp stem + append .npz for atomic rename"
  - "as_unit('ns').view(np.int64) for DatetimeIndex serialization (pandas 3.0 compat)"
patterns-established:
  - "Cache I/O: write to tmp in same directory, Path.replace() for atomicity"
  - "Metadata versioning: version field check in load_metadata()"
  - "Scaler persistence: torch.save with pickle_protocol=5, torch.load with weights_only=False"
requirements-completed: [MOD-01]
duration: 20min
completed: 2026-05-29
---

# Phase 07 Plan 01: Cache Utility Module Summary

**Cache utility module with deterministic key derivation, atomic file I/O, metadata versioning, and sklearn scaler persistence via torch.save/load**

## Performance

- **Duration:** 20 min
- **Started:** 2026-05-29T11:32:23Z
- **Completed:** 2026-05-29T11:52:00Z
- **Tasks:** 2 (TDD: RED + GREEN)
- **Files modified:** 3

## Accomplishments
- Created `utils/cache.py` with 7 public functions + 1 constant
- 23 unit tests covering all cache functions with full pass rate
- Exported all cache symbols via `utils/__init__.py`
- TDD gate compliance: test commit (`59bc73c`) precedes feat commit (`a46ec51`)

## Task Commits

1. **Task 1: RED phase - failing tests** - `59bc73c` (test)
2. **Task 2: GREEN phase - cache implementation** - `a46ec51` (feat)

## Files Created/Modified
- `src/tscollection/datasets/utils/cache.py` - Cache utilities: build_cache_key, resolve_cache_dir, atomic_save_npz, atomic_save_metadata, load_metadata, save_scaler, load_scaler, CACHE_SCHEMA_VERSION
- `tests/test_cache.py` - 23 tests covering all public functions
- `src/tscollection/datasets/utils/__init__.py` - Added cache exports to barrel

## Decisions Made
- Used `Path.replace()` over `os.replace()` to satisfy ruff PTH105
- Used `Path.open()` over `open()` to satisfy ruff PTH123
- `np.savez_compressed` appends `.npz` to the path; wrote to temp stem then renamed the `.npz` file
- DatetimeIndex serialization uses `as_unit('ns').view(np.int64)` for pandas 3.0 compatibility (internal unit is `us` not `ns`)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed build_cache_key to include hash prefix in return value**
- **Found during:** Task 2 (GREEN phase)
- **Issue:** Initial implementation computed `hash_prefix` but did not prepend it to the return string
- **Fix:** Changed `suffix_parts = [dataset_name]` to `suffix_parts = [hash_prefix, dataset_name]`
- **Files modified:** src/tscollection/datasets/utils/cache.py
- **Committed in:** a46ec51

**2. [Rule 1 - Bug] Fixed atomic_save_npz for np.savez_compressed .npz suffix behavior**
- **Found during:** Task 2 (GREEN phase)
- **Issue:** `np.savez_compressed` always appends `.npz` to the given path
- **Fix:** Write temp to stem (no suffix) then rename the resulting `.npz` file
- **Files modified:** src/tscollection/datasets/utils/cache.py
- **Committed in:** a46ec51

**3. [Rule 1 - Bug] Added parent directory creation in atomic save functions**
- **Found during:** Task 2 (GREEN phase)
- **Issue:** `Path.replace()` failed when parent directory did not exist
- **Fix:** Added `path.parent.mkdir(parents=True, exist_ok=True)` before each save
- **Files modified:** src/tscollection/datasets/utils/cache.py
- **Committed in:** a46ec51

**4. [Rule 1 - Bug] Fixed DatetimeIndex serialization for pandas 3.0**
- **Found during:** Task 2 (GREEN phase)
- **Issue:** `astype(np.int64)` returns microseconds in pandas 3.0
- **Fix:** Changed to `as_unit('ns').view(np.int64)` for nanosecond-precision serialization
- **Files modified:** tests/test_cache.py
- **Committed in:** a46ec51

---

**Total deviations:** 4 auto-fixed (Rule 1 - bugs)
**Impact on plan:** All fixes necessary for correctness. No scope creep.

## TDD Gate Compliance
- RED gate: `59bc73c` (test commit) -- 23 tests, all failing due to missing module
- GREEN gate: `a46ec51` (feat commit) -- 23 tests, all passing

## Next Phase Readiness
- Cache module is fully tested and linted clean
- Foundation for plans 07-02 through 07-07 which will wire cache into the module lifecycle

---
*Phase: 07-ddp-compliance*
*Completed: 2026-05-29*
