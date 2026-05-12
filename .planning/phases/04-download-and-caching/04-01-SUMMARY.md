---
phase: 04-download-and-caching
plan: 01
subsystem: download
tags: [requests, urllib3, hashlib, zipfile, caching, sha256, streaming]

requires:
  - phase: 03-pydantic-registry
    provides: Config classes with url/sha256 fields consumed by download module
  - phase: 04-download-and-caching
    plan: 00
    provides: Test infrastructure (mock_http_server, tmp_cache_dir fixtures) and 13 cache tests
provides:
  - get_cache_dir: Returns cache root with TSCOLLECTION_CACHE_DIR override
  - download_file: Streaming HTTP download with SHA256 validation and retry
  - file_exists_in_cache: Cache hit detection with optional hash verification
  - extract_archive: Safe ZIP extraction using zipfile
  - clear_cache_dir: Full dataset subdirectory removal via shutil.rmtree
  - Public API re-exports from download/__init__.py
affects: [04-02, 05-data-modules]

tech-stack:
  added: []
  patterns:
    - "requests.Session with urllib3.Retry for exponential backoff"
    - "Streaming SHA256 via hashlib during download"
    - "zipfile.ZipFile.extractall() for safe ZIP extraction"
    - "shutil.rmtree for full cache subdirectory removal"

key-files:
  created:
    - src/tscollection/datasets/download/cache.py
  modified:
    - src/tscollection/datasets/download/__init__.py

key-decisions:
  - "Using requests.Session (not urllib.request) per D-02 override -- requests already a project dependency"
  - "Keyword-only args for all public functions per project conventions"
  - "32 KB chunk size matching torchvision default for streaming downloads"
  - "file_exists_in_cache takes (cache_dir, sha256) to match test fixture interfaces"

patterns-established:
  - "Cache hit detection: check file existence + SHA256 match before HTTP call"
  - "Overwrite cache: unlink existing file before redownload"
  - "Warning on missing SHA256: log.warning when sha256=None"

requirements-completed: [DL-01, DL-02, DL-03, DL-04]

duration: 10min
completed: 2026-05-12
---

# Phase 4 Plan 01: Download Cache Primitives Summary

**Core download primitives with streaming SHA256 validation, exponential backoff retry via urllib3, safe ZIP extraction, and cache hit detection -- all 13 cache tests passing**

## Performance

- **Duration:** 10 min
- **Started:** 2026-05-12T09:29:41Z
- **Completed:** 2026-05-12T09:40:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Implemented cache.py with 5 public functions and 2 private helpers for download, caching, and extraction
- Updated download/__init__.py with proper re-exports following utils/__init__.py pattern
- All 13 cache tests pass; full regression suite (209 tests) green

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement cache.py with download primitives** - `12f64b1` (feat) -- 5 public + 2 private functions
2. **Task 2: Update download/__init__.py with re-exports** - `890f383` (feat) -- public API surface

## Files Created/Modified
- `src/tscollection/datasets/download/cache.py` -- Download primitives: get_cache_dir, download_file, file_exists_in_cache, extract_archive, clear_cache_dir with streaming SHA256, retry backoff, and safe extraction
- `src/tscollection/datasets/download/__init__.py` -- Re-exports all 5 public cache functions

## Decisions Made
- requests.Session with urllib3.Retry(total=3, backoff_factor=1) for exponential backoff (1s/2s/4s delays)
- 32 KB chunk size for streaming downloads (torchvision default)
- zipfile.ZipFile.extractall() for safe extraction (Python 3.12 zip-slip protection)
- shutil.rmtree for clear_cache_dir (full subdirectory removal per D-01)
- Module-level logger with `logging.getLogger(__name__)` for structured download logs

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] file_exists_in_cache signature adjusted for test compatibility**
- **Found during:** Task 1 (cache.py implementation)
- **Issue:** Plan specified `file_exists_in_cache(dataset_name, sha256_expected)` but tests use `file_exists_in_cache(cache_dir, sha256)` to work with tmp_cache_dir fixture
- **Fix:** Implemented with test-aligned signature: `cache_dir: Path, sha256: str | None`
- **Files modified:** src/tscollection/datasets/download/cache.py
- **Verification:** All 13 cache tests pass
- **Committed in:** 12f64b1

**2. [Rule 2 - Missing Critical] Plan stated 14 tests; test file has 13**
- **Found during:** Task 1 verification
- **Issue:** Wave 0 created test_download_cache.py with 13 tests (3 + 6 + 2 + 2), not 14 as stated in plan
- **Fix:** No code change needed -- plan text was slightly off. All existing tests pass.
- **Verification:** `pytest tests/test_download_cache.py` collects 13 items, all pass

---

**Total deviations:** 2 (1 signature alignment, 1 documentation accuracy)
**Impact on plan:** Signature change required for test compatibility. No scope creep.

## Issues Encountered
None -- implementation followed research patterns closely.

## Next Phase Readiness
- Cache primitives are complete and tested
- Plan 04-02 (family-specific downloaders: ucr_uea.py, forecasting.py) can now consume these primitives
- No blockers for wave 2

---
*Phase: 04-download-and-caching*
*Completed: 2026-05-12*
