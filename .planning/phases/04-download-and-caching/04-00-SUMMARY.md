---
phase: 04-download-and-caching
plan: 00
subsystem: testing
tags: [pytest, mock-server, http, cache, tdd, fixtures]

requires:
  - phase: 03-pydantic-registry
    provides: Config classes (ClassificationConfig, ForecastingConfig) used by downloader tests
provides:
  - mock_http_server fixture for network-free download testing
  - tmp_cache_dir fixture for isolated cache operations
  - 13 failing tests for cache primitives (test_download_cache.py)
  - 4 failing tests for UCR/UEA downloader (test_download_ucr_uea.py)
  - 3 failing tests for forecasting downloader (test_download_forecasting.py)
affects: [04-01, 04-02]

tech-stack:
  added: []
  patterns:
    - "Mock HTTP server fixture using threading.Thread + HTTPServer"
    - "Fixture test classes with requirement-ID prefixed docstrings"
    - "TDD RED state: tests written before implementation, fail on import"

key-files:
  created:
    - tests/test_download_cache.py
    - tests/test_download_ucr_uea.py
    - tests/test_download_forecasting.py
  modified:
    - tests/conftest.py
    - tests/test_conftest_fixtures.py

key-decisions:
  - "mock_http_server yields (base_url, file_hashes) tuple for clean fixture usage"
  - "test files import from non-existent modules intentionally (TDD RED state)"
  - "Requirement IDs (DL-01 through DL-04) used as docstring prefixes"

requirements-completed: [DL-01, DL-02, DL-03, DL-04]

duration: 15min
completed: 2026-05-12
---

# Phase 4 Plan 00: Download Test Infrastructure Summary

**Mock HTTP server fixture with SHA256 test files, tmp_cache_dir fixture, and 24 failing tests across 3 test files driving TDD for cache primitives and family-specific downloaders**

## Performance

- **Duration:** 15 min
- **Started:** 2026-05-12T09:16:40Z
- **Completed:** 2026-05-12T09:31:40Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- Added mock_http_server fixture: threaded HTTP server serving test_file.zip, test_file.csv, bad_file.zip with computed SHA256 hashes
- Added tmp_cache_dir fixture: isolated cache directory via TSCOLLECTION_CACHE_DIR env var
- Created test_download_cache.py with 13 tests covering get_cache_dir, download_file, file_exists_in_cache, and extract_archive
- Created test_download_ucr_uea.py with 4 tests for UCR/UEA classification downloader
- Created test_download_forecasting.py with 3 tests for forecasting CSV downloader

## Task Commits

Each task was committed atomically:

1. **Task 1: mock_http_server and tmp_cache_dir fixtures** - `1ffca82` (test) + `1b7ef3e` (feat) — TDD RED/GREEN
2. **Task 2: test_download_cache.py** - `42ba39a` (test) — 13 failing tests (RED state, module not yet created)
3. **Task 3: downloader test files** - `c0ed6aa` (test) — 7 failing tests (RED state, modules not yet created)

## Files Created/Modified
- `tests/conftest.py` — Added mock_http_server and tmp_cache_dir fixtures; moved config imports to module level
- `tests/test_conftest_fixtures.py` — Added 5 tests for new fixtures (TestMockHttpServer, TestTmpCacheDir)
- `tests/test_download_cache.py` — 13 tests across 4 classes: TestGetCacheDir (3), TestDownloadFile (6), TestFileExistsInCache (2), TestExtractArchive (2)
- `tests/test_download_ucr_uea.py` — 4 tests in TestDownloadUcrUea class
- `tests/test_download_forecasting.py` — 3 tests in TestDownloadForecasting class

## Decisions Made
- mock_http_server yields `(base_url, file_hashes)` tuple for straightforward unpacking in tests
- test_file.zip contains a single 'data.txt' for extraction verification
- bad_file.zip uses different content for SHA256 mismatch tests
- tmp_cache_dir uses monkeypatch.setenv (not os.environ) for automatic cleanup
- Requirement IDs prefixed in test docstrings (DL-01:, DL-02:, DL-03:, DL-04:)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Ruff import sorting required `from __future__ import annotations` before other imports
- TYPE_CHECKING blocks needed for Path imports when only used in annotations
- S108 lint rule flagged `/tmp/` paths; used `/cache/` for mock return values instead

## Next Phase Readiness
- Test infrastructure is complete — 24 total tests ready
- All tests fail on import (expected RED state) — Plans 04-01 and 04-02 will implement the modules
- Existing 189 tests remain green

---
*Phase: 04-download-and-caching*
*Completed: 2026-05-12*
