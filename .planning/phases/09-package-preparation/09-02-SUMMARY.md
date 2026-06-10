---
phase: 09
plan: 02
subsystem: packaging
tags: [license, metadata, pyproject, init, exports]
dependency_graph:
  requires: [09-01]
  provides: [package_metadata, license_file, citation_file, init_exports]
  affects: []
tech_stack:
  added: []
  patterns: [re-export, __all__]
key_files:
  created:
    - LICENSE
    - CITATION.cff
  modified:
    - pyproject.toml
    - src/chronocratic/datasets/__init__.py
    - uv.lock
decisions:
  - Removed License classifier from pyproject.toml — PEP 639 license expression supersedes it; 8 classifiers remain
  - All 49 symbols reconciled against actual submodule __all__ lists before writing __init__.py
metrics:
  duration: "10 minutes"
  completed: "2026-06-10T09:17:00Z"
---

# Phase 09 Plan 02: Package Metadata and Init Exports Summary

Completed package metadata (pyproject.toml), created LICENSE (BSD 3-Clause) and CITATION.cff, and expanded `__init__.py` to re-export all 49 public symbols from submodules.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create LICENSE and CITATION.cff | `0f3333e` | LICENSE, CITATION.cff |
| 2 | Complete pyproject.toml metadata | `2ecee22` | pyproject.toml, uv.lock |
| 3 | Expand __init__.py with re-exports | `17ba784` | src/chronocratic/datasets/__init__.py |

## Key Changes

**LICENSE**: BSD 3-Clause text with copyright "The Chronocratic Developers" 2024-2026.

**CITATION.cff**: CFF 1.2.0 metadata for `chronocratic-datasets` v0.1.0 with keywords (time-series, pytorch, lightning, machine-learning, datasets).

**pyproject.toml**: Added description, readme, license, license-files, keywords, authors, 8 classifiers, docs optional-dependency group (sphinx, pydata-sphinx-theme, myst-parser), and project URLs (Homepage, Documentation, Repository, Issues).

**`__init__.py`**: Expanded from a single-line docstring to a full package entry point with `__version__ = "0.1.0"` and 49 re-exported symbols across 5 categories (8 enums, 10 datatypes, 8 modules, 2 maps, 20 utils).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] License classifier conflicts with PEP 639 license expression**
- **Found during:** Task 2
- **Issue:** Including `"License :: OSI Approved :: BSD License"` classifier while using `license = "BSD-3-Clause"` causes setuptools to raise `InvalidConfigError` because the PEP 639 license expression supersedes trove license classifiers.
- **Fix:** Removed the license classifier. 8 classifiers remain (plan specified 9).
- **Files modified:** pyproject.toml
- **Commit:** `2ecee22`

## Verification

- `uv run pytest tests/test_package.py -v` — 6/6 passed
- `uv run pytest tests/test_enum_refactoring.py -v` — 21/21 passed
- `from chronocratic.datasets import ForecastingMode, ETTDataset, ETTDataModule, CLASSIFICATION_LOADER_MAP` — all resolve
- `len(chronocratic.datasets.__all__) == 49` — correct
- LICENSE and CITATION.cff exist with correct content

## Self-Check

**Result:** PASSED
- LICENSE: found
- CITATION.cff: found
- pyproject.toml: valid TOML with all metadata
- __init__.py: 130 lines, 49 symbols
- All commits found in git log
