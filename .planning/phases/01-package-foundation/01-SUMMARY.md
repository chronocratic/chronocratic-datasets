---
phase: 01-package-foundation
plan: 01
status: complete
completed_at: 2026-05-08
tasks_completed: 5
tests_passing: 6
one_liner: Package foundation — pyproject.toml config, 5 StrEnum classes, 9 __init__.py files with __all__ declarations, pytest setup
---

## Phase 1: Package Foundation — Summary

### Completed Tasks

- **Task 1:** Update pyproject.toml — added pydantic, scikit-learn, requests; broadened lightning/torch version constraints; fixed pytest config (testpaths, pythonpath)
- **Task 2:** Port enums from rbspaper — 5 StrEnum classes (TimeSeriesDatasetMode, SplittingStrategy, ScalingMethod, ForecastingMode, DistanceMetric) with clean tsdatasets imports
- **Task 3:** Create __init__.py skeleton — 9 files at all package levels with __all__ declarations
- **Task 4:** Package verification tests — 6 tests covering PKG-01, PKG-02, PKG-03
- **Task 5:** Reinstall and verify — `uv pip install -e .` + all tests pass, lint clean

### Key Decisions

- Renamed rbspaper enums for clarity (e.g., SplittingStrategy instead of TimeSeriesClassificationDatasetSplittingStrategy)
- Added ScalingMethod enum (new, not in rbspaper) for project constraint
- Broadened lightning to >=2.5,<3.0 and torch to >=2.4,<3.0
- Use tests/ directory (not test/) following pytest convention
- pythonpath = ["."] only — no redundant "src" with src-layout editable install

### Verification

- 6 tests passing (test_package.py)
- ruff lint clean on src/tsdatasets/ and tests/
- `import tsdatasets` works and exposes __version__ and enum types
- All __init__.py files have __all__ declarations
- No circular imports
