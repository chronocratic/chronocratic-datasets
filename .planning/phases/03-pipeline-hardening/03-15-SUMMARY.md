---
plan: 03-15
phase: 03-pipeline-hardening
status: complete
date: "2026-05-06"
self_check: passed
---

## Summary

Added output structure tests, CLI argument tests, and logging test. Updated smoke test to assert `.pipeline_state.json` exists after pipeline run.

## Key Files

| File | Action |
|------|--------|
| test/test_pipeline_core.py | modified |
| test/test_pipeline_state.py | modified |
| ruff.toml | modified |

## What Was Built

- `test_experiment_config_written` — verifies `experiment_config.json` written by pipeline
- `test_state_file_written` — verifies `.pipeline_state.json` written by pipeline
- `test_dataset_index_parsing` — verifies `--dataset_index` resolves via `_parse_args`
- `test_mutually_exclusive_args` — verifies `_resolve_dataset` rejects both dataset args
- `test_logging_creates_file` — verifies `setup_logging` creates log directory and FileHandler
- Updated `test_pipeline_smoke_run` to assert `.pipeline_state.json` existence
- Runner loaded via `importlib.util.spec_from_file_location` (no `__init__.py` in `runners/py/`)

## Self-Check

- [x] 5 new tests pass
- [x] 30 tests in test_pipeline_core.py pass
- [x] 30 tests in test_pipeline_state.py pass
- [x] ruff clean on both test files
- [x] ty clean on both test files
- [x] All 5 must-haves verified
