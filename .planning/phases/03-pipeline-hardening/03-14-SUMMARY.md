---
plan: 03-14
phase: 03-pipeline-hardening
status: complete
date: "2026-05-06"
self_check: passed
---

## Summary

Added the missing `force=True` resume override integration test. The other two must-haves (skip completed steps, run-all-without-state) were already covered by existing tests in `test_pipeline_core.py`.

## Key Files

| File | Action |
|------|--------|
| test/test_pipeline_core.py | modified |
| ruff.toml | modified |

## What Was Built

- Added `test_force_true_runs_all_steps_despite_previous_state` — verifies force=True ignores previous_state and runs all steps including train
- Extended ruff.toml per-file-ignores for test files (ARG001, NPY002, F401)

## Self-Check

- [x] `test_force_true_runs_all_steps_despite_previous_state` passes
- [x] Full suite: 28 tests in test_pipeline_core.py pass
- [x] ruff clean on both test files
- [x] ty clean on both test files
- [x] All 3 must-haves verified

## Deviations

Tests placed in `test_pipeline_core.py` rather than `test_pipeline_state.py` as the plan specified. Resume flow tests inherently test `run_experiment_pipeline` from core.py, and the existing test infrastructure (fakes, fixtures, model/data doubles) lives in core.py. Colocating the new test avoids duplicating infrastructure.
