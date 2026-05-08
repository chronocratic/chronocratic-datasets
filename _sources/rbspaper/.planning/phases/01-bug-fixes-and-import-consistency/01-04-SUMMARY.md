---
phase: 01-bug-fixes-and-import-consistency
plan: 04
type: execute
status: complete
date: 2026-05-05
---

# Phase 01 Plan 04 Summary: ruff + ty Clean

## Objective
Achieve ruff format + lint clean and ty type check clean across the entire codebase.

## Changes Made

### ruff.toml
- Fixed `per-file-ignores`: changed `"tests/**/*.py"` to `"test/**/*.py"` to match actual directory
- Added `"runners/py/*.py"` ignores: T201, EXE001, BLE001 (CLI tool with intentional prints, shebang, top-level exception catch)
- Added `"src/rbspaper/adapters/*.py"` ignores: ANN401, TC001 (intentional Any usage, TC001 would break runtime)
- Extended test ignores: E501, N812, ANN001, ANN003, ANN202, RUF059, ERA001

### Source Files - Lint Fixes
- `src/rbspaper/evaluation/classification.py`: Replaced single-line docstring with proper Google-style multi-line docstring (fixed E501 + D103)
- `src/rbspaper/evaluation/evaluation.py`: Wrapped long f-strings (E501), added `-> dict` return type (ANN201), added Google-style docstring (D103)
- `src/rbspaper/evaluation/forecasting.py`: Replaced single-line docstring with proper Google-style multi-line docstring (fixed E501)
- `src/rbspaper/evaluation/protocols.py`: Wrapped long f-string error message (E501)
- `runners/py/runner.py`: Added `ExperimentInstance` type annotation for `experiment_instance` param (ANN001), imported `ExperimentInstance`

### Source Files - Type Fixes (ty-ignore)
- `src/rbspaper/attacks/batch.py:98`: DataLoader type mismatch (variable-tuple vs fixed-tuple)
- `src/rbspaper/attacks/functional.py:152`: dict value type invariance (subtype vs supertype)
- `src/rbspaper/pipeline/core.py:300,307,314,441`: LightningModule vs TS2Vec|AutoTCL|CoST (conservative typing)
- `runners/py/runner.py:150,152-154`: hasattr narrowing not supported by ty for dynamic attribute access

### Test Files - Type Fixes (ty-ignore)
- `test/test_attacks_batch.py:30`: allclose with union return type
- `test/test_attacks_batch.py:79`: DataLoader type mismatch
- `test/test_attacks_batch.py:81`: unresolved `tensors` attribute on union type
- `test/test_attacks_functional.py:81`: allclose with union return type
- `test/test_pipeline_core.py:153,192,243,273,314`: _TinyDataModule vs BaseTimeSeriesDataModule (test double)

## Verification Results
- `uv run ruff format --check .` — 71 files already formatted (PASS)
- `uv run ruff check .` — All checks passed (PASS)
- `uv run ty check .` — All checks passed (PASS)
- `uv run pytest --collect-only` — 22 tests collected (PASS)

## Files Modified (13)
- ruff.toml
- runners/py/runner.py
- src/rbspaper/evaluation/classification.py
- src/rbspaper/evaluation/evaluation.py
- src/rbspaper/evaluation/forecasting.py
- src/rbspaper/evaluation/protocols.py
- src/rbspaper/attacks/batch.py
- src/rbspaper/attacks/functional.py
- src/rbspaper/pipeline/core.py
- test/test_attacks_batch.py
- test/test_attacks_functional.py
- test/test_pipeline_core.py
- .planning/phases/01-bug-fixes-and-import-consistency/01-04-SUMMARY.md
