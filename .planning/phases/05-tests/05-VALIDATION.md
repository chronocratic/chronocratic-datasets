---
name: 05-validation
description: Validation architecture for Phase 05 Tests
metadata:
  type: validation
  phase: 05
  date: 2026-05-13
---

# Phase 05: Tests - Validation Architecture

## Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 with pytest-cov (coverage 7.13.5) |
| Config file | `pyproject.toml` — testpaths=["tests"], pythonpath=["."] |
| Quick run | `uv run pytest tests/test_modules_forecasting.py -x` |
| Full suite | `uv run pytest --cov=tscollection.datasets --cov-report=term-missing` |

## Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command |
|--------|----------|-----------|-------------------|
| TST-01 | Dataset shapes/types correct | Unit | `uv run pytest tests/test_fixed_dataset.py tests/test_flexible_dataset.py -x` |
| TST-01 | Dataloader batch shapes | Integration | `uv run pytest tests/test_modules_forecasting.py -k "integration" -x` |
| TST-02 | Module properties after prepare_data | Integration | `uv run pytest tests/test_modules_forecasting.py -k "golden" -x` |
| TST-02 | num_features, num_time_series_features | Unit | `uv run pytest tests/test_modules_forecasting.py -k "setup" -x` |
| TST-03 | Utility functions output | Unit | `uv run pytest tests/test_transformations.py tests/test_utils_scaling.py -x` |
| TST-03 | transformations.py error paths | Unit | `uv run pytest tests/test_transformations.py -x` |

## Sampling Rate

- **Per task:** `uv run pytest tests/test_modules_forecasting.py -x`
- **Per wave:** `uv run pytest --cov=tscollection.datasets --cov-report=term-missing`
- **Phase gate:** All green + coverage >= 92%

## Coverage Targets

| Module | Current | Target |
|--------|---------|--------|
| forecasting.py | 49% | 85%+ |
| ett.py | 67% | 85%+ |
| weather.py | 73% | 85%+ |
| electricity.py | 88% | 95%+ |
| transformations.py | 75% | 85%+ |
| **TOTAL** | **87%** | **92%+** |
