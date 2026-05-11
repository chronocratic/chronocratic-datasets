---
phase: 02
plan: 00
subsystem: dataset-utilities-test-infrastructure
tags:
  - utilities
  - transformations
  - pytest-fixtures
  - test-stubs
dependency_graph:
  requires: []
  provides:
    - transform_utilities (convert_numpy_to_tensor, convert_data_to_np_array, expand_data_dimensionality)
    - compose_utilities (compose, FunctionComposer, get_num_samples_from_ts)
    - pytest_fixtures (synthetic_classification_df, synthetic_classification_labels, synthetic_forecast_data, synthetic_multivariate_data)
    - test_stubs (6 empty test files for Nyquist Rule compliance)
  affects:
    - tscollection.datasets.utils (re-export surface)
metrics:
  duration: 131s
  completed_date: "2026-05-11"
  tasks_completed: 3
  files_created: 11
  tests_added: 10
key_decisions:
  - Ported only compose, FunctionComposer, get_num_samples_from_ts from rbspaper/common.py per D-03; deferred load_json, flatten_list, AccumulatingTimerCallback to Phase 5
  - Added input validation to convert_numpy_to_tensor (TypeError for non-ndarray) per threat model T-02-00-01
  - Added bounds checking to expand_data_dimensionality per threat model T-02-00-02
  - Test stubs use module docstrings + single placeholder test pattern for Nyquist Rule compliance
---

# Phase 2 Plan 00: Utilities and Test Infrastructure Summary

Ported transform utilities from rbspaper (convert_numpy_to_tensor, convert_data_to_np_array, expand_data_dimensionality, compose, FunctionComposer, get_num_samples_from_ts) with input validation, created 4 synthetic pytest fixtures with shape/dtype tests, and added 6 test stub files for Nyquist Rule compliance across all Phase 2 dataset tests.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Port transform utilities | c12f455 | transformations.py, utils/common.py, utils/__init__.py |
| 2 | conftest.py fixtures + tests | 97a34a0 | tests/conftest.py, tests/test_conftest_fixtures.py |
| 3 | Test stub files | ca5bd82 | 6 test files under tests/ |

## Verification

- `uv run python -c "from tscollection.datasets.utils import compose, FunctionComposer, get_num_samples_from_ts, convert_numpy_to_tensor, convert_data_to_np_array, expand_data_dimensionality; print('OK')"` -- OK
- `uv run pytest tests/test_conftest_fixtures.py -x` -- 4 passed
- `uv run pytest tests/ -q --tb=short` -- 16 passed (6 Phase 1 + 4 fixtures + 6 stubs)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed DataFrame dtype check in fixture test**
- **Found during:** Task 2 verification
- **Issue:** `synthetic_classification_df.dtype` crashes because DataFrames use `.dtypes` (plural), not `.dtype`
- **Fix:** Changed to `all(d == np.float32 for d in synthetic_classification_df.dtypes)`
- **Files modified:** tests/test_conftest_fixtures.py
- **Commit:** 97a34a0

### Security Hardening (Threat Model)

**2. [Rule 2 - Missing validation] Added input type check to convert_numpy_to_tensor**
- **Threat:** T-02-00-01 (Spoofing)
- **Issue:** Original rbspaper code had no isinstance check on numpy input
- **Fix:** Added TypeError for non-ndarray inputs with helpful message
- **Files modified:** src/tscollection/datasets/datasets/transformations.py
- **Commit:** c12f455

**3. [Rule 2 - Missing validation] Added axis bounds check to expand_data_dimensionality**
- **Threat:** T-02-00-02 (Denial of Service)
- **Issue:** Original rbspaper code did not validate expand_dims_axis range
- **Fix:** Added ValueError when axis is out of range [0, ndim]
- **Files modified:** src/tscollection/datasets/datasets/transformations.py
- **Commit:** c12f455

## Known Stubs

None -- all test stubs are intentional placeholders documented with their scope in module docstrings.

## Threat Flags

None -- all threat model items (T-02-00-01, T-02-00-02) were mitigated; T-02-00-03 (FunctionComposer) was marked accept in the threat model and requires no action.
