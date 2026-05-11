---
phase: 02-dataset-classes
plan: 03
subsystem: dataset-core
tags:
  - pytorch-dataset
  - thin-wrapper
  - ucr
  - uea
  - ett
  - forecasting
  - classification
  - transformation
dependency_graph:
  requires:
    - phase: 02-dataset-classes
      plan: 02-00
      provides: "Transform utilities (convert_numpy_to_tensor, expand_data_dimensionality)"
    - phase: 02-dataset-classes
      plan: 02-02
      provides: "Dataset ABCs (FixedTimeSeriesDataset, FlexibleTimeSeriesDatasetSingleFile)"
  provides:
    - UCRClassificationUnivariateDataset (UCR univariate classification wrapper)
    - UEAClassificationMultivariateDataset (UEA multivariate classification wrapper)
    - ETTDataset (ETT forecasting wrapper with strategy injection)
    - datasets/__init__.py with 16-entry __all__ (all ABCs + wrappers + strategies)
    - Wrapper tests (test_ucr_dataset.py, test_ett_dataset.py, test_transformations.py)
  affects:
    - Phase 5 (modules — consume these wrappers)
    - Phase 6 (factory API — registry-driven instantiation)
tech-stack:
  added: []
  patterns:
    - Thin wrapper: UCR/UEA/ETT set domain defaults and delegate to ABC base
    - Strategy injection: ETTDataset creates ForecastingStrategySingleFile internally
key-files:
  created:
    - src/tscollection/datasets/datasets/ucr.py
    - src/tscollection/datasets/datasets/uea.py
    - src/tscollection/datasets/datasets/ett.py
  modified:
    - src/tscollection/datasets/datasets/__init__.py
    - tests/test_ucr_dataset.py
    - tests/test_ett_dataset.py
    - tests/test_transformations.py
key-decisions:
  - "Added forecast_horizon > 0 validation in ETTDataset (T-02-03-02 mitigation)"
  - "Followed rbspaper wrapper signatures verbatim with tscollection.datasets imports"
  - "UCR uses expand_dims_axis=1, UEA uses expand_dims_axis=None (no expansion for 3D arrays)"
requirements-completed:
  - DST-01
  - DST-02
  - DST-05
metrics:
  duration: 3min
  completed_date: "2026-05-11"
  tasks_completed: 3
  files_created: 3
  files_modified: 4
  tests_added: 10
---

# Phase 2 Plan 03: Concrete Dataset Wrappers Summary

Created three thin wrapper datasets (UCR, UEA, ETT) that set domain-specific defaults and delegate to the ABC bases, finalized the datasets/__init__.py with 16-entry exports, and added 10 verification tests covering wrapper behavior and transformation utilities.

## Performance

- **Duration:** ~3 min
- **Started:** 2026-05-11T09:57:06Z
- **Completed:** 2026-05-11T10:00:01Z
- **Tasks:** 3 (Task 1: TDD wrappers, Task 2: Export wiring, Task 3: Transformation tests)
- **Files created:** 3 (ucr.py, uea.py, ett.py)
- **Files modified:** 4 (__init__.py, test_ucr_dataset.py, test_ett_dataset.py, test_transformations.py)

## Accomplishments

- UCRClassificationUnivariateDataset: expand_dims_axis=1, yields (Tensor, int) for classification
- UEAClassificationMultivariateDataset: expand_dims_axis=None, yields (Tensor, label) for 3D arrays
- ETTDataset: injects ForecastingStrategySingleFile, yields (input, target) for forecasting
- datasets/__init__.py exports 16 classes (13 ABCs + strategies + 3 wrappers)
- 10 passing tests: 3 UCR wrapper, 3 ETT wrapper, 4 transformation utilities

## Task Commits

Each task was committed atomically:

1. **Task 1: Create concrete dataset wrappers (TDD)** - `5a4278b` (test: RED) + `0044cbd` (feat: GREEN)
2. **Task 2: Finalize export wiring** - `b0b6afc` (feat)
3. **Task 3: Implement transformation tests** - `f0faa75` (test)

## Files Created/Modified

- `src/tscollection/datasets/datasets/ucr.py` - UCRClassificationUnivariateDataset (42 lines, thin wrapper)
- `src/tscollection/datasets/datasets/uea.py` - UEAClassificationMultivariateDataset (42 lines, thin wrapper)
- `src/tscollection/datasets/datasets/ett.py` - ETTDataset (57 lines, with forecast_horizon validation)
- `src/tscollection/datasets/datasets/__init__.py` - 16-entry __all__, alphabetically sorted
- `tests/test_ucr_dataset.py` - 3 tests: yields data/label, without labels, length
- `tests/test_ett_dataset.py` - 3 tests: yields input/target, length, forecast horizon
- `tests/test_transformations.py` - 4 tests: float tensor, long tensor, expand dims, list to array

## Decisions Made

- Followed rbspaper wrapper signatures exactly (ucr_dataset.py, uea_dataset.py, ett_dataset.py)
- Used `from __future__ import annotations` in all 3 wrapper files (rbspaper convention)
- ETTDataset accepts `forecast_horizon` as a simple parameter (not user-controllable in malicious ways)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed UCR test expected shape**
- **Found during:** Task 1 (GREEN phase)
- **Issue:** Test expected sample.shape = (1, 50) but actual shape is (50, 1) because expand_dims_axis=1 adds a dimension at axis 1, not axis 0
- **Fix:** Updated test assertion to match actual rbspaper behavior: shape (50, 1)
- **Files modified:** tests/test_ucr_dataset.py
- **Commit:** 0044cbd

**2. [Rule 2 - Missing validation] Added forecast_horizon validation in ETTDataset**
- **Found during:** Task 1 (implementation)
- **Issue:** Threat model T-02-03-02 requires validating forecast_horizon > 0
- **Fix:** Added ValueError check in ETTDataset.__init__ when forecast_horizon <= 0
- **Files modified:** src/tscollection/datasets/datasets/ett.py
- **Commit:** 0044cbd

---

**Total deviations:** 2 auto-fixed (1 Rule 1 bug fix, 1 Rule 2 threat model hardening)
**Impact on plan:** Both fixes required for correctness; no scope creep.

## Issues Encountered

None — all tasks completed as specified.

## Known Stubs

None — all wrapper files are complete implementations, no placeholders.

## Threat Flags

None — all threat model items addressed:
- T-02-03-01 (UCR type validation): mitigated by inherited base class checks
- T-02-03-02 (forecast_horizon validation): mitigated via explicit ValueError check
- T-02-03-03 (transformations_sequence injection): accepted as per plan

## Next Phase Readiness

- All concrete dataset wrappers are complete and tested
- datasets/__init__.py exports 16 classes (all ABCs + wrappers + strategies)
- Full test suite passes (35 tests): `uv run pytest tests/ -x --tb=short -v`
- Ready for Phase 3 (Pydantic Registry) which will configure these wrappers via typed configs

## Self-Check

### Created Files
- [PASS] ucr.py exists with UCRClassificationUnivariateDataset class (42 lines, minimum 14)
- [PASS] uea.py exists with UEAClassificationMultivariateDataset class (42 lines, minimum 14)
- [PASS] ett.py exists with ETTDataset class (57 lines, minimum 30)

### Exports
- [PASS] datasets/__init__.py has __all__ with 16 entries
- [PASS] UCRClassificationUnivariateDataset importable from tscollection.datasets.datasets
- [PASS] ETTDataset importable from tscollection.datasets.datasets
- [PASS] UEAClassificationMultivariateDataset importable from tscollection.datasets.datasets

### Tests
- [PASS] test_ucr_dataset.py: 3 tests pass
- [PASS] test_ett_dataset.py: 3 tests pass
- [PASS] test_transformations.py: 4 tests pass
- [PASS] Full suite: 35 tests pass

### Commits
- [PASS] 5a4278b: test(02-03): add failing tests for concrete dataset wrappers
- [PASS] 0044cbd: feat(02-03): create concrete dataset wrappers
- [PASS] b0b6afc: feat(02-03): finalize datasets/__init__.py exports
- [PASS] f0faa75: test(02-03): implement transformation utility tests

### TDD Gate Compliance
- [PASS] Task 1: RED commit (5a4278b) before GREEN commit (0044cbd)

---
*Phase: 02-dataset-classes*
*Completed: 2026-05-11*
