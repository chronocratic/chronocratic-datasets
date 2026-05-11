---
phase: 03-pydantic-registry
plan: 01
subsystem: config
tags: [pydantic, strenum, frozen-models, field-validator, model-validator, httpurl]

requires:
  - phase: 01-package-foundation
    provides: Package structure, existing enums (StrEnum pattern), config/__init__.py stub
  - phase: 02-dataset-classes
    provides: Enum usage patterns, test infrastructure (conftest.py)
provides:
  - DatasetFamily and SplitMode StrEnum definitions
  - Frozen nested Pydantic models (ArffFilePattern, ClassificationFilePatterns)
  - Abstract DatasetConfig base with common fields and validators
  - ClassificationConfig intermediate with target_col_name, file_patterns, split_strategy
  - ForecastingConfig intermediate with split_mode, split_bounds, default_seq_len, default_horizon
  - Config test fixtures (sample_classification_config, sample_forecasting_config, sample_fractional_config)
affects: [03-02, 03-03, 03-04, 04-download, 05-modules, 06-factory]

tech-stack:
  added: []
  patterns:
    - "Layered frozen inheritance: DatasetConfig -> ClassificationConfig/ForecastingConfig -> family leaves"
    - "Nested Pydantic models for deep immutability (ArffFilePattern, ClassificationFilePatterns)"
    - "StrEnum for typed parameters (DatasetFamily, SplitMode)"
    - "@field_validator for single-field checks, @model_validator for cross-field validation"
    - "abc.ABC + abstract method to prevent DatasetConfig direct instantiation"

key-files:
  created:
    - src/tscollection/datasets/config/base.py
    - tests/test_config_enums.py
    - tests/test_config_base.py
  modified:
    - src/tscollection/datasets/enums/data.py
    - src/tscollection/datasets/enums/__init__.py
    - tests/conftest.py

key-decisions:
  - "All 8 rbspaper families in DatasetFamily enum (including exchange, traffic, illness) to prevent validation errors"
  - "Abstract _config_validate method on DatasetConfig to enforce ABC semantics with Pydantic"
  - "split_bounds typed as union (tuple[int,...] | tuple[float,...]) with runtime validation via model_validator"
  - "cache_key property computed from url and sha256 for Phase 4 download caching"

patterns-established:
  - "Frozen nested models: ArffFilePattern and ClassificationFilePatterns inherit ConfigDict(frozen=True)"
  - "Cross-field validators on intermediate classes (not base) to avoid Pitfall 5 (AttributeError on subclass-only fields)"
  - "Google-style docstrings with Args section on all config classes"

requirements-completed: [CFG-01, CFG-02, CFG-03]

duration: 5min
completed: 2026-05-11
---

# Phase 3 Plan 01: Enum Foundations and Config Hierarchy Summary

**DatasetFamily/SplitMode StrEnum enums, frozen Pydantic config hierarchy (DatasetConfig, ClassificationConfig, ForecastingConfig) with nested models and validators.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-11T11:42:18Z
- **Completed:** 2026-05-11T11:47:15Z
- **Tasks:** 3 (TDD: RED/GREEN for tasks 1 and 2, direct for task 3)
- **Files modified:** 6

## Accomplishments

- DatasetFamily (8 members) and SplitMode (2 members) StrEnum enums, exported from `tscollection.datasets.enums`
- Abstract `DatasetConfig` base with frozen Pydantic models, HttpUrl validation, sha256 format checks, and cache_key property
- `ClassificationConfig` intermediate enforcing data_form, with nested frozen file pattern models
- `ForecastingConfig` intermediate with dual-mode split validation (indexed integers, fractional sums)
- 27 new tests (7 enum + 20 config base) and 3 reusable fixtures for later plans

## Task Commits

Each task was committed atomically:

1. **Task 1: Add DatasetFamily and SplitMode enums** (TDD)
   - `c3de826` — test: add failing tests for DatasetFamily and SplitMode enums
   - `6159758` — feat: add DatasetFamily and SplitMode enums

2. **Task 2: Create config/base.py with hierarchy and nested models** (TDD)
   - `8a8b25c` — test: add failing tests for config base hierarchy
   - `81a2218` — feat: implement config base hierarchy with frozen Pydantic models

3. **Task 3: Update tests/conftest.py with config fixtures**
   - `dae1288` — feat: add config fixtures to conftest.py

## Files Created/Modified

- `src/tscollection/datasets/enums/data.py` — Added DatasetFamily (8 members) and SplitMode (2 members) StrEnum classes
- `src/tscollection/datasets/enums/__init__.py` — Exported new enums (alphabetical order)
- `src/tscollection/datasets/config/base.py` — NEW: ArffFilePattern, ClassificationFilePatterns, DatasetConfig (ABC), ClassificationConfig, ForecastingConfig with validators
- `tests/test_config_enums.py` — NEW: 7 tests for enum members, serialization, and importability
- `tests/test_config_base.py` — NEW: 20 tests for frozen behavior, validators, model_copy, cache_key
- `tests/conftest.py` — Added 3 config fixtures: sample_classification_config, sample_forecasting_config, sample_fractional_config

## Decisions Made

- **8-family DatasetFamily:** Included exchange, traffic, illness enum values (beyond the 5 core families) to prevent validation errors when those configs are created in future plans. Per RESEARCH.md Open Question 1 recommendation.
- **Abstract method for ABC semantics:** Added `_config_validate()` abstract method on DatasetConfig because `abc.ABC` alone does not block instantiation when Pydantic's `BaseModel` is involved — Pydantic's validation fires before ABC's check. This ensures `TypeError` on direct instantiation.
- **Union type for split_bounds:** Used `tuple[int, ...] | tuple[float, ...]` with a `@model_validator` for runtime type checking. Cleaner than separate indexed/fractional models since they share the same conceptual role.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Added abstract method to enforce ABC semantics**

- **Found during:** Task 2 (GREEN phase — `test_cannot_instantiate_directly` failed)
- **Issue:** `DatasetConfig` with `abc.ABC` mixin raised `ValidationError` (from missing required fields) instead of `TypeError` (from abstract class) when instantiated directly. Pydantic's `BaseModel` validation fires before ABC's instantiation guard, and without abstract methods, `abc.ABC` does not block `DatasetConfig()`.
- **Fix:** Added `@abc.abstractmethod def _config_validate(self) -> None` on DatasetConfig. Implemented as `pass` in both ClassificationConfig and ForecastingConfig. This makes ABC's guard trigger before Pydantic validation.
- **Files modified:** `src/tscollection/datasets/config/base.py`
- **Verification:** Test `test_cannot_instantiate_directly` now passes with `TypeError`
- **Committed in:** `81a2218` (part of Task 2 GREEN commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** The fix strengthens the abstract base class contract. No scope creep.

## Issues Encountered

- Initial HEAD check script failed due to short-vs-long git hash comparison in the merge-base assertion. The reset itself was correct; re-verified manually before proceeding.

## Known Stubs

None — all implementations are complete for this plan's scope.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: T-03-01 | src/tscollection/datasets/config/base.py | HttpUrl on DatasetConfig.url mitigates URL injection at model construction time |
| threat_flag: T-03-02 | src/tscollection/datasets/config/base.py | frozen=True on all models prevents runtime config tampering |

## Next Phase Readiness

- Enums are stable and importable. Plan 03-02 (classification configs) can inherit from ClassificationConfig.
- Base hierarchy is frozen and validated. Plan 03-03 (forecasting configs) can inherit from ForecastingConfig.
- Fixtures provide ready-made config instances for test reuse in later plans.

---
*Phase: 03-pydantic-registry*
*Completed: 2026-05-11*
