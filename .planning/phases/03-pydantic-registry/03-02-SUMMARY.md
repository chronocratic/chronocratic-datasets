---
phase: 03-pydantic-registry
plan: 02
subsystem: config
tags: [pydantic, frozen-models, classification, ucr, uea, nested-models]

requires:
  - phase: 03-pydantic-registry
    plan: 01
    provides: ClassificationConfig base, ArffFilePattern, ClassificationFilePatterns, DatasetFamily enum, SplittingStrategy enum
provides:
  - UCRConfig class with 3 frozen instances (Coffee, ECG200, FaceFour)
  - UEAConfig class with 2 frozen instances (BasicMotions, AtrialFibrillation)
  - data_form defaults: 'regular' for UCR, 'nested' for UEA
  - 45 total tests (26 UCR + 19 UEA)
affects: [03-03, 03-04, 04-download, 05-modules, 06-factory]

tech-stack:
  added: []
  patterns:
    - "Family-specific config: UCRConfig(UCR default, data_form='regular'), UEAConfig(UEA default, data_form='nested')"
    - "Shared frozen file patterns module constant to avoid per-instance allocation"

key-files:
  created:
    - src/tscollection/datasets/config/ucr.py
    - src/tscollection/datasets/config/uea.py
    - tests/test_config_ucr.py
    - tests/test_config_uea.py
    - src/tscollection/datasets/config/base.py (prerequisite from plan 01)
  modified:
    - src/tscollection/datasets/enums/data.py (prerequisite: DatasetFamily, SplitMode)
    - src/tscollection/datasets/enums/__init__.py (prerequisite: new exports)

key-decisions:
  - "data_form as class-level default (not @property) to avoid Pydantic field conflict"
  - "Shared file pattern constant (_UCR_FILE_PATTERNS, _UEA_FILE_PATTERNS) for memory efficiency"

patterns-established:
  - "Classification family configs: family default + data_form default + tasks default + num_classes Field(ge=1)"
  - "Module-level frozen file pattern constants shared across instances"

requirements-completed: [CFG-02]

duration: 7min
completed: 2026-05-11
---

# Phase 3 Plan 02: Classification Family Configs (UCR + UEA) Summary

**UCRConfig (3 instances) and UEAConfig (2 instances) with frozen Pydantic models, nested file patterns, and 45 tests covering inheritance, immutability, field values, and model_copy.**

## Performance

- **Duration:** 7 min
- **Started:** 2026-05-11T12:07:22Z
- **Completed:** 2026-05-11T12:14:08Z
- **Tasks:** 2 plan tasks + 1 prerequisite setup
- **Files modified:** 7

## Accomplishments

- UCRConfig inherits ClassificationConfig with `data_form='regular'` default and 3 frozen instances (Coffee, ECG200, FaceFour) with correct num_classes
- UEAConfig inherits ClassificationConfig with `data_form='nested'` default and 2 frozen instances (BasicMotions, AtrialFibrillation) with correct num_classes
- 45 total tests (26 UCR + 19 UEA) covering inheritance, frozen behavior, field values, file_patterns structure, URL validation, and model_copy
- Full test suite (85 tests) passes with no regressions

## Task Commits

Each task was committed atomically:

1. **Prerequisite: Plan 01 dependency files** -- `195bfd6` (chore)
   - base.py, enums/data.py (DatasetFamily, SplitMode), enums/__init__.py

2. **Task 1: UCR classification config with 3 instances** (TDD)
   - `f5011c2` -- test: add failing tests for UCR classification config
   - `7bbe1bf` -- feat: implement UCR classification config with 3 dataset instances

3. **Task 2: UEA classification config with 2 instances** (TDD)
   - `376dac0` -- test: add failing tests for UEA classification config
   - `6c967e6` -- feat: implement UEA classification config with 2 dataset instances

## Files Created/Modified

- `src/tscollection/datasets/config/ucr.py` -- UCRConfig class + UCR_COFFEE, UCR_ECG200, UCR_FACE_FOUR instances
- `src/tscollection/datasets/config/uea.py` -- UEAConfig class + UEA_BASIC_MOTIONS, UEA_ATRIAL_FIBRILLATION instances
- `tests/test_config_ucr.py` -- 26 tests for UCR config structure, values, and behavior
- `tests/test_config_uea.py` -- 19 tests for UEA config structure, values, and behavior
- `src/tscollection/datasets/config/base.py` -- Prerequisite: DatasetConfig, ClassificationConfig, ForecastingConfig hierarchy (from plan 01)
- `src/tscollection/datasets/enums/data.py` -- Prerequisite: Added DatasetFamily (8 members) and SplitMode (2 members) (from plan 01)
- `src/tscollection/datasets/enums/__init__.py` -- Prerequisite: Exported new enums (from plan 01)

## Decisions Made

- **data_form as class-level default:** Plan specified `@property data_form` for UCRConfig/UEAConfig, but Pydantic v2's field system does not allow properties to override inherited model fields. The property descriptor was returned as a raw object instead of being invoked. Resolved by using a class-level field default (`data_form: str = 'regular'` / `'nested'`) which satisfies the parent's `ClassificationConfig` validator and provides the same runtime behavior.

- **Shared file pattern constants:** Both UCR and UEA use a single `_FILE_PATTERNS` module constant shared across all instances. Since `ClassificationFilePatterns` and `ArffFilePattern` are frozen Pydantic models, sharing the same object is safe and avoids per-instance allocation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Plan 01 dependency files missing from worktree**

- **Found during:** Task 1 setup
- **Issue:** Plan 01 (wave 1) committed base.py, DatasetFamily enum, and SplitMode enum on a separate worktree branch (`worktree-agent-ab3f959825022c155`) that was not merged to main. The worktree base (`ecd3be8`) did not include these files, blocking all plan 02 tasks.
- **Fix:** Created base.py, updated enums/data.py and enums/__init__.py with plan 01's output. Committed as prerequisite chore.
- **Files modified:** `src/tscollection/datasets/config/base.py`, `src/tscollection/datasets/enums/data.py`, `src/tscollection/datasets/enums/__init__.py`
- **Committed in:** `195bfd6`

**2. [Rule 1 - Bug] Pydantic property conflict with inherited field**

- **Found during:** Task 1 (GREEN phase)
- **Issue:** Plan specified `@property data_form` returning 'regular' on UCRConfig. However, the parent class `DatasetConfig` defines `data_form: str | None = None` as a Pydantic field. Pydantic's `__getattr__` returns the field value (the property descriptor object) instead of invoking the property. Tests showed `config.data_form` returning `<property object at 0x...>` instead of `'regular'`.
- **Fix:** Replaced `@property` with class-level field default: `data_form: str = 'regular'`. Same runtime behavior (instances return 'regular'), and the parent's validator (`data_form is None`) is satisfied.
- **Files modified:** `src/tscollection/datasets/config/ucr.py`, `src/tscollection/datasets/config/uea.py`, `tests/test_config_ucr.py`
- **Committed in:** `7bbe1bf` (Task 1 GREEN commit)

---

**Total deviations:** 2 auto-fixed (1 Rule 3 blocking, 1 Rule 1 bug)
**Impact on plan:** Both deviations were necessary for correctness. The dependency gap was a worktree isolation issue. The property-to-field adjustment preserves the plan's semantic intent (data_form is a computed/family-specific value) while working within Pydantic's field model.

## Issues Encountered

- Pydantic v2 does not support overriding an inherited model field with a `@property`. The property descriptor is stored as the field value rather than being invoked during attribute access. This is a known limitation when mixing Python descriptors with Pydantic's attribute resolution.

## Known Stubs

None -- all implementations are complete for this plan's scope.

## Threat Flags

| Flag | File | Description |
|------|------|-------------|
| threat_flag: T-03-05 | src/tscollection/datasets/config/ucr.py, uea.py | HttpUrl on all config instances validates archive URLs at construction time |
| threat_flag: T-03-06 | src/tscollection/datasets/config/ucr.py, uea.py | frozen=True inherited from base prevents runtime mutation of num_classes, file_patterns |

## Next Phase Readiness

- UCRConfig and UEAConfig are stable, frozen, and fully tested. Plan 03-03 (forecasting configs) can follow the same pattern.
- Plan 03-04 (factory.py registry) can import these instances for `get_config()` lookup.
- Phase 4 (download) can read `url` and `sha256` fields.
- Phase 5 (modules) can read `target_col_name`, `file_patterns`, `data_form`, `split_strategy`.

---
*Phase: 03-pydantic-registry*
*Completed: 2026-05-11*
