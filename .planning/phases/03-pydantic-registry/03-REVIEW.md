---
phase: 03-pydantic-registry
reviewed: 2026-05-11T18:30:00Z
depth: deep
files_reviewed: 21
files_reviewed_list:
  - src/tscollection/datasets/config/base.py
  - src/tscollection/datasets/enums/data.py
  - src/tscollection/datasets/enums/__init__.py
  - src/tscollection/datasets/config/ucr.py
  - src/tscollection/datasets/config/uea.py
  - src/tscollection/datasets/config/ett.py
  - src/tscollection/datasets/config/electricity.py
  - src/tscollection/datasets/config/weather.py
  - src/tscollection/datasets/config/factory.py
  - src/tscollection/datasets/config/__init__.py
  - src/tscollection/datasets/__init__.py
  - tests/conftest.py
  - tests/test_config_enums.py
  - tests/test_config_base.py
  - tests/test_config_ucr.py
  - tests/test_config_uea.py
  - tests/test_config_ett.py
  - tests/test_config_electricity.py
  - tests/test_config_weather.py
  - tests/test_config_factory.py
  - tests/test_config_init.py
findings:
  critical: 2
  warning: 4
  info: 3
  total: 9
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-05-11T18:30:00Z
**Depth:** deep
**Files Reviewed:** 21
**Status:** issues_found

## Summary

Deep review of the Pydantic config registry introducing typed configuration for 11 time series datasets across 5 families. The implementation follows a clean layered inheritance model (DatasetConfig -> ClassificationConfig/ForecastingConfig -> family-specific configs) with frozen instances, enum-based parameters, and factory registry functions.

Two critical bugs were found: (1) a Pydantic v2 field_validator bypass allowing `split_bounds` defaults to skip length validation, producing invalid config instances, and (2) Python's `bool`-is-subclass-of-`int` allowing boolean values to pass INDEXED split_bounds validation. Four warnings cover input validation gaps (HTTPS not enforced, degenerate fractional splits allowed, unvalidated task strings), and three info items cover test scaffolding residue, naming inconsistency, and an undocumented `_config_validate` mechanism.

## Critical Issues

### CR-01: `split_bounds` default bypasses field validator -- creates invalid instances

**File:** `src/tscollection/datasets/config/base.py:196`
**Issue:** `ForecastingConfig.split_bounds` has a default value of `()` (empty tuple). The field validator `validate_split_bounds_length` (line 204-214) checks that the tuple has exactly 3 elements. However, in Pydantic v2, `@field_validator` does NOT run on default values when the field is not provided by the caller. This means any subclass or direct instantiation of ForecastingConfig that omits `split_bounds` will produce a model with an empty tuple that silently violates the length constraint.

Confirmed by runtime test: `ETTConfig(name='Test', url='...', forecast_column='OT', frequency='1h', num_features=7, tasks=('forecasting',))` creates an instance with `split_bounds=()` -- no error raised, even though the docstring says it must have 3 elements.

**Impact:** Downstream code accessing `split_bounds` on a forecasting config may receive an empty tuple, causing index errors or incorrect data splits. This defeats the validation guarantee.

**Fix:** Remove the default value and make `split_bounds` required, or move the length check into the `@model_validator(mode='after')` which always runs:

```python
# Option A: Make it required (recommended)
split_bounds: tuple[int, ...] | tuple[float, ...]

# Option B: Move length check to model_validator
@model_validator(mode='after')
def validate_split_consistency(self) -> ForecastingConfig:
    if len(self.split_bounds) != 3:
        raise ValueError(
            f'split_bounds must have exactly 3 elements, got {len(self.split_bounds)}'
        )
    # ... rest of existing validation
```

### CR-02: Boolean values accepted as INDEXED split_bounds due to `bool` being subclass of `int`

**File:** `src/tscollection/datasets/config/base.py:234`
**Issue:** The INDEXED mode validation at line 234 uses `isinstance(b, int)` to check split_bounds elements. In Python, `bool` is a subclass of `int`, so `isinstance(True, int)` returns `True`. Additionally, Pydantic's default coercion mode converts booleans to their integer equivalents (`True` -> 1, `False` -> 0). This means `split_bounds=(True, False, True)` is accepted and stored as `(1, 0, 1)`.

Confirmed by runtime test: `ForecastingConfig(..., split_mode=SplitMode.INDEXED, split_bounds=(True, False, True), ...)` produces a valid instance with `split_bounds=(1, 0, 1)`.

**Impact:** While Pydantic's coercion converts bools to ints (preventing runtime type confusion), this masks developer errors. Passing `(True, True, True)` would silently create split indices of `(1, 1, 1)` which is semantically meaningless but passes all validators.

**Fix:** Add explicit bool rejection in the field validator:

```python
@field_validator('split_bounds')
@classmethod
def validate_split_bounds_length(
    cls, v: tuple[int, ...] | tuple[float, ...]
) -> tuple[int, ...] | tuple[float, ...]:
    if len(v) != 3:
        raise ValueError(
            f'split_bounds must have exactly 3 elements, got {len(v)}'
        )
    # Reject booleans (bool is subclass of int)
    if any(isinstance(b, bool) for b in v):
        raise ValueError(
            'split_bounds must not contain boolean values'
        )
    return v
```

## Warnings

### WR-01: HttpUrl does not enforce HTTPS-only schemes

**File:** `src/tscollection/datasets/config/base.py:97`
**Issue:** The `url` field uses `HttpUrl` from Pydantic, which accepts both `http://` and `https://` schemes. All current instances use HTTPS, but nothing prevents a contributor from adding an HTTP-only URL. Given that these URLs are used for downloading dataset archives, this is a security concern -- HTTP URLs are vulnerable to MITM attacks.

**Fix:** Add a field validator enforcing HTTPS:

```python
@field_validator('url')
@classmethod
def validate_https_only(cls, v: HttpUrl) -> HttpUrl:
    if str(v).startswith('http://'):
        raise ValueError('URL must use HTTPS scheme')
    return v
```

### WR-02: Degenerate fractional splits (1.0, 0.0, 0.0) pass validation

**File:** `src/tscollection/datasets/config/base.py:227-231`
**Issue:** The fractional split validator checks that values sum to approximately 1.0 with a 0.01 tolerance. This allows `(1.0, 0.0, 0.0)` to pass, which represents a dataset with no validation or test split. While this is technically a valid sum, it would produce broken data pipelines.

**Fix:** Add a minimum threshold per split component:

```python
if self.split_mode == SplitMode.FRACTIONAL:
    total = sum(self.split_bounds)  # type: ignore[arg-type]
    if abs(total - 1.0) > 0.01:
        raise ValueError(
            f'Fractional split_bounds must sum to 1.0, got {total}'
        )
    if any(b < 0.05 for b in self.split_bounds):  # type: ignore[union-attr]
        raise ValueError(
            'Each fractional split component must be >= 0.05 '
            f'(got {self.split_bounds})'
        )
```

### WR-03: `tasks` field accepts arbitrary strings without validation

**File:** `src/tscollection/datasets/config/base.py:101`
**Issue:** The `tasks` field is typed as `tuple[str, ...]` with no constraints. While current instances use values like `'classification'`, `'forecasting'`, and `'representation'`, nothing prevents a typo or invalid value (e.g., `'classfication'` or `'unknown_task'`). This is especially relevant since tasks are described in the docstring as "supported task types" suggesting a finite set.

**Fix:** Define a `TaskType` StrEnum and use it:

```python
class TaskType(StrEnum):
    CLASSIFICATION = 'classification'
    FORECASTING = 'forecasting'
    REPRESENTATION = 'representation'

# Then in DatasetConfig:
tasks: tuple[TaskType, ...]
```

### WR-04: `num_classes=0` on ForecastingConfig may mask missing data

**File:** `src/tscollection/datasets/config/base.py:99`
**Issue:** The base `DatasetConfig` allows `num_classes=0` (via `Field(ge=0)`), and `ForecastingConfig` does not override this constraint. While forecasting datasets don't have class labels, having `num_classes=0` is semantically different from "not applicable." If downstream code checks `if cfg.num_classes > 0` to decide whether to load labels, a forecasting config with accidentally set `num_classes=1` would incorrectly trigger classification logic.

**Fix:** Either override with `Field(ge=0, default=0, frozen=True)` in ForecastingConfig to document the intent explicitly, or use `num_classes: int = Field(default=0, ge=0)` with a docstring noting that 0 means "not a classification task."

## Info

### IN-01: Stale `try/except ImportError` scaffolding in factory tests

**File:** `tests/test_config_factory.py:13-20`
**Issue:** The test file wraps all factory imports in a `try/except ImportError` block with `pytest.skip("factory.py not yet implemented", allow_module_level=True)`. This was likely added during incremental development. Now that factory.py is complete, this scaffolding should be removed -- if factory.py goes missing, the test should fail loudly rather than skip silently.

**Fix:** Remove the try/except and import directly:

```python
from tscollection.datasets.config.factory import (
    CONFIGS,
    get_config,
    list_configs,
)
```

### IN-02: `ELECTRICITY_LOAD` name uses lowercase while dataset name is "electricity"

**File:** `src/tscollection/datasets/config/electricity.py:54`
**Issue:** The `name` field is `'electricity'` (lowercase) while all other dataset names use their display casing (e.g., `'Coffee'`, `'ETTh1'`, `'BasicMotions'`). This inconsistency is not a bug but could cause confusion in the factory lookup -- `get_config(name='Electricity')` would fail while `get_config(name='electricity')` works, breaking the pattern.

**Fix:** Consider using `'Electricity'` (capitalized) for the name field to match the convention used by other dataset instances, or document explicitly that the electricity dataset uses lowercase.

### IN-03: `_config_validate` abstract method is never called

**File:** `src/tscollection/datasets/config/base.py:113-120`
**Issue:** The `_config_validate` method on `DatasetConfig` is marked `@abc.abstractmethod` and overridden as `pass` in both `ClassificationConfig` and `ForecastingConfig`. However, this method is never invoked during model construction. It exists solely to make `DatasetConfig` abstract (preventing direct instantiation). While this is a valid ABC pattern, it is undocumented that the method serves as an instantiation barrier rather than an actual validation hook. The docstring says "Subclasses should call `super()._config_validate()` if they add their own validation logic" -- but since no code path ever calls `_config_validate`, this guidance is misleading.

**Fix:** Either remove the abstract method and use a different mechanism to prevent direct instantiation (e.g., raise `NotImplementedError` in `__init__`), or rename it to something like `_prevent_direct_instantiation` with a clear docstring that it is not meant to be called.

## Cross-File Analysis

### Import Graph

The import dependency graph is acyclic and well-structured:

```
enums/data.py (no internal deps)
    ^
    |
enums/__init__.py -> data.py
    ^
    |
config/base.py -> enums/data.py
    ^
    |
config/ucr.py, uea.py -> base.py + enums/data.py
config/ett.py, electricity.py, weather.py -> base.py + enums/__init__.py
    ^
    |
config/factory.py -> all family configs + base.py + enums/data.py
    ^
    |
config/__init__.py -> base.py + factory.py + all family configs
    ^
    |
datasets/__init__.py -> enums/__init__.py
```

No circular dependencies detected. Factory imports are explicit (not auto-discovered), which is correct per D-02.

### Type Consistency at Boundaries

- `get_config()` returns `DatasetConfig` (base type) rather than a Union or Generic. Callers receive the base type and must check `isinstance()` to access family-specific fields. This is acceptable for the current use case but worth noting.
- `list_configs()` returns `list[DatasetConfig]` -- same consideration applies.
- Type annotations across the hierarchy are consistent: `ClassificationConfig` and `ForecastingConfig` both override `family` with `DatasetFamily = <specific_member>`, which is valid Pydantic v2 behavior.

### Error Propagation

- Factory `get_config()` catches `KeyError` and re-raises with a helpful message listing available configs. Good pattern.
- Pydantic `ValidationError` propagates naturally from config construction. All validators raise clear messages.

### Test Coverage Gaps

The following scenarios are not covered by the 144 config-specific tests:
- `ForecastingConfig` created without `split_bounds` (CR-01)
- Boolean values in split_bounds (CR-02)
- `get_config()` with case variations (e.g., `'coffee'` vs `'Coffee'`)
- Cross-type validation (passing `DatasetFamily` where `SplitMode` is expected)
- `model_copy()` with invalid updates (e.g., setting `num_classes=-1`)
- `list_configs()` for families with no registered configs (e.g., `DatasetFamily.EXCHANGE`)

---

_Reviewed: 2026-05-11T18:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
