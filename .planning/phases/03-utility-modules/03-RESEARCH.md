# Phase 3: Utility Modules - Research

**Researched:** 2026-05-13
**Domain:** Python data utilities -- ARFF I/O, scaling, time features, variable-length processing
**Confidence:** HIGH

## Summary

This phase ports four utility modules from `_sources/rbspaper/` into `tscollection.datasets.utils/` with improved file separation, full type hints, Google-style docstrings, and keyword-only function signatures per CLAUDE.md. The four modules are: `arff.py` (ARFF reading), `scaling.py` (data scaling via scikit-learn), `features.py` (time feature extraction from DatetimeIndex), and `general.py` (collation and variable-length handling). Additionally, `flatten_list_of_np_arrays` must be added to the existing `common.py` and the `DataForm` enum must be added to `enums/data.py`.

**Primary recommendation:** Port in dependency order: (1) add `DataForm` enum, (2) add `flatten_list_of_np_arrays` to `common.py`, (3) `arff.py` and `features.py` (self-contained), (4) `scaling.py` (depends on step 2), (5) `general.py` (depends on torch), (6) update `__init__.py` exports.

## User Constraints (from CONTEXT.md)

### Locked Decisions

| ID | Decision |
|----|----------|
| D-01 | Keep separate files per concern: `arff.py`, `scaling.py`, `features.py`, `general.py` |
| D-02 | Existing `common.py` stays as-is -- contains `compose`, `FunctionComposer`, `get_num_samples_from_ts` |
| D-03 | No merging files -- each utility module has a clear domain boundary |
| D-04 | `flatten_list_of_np_arrays` must be added to `common.py` -- `scaling.py` depends on it via import |
| D-05 | `create_data_scaler()` uses `ScalingMethod` and `DataForm` enums from our enums package instead of strings -- type-safe at call time |
| D-06 | `ScalingMethod` values stay as `'minmax'` and `'standard'` -- update the ported function to compare against enum values, not rbspaper source strings (`'min_max'`/`'standardization'`) |
| D-07 | Add `DataForm` enum (REGULAR, NESTED, MULTI_FILES) to `enums/data.py` -- import from there in `scaling.py` |
| D-08 | Port all 4 utility files now (arff, scaling, features, general) -- Phase 4 data modules can start immediately without waiting |
| D-09 | Full CLAUDE.md compliance -- keyword-only args (`*`), full type hints, clean imports, Google-style docstrings |
| D-10 | Remove `from __future__ import annotations` unless needed for circular imports |
| D-11 | Convert `src.rbspaper` imports to `tscollection.datasets` relative imports |

### Claude's Discretion

No specific requirements -- port with CLAUDE.md style improvements.

### Deferred Ideas (OUT OF SCOPE)

None -- discussion stayed within phase scope.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| UTI-01 | ARFF file reading with dtype processing (nominal/numeric) | `arff.py` port: `read_arff_as_df`, `process_df_according_to_dtypes` |
| UTI-02 | Data scaling -- `create_data_scaler()` for regular, nested, multi-file data | `scaling.py` port with enum wiring per D-05, D-06, D-07 |
| UTI-03 | Time feature extraction from DatetimeIndex | `features.py` port: `extract_time_features` |
| UTI-04 | Variable-length series processing -- centering, collation | `general.py` port: `custom_collate_fn`, `centralize_variable_length_series`, `process_data_with_varying_sequence_lengths_single` |
| UTI-05 | Each utility is in a separate file with proper `__all__` exports | D-01, D-03 file organization decisions |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| ARFF reading/dtype processing | Pure Python utility | — | File I/O + data transformation; no framework dependency |
| Data scaling (MinMax/Standard) | Pure Python utility | — | Wraps scikit-learn; returns callables, no state leakage |
| Time feature extraction | Pure Python utility | — | Pandas-only; produces numpy arrays for downstream use |
| Variable-length centering | Pure Python utility | — | Numpy-only; pure function, no side effects |
| DataLoader collation | PyTorch utility | — | Uses `default_collate`; consumed by Phase 4 DataModules |

## Standard Stack

### Core Dependencies (already in pyproject.toml)

| Library | Version | Purpose | Verified |
|---------|---------|---------|----------|
| numpy | 2.4.4 | Array operations in all modules | [VERIFIED: uv run] |
| pandas | 3.0.2 | DataFrame handling, DatetimeIndex | [VERIFIED: uv run] |
| scipy | 1.17.1 | `scipy.io.arff.loadarff` for ARFF parsing | [VERIFIED: uv run] |
| scikit-learn | 1.8.0 | `MinMaxScaler`, `StandardScaler` | [VERIFIED: uv run] |
| torch | 2.8.0 | `default_collate` for DataLoader integration | [VERIFIED: uv run] |

### Supporting

| Library | Version | Purpose | When Used |
|---------|---------|---------|-----------|
| pytest | >=8.2 | Unit testing all utility functions | Phase 5 tests |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `scipy.io.arff` | Manual ARFF parser | Unnecessary complexity; scipy handles edge cases |
| Custom scaling | hand-written min-max logic | scikit-learn handles axis, partial_fit, inverse_transform |
| Custom collation | manual tensor stacking | `default_collate` handles nested dicts, tuples correctly |

## Architecture Patterns

### Module Dependency Graph

```
common.py (existing + flatten_list_of_np_arrays)
    ^
    |
scaling.py -- needs flatten_list_of_np_arrays from common.py
    |
    | uses DataForm enum from enums/data.py
    | uses ScalingMethod enum from enums/data.py

arff.py -- self-contained, no cross-utility imports
    |
    | uses scipy.io.arff, pandas

features.py -- self-contained, no cross-utility imports
    |
    | uses pandas, numpy

general.py -- uses torch.utils.data.dataloader.default_collate
    |
    | uses numpy, pandas
```

### File-to-Function Mapping

| File | Functions/Classes | Requirement |
|------|-------------------|-------------|
| `enums/data.py` | `DataForm` (new) | D-07 |
| `utils/common.py` | `flatten_list_of_np_arrays` (add) | D-04 |
| `utils/arff.py` | `read_arff_as_df`, `process_df_according_to_dtypes` | UTI-01 |
| `utils/scaling.py` | `create_data_scaler`, `_get_scaler`, `_scale_regular_data`, `_scale_regular_data_and_return_same_type`, `_scale_multi_file_data`, `_scale_nested_data_all_dimensions` | UTI-02 |
| `utils/features.py` | `extract_time_features` | UTI-03 |
| `utils/general.py` | `custom_collate_fn`, `centralize_variable_length_series`, `process_data_with_varying_sequence_lengths_single` | UTI-04 |
| `utils/__init__.py` | Export wiring for all modules | UTI-05 |

### Recommended Project Structure

```
src/tscollection/datasets/
├── enums/
│   ├── __init__.py          # export DataForm
│   └── data.py              # add DataForm enum
├── utils/
│   ├── __init__.py          # export all utility functions
│   ├── common.py            # add flatten_list_of_np_arrays
│   ├── arff.py              # NEW
│   ├── scaling.py           # NEW
│   ├── features.py          # NEW
│   └── general.py           # NEW
```

### Style Patterns

All functions follow CLAUDE.md conventions:

1. **Keyword-only arguments** after `*` separator for functions with optional params
2. **Full type hints** on all parameters and return values
3. **Google-style docstrings** with Args/Returns sections
4. **`__all__` exports** in every module file
5. **`TYPE_CHECKING` guards** for heavy imports (numpy, typing)
6. **No `from __future__ import annotations`** unless circular imports require it

## Source Code Analysis

### arff.py (UTI-01)

**Source:** `_sources/rbspaper/src/rbspaper/data/utils/arff.py` (49 lines)

**Functions:**

| Function | Signature | Dependencies | Notes |
|----------|-----------|--------------|-------|
| `read_arff_as_df` | `(arff_file_path: Path \| str) -> tuple[pd.DataFrame, Any]` | `scipy.io.arff`, `pandas` | Lazy-imports scipy. Returns DataFrame + metadata. |
| `process_df_according_to_dtypes` | `(df_data, meta, dtypes_functions_map) -> pd.DataFrame` | `pandas` | Iterates meta.names(), applies mapped callables. |

**Changes needed:**
- Convert to keyword-only where appropriate (`process_df_according_to_dtypes` has 3 params, all required -- keep positional)
- Add `TYPE_CHECKING` guard for `Any` import
- Remove `from __future__ import annotations` (no circular imports)
- Verify `scipy.io.arff.loadarff` returns `bytes` for nominal columns -- the caller must handle decoding. This is verified: nominal columns return `b'a'`, `b'b'`, etc.

### scaling.py (UTI-02)

**Source:** `_sources/rbspaper/src/rbspaper/data/utils/scaling.py` (264 lines)

**Functions:**

| Function | Signature | Dependencies | Notes |
|----------|-----------|--------------|-------|
| `create_data_scaler` | `(*, scale, scaling_range, scaling_method, data_form) -> Callable` | Internal helpers | Factory returning `scale_data` closure. |
| `_get_scaler` | `(scaling_method, scaling_range) -> MinMaxScaler \| StandardScaler` | sklearn | Private helper. |
| `_scale_regular_data` | `(train, valid, test, method, range) -> tuple` | sklearn | 2-D scaling. |
| `_scale_regular_data_and_return_same_type` | `(train, valid, test, method, range) -> tuple` | sklearn, pandas | Preserves DataFrame type. |
| `_scale_multi_file_data` | `(train, valid, test, method, range) -> tuple` | sklearn, numpy, `flatten_list_of_np_arrays` | 1-D list scaling. |
| `_scale_nested_data_all_dimensions` | `(train, valid, test, method, range) -> tuple` | sklearn, numpy | 3-D scaling. |

**Changes needed:**
- **D-05:** Replace string `scaling_method` param with `ScalingMethod` enum type
- **D-06:** Compare against `ScalingMethod.MINMAX.value` ('minmax') and `ScalingMethod.STANDARD.value` ('standard') -- NOT `'min_max'`/`'standardization'`
- **D-07:** Import `DataForm` from `tscollection.datasets.enums.data` -- replace `DataFormEnum` class definition
- **D-04:** Import `flatten_list_of_np_arrays` from `tscollection.datasets.utils.common`
- Remove internal `DataFormEnum` class definition (now in enums/data.py)
- `create_data_scaler` already uses keyword-only (`*`) -- good
- Private helpers should NOT be in `__all__`

### features.py (UTI-03)

**Source:** `_sources/rbspaper/src/rbspaper/data/utils/features.py` (35 lines)

**Functions:**

| Function | Signature | Dependencies | Notes |
|----------|-----------|--------------|-------|
| `extract_time_features` | `(datetime_index: pd.DatetimeIndex) -> np.ndarray` | `numpy`, `pandas` | Returns (N, 7) float32 array. |

**Changes needed:**
- Minimal. Function is already clean.
- Uses `series.dt.isocalendar().week.to_numpy()` -- verified working on pandas 3.0.2
- Remove `from __future__ import annotations` (no circular imports)
- Add keyword-only marker: `def extract_time_features(*, datetime_index: ...)` -- actually, with a single required param, keyword-only adds no value. Keep positional.

### general.py (UTI-04)

**Source:** `_sources/rbspaper/src/rbspaper/data/utils/general.py` (108 lines)

**Functions:**

| Function | Signature | Dependencies | Notes |
|----------|-----------|--------------|-------|
| `custom_collate_fn` | `(batch, desired_batch_size) -> Any` | `torch.utils.data.dataloader.default_collate` | Pads last batch by cycling. |
| `centralize_variable_length_series` | `(series_batch: np.ndarray) -> np.ndarray` | `numpy` | Centers valid data in NaN-padded sequences. |
| `process_data_with_varying_sequence_lengths_single` | `(data: np.ndarray \| pd.DataFrame) -> np.ndarray \| pd.DataFrame` | `numpy`, `pandas` | 2-D/3-D handling + centering. |

**Changes needed:**
- `custom_collate_fn`: Add `*` for keyword-only on `desired_batch_size`
- `centralize_variable_length_series`: Single required arg -- no change needed
- `process_data_with_varying_sequence_lengths_single`: Single required arg -- no change needed
- Remove `from __future__ import annotations` (no circular imports)
- `default_collate` import should use `TYPE_CHECKING`? No -- it's used at runtime. Keep as regular import.

### common.py additions

**Source:** `_sources/rbspaper/src/rbspaper/data/utils/common.py` (lines 107-116)

**Function to add:**

| Function | Signature | Dependencies | Notes |
|----------|-----------|--------------|-------|
| `flatten_list_of_np_arrays` | `(list_of_np_arrays: list[np.ndarray]) -> np.ndarray` | `numpy` | Concatenates and ravel(). |

**Changes needed:**
- Add to existing `common.py` __all__ list
- Add function implementation
- Current `common.py` uses `from __future__ import annotations` and `TYPE_CHECKING` guard for numpy -- keep this pattern, since `flatten_list_of_np_arrays` has a runtime numpy dependency. Move `import numpy as np` out of TYPE_CHECKING or keep it conditional. Actually, the function uses `np.concatenate` at runtime, so numpy cannot be behind TYPE_CHECKING.

**IMPORTANT:** Current `common.py` has `import numpy as np` inside `TYPE_CHECKING`, but `get_num_samples_from_ts` and `flatten_list_of_np_arrays` both need numpy at runtime. The existing code works because `get_num_samples_from_ts` only uses `len()` which doesn't need numpy. But `flatten_list_of_np_arrays` calls `np.concatenate()` at runtime. The import must be moved OUT of `TYPE_CHECKING`.

### enums/data.py additions

**Per D-07**, add `DataForm` enum:

```python
class DataForm(StrEnum):
    """Enum for the form (shape) of the data.

    Attributes:
        REGULAR: 2-D tabular data (samples x features).
        NESTED: 3-D array data (samples x timesteps x features).
        MULTI_FILES: List of 1-D arrays from multiple files.
    """

    REGULAR = 'regular'
    NESTED = 'nested'
    MULTI_FILES = 'multi_files'
```

Also update `enums/__init__.py` and `datasets/__init__.py` to export `DataForm`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ARFF parsing | Custom ARFF parser | `scipy.io.arff.loadarff` | Handles relation headers, attribute types, missing values (@, ?), string escaping |
| Data scaling | Manual min-max / z-score math | `sklearn.preprocessing.MinMaxScaler`, `StandardScaler` | Handles axis, partial_fit, inverse_transform, feature-wise fitting |
| DataLoader collation | Manual tensor stacking | `torch.utils.data.dataloader.default_collate` | Handles nested dicts, tuples, lists of tensors with correct device placement |
| Function composition | Lambda chains | Existing `FunctionComposer` / `compose` | Already ported in Phase 2 with proper filtering of None |

## Common Pitfalls

### Pitfall 1: ScalingMethod Enum Value Mismatch
**What goes wrong:** Using source string values `'min_max'` / `'standardization'` instead of our enum values `'minmax'` / `'standard'`.
**Why it happens:** The source code compares `scaling_method == 'min_max'` but our `ScalingMethod` enum has `MINMAX = 'minmax'`.
**How to avoid:** In `_get_scaler()`, compare against `ScalingMethod.MINMAX` and `ScalingMethod.STANDARD` enum members (or their `.value`). Per D-06.
**Warning signs:** Tests fail with "Unsupported scaling method" even when the correct enum is passed.

### Pitfall 2: Nominal ARFF Columns Are Bytes
**What goes wrong:** `scipy.io.arff.loadarff` returns nominal (string) column values as `bytes` objects (e.g., `b'a'` not `'a'`).
**Why it happens:** scipy reads ARFF as binary internally.
**How to avoid:** Document this in `read_arff_as_df` docstring. The Phase 4 consumer must provide a decode function in `dtypes_functions_map` for nominal columns.
**Warning signs:** DataFrame shows `b'value'` instead of `'value'` for categorical columns.

### Pitfall 3: numpy Import Behind TYPE_CHECKING
**What goes wrong:** `flatten_list_of_np_arrays` calls `np.concatenate()` at runtime but numpy is only imported inside `TYPE_CHECKING`.
**Why it happens:** Current `common.py` puts `import numpy as np` inside `TYPE_CHECKING` because `get_num_samples_from_ts` only uses `len()`.
**How to avoid:** Move `import numpy as np` to a top-level import in `common.py` when adding `flatten_list_of_np_arrays`.
**Warning signs:** `NameError: name 'np' is not defined` at runtime.

### Pitfall 4: pandas 3.0 Frequency Alias Change
**What goes wrong:** Test code using `pd.date_range(freq='H')` crashes because uppercase frequency aliases were removed.
**Why it happens:** pandas 3.0 deprecated capital frequency aliases (H -> h, D -> d, etc.).
**How to avoid:** Use lowercase frequency aliases in all test code: `freq='h'` instead of `freq='H'`.
**Warning signs:** `ValueError: Invalid frequency: H. Did you mean h?`

### Pitfall 5: ROADMAP vs. CONTEXT File Mapping Discrepancy
**What goes wrong:** ROADMAP.md describes `features.py` as containing `extract_time_features, custom_collate_fn` but CONTEXT D-01 says `general.py` contains collation utilities.
**Why it happens:** ROADMAP was written before the detailed discussion that produced CONTEXT.md decisions.
**How to avoid:** Follow CONTEXT.md D-01: `custom_collate_fn`, `centralize_variable_length_series`, `process_data_with_varying_sequence_lengths_single` go in `general.py`. `extract_time_features` goes in `features.py`.
**Warning signs:** Functions in wrong files during plan review.

### Pitfall 6: DataFormEnum Name Change
**What goes wrong:** Copying the `DataFormEnum` class name from source instead of using `DataForm` as specified in D-07.
**Why it happens:** Source code defines `class DataFormEnum(str, Enum)`.
**How to avoid:** Use `class DataForm(StrEnum)` to match the naming convention of existing enums in `data.py` (no "Enum" suffix).
**Warning signs:** Import errors in Phase 4 when looking for `DataForm`.

## Code Examples

### DataForm Enum (enums/data.py addition)

```python
from enum import StrEnum

class DataForm(StrEnum):
    """Enum for the form (shape) of the data.

    Attributes:
        REGULAR: 2-D tabular data (samples x features).
        NESTED: 3-D array data (samples x timesteps x features).
        MULTI_FILES: List of 1-D arrays from multiple files.
    """

    REGULAR = 'regular'
    NESTED = 'nested'
    MULTI_FILES = 'multi_files'
```

### flatten_list_of_np_arrays (common.py addition)

```python
def flatten_list_of_np_arrays(list_of_np_arrays: list[np.ndarray]) -> np.ndarray:
    """Flatten a list of numpy arrays into a single 1-D array.

    Args:
        list_of_np_arrays: A list of numpy arrays.

    Returns:
        A single flattened numpy array.
    """
    return np.concatenate(list_of_np_arrays).ravel()
```

### _get_scaler with enum comparison (scaling.py change)

Source uses:
```python
if scaling_method == 'min_max':      # OLD
if scaling_method == 'standardization':  # OLD
```

Must change to:
```python
if scaling_method == ScalingMethod.MINMAX:        # NEW -- compares to 'minmax'
if scaling_method == ScalingMethod.STANDARD:      # NEW -- compares to 'standard'
```

### create_data_scaler signature (scaling.py change)

Source uses strings:
```python
def create_data_scaler(
    *,
    scale: bool,
    scaling_range: tuple[float, float],
    scaling_method: str = 'min_max',
    data_form: str = DataFormEnum.REGULAR.value,
) -> Callable:
```

Ported uses enums:
```python
def create_data_scaler(
    *,
    scale: bool,
    scaling_range: tuple[float, float],
    scaling_method: ScalingMethod = ScalingMethod.MINMAX,
    data_form: DataForm = DataForm.REGULAR,
) -> Callable:
```

## Consumer Analysis

### How abstract.py (Phase 4) Consumes Utilities

From `_sources/rbspaper/src/rbspaper/data/modules/abstract.py`:

| Import | Used By | Phase 4 Impact |
|--------|---------|---------------|
| `custom_collate_fn` | `_get_custom_collate_fn()` via `functools.partial` | Needs `general.py` |
| `extract_time_features` | `BaseForecastingTimeSeriesDataModule.setup()` | Needs `features.py` |
| `load_json` | `BaseClassificationTimeSeriesDataModule.__init__()` | From `common.py` (not ported -- Phase 4 must import from existing source or add to common.py) |
| `process_data_with_varying_sequence_lengths_single` | `_process_data_with_varying_sequence_lengths()` | Needs `general.py` |
| `separate_target_feature_from_df` | `_separate_target_feature` via `partial` | From `common.py` in source (not currently in our common.py) |
| `create_data_scaler` | `BaseTimeSeriesDataModule.setup()` | Needs `scaling.py` |

**Note on `load_json` and `separate_target_feature_from_df`:** These are in the source `common.py` but NOT in our current `common.py`. Phase 4's `abstract.py` imports them. The plan should verify whether Phase 4 needs these added or imports them from elsewhere. Per D-02, "existing common.py stays as-is" refers to what is currently there; adding new functions from the source is within scope for completeness.

## Runtime State Inventory

Not applicable -- this is a greenfield port phase, not a rename/refactor.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| numpy | all modules | -- | 2.4.4 | -- |
| pandas | arff, scaling, features, general | -- | 3.0.2 | -- |
| scipy | arff.py | -- | 1.17.1 | -- |
| scikit-learn | scaling.py | -- | 1.8.0 | -- |
| torch | general.py (default_collate) | -- | 2.8.0 | -- |
| pytest | tests | -- | >=8.2 | -- |
| ruff | linting | -- | >=0.15.9 | -- |
| uv | task execution | -- | -- | -- |

All dependencies are listed in `pyproject.toml` and verified available in the uv environment. No blocking dependencies.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=8.2 |
| Config file | `pyproject.toml` [tool.pytest.ini_options] section |
| Quick run command | `uv run pytest tests/ -x -q` |
| Full suite command | `uv run pytest tests/ -x --cov=tscollection.datasets` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| UTI-01 | `read_arff_as_df` returns DataFrame + metadata | unit | `pytest tests/test_utils.py::test_read_arff_as_df -x` | Gap -- Wave 0 |
| UTI-01 | `process_df_according_to_dtypes` applies transformations | unit | `pytest tests/test_utils.py::test_process_df_according_to_dtypes -x` | Gap -- Wave 0 |
| UTI-02 | `create_data_scaler` with REGULAR returns scaled arrays | unit | `pytest tests/test_utils.py::test_create_data_scaler_regular -x` | Gap -- Wave 0 |
| UTI-02 | `create_data_scaler` with NESTED preserves 3-D shape | unit | `pytest tests/test_utils.py::test_create_data_scaler_nested -x` | Gap -- Wave 0 |
| UTI-02 | `create_data_scaler` with MULTI_FILES scales list | unit | `pytest tests/test_utils.py::test_create_data_scaler_multi_files -x` | Gap -- Wave 0 |
| UTI-02 | `create_data_scaler` with scale=False returns unchanged | unit | `pytest tests/test_utils.py::test_create_data_scaler_no_scale -x` | Gap -- Wave 0 |
| UTI-02 | ScalingMethod enum wiring works correctly | unit | `pytest tests/test_utils.py::test_scaling_method_enum -x` | Gap -- Wave 0 |
| UTI-03 | `extract_time_features` returns (N, 7) float32 array | unit | `pytest tests/test_utils.py::test_extract_time_features -x` | Gap -- Wave 0 |
| UTI-04 | `custom_collate_fn` pads last batch | unit | `pytest tests/test_utils.py::test_custom_collate_fn -x` | Gap -- Wave 0 |
| UTI-04 | `centralize_variable_length_series` centers NaN-padded data | unit | `pytest tests/test_utils.py::test_centralize_variable_length -x` | Gap -- Wave 0 |
| UTI-04 | `process_data_with_varying_sequence_lengths_single` handles 2-D/3-D | unit | `pytest tests/test_utils.py::test_process_varying_lengths -x` | Gap -- Wave 0 |
| UTI-05 | All modules export via `__all__` | unit | `pytest tests/test_utils.py::test_all_exports -x` | Gap -- Wave 0 |
| UTI-05 | `DataForm` enum importable from package root | unit | `pytest tests/test_utils.py::test_dataform_enum -x` | Gap -- Wave 0 |
| D-04 | `flatten_list_of_np_arrays` in common.py works | unit | `pytest tests/test_utils.py::test_flatten_list_of_np_arrays -x` | Gap -- Wave 0 |

### Sampling Rate
- **Per task commit:** `uv run pytest tests/test_utils.py -x -q` (runs only utility tests)
- **Per wave merge:** `uv run pytest tests/ -x -q` (full suite including existing tests)
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_utils.py` -- covers UTI-01 through UTI-05, D-04
- [ ] Synthetic ARFF fixture in `tests/conftest.py` -- temporary ARFF file for `read_arff_as_df` tests
- [ ] `DataForm` enum export test

## Security Domain

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | Type hints + runtime isinstance checks in scaling helpers |
| All others | no | Utilities are data processing only; no auth, session, or crypto |

### Known Threat Patterns

| Pattern | Risk | Mitigation |
|---------|------|-----------|
| Path traversal via ARFF file path | Low | `scipy.io.arff.loadarff` uses standard file open; caller controls paths |
| Large file memory exhaustion | Low | ARFF loading reads entire file; Phase 4 consumers should validate file sizes |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `load_json` and `separate_target_feature_from_df` from source common.py may be needed by Phase 4 | Consumer Analysis | Phase 4 plan must include adding them; low risk since Phase 4 research will re-verify |
| A2 | No existing tests directory structure prevents adding `test_utils.py` | Validation Architecture | Very low -- current tests/ directory is open |
| A3 | `scipy.io.arff` behavior is stable across scipy 1.13-1.17 | Standard Stack | Low -- verified against scipy 1.17.1 |

## Open Questions (RESOLVED)

1. **Should `load_json` and `separate_target_feature_from_df` be added to `common.py` now?**
   - What we know: Phase 4's `abstract.py` imports both from source common.py
   - What's unclear: Whether Phase 4 will add them or expect them from Phase 3
   - Recommendation: Add them now for completeness -- they are small, pure functions. Phase 4 research can confirm.
   - **RESOLVED:** Deferred to Phase 4. Phase 4 research will confirm whether common.py needs these functions. Including them now adds scope without verified demand.

2. **Should `DataForm` be exported from the package root (`tscollection.datasets`)?**
   - What we know: Other enums are exported from the package root
   - What's unclear: Whether utility-only enums should be at the root
   - Recommendation: Yes -- follow the existing pattern of exporting all enums from `__init__.py`
   - **RESOLVED:** Yes. Plan 03 Task 1 (03-03) exports DataForm from `tscollection.datasets` root following existing enum pattern.

## Sources

### Primary (HIGH confidence)
- Source files read directly: `_sources/rbspaper/src/rbspaper/data/utils/arff.py`, `scaling.py`, `features.py`, `general.py`, `common.py`
- Source consumer: `_sources/rbspaper/src/rbspaper/data/modules/abstract.py`
- Existing codebase: `src/tscollection/datasets/utils/common.py`, `src/tscollection/datasets/enums/data.py`
- Environment verified: `uv run` -- numpy 2.4.4, pandas 3.0.2, scipy 1.17.1, sklearn 1.8.0, torch 2.8.0

### Secondary (MEDIUM confidence)
- `pyproject.toml` dependency declarations
- `tests/conftest.py` fixture patterns

### Tertiary (LOW confidence)
- None -- all claims traced to source files or verified in environment

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all versions verified via `uv run`
- Architecture: HIGH -- source code fully analyzed, dependency graph mapped
- Pitfalls: HIGH -- pandas 3.0 freq alias change verified, bytes/nominal issue verified, TYPE_CHECKING/numpy conflict identified by code inspection

**Research date:** 2026-05-13
**Valid until:** 2026-06-12 (30 days -- stable ecosystem; no rapid version churn expected)

---

## RESEARCH COMPLETE

**Phase:** 03 - Utility Modules
**Confidence:** HIGH

### Key Findings

1. **Four modules to port** (arff, scaling, features, general) plus `flatten_list_of_np_arrays` to `common.py` and `DataForm` enum to `enums/data.py`. Total: ~530 source lines across 6 files.

2. **Critical enum value mapping change:** Source uses `'min_max'`/`'standardization'` strings; our `ScalingMethod` enum uses `'minmax'`/`'standard'`. All `_get_scaler()` comparisons must be updated. This is the most likely source of bugs.

3. **numpy import must move out of TYPE_CHECKING** in `common.py` when adding `flatten_list_of_np_arrays`, because `np.concatenate()` is called at runtime. Current code works only because `get_num_samples_from_ts` uses `len()`.

4. **pandas 3.0 breaks uppercase frequency aliases:** Any test code using `freq='H'` must use `freq='h'`. Verified that `extract_time_features` (using `.dt.isocalendar().week`) works correctly on pandas 3.0.2.

5. **scipy.io.arff returns bytes for nominal columns:** Documented but not changed -- the `dtypes_functions_map` pattern delegates decoding to the caller. Phase 4 consumers must handle this.

6. **Consumer analysis reveals** `load_json` and `separate_target_feature_from_df` are imported by Phase 4's abstract.py but not yet in our common.py. Open question whether to add them now.

7. **Test infrastructure:** All tests are gaps (Wave 0). No `test_utils.py` exists yet. pytest is configured, conftest.py has fixtures for data shapes.

### File Created
`/Users/skaf/VSCodeProjects/tsdatasets/.planning/phases/03-utility-modules/03-RESEARCH.md`

### Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Standard Stack | HIGH | All versions verified in uv environment |
| Architecture | HIGH | Full source code analysis, dependency mapping complete |
| Pitfalls | HIGH | Runtime-verified: pandas 3.0 freq, scipy bytes, numpy TYPE_CHECKING |

### Open Questions
- Whether `load_json` and `separate_target_feature_from_df` should be added to `common.py` in this phase (recommended: yes).
- Whether `DataForm` should be exported from package root (recommended: yes, following existing pattern).

### Ready for Planning
Research complete. Planner can now create PLAN.md files.
