---
phase: 03-utility-modules
verified: 2026-05-13T10:00:00Z
status: passed
score: 14/14 must-haves verified
overrides_applied: 0
---

# Phase 3: Utility Modules Verification Report

**Phase Goal:** Port utility modules from `_sources/` with improved file separation and style.
**Verified:** 2026-05-13T10:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

Merged must-haves from ROADMAP success criteria and all three plan must_haves sections.

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | DataForm enum has REGULAR, NESTED, MULTI_FILES members with StrEnum base | VERIFIED | `enums/data.py` lines 34-45: `class DataForm(StrEnum)` with 3 members; import test passes |
| 2 | flatten_list_of_np_arrays concatenates list of arrays into single 1-D array | VERIFIED | `common.py` lines 67-78: `np.concatenate(list_of_np_arrays).ravel()`; behavioral test confirms `[1,2,3,4]` output |
| 3 | read_arff_as_df returns tuple of (DataFrame, metadata) from valid ARFF file | VERIFIED | `arff.py` lines 12-31: lazy scipy import, returns `(df, meta)`; behavioral test with temp ARFF file passes |
| 4 | extract_time_features returns (N, 7) float32 array from DatetimeIndex | VERIFIED | `features.py` lines 9-33: `np.stack(...).astype(np.float32)`; behavioral test confirms `(10, 7)` shape |
| 5 | create_data_scaler with ScalingMethod.MINMAX returns MinMaxScaler-based closure | VERIFIED | `scaling.py` lines 87-88: `if scaling_method == ScalingMethod.MINMAX: return MinMaxScaler(...)`; behavioral test confirms scaling to [0, 1] |
| 6 | create_data_scaler with DataForm.NESTED preserves 3-D array shape | VERIFIED | `scaling.py` lines 223-265: `_scale_nested_data_all_dimensions` reshapes and restores; test confirms shape preservation |
| 7 | create_data_scaler with scale=False returns data unchanged | VERIFIED | `scaling.py` lines 42-43: `if not scale: return train_data, valid_data, test_data`; test confirms identity return |
| 8 | _get_scaler compares against ScalingMethod enum members (not source strings) | VERIFIED | `scaling.py` line 87: `scaling_method == ScalingMethod.MINMAX`; line 89: `scaling_method == ScalingMethod.STANDARD` |
| 9 | custom_collate_fn pads short batches by cycling samples | VERIFIED | `general.py` lines 16-36: cycle logic with `batch.append()`; test confirms padding to desired size |
| 10 | centralize_variable_length_series centers NaN-padded data | VERIFIED | `general.py` lines 39-64: offset calculation and shift via `np.ogrid`; test confirms shape preservation |
| 11 | All 11 utility symbols importable from tscollection.datasets.utils | VERIFIED | `utils/__init__.py` exports all 11; import test passes |
| 12 | DataForm importable from tscollection.datasets.enums and tscollection.datasets root | VERIFIED | `enums/__init__.py` line 4, `datasets/__init__.py` line 8: both export DataForm |
| 13 | ARFF reader handles nominal/numeric dtypes correctly (ROADMAP SC) | VERIFIED | Behavioral test: nominal columns return bytes, numeric columns return float; `process_df_according_to_dtypes` applies mapped transformations |
| 14 | create_data_scaler() returns callable that scales train/valid/test splits (ROADMAP SC) | VERIFIED | Behavioral test: callable accepts (train, valid, test), returns scaled tuple with values in [0, 1] |

**Score:** 14/14 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/tscollection/datasets/enums/data.py` | DataForm enum | VERIFIED | 75 lines, DataForm(StrEnum) at lines 34-45 |
| `src/tscollection/datasets/utils/common.py` | flatten_list_of_np_arrays | VERIFIED | 79 lines, top-level numpy import (line 7), function at lines 67-78 |
| `src/tscollection/datasets/utils/arff.py` | read_arff_as_df, process_df_according_to_dtypes | VERIFIED | 57 lines, lazy scipy import (line 27), __all__ correct |
| `src/tscollection/datasets/utils/features.py` | extract_time_features | VERIFIED | 33 lines, pandas 3.0 compatible (isocalendar().week), __all__ correct |
| `src/tscollection/datasets/utils/scaling.py` | create_data_scaler with enum wiring | VERIFIED | 265 lines, imports DataForm/ScalingMethod from enums, no local DataFormEnum |
| `src/tscollection/datasets/utils/general.py` | custom_collate_fn, centralize_variable_length_series, process_data_with_varying_sequence_lengths_single | VERIFIED | 103 lines, keyword-only desired_batch_size, default_collate import |
| `src/tscollection/datasets/utils/__init__.py` | All 11 utility exports | VERIFIED | 33 lines, 11 symbols in __all__, grouped by source module |
| `src/tscollection/datasets/enums/__init__.py` | DataForm export | VERIFIED | 23 lines, DataForm in imports and __all__ |
| `src/tscollection/datasets/__init__.py` | DataForm root export | VERIFIED | 28 lines, DataForm in imports and __all__ |
| `tests/test_utils_common.py` | DataForm and flatten tests | VERIFIED | 66 lines, 7 tests |
| `tests/test_utils_arff.py` | ARFF reader tests | VERIFIED | 112 lines, 4 tests |
| `tests/test_utils_features.py` | Time feature tests | VERIFIED | 57 lines, 5 tests |
| `tests/test_utils_scaling.py` | Scaling tests | VERIFIED | 245 lines, 12 tests |
| `tests/test_utils_general.py` | General utility tests | VERIFIED | 161 lines, 12 tests |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| common.py | numpy | `import numpy as np` (top-level, line 7) | WIRED | Not behind TYPE_CHECKING |
| arff.py | scipy.io.arff | `from scipy.io import arff` (lazy, line 27) | WIRED | Inside read_arff_as_df function |
| features.py | numpy, pandas | `import numpy as np`, `import pandas as pd` (lines 3-4) | WIRED | Top-level imports |
| scaling.py | enums/data.py | `from tscollection.datasets.enums.data import DataForm, ScalingMethod` (line 10) | WIRED | Uses enum members for comparison |
| scaling.py | common.py | `from tscollection.datasets.utils.common import flatten_list_of_np_arrays` (line 11) | WIRED | Called at line 202 in _scale_multi_file_data |
| general.py | torch | `from torch.utils.data.dataloader import default_collate` (line 7) | WIRED | Used at line 36 in custom_collate_fn |
| utils/__init__.py | arff.py | `from tscollection.datasets.utils.arff import ...` (lines 3-5) | WIRED | Both symbols exported |
| utils/__init__.py | scaling.py | `from tscollection.datasets.utils.scaling import create_data_scaler` (line 19) | WIRED | Factory exported |
| enums/__init__.py | data.py | `DataForm` in import block (line 4) and __all__ (line 15) | WIRED | Alphabetical ordering |
| datasets/__init__.py | enums/__init__.py | `DataForm` in import block (line 8) and __all__ (line 19) | WIRED | Package root re-export |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| arff.py:read_arff_as_df | df_data, meta | scipy.io.arff.loadarff(arff_file_path) | FLOWING -- real file parse |
| arff.py:process_df_according_to_dtypes | df_data[col_name] | dtypes_functions_map[col_type](df_data[col_name]) | FLOWING -- caller-provided transforms applied |
| features.py:extract_time_features | stacked array | np.stack([minute, hour, ...]).astype(float32) | FLOWING -- real DatetimeIndex properties |
| scaling.py:create_data_scaler | scale_data closure | _scale_* helpers using sklearn scalers | FLOWING -- fits and transforms via sklearn |
| scaling.py:_scale_multi_file_data | combined | flatten_list_of_np_arrays(train_arrays) | FLOWING -- uses common.py import (line 202) |
| general.py:custom_collate_fn | batch | default_collate(batch) | FLOWING -- torch collate on padded batch |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| DataForm enum values | `assert DataForm.REGULAR == 'regular'` | Pass | PASS |
| flatten_list_of_np_arrays | `[1,2] + [3,4] -> [1,2,3,4]` | Pass | PASS |
| 11 utility symbols from utils | Import all 11 symbols | All resolve | PASS |
| DataForm from package root | `from tscollection.datasets import DataForm` | Resolves | PASS |
| extract_time_features shape | `(10, 7) float32` | Confirmed | PASS |
| create_data_scaler MINMAX | Scaled values in [0, 1] | Confirmed | PASS |
| Full test suite | `uv run pytest tests/ -x -q` | 76 passed | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-----------|-------------|--------|----------|
| UTI-01 | Plan 01 | ARFF file reading with dtype processing | VERIFIED | arff.py with read_arff_as_df, process_df_according_to_dtypes; 4 tests in test_utils_arff.py |
| UTI-02 | Plan 02 | Data scaling -- create_data_scaler() | VERIFIED | scaling.py with enum-wired factory; 12 tests in test_utils_scaling.py |
| UTI-03 | Plan 01 | Time feature extraction from DatetimeIndex | VERIFIED | features.py with extract_time_features; 5 tests in test_utils_features.py |
| UTI-04 | Plan 02 | Variable-length series processing | VERIFIED | general.py with collation and centering; 12 tests in test_utils_general.py |
| UTI-05 | Plan 03 | Each utility in separate file with __all__ exports | VERIFIED | 4 modules + common.py, each with __all__; utils/__init__.py wires 11 symbols |

No orphaned requirements -- all Phase 3 requirements (UTI-01 through UTI-05) appear in plan frontmatter.

### Anti-Patterns Found

None. No TBD, FIXME, XXX, TODO, PLACEHOLDER, empty returns, hardcoded empty data, or console.log stubs in any utility module.

### CLAUDE.md Style Compliance

| Rule | File | Status |
|------|------|--------|
| D-10: No `from __future__ import annotations` in new files | arff.py, scaling.py, features.py, general.py | COMPLIANT |
| D-07: No local DataFormEnum in scaling.py | scaling.py | COMPLIANT |
| D-06: Enum member comparisons (not strings) | scaling.py lines 87, 89 | COMPLIANT |
| D-09: Keyword-only args on multi-param functions | general.py custom_collate_fn (line 16: `*, desired_batch_size`) | COMPLIANT |
| Google-style docstrings | All modules | COMPLIANT |
| `__all__` exports in every module | All modules | COMPLIANT |
| Type hints on all functions | All modules | COMPLIANT |
| snake_case for functions, PascalCase for classes | All modules | COMPLIANT |

### Human Verification Required

None -- all deliverables are programmatically verifiable.

### Gaps Summary

No gaps found. All 14 must-have truths verified against the codebase. All artifacts exist, are substantive, are properly wired, and produce real data flows. All 5 requirements (UTI-01 through UTI-05) are satisfied with comprehensive tests. Full test suite (76 tests) passes.

---

_Verified: 2026-05-13T10:00:00Z_
_Verifier: Claude (gsd-verifier)_
