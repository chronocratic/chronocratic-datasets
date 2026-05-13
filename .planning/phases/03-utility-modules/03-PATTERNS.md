# Phase 3: Utility Modules - Pattern Map

**Mapped:** 2026-05-13
**Files analyzed:** 8 (4 new, 2 modified, 4 test)
**Analogs found:** 8 / 8

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/tscollection/datasets/utils/arff.py` | utility | file-I/O | `src/tscollection/datasets/datasets/transformations.py` | exact |
| `src/tscollection/datasets/utils/scaling.py` | utility | transform | `src/tscollection/datasets/datasets/transformations.py` | exact |
| `src/tscollection/datasets/utils/features.py` | utility | transform | `src/tscollection/datasets/utils/common.py` | role-match |
| `src/tscollection/datasets/utils/general.py` | utility | transform | `src/tscollection/datasets/datasets/transformations.py` | exact |
| `src/tscollection/datasets/utils/common.py` | utility | transform | itself (extend existing) | exact |
| `src/tscollection/datasets/utils/__init__.py` | config | request-response | itself (extend existing) | exact |
| `src/tscollection/datasets/enums/data.py` | model | CRUD | itself (extend existing) | exact |
| `tests/test_utils_arff.py` | test | CRUD | `tests/test_transformations.py` | exact |
| `tests/test_utils_scaling.py` | test | CRUD | `tests/test_transformations.py` | exact |
| `tests/test_utils_features.py` | test | CRUD | `tests/test_transformations.py` | exact |
| `tests/test_utils_general.py` | test | CRUD | `tests/test_strategies.py` | role-match |

## Pattern Assignments

### `src/tscollection/datasets/utils/arff.py` (utility, file-I/O)

**Analog:** `src/tscollection/datasets/datasets/transformations.py`

**Module docstring** (transformations.py line 1):
```python
"""Data transformation helpers for PyTorch datasets."""
```
New file should use:
```python
"""ARFF file reading utilities for time series datasets."""
```

**Imports pattern** (transformations.py lines 3-7):
```python
from __future__ import annotations

import numpy as np
import torch

__all__ = ['convert_data_to_np_array', 'convert_numpy_to_tensor', 'expand_data_dimensionality']
```
Note: arff.py does NOT need `from __future__ import annotations` per D-10 (no circular imports). Pandas imports are used at runtime. Scipy is lazy-imported inside functions.

**Lazy import pattern** (rbspaper arff.py lines 26-28):
```python
def read_arff_as_df(arff_file_path: Path | str) -> tuple[pd.DataFrame, Any]:
    from scipy.io import arff

    data, meta = arff.loadarff(arff_file_path)
```
Scipy is imported inside the function to avoid top-level dependency on scipy.

**Function signature pattern** (transformations.py lines 11-12):
```python
def convert_numpy_to_tensor(data: np.ndarray, dtype: str = 'float') -> torch.Tensor:
```
Type hints on all params and return. Default values use positional params.

**`__all__` export pattern** (transformations.py line 8):
```python
__all__ = ['convert_data_to_np_array', 'convert_numpy_to_tensor', 'expand_data_dimensionality']
```
Public functions listed alphabetically. Only public API (no private helpers).

**New file `__all__`:**
```python
__all__ = ['process_df_according_to_dtypes', 'read_arff_as_df']
```

**TYPE_CHECKING for `Any`** (rbspaper arff.py lines 13-14):
```python
if TYPE_CHECKING:
    from typing import Any
```
`Any` is only needed for type hints, not at runtime.

---

### `src/tscollection/datasets/utils/scaling.py` (utility, transform)

**Analog:** `src/tscollection/datasets/datasets/transformations.py` (structure) + `src/tscollection/datasets/enums/data.py` (enum wiring)

**Imports pattern** (transformations.py lines 3-7, adapted):
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tscollection.datasets.enums.data import DataForm, ScalingMethod
from tscollection.datasets.utils.common import flatten_list_of_np_arrays
```
Key differences from source:
- `src.rbspaper.data.utils.common` becomes `tscollection.datasets.utils.common` (D-11)
- `DataFormEnum` becomes `DataForm` from `enums/data.py` (D-07)
- No `from __future__ import annotations` (D-10)

**TYPE_CHECKING guard** (rbspaper scaling.py lines 16-18):
```python
if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any
```

**`__all__` export** -- only public API:
```python
__all__ = ['create_data_scaler']
```
Private helpers (`_get_scaler`, `_scale_regular_data`, `_scale_regular_data_and_return_same_type`, `_scale_multi_file_data`, `_scale_nested_data_all_dimensions`) are NOT exported.

**Enum wiring pattern** (data.py lines 19-24):
```python
class ScalingMethod(StrEnum):
    """Method for data scaling."""

    NONE = 'none'
    MINMAX = 'minmax'
    STANDARD = 'standard'
```
New `create_data_scaler` signature uses enum members directly (D-05, D-06):
```python
def create_data_scaler(
    *,
    scale: bool,
    scaling_range: tuple[float, float],
    scaling_method: ScalingMethod = ScalingMethod.MINMAX,
    data_form: DataForm = DataForm.REGULAR,
) -> Callable:
```

**Enum comparison pattern** -- source uses string comparison (rbspaper scaling.py lines 104-107):
```python
if scaling_method == 'min_max':          # OLD source
    return MinMaxScaler(feature_range=scaling_range)
if scaling_method == 'standardization':  # OLD source
    return StandardScaler()
```
Must change to enum comparison (D-06):
```python
if scaling_method == ScalingMethod.MINMAX:        # 'minmax'
    return MinMaxScaler(feature_range=scaling_range)
if scaling_method == ScalingMethod.STANDARD:      # 'standard'
    return StandardScaler()
```

**DataForm enum comparison** -- source uses `DataFormEnum.REGULAR.value` (rbspaper scaling.py line 62):
```python
if data_form == DataFormEnum.REGULAR.value:  # OLD source
```
Must change to direct enum comparison:
```python
if data_form == DataForm.REGULAR:  # NEW - StrEnum auto-coerces
```

**Keyword-only factory pattern** (rbspaper scaling.py lines 35-41):
```python
def create_data_scaler(
    *,
    scale: bool,
    scaling_range: tuple[float, float],
    scaling_method: str = 'min_max',
    data_form: str = DataFormEnum.REGULAR.value,
) -> Callable:
```
The `*` forces all params to be keyword-only. The returned `scale_data` closure accepts positional args but the factory is keyword-only.

---

### `src/tscollection/datasets/utils/features.py` (utility, transform)

**Analog:** `src/tscollection/datasets/utils/common.py` (same utils package)

**Module docstring** (common.py line 1):
```python
"""Common utility functions for time series data processing."""
```
New file should use:
```python
"""Time feature extraction utilities for forecasting datasets."""
```

**Imports pattern** -- no `from __future__ import annotations` (D-10):
```python
import numpy as np
import pandas as pd

__all__ = ['extract_time_features']
```

**Function signature** (rbspaper features.py lines 11-12):
```python
def extract_time_features(datetime_index: pd.DatetimeIndex) -> np.ndarray:
```
Single required positional arg -- no `*` keyword-only separator needed.

**Google-style docstring** (rbspaper features.py lines 12-21):
```python
    """Extract cyclical time features from a DatetimeIndex.

    Produces a 2-D array with columns: minute, hour, dayofweek,
    day, dayofyear, month, week.

    Args:
        datetime_index: A pandas DatetimeIndex.

    Returns:
        2-D numpy array of shape (len(index), 7) with dtype float32.
    """
```

**Core pattern** (rbspaper features.py lines 23-35):
```python
    series = datetime_index.to_series()
    return np.stack(
        [
            series.dt.minute.to_numpy(),
            series.dt.hour.to_numpy(),
            series.dt.dayofweek.to_numpy(),
            series.dt.day.to_numpy(),
            series.dt.dayofyear.to_numpy(),
            series.dt.month.to_numpy(),
            series.dt.isocalendar().week.to_numpy(),
        ],
        axis=1,
    ).astype(np.float32)
```
Self-contained. No cross-utility imports. Uses `isocalendar().week` (verified pandas 3.0.2 compatible).

---

### `src/tscollection/datasets/utils/general.py` (utility, transform)

**Analog:** `src/tscollection/datasets/datasets/transformations.py` (structure + data processing)

**Module docstring**:
```python
"""General data utilities: collation, variable-length handling."""
```

**Imports pattern** -- `default_collate` is used at runtime, NOT behind TYPE_CHECKING:
```python
import numpy as np
import pandas as pd
from torch.utils.data.dataloader import default_collate

__all__ = [
    'centralize_variable_length_series',
    'custom_collate_fn',
    'process_data_with_varying_sequence_lengths_single',
]

if TYPE_CHECKING:
    from typing import Any
```

**Function with keyword-only separator** (rbspaper general.py lines 21-22):
```python
def custom_collate_fn(batch: list[Any], desired_batch_size: int) -> Any:
```
`batch` is required positional. `desired_batch_size` should be keyword-only per D-09:
```python
def custom_collate_fn(batch: list[Any], *, desired_batch_size: int) -> Any:
```

**Google-style docstring pattern** (rbspaper general.py lines 22-32):
```python
    """Collate function that pads the last batch by cycling samples.

    If the current batch is smaller than *desired_batch_size*, extra
    samples are appended by cycling backwards through the batch.

    Args:
        batch: A list of samples returned by the dataset.
        desired_batch_size: Target batch size.

    Returns:
        Standard collated tensor batch.
    """
```

**`__all__` alphabetical order** -- all three public functions, sorted alphabetically.

---

### `src/tscollection/datasets/utils/common.py` MODIFY (utility, transform)

**Analog:** itself (existing file)

**Current state** (lines 1-60):
```python
"""Common utility functions for time series data processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

__all__ = ['FunctionComposer', 'compose', 'get_num_samples_from_ts']
```

**Critical change** -- `import numpy as np` must move OUT of `TYPE_CHECKING` because `flatten_list_of_np_arrays` calls `np.concatenate()` at runtime. Research Pitfall 3.

**New imports pattern:**
```python
"""Common utility functions for time series data processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
```

**New `__all__`:**
```python
__all__ = ['FunctionComposer', 'compose', 'flatten_list_of_np_arrays', 'get_num_samples_from_ts']
```

**New function** (rbspaper common.py lines 107-116):
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
Add after existing functions (after `compose`, before end of file).

---

### `src/tscollection/datasets/utils/__init__.py` MODIFY (config, request-response)

**Analog:** itself (existing file)

**Current state** (lines 1-13):
```python
"""Utility functions for data processing."""

from tscollection.datasets.utils.common import (
    compose,
    FunctionComposer,
    get_num_samples_from_ts,
)

__all__ = [
    'FunctionComposer',
    'compose',
    'get_num_samples_from_ts',
]
```

**New wiring pattern** -- mirror the enums `__init__.py` structure (enums/__init__.py lines 3-21):
```python
from tscollection.datasets.utils.arff import (
    process_df_according_to_dtypes,
    read_arff_as_df,
)
from tscollection.datasets.utils.common import (
    compose,
    flatten_list_of_np_arrays,
    FunctionComposer,
    get_num_samples_from_ts,
)
from tscollection.datasets.utils.features import extract_time_features
from tscollection.datasets.utils.general import (
    centralize_variable_length_series,
    custom_collate_fn,
    process_data_with_varying_sequence_lengths_single,
)
from tscollection.datasets.utils.scaling import create_data_scaler

__all__ = [
    'FunctionComposer',
    'centralize_variable_length_series',
    'compose',
    'create_data_scaler',
    'custom_collate_fn',
    'extract_time_features',
    'flatten_list_of_np_arrays',
    'get_num_samples_from_ts',
    'process_data_with_varying_sequence_lengths_single',
    'process_df_according_to_dtypes',
    'read_arff_as_df',
]
```
Alphabetical order in both imports and `__all__`.

---

### `src/tscollection/datasets/enums/data.py` MODIFY (model, CRUD)

**Analog:** itself (existing file)

**Current pattern** (lines 1-61):
```python
from enum import StrEnum


class TimeSeriesDatasetMode(StrEnum):
    """Mode for how the dataset yields samples."""

    WITH_LABELS = 'with_labels'
    WITHOUT_LABELS = 'without_labels'
    FORECASTING = 'forecasting'


class ScalingMethod(StrEnum):
    """Method for data scaling."""

    NONE = 'none'
    MINMAX = 'minmax'
    STANDARD = 'standard'
```

**New enum pattern** -- add `DataForm` following same StrEnum convention, no "Enum" suffix (D-07, Pitfall 6):
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
Add after existing classes. Insert in alphabetical position (after `DatasetFamily`, before `DistanceMetric`).

**Also modify** `src/tscollection/datasets/enums/__init__.py` to add `DataForm` to imports and `__all__`:
```python
from tscollection.datasets.enums.data import (
    DataForm,
    DatasetFamily,
    DistanceMetric,
    ForecastingMode,
    ScalingMethod,
    SplitMode,
    SplittingStrategy,
    TimeSeriesDatasetMode,
)

__all__ = [
    'DataForm',
    'DatasetFamily',
    'DistanceMetric',
    'ForecastingMode',
    'ScalingMethod',
    'SplitMode',
    'SplittingStrategy',
    'TimeSeriesDatasetMode',
]
```

**Also modify** `src/tscollection/datasets/__init__.py` to add `DataForm` to re-exports:
```python
from tscollection.datasets.enums import (
    DataForm,
    DatasetFamily,
    DistanceMetric,
    ForecastingMode,
    ScalingMethod,
    SplitMode,
    SplittingStrategy,
    TimeSeriesDatasetMode,
)

__all__ = [
    'DataForm',
    'DatasetFamily',
    'DistanceMetric',
    'ForecastingMode',
    'ScalingMethod',
    'SplitMode',
    'SplittingStrategy',
    'TimeSeriesDatasetMode',
    '__version__',
]
```

---

### `tests/test_utils_arff.py` (test, CRUD)

**Analog:** `tests/test_transformations.py`

**Module docstring pattern** (test_transformations.py lines 1-5):
```python
"""Tests for data transformation utilities (DST-01, DST-02).

Verifies that convert_numpy_to_tensor, expand_data_dimensionality,
and convert_data_to_np_array produce correct output types and shapes.
"""
```

**Imports pattern** (test_transformations.py lines 7-14):
```python
import numpy as np
import torch

from tscollection.datasets.datasets.transformations import (
    convert_data_to_np_array,
    convert_numpy_to_tensor,
    expand_data_dimensionality,
)
```

**Test function signature** (test_transformations.py lines 17-22):
```python
def test_convert_numpy_to_tensor_float():
    """DST-01: convert_numpy_to_tensor returns torch.Tensor with dtype torch.float."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = convert_numpy_to_tensor(data, dtype='float')
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float
```

**Requirements ID prefix pattern** (test_transformations.py): Tests reference requirement IDs like `DST-01:` in docstrings.
New tests should use `UTI-01:` prefix.

**New test structure:**
```python
"""Tests for ARFF utility functions (UTI-01).

Verifies that read_arff_as_df and process_df_according_to_dtypes
correctly parse ARFF files and transform DataFrame columns.
"""

import numpy as np
import pandas as pd
import pytest

from tscollection.datasets.utils.arff import (
    process_df_according_to_dtypes,
    read_arff_as_df,
)


def test_read_arff_as_df_returns_dataframe_and_metadata(tmp_path):
    """UTI-01: read_arff_as_df returns a DataFrame and ARFF metadata."""
    ...
```

---

### `tests/test_utils_scaling.py` (test, CRUD)

**Analog:** `tests/test_transformations.py`

**Test imports:**
```python
import numpy as np
import pandas as pd
import pytest

from tscollection.datasets.enums.data import DataForm, ScalingMethod
from tscollection.datasets.utils.scaling import create_data_scaler
```

**Test structure** -- follows `test_strategies.py` pattern (lines 22-31) with descriptive docstrings and `-> None` return type:
```python
def test_create_data_scaler_regular_minmax() -> None:
    """UTI-02: create_data_scaler with REGULAR data and MinMax scaling.

    Train data is fit, all splits are transformed.
    Values should be in the specified range.
    """
    scaler_fn = create_data_scaler(
        scale=True,
        scaling_range=(0.0, 1.0),
        scaling_method=ScalingMethod.MINMAX,
        data_form=DataForm.REGULAR,
    )
    ...
```

**Section separator pattern** (test_strategies.py lines 17-19):
```python
# --------------------------------------------------------------------------- #
# ForecastingStrategySingleFile tests                                          #
# --------------------------------------------------------------------------- #
```
Use this pattern to group tests by data form (REGULAR, NESTED, MULTI_FILES).

---

### `tests/test_utils_features.py` (test, CRUD)

**Analog:** `tests/test_transformations.py`

**Test imports:**
```python
import numpy as np
import pandas as pd

from tscollection.datasets.utils.features import extract_time_features
```

**Test pattern** -- create DatetimeIndex, call function, assert shape and dtype:
```python
def test_extract_time_features_shape() -> None:
    """UTI-03: extract_time_features returns (N, 7) float32 array."""
    dti = pd.date_range('2020-01-01', periods=10, freq='h')  # lowercase per pandas 3.0
    result = extract_time_features(dti)
    assert result.shape == (10, 7)
    assert result.dtype == np.float32
```

**CRITICAL: pandas 3.0 frequency alias** (Research Pitfall 4):
```python
# WRONG (pandas 3.0):
pd.date_range(freq='H')
# CORRECT:
pd.date_range(freq='h')
```

---

### `tests/test_utils_general.py` (test, CRUD)

**Analog:** `tests/test_strategies.py` (has section separators, multiple assertion patterns)

**Test imports:**
```python
import numpy as np
import pandas as pd
import torch
import pytest

from tscollection.datasets.utils.general import (
    centralize_variable_length_series,
    custom_collate_fn,
    process_data_with_varying_sequence_lengths_single,
)
```

**Test structure** -- follows section separator pattern:
```python
# --------------------------------------------------------------------------- #
# custom_collate_fn tests                                                      #
# --------------------------------------------------------------------------- #


def test_custom_collate_fn_pads_last_batch() -> None:
    """UTI-04: custom_collate_fn pads short batches by cycling."""
    batch = [torch.tensor([1.0]), torch.tensor([2.0])]
    result = custom_collate_fn(batch, desired_batch_size=4)
    assert result.shape[0] == 4


# --------------------------------------------------------------------------- #
# centralize_variable_length_series tests                                       #
# --------------------------------------------------------------------------- #


def test_centralize_variable_length_series() -> None:
    """UTI-04: Centering shifts valid data to middle of sequence."""
    ...
```

## Shared Patterns

### Module Structure
**Source:** `src/tscollection/datasets/datasets/transformations.py`
**Apply to:** All new utility files (`arff.py`, `scaling.py`, `features.py`, `general.py`)
```python
"""Module docstring -- one line describing purpose."""

import numpy as np
import pandas as pd

__all__ = ['function_a', 'function_b']

if TYPE_CHECKING:
    from typing import Any


def function_a(param: SomeType) -> ReturnType:
    """Google-style docstring.

    Args:
        param: Description.

    Returns:
        Description.
    """
    ...
```

### Enum Pattern
**Source:** `src/tscollection/datasets/enums/data.py`
**Apply to:** `DataForm` enum addition
```python
class DataForm(StrEnum):
    """Docstring.

    Attributes:
        MEMBER: Description.
    """

    MEMBER = 'member_value'
```
- Uses `StrEnum` (not `str, Enum`)
- No "Enum" suffix in class name
- Lowercase snake_case values
- UPPER_CASE member names

### Import Wiring Pattern
**Source:** `src/tscollection/datasets/enums/__init__.py`
**Apply to:** `utils/__init__.py`, `enums/__init__.py`, `datasets/__init__.py`
- Alphabetical order in import blocks
- Alphabetical order in `__all__` lists
- Group imports by source module

### Test Pattern
**Source:** `tests/test_transformations.py`, `tests/test_strategies.py`
**Apply to:** All new test files
- Module docstring lists requirement IDs and tested functions
- Test functions prefixed with `test_` and descriptive name
- Docstrings include requirement ID prefix (`UTI-01:`, `UTI-02:`, etc.)
- Keyword arguments for function calls per CLAUDE.md
- Lowercase frequency aliases (`freq='h'` not `freq='H'`)
- Section separators for grouping tests by concern

### No `from __future__ import annotations`
**Source:** D-10 decision
**Apply to:** All new utility files (`arff.py`, `scaling.py`, `features.py`, `general.py`)
Only existing files that were created before this decision use it. New files should omit it unless circular imports occur.

### TYPE_CHECKING Guard
**Source:** `src/tscollection/datasets/utils/common.py` (lines 5-10)
**Apply to:** Files that import types only for annotations
```python
if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Callable
```
Runtime-needed imports (numpy, pandas, torch, sklearn) must be top-level.

## No Analog Found

All files have close analogs in the codebase. No files require RESEARCH.md patterns as primary source.

## Metadata

**Analog search scope:** `src/tscollection/datasets/`, `tests/`, `_sources/rbspaper/src/rbspaper/data/utils/`
**Files scanned:** 16 source files, 10 test files, 6 rbspaper source files
**Pattern extraction date:** 2026-05-13

---

## PATTERN MAPPING COMPLETE

**Phase:** 03 - Utility Modules
**Files classified:** 11
**Analogs found:** 11 / 11

### Coverage
- Files with exact analog: 7
- Files with role-match analog: 4
- Files with no analog: 0

### Key Patterns Identified
- All utility modules use module-level docstring + `__all__` + Google-style docstrings
- `StrEnum` without "Enum" suffix for new enums (matches existing enum pattern in data.py)
- No `from __future__ import annotations` in new files per D-10
- Runtime numpy import must be top-level, not behind TYPE_CHECKING (Pitfall 3)
- `ScalingMethod` and `DataForm` enums replace string comparisons (D-05, D-06, D-07)
- Keyword-only separator (`*`) for optional params in function signatures (D-09)
- Tests use requirement ID prefix in docstrings (`UTI-01:`) and lowercase pandas freq aliases
- Import wiring uses alphabetical order in both `from ... import` blocks and `__all__` lists
- Lazy import pattern for scipy inside `read_arff_as_df()` to avoid top-level dependency

### File Created
`/Users/skaf/VSCodeProjects/tsdatasets/.planning/phases/03-utility-modules/03-PATTERNS.md`

### Ready for Planning
Pattern mapping complete. Planner can now reference analog patterns in PLAN.md files.
