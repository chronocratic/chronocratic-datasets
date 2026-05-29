# Phase 07: DDP Compliance + `_full_data` Split - Pattern Map

**Mapped:** 2026-05-29
**Files analyzed:** 13 (3 new, 10 modified)
**Analogs found:** 13 / 13

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/tscollection/datasets/utils/cache.py` | utility | file-I/O | `src/tscollection/datasets/utils/scaling.py` | role-match |
| `src/tscollection/datasets/utils/__init__.py` | config | request-response | `src/tscollection/datasets/utils/__init__.py` (self) | exact |
| `src/tscollection/datasets/modules/_base/base.py` | component | request-response | `src/tscollection/datasets/modules/_base/base.py` (self) | exact |
| `src/tscollection/datasets/modules/_base/forecasting.py` | component | CRUD | `src/tscollection/datasets/modules/_base/forecasting.py` (self) | exact |
| `src/tscollection/datasets/modules/_base/classification.py` | component | CRUD | `src/tscollection/datasets/modules/_base/classification.py` (self) | exact |
| `src/tscollection/datasets/modules/ett.py` | component | CRUD | `src/tscollection/datasets/modules/ett.py` (self) | exact |
| `src/tscollection/datasets/modules/weather.py` | component | CRUD | `src/tscollection/datasets/modules/weather.py` (self) | exact |
| `src/tscollection/datasets/modules/electricity.py` | component | CRUD | `src/tscollection/datasets/modules/electricity.py` (self) | exact |
| `src/tscollection/datasets/modules/ucr.py` | component | CRUD | `src/tscollection/datasets/modules/ucr.py` (self) | exact |
| `src/tscollection/datasets/modules/uea.py` | component | CRUD | `src/tscollection/datasets/modules/uea.py` (self) | exact |
| `tests/test_cache.py` | test | file-I/O | `tests/test_utils_scaling.py` | role-match |
| `tests/test_ddp_compliance.py` | test | event-driven | `tests/test_modules_forecasting.py` (integration tests) | role-match |
| `tests/conftest.py` | config | file-I/O | `tests/conftest.py` (self) | exact |

## Pattern Assignments

### `src/tscollection/datasets/utils/cache.py` (utility, file-I/O)

**Analog:** `src/tscollection/datasets/utils/scaling.py`

**Docstring + imports pattern** (scaling.py lines 1-13):
```python
"""Data scaling strategies for time series datasets."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tscollection.datasets.enums.data import DataForm, ScalingMethod
from tscollection.datasets.utils.common import flatten_list_of_np_arrays

__all__ = ['create_data_scaler']
```
Apply: cache.py will use a module-level docstring, stdlib imports first (`hashlib`, `json`, `os`), then third-party (`numpy`, `torch`, `pandas`), then project imports. `__all__` list at top.

**Pure function with keyword-only args pattern** (scaling.py lines 16-22):
```python
def create_data_scaler(
    *,
    scale: bool,
    scaling_range: tuple[float, float],
    scaling_method: ScalingMethod = ScalingMethod.MINMAX,
    data_form: DataForm = DataForm.REGULAR,
) -> Callable:
```
Apply: All cache functions use `*` to enforce keyword-only args, full type hints, return types, Google-style docstrings.

**Private helper pattern** (scaling.py lines 74-91):
```python
def _get_scaler(
    scaling_method: ScalingMethod, scaling_range: tuple[float, float]
) -> MinMaxScaler | StandardScaler:
    """Instantiate the appropriate sklearn scaler.

    Args:
        scaling_method: Scaling algorithm identifier.
        scaling_range: Target range for MinMaxScaler.

    Returns:
        A scaler instance ready for fitting.
    """
    if scaling_method == ScalingMethod.MINMAX:
        return MinMaxScaler(feature_range=scaling_range)
    if scaling_method == ScalingMethod.STANDARD:
        return StandardScaler()
    msg = f'Unsupported scaling method: {scaling_method}'
    raise ValueError(msg)
```
Apply: Private helpers (prefixed `_`) used internally, error messages use `msg = f'...'` then `raise ValueError(msg)` pattern.

**Analog:** `src/tscollection/datasets/utils/features.py` (simpler utility pattern)

**Module-level constant + `__all__` pattern** (features.py lines 1-8):
```python
"""Time feature extraction utilities for forecasting datasets."""

import numpy as np
import pandas as pd

__all__ = ['TIME_FEATURE_COUNT', 'extract_time_features']

TIME_FEATURE_COUNT: int = 7
```
Apply: cache.py exports constant `CACHE_SCHEMA_VERSION: int = 1`.

**Analog:** `src/tscollection/datasets/utils/general.py` (file-I/O utility pattern)

**`__all__` + pure functions pattern** (general.py lines 1-13):
```python
"""General data utilities: collation, variable-length handling."""

from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data.dataloader import default_collate

__all__ = [
    'centralize_variable_length_series',
    'custom_collate_fn',
    'process_data_with_varying_sequence_lengths_single',
]
```
Apply: cache.py follows same import ordering: stdlib -> typing -> third-party -> project imports.

---

### `src/tscollection/datasets/utils/__init__.py` (config, request-response)

**Analog:** `src/tscollection/datasets/utils/__init__.py` (self -- existing file)

**Current barrel export pattern** (lines 1-33):
```python
"""Utility functions for data processing."""

from tscollection.datasets.utils.arff import process_df_according_to_dtypes, read_arff_as_df
from tscollection.datasets.utils.common import (
    compose,
    flatten_list_of_np_arrays,
    FunctionComposer,
    get_num_samples_from_ts,
    separate_target_feature_from_df,
)
from tscollection.datasets.utils.features import TIME_FEATURE_COUNT, extract_time_features
from tscollection.datasets.utils.general import (
    centralize_variable_length_series,
    custom_collate_fn,
    process_data_with_varying_sequence_lengths_single,
)
from tscollection.datasets.utils.scaling import create_data_scaler

__all__ = [
    'FunctionComposer',
    'TIME_FEATURE_COUNT',
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
    'separate_target_feature_from_df',
]
```
Apply: Add `from tscollection.datasets.utils.cache import (...)` import block and append cache exports to `__all__` in alphabetical order. Note: `__all__` is sorted alphabetically (F, T, c, c, c, c, e, f, g, p, p, r, s).

---

### `src/tscollection/datasets/modules/_base/base.py` (component, request-response)

**Analog:** `src/tscollection/datasets/modules/_base/base.py` (self -- existing file)

**Constructor pattern with keyword-only args** (lines 54-89):
```python
    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int | None,
        valid_size: float,
        test_size: float,
        shuffle: bool,
        scale_data: bool,
        data_scaling_method: ScalingMethod = ScalingMethod.MINMAX,
        data_scaling_range: tuple[float, float] = (0, 1),
        num_workers: int = 0,
        data_form: DataForm = DataForm.REGULAR,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        # ... assign params ...
        self._setup_completed_stages: set[str | None] = set()
        self._prepare_data_called: bool = False
        self._scaler_cache: Callable[..., tuple[Any, Any, Any]] | None = None
```
Apply: Add `cache_dir: Path | None = None` param after `data_form`. Store as `self._cache_dir`. Initialize new attrs `_full_data_raw`, `_time_index`, `_full_data_scaled` to None.

**Class attribute for Lightning hook pattern** (not yet in base, but per D-05):
Apply: Add `prepare_data_per_node: bool = True` as class attribute, same pattern as Lightning's convention. Place after docstring, before `__init__`.

**Property pattern** (lines 95-123):
```python
    @property
    def name(self) -> str | None:
        """Dataset name."""
        return self._dataset_name

    @property
    def sequence_length(self) -> int | None:
        """Sequence length (read-only)."""
        return self._seq_len
```
Apply: No new properties needed for base; `_full_data_raw` etc. are internal.

**`prepare_data()` template pattern** (lines 152-169):
```python
    def prepare_data(self) -> None:
        if self._prepare_data_called:
            return
        self._do_prepare_data()
        self._finalize_prepare_data()
        self._prepare_data_called = True
```
Apply: Unchanged structure. `_do_prepare_data()` in concrete modules will now write to cache instead of `self._full_data`.

**`prepare_dimensions()` + `_compute_dimensions()` pattern** (lines 175-205):
```python
    def prepare_dimensions(self) -> tuple[int | None, int | None]:
        if self._num_features is not None:
            return self._num_features, self._seq_len
        return self._compute_dimensions()

    def _compute_dimensions(self) -> tuple[int | None, int | None]:
        return self._num_features, self._seq_len
```
Apply: Replace `_compute_dimensions()` to read `metadata.json` via `load_metadata()`. Pattern: check cached attrs first, then disk read, raise if missing.

**`reset()` pattern** (lines 296-305):
```python
    def reset(self) -> None:
        """Clear lifecycle sentinels to allow re-use of this DataModule."""
        self._setup_completed_stages.clear()
        self._prepare_data_called = False
```
Apply: Extend to also clear `_full_data_raw`, `_time_index`, `_full_data_scaled`, `_data_scaler_cache`, `_ts_feature_scaler_cache`, and data sample attrs.

---

### `src/tscollection/datasets/modules/_base/forecasting.py` (component, CRUD)

**Analog:** `src/tscollection/datasets/modules/_base/forecasting.py` (self -- existing file)

**Constructor + attrs pattern** (lines 51-83):
```python
    def __init__(
        self,
        *,
        batch_size: int = 32,
        seq_len: int = 128,
        # ...
        mode: ForecastingMode = ForecastingMode.UNIVARIATE,
    ) -> None:
        super().__init__(batch_size=batch_size, seq_len=seq_len, ...)
        self._mode = mode
        self._train_slice: slice | None = None
        self._valid_slice: slice | None = None
        self._test_slice: slice | None = None
        self._full_data: np.ndarray | pd.DataFrame | None = None
        self._num_time_series_features: int | None = None
        self._data_scaler_cache: MinMaxScaler | StandardScaler | None = None
        self._ts_feature_scaler_cache: MinMaxScaler | StandardScaler | None = None
```
Apply: Replace `self._full_data` with three typed attrs: `self._full_data_raw: np.ndarray | None = None`, `self._time_index: pd.DatetimeIndex | None = None`, `self._full_data_scaled: np.ndarray | None = None`. Keep scaler caches.

**`_compute_dimensions()` isinstance branch to eliminate** (lines 134-154):
```python
    def _compute_dimensions(self) -> tuple[int | None, int | None]:
        if self._full_data is None:
            return None, self._seq_len
        if isinstance(self._full_data, pd.DataFrame):
            raw_cols = self._full_data.shape[-1]
            n_features = raw_cols + TIME_FEATURE_COUNT
            self._num_features = n_features
            return n_features, self._seq_len
        raw_cols = self._full_data.shape[-1]
        self._num_features = raw_cols
        return raw_cols, self._seq_len
```
Apply: After phase 7, replace with `prepare_dimensions()` reading from `metadata.json` (base class handles this). The forecasting override of `_compute_dimensions()` is removed entirely -- the base class reads metadata.

**`setup()` isinstance branches to eliminate** (lines 206-211):
```python
        if isinstance(self._full_data, pd.DataFrame):
            time_index = self._full_data.index
            full_array = self._full_data.to_numpy()
        else:
            time_index = None
            full_array = self._full_data
```
Apply: After phase 7, `_full_data_raw` is always `np.ndarray`, `_time_index` is always `pd.DatetimeIndex | None`. No isinstance needed: `time_index = self._time_index`, `full_array = self._full_data_raw`.

**`_finalize_prepare_data()` pattern** (lines 340-347):
```python
    def _finalize_prepare_data(self) -> None:
        """Hook called after ``_do_prepare_data()`` to set data slices."""
        self._set_data_slices()
```
Apply: CRITICAL CHANGE per Pitfall 5. For Weather/Electricity, `_set_data_slices()` depends on `len(self._full_data)`. After phase 7, `_full_data_raw` is not available until `setup()`. Move `_set_data_slices()` call from `_finalize_prepare_data()` into `setup()`, right after cache read. `_finalize_prepare_data()` becomes no-op for forecasting (or only ETT uses it since ETT slices are variant-based, not data-length-based).

**`_prepare_data_scaler()` pattern** (lines 296-314):
```python
    def _prepare_data_scaler(self) -> MinMaxScaler | StandardScaler:
        if self.data_scaling_method == ScalingMethod.MINMAX:
            return MinMaxScaler(feature_range=self.data_scaling_range)
        if self.data_scaling_method == ScalingMethod.STANDARD:
            return StandardScaler()
        msg = f'Unsupported scaling method for forecasting: {self.data_scaling_method}'
        raise ValueError(msg)
```
Apply: Unchanged. Add new helpers `_save_scaler_to_cache()` and `_load_scaler_from_cache()` that wrap `save_scaler()` / `load_scaler()` from cache.py.

**`_split_data()` pattern** (lines 321-338):
```python
    def _split_data(self) -> None:
        assert self._full_data is not None
        train_data = self._full_data[:, self._train_slice]
        self._train_data_samples = train_data
```
Apply: Change `self._full_data` to `self._full_data_scaled` (after phase 7, scaled data is separate attr).

---

### `src/tscollection/datasets/modules/_base/classification.py` (component, CRUD)

**Analog:** `src/tscollection/datasets/modules/_base/classification.py` (self -- existing file)

**Constructor pattern** (lines 64-104):
```python
    def __init__(
        self,
        *,
        dataset_folder_path: Path,
        batch_size: int = 32,
        # ...
        data_form: DataForm = DataForm.REGULAR,
    ) -> None:
        super().__init__(batch_size=batch_size, seq_len=None, ...)
        self.dataset_folder_path = dataset_folder_path
        # ... classification-specific attrs ...
```
Apply: Add `_cache_dir` inherited from base (via `cache_dir` constructor param). No new classification-specific attrs needed beyond what base provides.

**`_compute_dimensions()` pattern** (lines 143-160):
```python
    def _compute_dimensions(self) -> tuple[int | None, int | None]:
        if self._train_data_samples is None:
            msg = 'prepare_dimensions() requires prepare_data() to have run first'
            raise RuntimeError(msg)
        return self._num_features, self._seq_len
```
Apply: After phase 7, replace with `metadata.json` read pattern (base class handles). Classification overrides `_compute_dimensions()` to read metadata, falling back to RuntimeError if not available.

**`_process_data_with_varying_sequence_lengths()` pattern** (lines 193-209):
```python
    def _process_data_with_varying_sequence_lengths(self) -> None:
        self._train_data_samples = process_data_with_varying_sequence_lengths_single(
            data=self._train_data_samples
        )
```
Apply: Unchanged logic. Called from `_do_prepare_data()` in concrete modules before caching. Classification caches post-processed data (per RESEARCH.md recommendation).

---

### `src/tscollection/datasets/modules/ett.py` (component, CRUD)

**Analog:** `src/tscollection/datasets/modules/ett.py` (self -- existing file)

**Class-level attr pattern** (line 60):
```python
    _full_data: pd.DataFrame | np.ndarray | None = None
```
Apply: Remove this. After phase 7, forecasting base declares `_full_data_raw`, `_time_index`, `_full_data_scaled`.

**Constructor pattern** (lines 62-95):
```python
    def __init__(
        self,
        *,
        dataset_file_path: Path,
        variant: str,
        seq_len: int = 128,
        mode: ForecastingMode = ForecastingMode.UNIVARIATE,
        # ...
    ) -> None:
        if variant not in VALID_ETT_VARIANTS:
            msg = f'Unknown ETT variant: {variant!r}. Must be one of {sorted(VALID_ETT_VARIANTS)}'
            raise ValueError(msg)
        super().__init__(batch_size=batch_size, seq_len=seq_len, ...)
        self.dataset_file_path = dataset_file_path
        self.variant = variant
```
Apply: Add `cache_dir` param, pass to `super()`. Add `_cache_key` computation in constructor (from variant, seq_len, mode, scaling params).

**`_set_data_slices()` -- variant-based (independent of data length)** (lines 101-114):
```python
    def _set_data_slices(self) -> None:
        if self.variant in {'ETTh1', 'ETTh2'}:
            self._train_slice = slice(None, 12 * 30 * 24)
            self._valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
            self._test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
        else:  # ETTm1, ETTm2
            self._train_slice = slice(None, 12 * 30 * 24 * 4)
            self._valid_slice = slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
            self._test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)
```
Apply: Since ETT slices are variant-based (not data-length-based), they CAN be set in `_finalize_prepare_data()` (no dependency on `_full_data_raw`). However, for consistency, move to `setup()` after cache read.

**`_transform_data()` -- eliminate isinstance** (lines 116-125):
```python
    def _transform_data(self) -> None:
        assert self._full_data is not None, '_full_data was not set by prepare_data()'
        if isinstance(self._full_data, pd.DataFrame):
            self._full_data = self._full_data.to_numpy()
        if isinstance(self._full_data, np.ndarray):
            self._full_data = np.expand_dims(self._full_data, axis=0)
```
Apply: After phase 7, `_full_data_raw` is always `np.ndarray` (loaded from cache). Simplify to: `self._full_data_scaled = np.expand_dims(self._full_data_raw, axis=0)`. No isinstance check.

**`_do_prepare_data()` -- write cache instead of self._full_data** (lines 131-150):
```python
    def _do_prepare_data(self) -> None:
        if not self.dataset_file_path.exists():
            msg = f'Dataset file not found: {self.dataset_file_path}'
            raise FileNotFoundError(msg)
        self._dataset_name = self.variant
        df = pd.read_csv(self.dataset_file_path, parse_dates=True, index_col='date')
        if self._mode == ForecastingMode.UNIVARIATE:
            df = df[['OT']]
        self._full_data = df
```
Apply: Keep CSV read logic. Replace `self._full_data = df` with cache write: extract data as `df.to_numpy()`, extract index as `df.index.astype(np.int64).to_numpy()`, call `atomic_save_npz(cache_path, data=data, index=index_ns)`, write `metadata.json` via `atomic_save_metadata()`. Set `_dataset_name` and `_time_index` as before.

**Dataloader pattern** (lines 156-231):
```python
    def train_dataloader(self, *, mode=..., shuffle=..., strict_batch_size=..., extra_args=...) -> DataLoader:
        tensor = torch.from_numpy(self._train_data_samples).to(torch.float32)
        return self._process_train_dataloader(dataset_object=TensorDataset(tensor), ...)
```
Apply: Unchanged. Dataloaders use `_train_data_samples` etc. which are set by `_split_data()` in `setup()`.

---

### `src/tscollection/datasets/modules/weather.py` (component, CRUD)

**Analog:** `src/tscollection/datasets/modules/weather.py` (self -- existing file)

**`_set_data_slices()` -- DEPENDS on `len(self._full_data)`** (lines 86-92):
```python
    def _set_data_slices(self) -> None:
        assert self._full_data is not None, '_full_data was not set by prepare_data()'
        num_samples = len(self._full_data)
        self._train_slice = slice(None, int(0.6 * num_samples))
        self._valid_slice = slice(int(0.6 * num_samples), int(0.8 * num_samples))
        self._test_slice = slice(int(0.8 * num_samples), None)
```
Apply: CRITICAL -- must move to `setup()` after cache read. Change `self._full_data` to `self._full_data_raw`. Remove assertion (cache guarantees data exists).

**`_transform_data()` -- expand_dims(axis=0)** (lines 94-104):
```python
    def _transform_data(self) -> None:
        assert self._full_data is not None, '_full_data was not set by prepare_data()'
        if isinstance(self._full_data, pd.DataFrame):
            self._full_data = self._full_data.to_numpy()
        if isinstance(self._full_data, np.ndarray):
            self._full_data = np.expand_dims(self._full_data, axis=0)
```
Apply: Same simplification as ETT. `_full_data_raw` is always `np.ndarray`: `self._full_data_scaled = np.expand_dims(self._full_data_raw, axis=0)`.

**`_do_prepare_data()` -- same CSV read + cache write pattern as ETT** (lines 110-127):
```python
    def _do_prepare_data(self) -> None:
        if not self.dataset_file_path.exists():
            msg = f'Dataset file not found: {self.dataset_file_path}'
            raise FileNotFoundError(msg)
        self._dataset_name = self.dataset_file_path.name
        df = pd.read_csv(self.dataset_file_path, parse_dates=True, index_col='date')
        if self._mode == ForecastingMode.UNIVARIATE:
            df = df.iloc[:, -1:]
        self._full_data = df
```
Apply: Same cache-write pattern as ETT. Note: `_dataset_name` derives from filename (not variant).

---

### `src/tscollection/datasets/modules/electricity.py` (component, CRUD)

**Analog:** `src/tscollection/datasets/modules/electricity.py` (self -- existing file)

**`_set_data_slices()` -- DEPENDS on `len(self._full_data)`** (lines 86-92):
```python
    def _set_data_slices(self) -> None:
        assert self._full_data is not None, '_full_data was not set by prepare_data()'
        num_samples = len(self._full_data)
        self._train_slice = slice(None, int(0.6 * num_samples))
        self._valid_slice = slice(int(0.6 * num_samples), int(0.8 * num_samples))
        self._test_slice = slice(int(0.8 * num_samples), None)
```
Apply: Same as Weather -- move to `setup()` after cache read. Use `self._full_data_raw`.

**`_transform_data()` -- transpose + expand_dims(axis=-1)** (lines 94-105):
```python
    def _transform_data(self) -> None:
        assert self._full_data is not None, '_full_data was not set by prepare_data()'
        if isinstance(self._full_data, pd.DataFrame):
            self._full_data = self._full_data.to_numpy()
        if isinstance(self._full_data, np.ndarray):
            self._full_data = self._full_data.T
            self._full_data = np.expand_dims(self._full_data, axis=-1)
```
Apply: Simplify: `self._full_data_scaled = np.expand_dims(self._full_data_raw.T, axis=-1)`.

**`_do_prepare_data()` -- semicolon CSV + resampling + filtering** (lines 111-135):
```python
    def _do_prepare_data(self) -> None:
        if not self.dataset_file_path.exists():
            msg = f'Dataset file not found: {self.dataset_file_path}'
            raise FileNotFoundError(msg)
        self._dataset_name = 'ElectricityLoad'
        df = pd.read_csv(self.dataset_file_path, parse_dates=True, sep=';', decimal=',', index_col=[0])
        df = df.resample('1h', closed='right').sum()
        df = df.loc[:, (df != 0).any(axis=0)]
        df.index = df.index.rename('date')
        df = df['2012':]
        if self._mode == ForecastingMode.UNIVARIATE:
            df = df[['MT_001']]
        self._full_data = df
```
Apply: Same cache-write pattern. Key difference: `_dataset_name` is hardcoded `'ElectricityLoad'` (not derived from filename). CSV has semicolon delimiter + comma decimal. Data is resampled and filtered before caching.

---

### `src/tscollection/datasets/modules/ucr.py` (component, CRUD)

**Analog:** `src/tscollection/datasets/modules/ucr.py` (self -- existing file)

**`_do_prepare_data()` -- ARFF read + split + variable-length processing** (lines 151-240):
```python
    def _do_prepare_data(self) -> None:
        if not self.dataset_folder_path.exists():
            msg = f'Dataset folder not found: {self.dataset_folder_path}'
            raise FileNotFoundError(msg)
        self._dataset_name = self.dataset_folder_path.name
        arff_train = self.dataset_folder_path / f'{self._dataset_name}_TRAIN.arff'
        arff_test = self.dataset_folder_path / f'{self._dataset_name}_TEST.arff'
        train_data = self._read_arff_file_as_df(arff_train)
        test_data = self._read_arff_file_as_df(arff_test)
        # ... clean, split, separate target, compute dims, create val split ...
        self._process_data_with_varying_sequence_lengths()
```
Apply: Keep all ARFF read + processing logic. After processing (variable-length centering), cache the result: `atomic_save_npz(cache_path, train_samples=train_data, train_labels=train_labels, test_samples=test_data, test_labels=test_labels, valid_samples=valid_data, valid_labels=valid_labels)`. Write `metadata.json` with dims. In `setup()`, read from cache.

**Key difference from forecasting:** Classification caches post-processed, already-split data. `setup()` becomes a trivial cache read + scaler application.

**`_datatype_handling_functions_map` pattern** (lines 102-114):
```python
    def _initiate_datatypes_handling_functions_map(self) -> None:
        self._datatype_handling_functions_map = defaultdict(
            lambda: lambda x: x,
            {
                'nominal': lambda x: x.str.decode('utf-8').astype('category').astype('int64'),
                'numeric': lambda x: x.astype('float64'),
            },
        )
```
Apply: Unchanged. This is UCR-specific ARFF parsing logic.

---

### `src/tscollection/datasets/modules/uea.py` (component, CRUD)

**Analog:** `src/tscollection/datasets/modules/uea.py` (self -- existing file)

**`_do_prepare_data()` -- scipy ARFF + nested processing** (lines 165-267):
```python
    def _do_prepare_data(self) -> None:
        if not self.dataset_folder_path.exists():
            msg = f'Dataset folder not found: {self.dataset_folder_path}'
            raise FileNotFoundError(msg)
        self._dataset_name = self.dataset_folder_path.name
        # ... read via scipy.io.arff, process, split ...
        self._process_data_with_varying_sequence_lengths()
        # Convert labels to pd.Series with category dtype
        self._train_data_labels = pd.Series(self._train_data_labels, dtype='category')
        # Compute module state
        self._num_classes = len(self._train_data_labels.unique())
        self._seq_len, self._num_features = self._train_data_samples[0].shape
```
Apply: Same pattern as UCR -- cache post-processed data. Key difference: UEA uses scipy `arff.loadarff` (not pandas-based ARFF reader). Data is 3-D numpy arrays `(samples, timesteps, features)`. Labels are `pd.Series` with category dtype. Cache needs to handle 3-D arrays (numpy savez_compressed handles this natively).

**`_process_stacked_data()` pattern** (lines 125-159):
```python
    def _process_stacked_data(self, data: Any) -> tuple[np.ndarray, np.ndarray]:
        processed_data: list[np.ndarray] = []
        labels: list[str] = []
        for sample, label in data:
            # ... decode bytes, build arrays ...
        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)
        output_data = np.array(processed_data).astype(np.float32).swapaxes(1, 2)
        return output_data, np.array(encoded_labels)
```
Apply: Unchanged. This is UEA-specific nested ARFF parsing.

---

### `tests/test_cache.py` (test, file-I/O)

**Analog:** `tests/test_utils_scaling.py`

**Module-level import + no class wrapping for simple tests** (scaling test lines 1-14):
```python
import numpy as np
import pandas as pd
import pytest

from tscollection.datasets.enums.data import DataForm, ScalingMethod
```
Apply: test_cache.py uses standalone `def test_...()` functions for simple utility tests. No class wrapper needed for flat utility functions.

**Import verification test** (scaling test lines 19-23):
```python
def test_create_data_scaler_import() -> None:
    """create_data_scaler is importable from scaling module."""
    from tscollection.datasets.utils.scaling import create_data_scaler
    assert callable(create_data_scaler)
```
Apply: Start with import verification tests for each cache function.

**Enum comparison test pattern** (scaling test lines 204-229):
```python
def test_get_scaler_minmax_enum() -> None:
    """_get_scaler accepts ScalingMethod.MINMAX enum member."""
    from sklearn.preprocessing import MinMaxScaler
    from tscollection.datasets.utils.scaling import _get_scaler
    scaler = _get_scaler(scaling_method=ScalingMethod.MINMAX, scaling_range=(0.0, 1.0))
    assert isinstance(scaler, MinMaxScaler)

def test_get_scaler_invalid_raises() -> None:
    """_get_scaler raises ValueError for unknown method."""
    from tscollection.datasets.utils.scaling import _get_scaler
    with pytest.raises(ValueError, match='Unsupported scaling method'):
        _get_scaler(scaling_method=ScalingMethod.NONE, scaling_range=(0.0, 1.0))
```
Apply: Use `pytest.raises` for error cases. Test `load_metadata()` version mismatch raises `ValueError`.

**`__all__` export test** (scaling test lines 237-241):
```python
def test_all_exports() -> None:
    """__all__ exports only create_data_scaler (not private helpers)."""
    import tscollection.datasets.utils.scaling as scaling_mod
    assert scaling_mod.__all__ == ['create_data_scaler']
```
Apply: Test `cache.__all__` includes all expected exports.

**`tmp_path` fixture for file I/O tests** (see `test_modules_forecasting.py` lines 24-48):
```python
@pytest.fixture
def synthetic_csv_file(tmp_path: Path) -> Path:
    csv_file = tmp_path / 'synthetic.csv'
    dates = pd.date_range('2020-01-01', periods=100, freq='h')
    df = pd.DataFrame({'date': dates, 'col1': np.random.randn(100), 'col2': np.random.randn(100)})
    df.to_csv(csv_file, index=False)
    return csv_file
```
Apply: Use `tmp_path` for all cache write/read tests. Create temp dirs that mimic `~/.cache/tsdatasets/<dataset>/`.

**Round-trip test pattern** (test_utils_common.py lines 44-55):
```python
def test_flatten_two_arrays() -> None:
    result = flatten_list_of_np_arrays(list_of_np_arrays=[np.array([1, 2]), np.array([3, 4])])
    assert list(result) == [1, 2, 3, 4]
```
Apply: Test `atomic_save_npz` + `np.load` round-trip. Test `save_scaler` + `load_scaler` round-trip. Test DatetimeIndex serialization round-trip.

---

### `tests/test_ddp_compliance.py` (test, event-driven)

**Analog:** `tests/test_modules_forecasting.py` (integration test classes)

**Class-based integration tests with fixtures** (forecasting test lines 468-631):
```python
class TestETTGoldenPathIntegration:
    """Integration tests exercising the full ETT forecasting pipeline."""

    @pytest.fixture
    def ett_csv_file(self, tmp_path: Path) -> Path:
        csv_file = tmp_path / 'ETT_synthetic.csv'
        dates = pd.date_range('2016-01-01', periods=500, freq='h')
        # ... create synthetic data ...
        return csv_file

    def test_ett_univariate_golden_path(self, ett_csv_file: Path) -> None:
        from tscollection.datasets.modules.ett import ETTDataModule
        module = ETTDataModule(dataset_file_path=ett_csv_file, variant='ETTh1', ...)
        module.prepare_data()
        module.setup(stage='fit')
        assert module._train_data_samples is not None
```
Apply: DDP test uses class-based structure. Key difference: DDP test uses `torch.multiprocessing.spawn` with gloo backend. Worker function must be defined at module level (not inside class method) per Pitfall 3.

**Module-level pattern for test helpers** (see `test_modules_ucr.py` lines 170-218):
```python
def test_setup_idempotent(tmp_path: Path) -> None:
    """UCR: setup('fit') called twice produces identical train samples."""
    from tscollection.datasets.modules.ucr import UCRClassificationDataModule
    # ... standalone function test ...
```
Apply: The `_ddp_worker` function must be at module level. Test functions call `mp.spawn(_ddp_worker, ...)`.

**Attribute injection pattern to bypass I/O** (forecasting test lines 646-676):
```python
def test_setup_numpy_full_data_skips_time_features(self) -> None:
    module = ETTDataModule(dataset_file_path=Path('/nonexistent/dummy.csv'), ...)
    rng = np.random.default_rng(42)
    module._full_data = rng.standard_normal((100, 5)).astype(np.float32)
    module._train_slice = slice(None, 60)
    module.setup(stage='fit')
    assert module.num_time_series_features == 0
```
Apply: DDP test uses cache files (not attribute injection). Worker simulates `prepare_data()` writing cache, then `setup()` reading it.

---

### `tests/conftest.py` (config, file-I/O)

**Analog:** `tests/conftest.py` (self -- existing file)

**Current fixture pattern** (lines 1-43):
```python
"""Shared pytest fixtures for dataset tests.

Provides synthetic numpy/pandas data matching real dataset shapes
for unit testing without file I/O or downloads.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_classification_df() -> pd.DataFrame:
    """Return a DataFrame of shape (10, 50) with dtype float32."""
    return pd.DataFrame(np.random.default_rng().standard_normal((10, 50)).astype(np.float32))
```
Apply: Add `synthetic_cache_dir` fixture that creates temp dir with pre-populated cache files (npz + metadata.json + scaler.pt). Use `tmp_path` and cache helper functions. Follow same pattern: docstring with shape/dtype, `np.random.default_rng()` for reproducibility.

## Shared Patterns

### Import Order
**Source:** All existing utility modules (scaling.py, features.py, general.py, common.py)
**Apply to:** `cache.py`, all modified files
```python
# 1. __future__ (only if needed for circular imports)
# 2. Docstring
# 3. stdlib imports (hashlib, json, os, pathlib, tempfile, typing)
# 4. Third-party imports (numpy, pandas, torch, sklearn)
# 5. Project imports (from tscollection.datasets...)
# 6. __all__ list
```

### Keyword-Only Args
**Source:** `scaling.py` line 17, `base.py` line 56, all constructors
**Apply to:** All new functions in `cache.py`, all modified constructors
```python
def function_name(
    *,
    param1: type,
    param2: type = default,
) -> return_type:
```

### Error Message Pattern
**Source:** `scaling.py` lines 68-69, 90-91, `base.py` line 256
**Apply to:** `cache.py` validation, `forecasting.py` error handling
```python
msg = f'Error description: {detail}'
raise ValueError(msg)
```

### Google-Style Docstrings
**Source:** All existing modules
**Apply to:** All new functions/classes
```python
"""Short description.

Longer description if needed.

Args:
    param1: Description.
    param2: Description.

Returns:
    Description of return value.

Raises:
    ValueError: When condition.
"""
```

### Idempotency Sentinel Pattern
**Source:** `base.py` lines 165-169
**Apply to:** Cache-aware `prepare_data()` -- still guard with `_prepare_data_called`
```python
if self._prepare_data_called:
    return
self._do_prepare_data()
self._finalize_prepare_data()
self._prepare_data_called = True
```

### Stage Validation Pattern
**Source:** `base.py` lines 254-258, `forecasting.py` lines 186-195
**Apply to:** All `setup()` methods
```python
if stage not in ('fit', 'validate', 'test', 'predict', None):
    msg = f'Unknown stage: {stage!r}'
    raise ValueError(msg)
if stage in self._setup_completed_stages:
    return
```

### Test Fixture Pattern
**Source:** `test_modules_forecasting.py` lines 24-48, `test_modules_ucr.py` lines 67-100
**Apply to:** `test_cache.py`, `test_ddp_compliance.py`, `conftest.py` additions
```python
@pytest.fixture
def synthetic_cache_dir(tmp_path: Path) -> Path:
    """Create temp dir with pre-populated cache files."""
    # Write npz, metadata.json, scaler.pt
    return tmp_path
```

### Private Helper Convention
**Source:** `scaling.py` `_get_scaler()`, `_scale_regular_data()`, etc.
**Apply to:** `cache.py` -- helpers like `_serialize_params()` if needed
```python
def _helper_name(param: type) -> return_type:
    """Private helper. Not in __all__."""
```

## No Analog Found

All 13 files have close analogs in the existing codebase. No files require RESEARCH.md patterns exclusively.

Note: `tests/test_ddp_compliance.py` has no existing DDP test analog in the codebase. The test pattern is derived from RESEARCH.md Example C (mp.spawn + gloo + 2 ranks). The structure follows the existing integration test patterns in `test_modules_forecasting.py`.

## Metadata

**Analog search scope:** `src/tscollection/datasets/utils/`, `src/tscollection/datasets/modules/_base/`, `src/tscollection/datasets/modules/`, `tests/`
**Files scanned:** 13 source files read, 8 test files reviewed
**Pattern extraction date:** 2026-05-29

## PATTERN MAPPING COMPLETE

**Phase:** 07 - ddp-compliance
**Files classified:** 13
**Analogs found:** 13 / 13

### Coverage
- Files with exact analog (self-modification): 8
- Files with role-match analog: 3 (cache.py from scaling.py/general.py, test_cache.py from test_utils_scaling.py, test_ddp_compliance.py from test_modules_forecasting.py)
- Files with no analog: 0 (test_ddp_compliance.py uses RESEARCH.md Example C for DDP-specific pattern, but test structure follows existing analogs)

### Key Patterns Identified
- All utility functions use keyword-only args (`*`) with full type hints and Google-style docstrings.
- `__all__` exports at module top, alphabetically sorted in `__init__.py`.
- Import order: `__future__` (optional) -> stdlib -> third-party -> project.
- Error handling: `msg = f'...'` then `raise ValueError(msg)` pattern.
- Private helpers prefixed with `_`, not exported in `__all__`.
- Test fixtures use `tmp_path` for file I/O, `np.random.default_rng()` for reproducibility.
- Constructor patterns pass all params to `super()` via keyword arguments.
- Stage validation + idempotency sentinel pattern in all `setup()` methods.
- DDP worker function must be module-level (not nested) for `mp.spawn` pickling.

### File Created
`/Users/skaf/VSCodeProjects/tsdatasets/.planning/phases/07-ddp-compliance/07-PATTERNS.md`

### Ready for Planning
Pattern mapping complete. Planner can now reference analog patterns in PLAN.md files.
