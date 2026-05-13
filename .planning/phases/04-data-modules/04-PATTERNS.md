# Phase 04: Data Modules - Pattern Map

**Mapped:** 2026-05-13
**Files analyzed:** 13 (10 new, 3 modified)
**Analogs found:** 13 / 13

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/.../modules/classes/base.py` | class (base) | request-response (Lightning lifecycle) | `src/.../datasets/classes/fixed.py` | role-match |
| `src/.../modules/classes/classification.py` | class (base) | CRUD, file I/O | `src/.../datasets/classes/fixed.py` | role-match |
| `src/.../modules/classes/forecasting.py` | class (base) | transform, file I/O | `src/.../datasets/classes/flexible.py` | role-match |
| `src/.../modules/ucr.py` | module (concrete) | file I/O, CRUD | `src/.../datasets/ucr.py` | exact |
| `src/.../modules/uea.py` | module (concrete) | file I/O, transform | `src/.../datasets/uea.py` | exact |
| `src/.../modules/ett.py` | module (concrete) | file I/O, streaming | `src/.../datasets/ett.py` | exact |
| `src/.../modules/electricity.py` | module (concrete) | file I/O, transform | `src/.../modules/ett.py` (TBD) | partial |
| `src/.../modules/weather.py` | module (concrete) | file I/O, transform | `src/.../modules/ett.py` (TBD) | partial |
| `src/.../modules/__init__.py` | config (exports) | request-response | `src/.../datasets/__init__.py` | exact |
| `src/.../modules/classes/__init__.py` | config (exports) | request-response | `src/.../datasets/classes/__init__.py` | exact |
| `src/.../enums/data.py` (modified) | enum | CRUD | `src/.../enums/data.py` (self) | exact |
| `src/.../enums/__init__.py` (modified) | config (exports) | request-response | `src/.../enums/__init__.py` (self) | exact |
| `src/.../utils/common.py` (modified) | utility | transform | `src/.../utils/common.py` (self) | exact |
| `src/.../utils/__init__.py` (modified) | config (exports) | request-response | `src/.../utils/__init__.py` (self) | exact |
| `src/.../datasets/__init__.py` (modified) | config (exports) | request-response | `src/.../datasets/__init__.py` (self) | exact |

Note: `src/.../` abbreviates `src/tscollection/datasets/`.

## Pattern Assignments

### `src/tscollection/datasets/modules/classes/base.py` (class, Lightning lifecycle)

**Analog:** `src/tscollection/datasets/datasets/classes/fixed.py` (for import style, ABC pattern, docstrings, `__all__`)

**Imports pattern** (from `datasets/classes/fixed.py`, lines 1-27):
```python
"""Abstract base classes for fixed-length time series datasets.

Provides the ``TimeSeriesDataset`` root ABC and the
``FixedTimeSeriesDataset`` hierarchy (univariate and multivariate) for
classification tasks in which each sample is an independent, fixed-length
time series.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, ClassVar, TYPE_CHECKING

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from tscollection.datasets.datasets.transformations import (
    expand_data_dimensionality,
)
from tscollection.datasets.enums import TimeSeriesDatasetMode
from tscollection.datasets.utils import compose

if TYPE_CHECKING:
    from collections.abc import Callable
```

**Adaptation for base.py:** Replace `from torch.utils.data import Dataset` with `import lightning.pytorch as pl`. Import `DataLoader` from `torch.utils.data`. Import `custom_collate_fn`, `process_data_with_varying_sequence_lengths_single` from `tscollection.datasets.utils`. Import `create_data_scaler` from `tscollection.datasets.utils.scaling`. Use `partial` from `functools` for the collate function binding.

**Core pattern** (from `datasets/classes/fixed.py`, lines 39-138):
```python
class TimeSeriesDataset(Dataset[Any], ABC):
    """Abstract base for all time series datasets.
    ...
    """

    _get_sample_fun_map: ClassVar[dict[TimeSeriesDatasetMode, str]] = {
        TimeSeriesDatasetMode.WITHOUT_LABELS: '_get_sample_1',
        TimeSeriesDatasetMode.WITH_LABELS: '_get_sample_2',
        TimeSeriesDatasetMode.FORECASTING: '_get_sample_3',
    }

    _data: np.ndarray | list[np.ndarray] | pd.DataFrame
    _labels: np.ndarray | list[np.ndarray] | pd.Series | pd.DataFrame | None

    def __init__(
        self,
        data: ...,
        labels: ...,
        ...
    ) -> None:
        super().__init__()
        self._data = data
        ...

    @abstractmethod
    def _go_to_idx(self, idx: int) -> None:
        """Position internal cursor at index *idx*."""
```

**Adaptation:** `BaseTimeSeriesDataModule` inherits from `pl.LightningDataModule` and `ABC`. Store constructor params as instance attrs. Provide `_get_custom_collate_fn()`, `_process_train_dataloader()`, `_process_val_dataloader()`, `_process_test_dataloader()`. Provide `sequence_len`, `n_features`, `train_data_samples`, `test_data_samples`, `valid_data_samples` as `@property` accessors. See rbspaper `abstract.py` lines 41-198 for full method signatures.

**Dataloader construction pattern** (from rbspaper `abstract.py`, lines 142-198):
```python
def _process_train_dataloader(
    self,
    *,
    dataset_object: Any,
    shuffle: bool | None = None,
    strict_batch_size: bool = False,
    extra_args: dict | None = None,
) -> DataLoader:
    if shuffle is None:
        shuffle = self.shuffle
    dataloader_args: dict[str, Any] = {
        'dataset': dataset_object,
        'batch_size': self.batch_size,
        'num_workers': self.num_workers,
        'shuffle': shuffle,
        **(extra_args or {}),
    }
    if self.num_workers > 0:
        dataloader_args['persistent_workers'] = True
    if strict_batch_size:
        dataloader_args['collate_fn'] = self._get_custom_collate_fn()
    return DataLoader(**dataloader_args)

def _process_valid_dataloader(
    self,
    *,
    dataset_object: Any,
    strict_batch_size: bool = False,
    extra_args: dict | None = None,
) -> DataLoader | None:
    if self.valid_size == 0.0:
        return None
    return self._process_test_dataloader(
        dataset_object=dataset_object,
        strict_batch_size=strict_batch_size,
        extra_args=extra_args,
    )
```

---

### `src/tscollection/datasets/modules/classes/classification.py` (class, CRUD + file I/O)

**Analog:** `src/tscollection/datasets/modules/classes/base.py` (inherits from it) + rbspaper `abstract.py` lines 201-301

**Imports pattern:** Same module-level style as `base.py`. Additionally import `ScalingMethod` from `tscollection.datasets.enums.data`, `ClassificationSplittingStrategy` from the same, `DataForm` from the same. Import `create_data_scaler` from `tscollection.datasets.utils.scaling`. Import `process_data_with_varying_sequence_lengths_single` from `tscollection.datasets.utils`. Import `separate_target_feature_from_df` from `tscollection.datasets.utils.common`.

**Core pattern** (from rbspaper `abstract.py`, lines 201-301, adapted per D-01, D-02, D-03, D-04):
```python
class BaseClassificationTimeSeriesDataModule(BaseTimeSeriesDataModule, ABC):
    """Base datamodule for classification datasets (UCR/UEA).

    Manages train/val/test splitting, target column extraction, and
    variable-length sequence handling.

    Args:
        dataset_folder_path: Path to the dataset directory.
        target_column_name: Name of the label column in ARFF data.
        batch_size: Batch size for dataloaders.
        valid_size: Fraction of training data for validation.
        ...
    """

    def __init__(
        self,
        *,
        dataset_folder_path: Path,
        target_column_name: str,
        batch_size: int,
        valid_size: float,
        shuffle: bool,
        scale_data: bool,
        data_scaling_method: ScalingMethod,
        data_scaling_range: tuple[float, float],
        data_form: DataForm,
        splitting_strategy: ClassificationSplittingStrategy,
        test_size: float,
        num_workers: int,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            seq_len=None,
            valid_size=valid_size,
            test_size=test_size,
            shuffle=shuffle,
            scale_data=scale_data,
            data_scaling_method=data_scaling_method,
            data_scaling_range=data_scaling_range,
            num_workers=num_workers,
            data_form=data_form,
        )
        self.dataset_folder_path = dataset_folder_path
        self.target_column_name = target_column_name
        self.splitting_strategy = splitting_strategy
        self._separate_target_feature = partial(
            separate_target_feature_from_df, target_feature_name=self.target_column_name
        )
        self._num_classes: int | None = None
        self._train_data_labels: Any = None
        self._test_data_labels: Any = None
        self._valid_data_labels: Any = None

    @property
    def num_classes(self) -> int | None:
        return self._num_classes

    def setup(self, stage: str) -> None:
        scaler = create_data_scaler(
            scale=self.scale_data,
            scaling_range=self.data_scaling_range,
            scaling_method=self.data_scaling_method,
            data_form=self._data_form,
        )
        (
            self._train_data_samples,
            self._valid_data_samples,
            self._test_data_samples,
        ) = scaler(
            self._train_data_samples,
            self._valid_data_samples,
            self._test_data_samples,
        )
```

**Key deviations from rbspaper source:**
- No `dataset_config_path` parameter. Constructor accepts `target_column_name` and `data_form` directly (D-01, D-02).
- `data_scaling_method` is `ScalingMethod` enum, not `str` (D-03).
- `splitting_strategy` is `ClassificationSplittingStrategy`, not `SplittingStrategy` (D-04).
- `setup()` signature uses `stage: str` (Lightning 2.5.6 standard), not `stage: str | None` (D-09).

---

### `src/tscollection/datasets/modules/classes/forecasting.py` (class, transform + file I/O)

**Analog:** `src/tscollection/datasets/modules/classes/base.py` (inherits from it) + rbspaper `abstract.py` lines 304-445

**Imports pattern:** Same module-level style. Import `ForecastingMode`, `ScalingMethod` from `tscollection.datasets.enums.data`. Import `extract_time_features` from `tscollection.datasets.utils.features`. Import `MinMaxScaler`, `StandardScaler` from `sklearn.preprocessing`.

**Core pattern** (from rbspaper `abstract.py`, lines 304-445, adapted per D-05, D-09, D-10):
```python
class BaseForecastingTimeSeriesDataModule(BaseTimeSeriesDataModule, ABC):
    """Base datamodule for forecasting datasets (ETT, Electricity, Weather).

    Manages time series slicing, feature extraction, and per-split
    scaling trained only on the training portion.

    Args:
        batch_size: Batch size.
        seq_len: Input window length.
        ...
        mode: UNIVARIATE or MULTIVARIATE forecasting mode.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int,
        valid_size: float,
        test_size: float,
        shuffle: bool,
        scale_data: bool,
        data_scaling_method: ScalingMethod,
        data_scaling_range: tuple[float, float],
        num_workers: int,
        mode: ForecastingMode,
    ) -> None:
        super().__init__(...)
        self._mode = mode
        self._train_slice: slice | None = None
        self._valid_slice: slice | None = None
        self._test_slice: slice | None = None
        self._full_data: np.ndarray | pd.DataFrame | None = None
        self._num_time_series_features: int | None = None

    @abstractmethod
    def _set_data_slices(self) -> None:
        """Define train/valid/test slice boundaries."""

    @abstractmethod
    def _transform_data(self) -> None:
        """Transform _full_data after scaling (e.g. reshape)."""

    def _prepare_data_scaler(self) -> StandardScaler | MinMaxScaler:
        # NOTE: Uses ScalingMethod enum, not strings
        if self.data_scaling_method == ScalingMethod.STANDARD:
            return StandardScaler()
        if self.data_scaling_method == ScalingMethod.MINMAX:
            return MinMaxScaler(feature_range=self.data_scaling_range)
        raise ValueError(f'Unsupported scaling method: {self.data_scaling_method}')

    def setup(self, stage: str) -> None:
        # Full pattern from rbspaper abstract.py lines 402-437, adapted
        ...

    def _calculate_num_features(self) -> None:
        assert self._full_data is not None
        self._num_features = self._full_data.shape[-1]

    def _post_prepare_data(self) -> None:
        self._set_data_slices()
```

**Key deviation:** `data_scaling_method` is `ScalingMethod` enum, so `_prepare_data_scaler()` must compare against `ScalingMethod.STANDARD` and `ScalingMethod.MINMAX` -- NOT against strings `'standardization'` and `'min_max'`. Since `ScalingMethod` is a `StrEnum`, it compares equal to its string value, but using the enum member is the correct approach per D-03.

---

### `src/tscollection/datasets/modules/ucr.py` (module, file I/O + CRUD)

**Analog:** `src/tscollection/datasets/datasets/ucr.py` (thin wrapper pattern) + rbspaper `ucr_datamodule.py`

**Imports pattern** (from `datasets/ucr.py`, lines 1-21):
```python
"""UCR univariate classification dataset.

Thin wrapper around FixedTimeSeriesDatasetUnivariate that sets domain
defaults for UCR-style time series classification tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tscollection.datasets.datasets.classes.fixed import FixedTimeSeriesDatasetUnivariate
from tscollection.datasets.datasets.transformations import convert_numpy_to_tensor

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from tscollection.datasets.enums import TimeSeriesDatasetMode
```

**Adaptation for ucr.py module:** Import `BaseClassificationTimeSeriesDataModule` from `modules/classes/classification.py`. Import `UCRClassificationUnivariateDataset` from `datasets/ucr.py`. Import `read_arff_as_df`, `process_df_according_to_dtypes` from `tscollection.datasets.utils.arff`. Import `DataForm`, `ScalingMethod`, `ClassificationSplittingStrategy`, `TimeSeriesDatasetMode` from `tscollection.datasets.enums`.

**Core pattern** (from rbspaper `ucr_datamodule.py`, lines 25-216, adapted per D-01, D-02, D-07, D-16):
```python
class UCRClassificationDataModule(BaseClassificationTimeSeriesDataModule):
    """LightningDataModule for UCR univariate classification datasets.

    Reads train/test ARFF files, applies optional manual re-splitting,
    creates a validation split, and handles variable-length series.

    Args:
        dataset_folder_path: Path to the dataset ARFF directory.
        target_column_name: Name of the label column.
        batch_size: Batch size.
        ...
    """

    def __init__(
        self,
        *,
        dataset_folder_path: Path,
        target_column_name: str,
        batch_size: int = 32,
        valid_size: float = 0.1,
        shuffle: bool = False,
        scale_data: bool = True,
        data_scaling_method: ScalingMethod = ScalingMethod.MINMAX,
        data_scaling_range: tuple[float, float] = (0, 1),
        splitting_strategy: ClassificationSplittingStrategy = (
            ClassificationSplittingStrategy.AS_DEFINED
        ),
        test_size: float = 0.5,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            dataset_folder_path=dataset_folder_path,
            target_column_name=target_column_name,
            batch_size=batch_size,
            valid_size=valid_size,
            shuffle=shuffle,
            scale_data=scale_data,
            data_scaling_method=data_scaling_method,
            data_scaling_range=data_scaling_range,
            data_form=DataForm.REGULAR,       # D-02: hardcoded per subclass
            splitting_strategy=splitting_strategy,
            test_size=test_size,
            num_workers=num_workers,
        )

    def prepare_data(self) -> None:
        # D-16: Fail fast with descriptive errors
        if not self.dataset_folder_path.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {self.dataset_folder_path}"
            )
        # Per D-01: Hardcoded ARFF patterns, no JSON config
        self._dataset_name = self.dataset_folder_path.name
        arff_train = self.dataset_folder_path / f"{self._dataset_name}_TRAIN.arff"
        arff_test = self.dataset_folder_path / f"{self._dataset_name}_TEST.arff"
        # ... rest follows rbspaper ucr_datamodule.py lines 97-170
```

**Property accessor pattern** (D-11): Use `sequence_length`, `num_features`, `num_classes` as full-name properties on the base class; UCR module just sets `_seq_len`, `_num_features`, `_num_classes` internal attrs in `prepare_data()`.

**Dataloader methods pattern** (from rbspaper `ucr_datamodule.py`, lines 172-216):
```python
def train_dataloader(
    self,
    *,
    mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.WITHOUT_LABELS,
    shuffle: bool | None = None,
    strict_batch_size: bool = True,
    extra_args: dict | None = None,
) -> DataLoader:
    dataset = UCRClassificationUnivariateDataset(
        data=self._train_data_samples,
        labels=self._train_data_labels,
        mode=mode,
    )
    return self._process_train_dataloader(
        dataset_object=dataset,
        shuffle=shuffle,
        strict_batch_size=strict_batch_size,
        extra_args=extra_args,
    )
```

---

### `src/tscollection/datasets/modules/uea.py` (module, file I/O + transform)

**Analog:** `src/tscollection/datasets/datasets/uea.py` (thin wrapper pattern) + rbspaper `uea_datamodule.py`

**Imports pattern:** Same structure as `ucr.py`. Import `BaseClassificationTimeSeriesDataModule`. Import `UEAClassificationMultivariateDataset` from `datasets/uea.py`. Import `arff` from `scipy.io` (D-12: raw scipy, not `arff.py` utility). Import `LabelEncoder` from `sklearn.preprocessing`.

**Core pattern** (from rbspaper `uea_datamodule.py`, lines 31-249, adapted per D-01, D-02, D-12):
```python
class UEAClassificationDataModule(BaseClassificationTimeSeriesDataModule):
    """LightningDataModule for UEA multivariate classification datasets.

    Reads multi-dimensional ARFF files, encodes labels, and manages
    splits with variable-length handling.
    """

    def __init__(
        self,
        *,
        dataset_folder_path: Path,
        target_column_name: str,
        batch_size: int = 32,
        ...
        data_form: DataForm = DataForm.NESTED,       # D-02: hardcoded as NESTED
        ...
    ) -> None:
        super().__init__(
            ...
            data_form=DataForm.NESTED,
            ...
        )

    def _process_stacked_data(self, data: Any) -> tuple[np.ndarray, np.ndarray]:
        # D-12: Internal to UEA module, NOT extracted to arff.py
        processed_data: list[np.ndarray] = []
        labels: list[str] = []
        for sample, label in data:
            sample_list = []
            for point in sample:
                point = point.tolist()
                point = [
                    float(d.decode('utf-8')) if isinstance(d, bytes) else float(d)
                    for d in point
                ]
                sample_list.append(point)
            processed_data.append(np.array(sample_list))
            labels.append(
                label.decode('utf-8') if isinstance(label, bytes) else label
            )

        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)
        output_data = np.array(processed_data).astype(np.float32).swapaxes(1, 2)
        return output_data, np.array(encoded_labels)
```

**Note:** UEA does NOT use `read_arff_as_df` or `process_df_according_to_dtypes` from `arff.py`. It uses raw `scipy.io.arff.loadarff()` per D-12. The `_separate_target_feature` partial from classification base is NOT used since UEA data is ndarray-based (not DataFrame). Instead, `_process_stacked_data` returns both data and labels directly.

**Dataloader methods:** Same pattern as UCR but use `UEAClassificationMultivariateDataset`.

---

### `src/tscollection/datasets/modules/ett.py` (module, file I/O + streaming)

**Analog:** `src/tscollection/datasets/datasets/ett.py` (thin wrapper pattern) + rbspaper `ett_datamodule.py`

**Imports pattern:** Same structure. Import `BaseForecastingTimeSeriesDataModule` from `modules/classes/forecasting.py`. Import `ForecastingMode`, `ScalingMethod`, `TimeSeriesDatasetMode` from `tscollection.datasets.enums`. Import `TensorDataset` from `torch.utils.data`, `torch` from `torch`.

**Core pattern** (from rbspaper `ett_datamodule.py`, lines 18-140, adapted per D-06, D-07, D-13):
```python
class ETTDataModule(BaseForecastingTimeSeriesDataModule):
    """LightningDataModule for ETT forecasting datasets.

    Supports ETTh1, ETTh2 (hourly) and ETTm1, ETTm2 (15-min). Uses
    standard 16-month / 4-month / 4-month splits.

    Args:
        dataset_file_path: Path to the CSV file (Path only, D-07).
        variant: Explicit variant name (D-06).
        ...
    """

    _full_data: pd.DataFrame | np.ndarray | None = None

    def __init__(
        self,
        *,
        dataset_file_path: Path,                    # D-07: Path only
        variant: str,                                # D-06: explicit variant
        seq_len: int = 128,
        mode: ForecastingMode = ForecastingMode.UNIVARIATE,
        batch_size: int = 32,
        scale_data: bool = True,
        data_scaling_method: ScalingMethod = ScalingMethod.MINMAX,
        data_scaling_range: tuple[float, float] = (0, 1),
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            seq_len=seq_len,
            valid_size=0.0,                          # Fixed splits, unused
            test_size=0.0,                           # Fixed splits, unused
            shuffle=False,
            scale_data=scale_data,
            data_scaling_method=data_scaling_method,
            data_scaling_range=data_scaling_range,
            num_workers=num_workers,
            mode=mode,
        )
        self.dataset_file_path = dataset_file_path
        self.variant = variant

    def _set_data_slices(self) -> None:
        # D-06: Use self.variant, NOT self.dataset_file_path.stem
        if self.variant in {'ETTh1', 'ETTh2'}:
            self._train_slice = slice(None, 12 * 30 * 24)
            self._valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
            self._test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
        elif self.variant in {'ETTm1', 'ETTm2'}:
            self._train_slice = slice(None, 12 * 30 * 24 * 4)
            self._valid_slice = slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
            self._test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)

    def _transform_data(self) -> None:
        if self._full_data is None:
            raise RuntimeError('_full_data was not set by prepare_data()')
        if isinstance(self._full_data, pd.DataFrame):
            self._full_data = self._full_data.to_numpy()
        if isinstance(self._full_data, np.ndarray):
            self._full_data = np.expand_dims(self._full_data, axis=0)

    def prepare_data(self) -> None:
        # D-16: Fail fast
        if not self.dataset_file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.dataset_file_path}"
            )
        # D-06: Use explicit variant, not filename parsing
        self._dataset_name = self.variant
        df = pd.read_csv(self.dataset_file_path, parse_dates=True, index_col='date')

        if self._mode == ForecastingMode.UNIVARIATE:
            df = df[['OT']]

        self._full_data = df
        self._post_prepare_data()
```

**Dataloader methods pattern** (D-13: TensorDataset, not proper dataset class):
```python
def train_dataloader(
    self,
    *,
    mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.FORECASTING,
    shuffle: bool | None = None,
    strict_batch_size: bool = False,
    extra_args: dict | None = None,
) -> DataLoader:
    tensor = torch.from_numpy(self._train_data_samples).to(torch.float32)
    return self._process_train_dataloader(
        dataset_object=TensorDataset(tensor),
        shuffle=shuffle,
        strict_batch_size=strict_batch_size,
        extra_args=extra_args,
    )
```

---

### `src/tscollection/datasets/modules/electricity.py` (module, file I/O + transform)

**Analog:** `src/tscollection/datasets/modules/ett.py` (same base class, same dataloader pattern) + rbspaper `electricity_load_datamodule.py`

**Pattern copy from `ett.py`:** All imports, constructor structure, dataloader methods (TensorDataset), `_transform_data` signature.

**Differences from ETT:**
- `_set_data_slices()`: Uses 60/20/20 fractional split of `len(self._full_data)`.
- `_transform_data()`: Transpose + expand last axis (`.T` + `expand_dims(axis=-1)`) instead of just `expand_dims(axis=0)`.
- `prepare_data()`: Uses `sep=';'`, `decimal=','`, resampling, column filtering.
- Univariate column is `'MT_001'` (not `'OT'`).

**Core pattern** (from rbspaper `electricity_load_datamodule.py`, lines 18-144):
```python
class ElectricityLoadModule(BaseForecastingTimeSeriesDataModule):
    _full_data: pd.DataFrame | np.ndarray | None = None

    def __init__(
        self,
        *,
        dataset_file_path: Path,
        seq_len: int = 128,
        mode: ForecastingMode = ForecastingMode.UNIVARIATE,
        batch_size: int = 32,
        scale_data: bool = True,
        data_scaling_method: ScalingMethod = ScalingMethod.MINMAX,
        data_scaling_range: tuple[float, float] = (0, 1),
        num_workers: int = 0,
    ) -> None:
        ...

    def _set_data_slices(self) -> None:
        if self._full_data is None:
            raise RuntimeError('_full_data was not set by prepare_data()')
        num_samples = len(self._full_data)
        self._train_slice = slice(None, int(0.6 * num_samples))
        self._valid_slice = slice(int(0.6 * num_samples), int(0.8 * num_samples))
        self._test_slice = slice(int(0.8 * num_samples), None)

    def _transform_data(self) -> None:
        # Climate-specific: .T + expand_dims(axis=-1)
        if isinstance(self._full_data, pd.DataFrame):
            self._full_data = self._full_data.to_numpy()
        self._full_data = self._full_data.T
        self._full_data = np.expand_dims(self._full_data, axis=-1)
```

---

### `src/tscollection/datasets/modules/weather.py` (module, file I/O + transform)

**Analog:** `src/tscollection/datasets/modules/ett.py` + rbspaper `weather_datamodule.py`

**Pattern copy from `ett.py`:** Same structure. Dataloaders use TensorDataset.

**Differences from ETT:**
- `_set_data_slices()`: 60/20/20 fractional split (same as Electricity).
- `_transform_data()`: Just `expand_dims(axis=0)` (same as ETT).
- `prepare_data()`: Univariate mode selects last column (`df.iloc[:, -1:]`), not `'OT'`.
- `self._dataset_name = self.dataset_file_path.name` (full filename, not stem).

**Core pattern** (from rbspaper `weather_datamodule.py`, lines 18-136):
```python
class WeatherModule(BaseForecastingTimeSeriesDataModule):
    _full_data: pd.DataFrame | np.ndarray | None = None

    def _set_data_slices(self) -> None:
        if self._full_data is None:
            raise RuntimeError('_full_data was not set by prepare_data()')
        num_samples = len(self._full_data)
        self._train_slice = slice(None, int(0.6 * num_samples))
        self._valid_slice = slice(int(0.6 * num_samples), int(0.8 * num_samples))
        self._test_slice = slice(int(0.8 * num_samples), None)

    def _transform_data(self) -> None:
        # Weather-specific: just expand_dims(axis=0)
        if isinstance(self._full_data, pd.DataFrame):
            self._full_data = self._full_data.to_numpy()
        self._full_data = np.expand_dims(self._full_data, axis=0)

    def prepare_data(self) -> None:
        if not self.dataset_file_path.exists():
            raise FileNotFoundError(...)
        self._dataset_name = self.dataset_file_path.name
        df = pd.read_csv(self.dataset_file_path, parse_dates=True, index_col='date')
        if self._mode == ForecastingMode.UNIVARIATE:
            df = df.iloc[:, -1:]
        self._full_data = df
        self._post_prepare_data()
```

---

### `src/tscollection/datasets/modules/__init__.py` (config, exports)

**Analog:** `src/tscollection/datasets/datasets/__init__.py` (lines 1-39)

**Pattern to copy:**
```python
"""Time series dataset classes (PyTorch Dataset)."""

from tscollection.datasets.datasets.classes import (
    FixedTimeSeriesDataset,
    ...
)
from tscollection.datasets.datasets.ett import ETTDataset
from tscollection.datasets.datasets.ucr import UCRClassificationUnivariateDataset
from tscollection.datasets.datasets.uea import UEAClassificationMultivariateDataset

__all__ = [
    'ClassificationStrategyMultipleFiles',
    ...
    'ETTDataset',
    'UCRClassificationUnivariateDataset',
    'UEAClassificationMultivariateDataset',
]
```

**Adaptation for modules:** Import base classes from `modules/classes`, then concrete modules from `modules/ucr`, `modules/uea`, `modules/ett`, `modules/electricity`, `modules/weather`. Build `__all__` with all exports sorted alphabetically.

---

### `src/tscollection/datasets/modules/classes/__init__.py` (config, exports)

**Analog:** `src/tscollection/datasets/datasets/classes/__init__.py` (lines 1-37)

**Pattern to copy:**
```python
"""Abstract base classes for time series datasets."""

from tscollection.datasets.datasets.classes.fixed import (
    FixedTimeSeriesDataset,
    FixedTimeSeriesDatasetMultivariate,
    FixedTimeSeriesDatasetUnivariate,
    TimeSeriesDataset,
)
from tscollection.datasets.datasets.classes.flexible import (
    FlexibleTimeSeriesDataset,
    FlexibleTimeSeriesDatasetMultipleFiles,
    FlexibleTimeSeriesDatasetSingleFile,
)
from tscollection.datasets.datasets.classes.strategies import (
    ClassificationStrategyMultipleFiles,
    ...
)

__all__ = [
    'ClassificationStrategyMultipleFiles',
    ...
]
```

**Adaptation:** Import from `modules/classes/base`, `modules/classes/classification`, `modules/classes/forecasting`. Export `BaseTimeSeriesDataModule`, `BaseClassificationTimeSeriesDataModule`, `BaseForecastingTimeSeriesDataModule`.

---

### `src/tscollection/datasets/enums/data.py` (modified, enum)

**Analog:** Self (same file). See current content at lines 1-75.

**Change:** Rename `SplittingStrategy` class to `ClassificationSplittingStrategy` (D-04). Keep all enum values identical. This is a simple rename:

```python
class ClassificationSplittingStrategy(StrEnum):
    """Strategy for train/test data splitting (classification only)."""

    AS_DEFINED = 'as_defined'
    MANUAL = 'manual'
```

**Error handling pattern:** The rename is backward-compatible since `ClassificationSplittingStrategy` is a `StrEnum` -- existing code that compares against string values (`'as_defined'`, `'manual'`) will still work.

---

### `src/tscollection/datasets/enums/__init__.py` (modified, exports)

**Analog:** Self (same file). Replace `SplittingStrategy` with `ClassificationSplittingStrategy` in both the import and `__all__`.

```python
from tscollection.datasets.enums.data import (
    ClassificationSplittingStrategy,  # was SplittingStrategy
    ...
)

__all__ = [
    'ClassificationSplittingStrategy',  # was SplittingStrategy
    ...
]
```

---

### `src/tscollection/datasets/utils/common.py` (modified, utility)

**Analog:** Self (same file). Add `separate_target_feature_from_df` function.

**Pattern to copy from rbspaper `common.py`, lines 119-133:**
```python
def separate_target_feature_from_df(
    df: pd.DataFrame, target_feature_name: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate target feature from a DataFrame.

    Args:
        df: Source DataFrame containing the target column.
        target_feature_name: Name of the target column.

    Returns:
        A tuple of (features DataFrame, target Series).
    """
    target_feature = df[target_feature_name]
    features = df.drop(target_feature_name, axis=1)
    return features, target_feature
```

**Also update `__all__`** to include `'separate_target_feature_from_df'`.

**Import update:** Add `import pandas as pd` (currently only used in `TYPE_CHECKING` block in common.py -- need to check if pandas is already imported runtime). Looking at current `common.py`, `pandas` is NOT imported at runtime. Need to add `import pandas as pd` outside `TYPE_CHECKING`.

---

### `src/tscollection/datasets/utils/__init__.py` (modified, exports)

**Analog:** Self (same file). Add `separate_target_feature_from_df` to imports and `__all__`.

```python
from tscollection.datasets.utils.common import (
    compose,
    flatten_list_of_np_arrays,
    FunctionComposer,
    get_num_samples_from_ts,
    separate_target_feature_from_df,  # NEW
)
```

And in `__all__`, add `'separate_target_feature_from_df'` in alphabetical position.

---

### `src/tscollection/datasets/__init__.py` (modified, exports)

**Analog:** Self (same file). Replace `SplittingStrategy` with `ClassificationSplittingStrategy` in both the import and `__all__`.

```python
from tscollection.datasets.enums import (
    ClassificationSplittingStrategy,  # was SplittingStrategy
    ...
)

__all__ = [
    'ClassificationSplittingStrategy',  # was SplittingStrategy
    ...
]
```

## Shared Patterns

### LightningDataModule Lifecycle (D-09)
**Source:** rbspaper `abstract.py` lines 41-198 (restructured)
**Apply to:** All module files (`base.py`, `classification.py`, `forecasting.py`, `ucr.py`, `uea.py`, `ett.py`, `electricity.py`, `weather.py`)

Pattern: `prepare_data()` does file validation (raise `FileNotFoundError`) and raw data loading. `setup(stage: str)` does scaling and final data preparation. This deviates from rbspaper which does everything in `prepare_data()`.

```python
def prepare_data(self) -> None:
    if not self.dataset_file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {self.dataset_file_path}")
    # Load raw data into self._full_data or self._train_data_samples etc.
    ...

def setup(self, stage: str) -> None:
    # Apply scaling, extract features, finalize splits
    ...
```

### Enum-Based Parameters (D-03, D-04, D-05)
**Source:** `src/tscollection/datasets/enums/data.py`
**Apply to:** All module constructors and base class `__init__` methods

- `ScalingMethod.MINMAX` / `ScalingMethod.STANDARD` (not strings)
- `ClassificationSplittingStrategy.AS_DEFINED` / `ClassificationSplittingStrategy.MANUAL`
- `ForecastingMode.UNIVARIATE` / `ForecastingMode.MULTIVARIATE`
- `DataForm.REGULAR` / `DataForm.NESTED` (hardcoded per subclass, D-02)

### `persistent_workers` Guard (Pitfall 6)
**Source:** rbspaper `abstract.py` lines 159-160
**Apply to:** `_process_train_dataloader()`, `_process_test_dataloader()` in `base.py`

```python
if self.num_workers > 0:
    dataloader_args['persistent_workers'] = True
```

Never add `persistent_workers` to the default args dict unconditionally.

### Dataloader Method Signatures (D-14)
**Source:** rbspaper `ucr_datamodule.py` lines 172-216
**Apply to:** All concrete module `train_dataloader()`, `val_dataloader()`, `test_dataloader()` methods

```python
def train_dataloader(
    self,
    *,
    mode: TimeSeriesDatasetMode,
    shuffle: bool | None = None,
    strict_batch_size: bool,
    extra_args: dict | None = None,
) -> DataLoader:
```

All params are keyword-only (`*`). Classification: default `mode=WITHOUT_LABELS`. Forecasting: default `mode=FORECASTING`.

### Property Naming (D-11)
**Source:** ROADMAP.md MOD-05
**Apply to:** `base.py`, `classification.py`

Public properties: `sequence_length`, `num_features`, `num_classes`.
Internal attributes: `_seq_len`, `_num_features`, `_num_classes`.

### `custom_collate_fn` Binding via `functools.partial`
**Source:** rbspaper `abstract.py` lines 126-129
**Apply to:** `base.py`

```python
def _get_custom_collate_fn(self, desired_batch_size: int | None = None) -> Any:
    if desired_batch_size is None:
        desired_batch_size = self.batch_size
    return partial(custom_collate_fn, desired_batch_size=desired_batch_size)
```

### Validation Split Fallback (stratify error handling)
**Source:** rbspaper `ucr_datamodule.py` lines 139-168
**Apply to:** `ucr.py`, `uea.py`

When `train_test_split` with `stratify=` raises `ValueError` because `test_size` is too small to cover all classes, fall back to `test_size = number_of_classes`.

### Forecasting `_post_prepare_data()` Hook
**Source:** rbspaper `abstract.py` line 444
**Apply to:** `ett.py`, `electricity.py`, `weather.py`

At the end of each concrete module's `prepare_data()`, call `self._post_prepare_data()` which invokes `_set_data_slices()`. This ensures slice boundaries are set before `setup()` runs.

### Docstring Style
**Source:** All existing `datasets/` and `utils/` files
**Apply to:** All new module files

Google-style docstrings. Module-level docstring as a triple-quoted string at the top. Class docstrings describe purpose and list Args. Method docstrings describe behavior. Use `*param_name*` for emphasis within docstrings.

## No Analog Found

All 13 files have close analogs. The forecasting modules (`electricity.py`, `weather.py`) have no existing module in our codebase yet, but they closely follow the `ett.py` pattern which is being created in this phase, and their rbspaper source is the canonical reference.

## Metadata

**Analog search scope:** `src/tscollection/datasets/`, `_sources/rbspaper/src/rbspaper/data/modules/`
**Files scanned:** 35 (existing source) + 12 (rbspaper source)
**Pattern extraction date:** 2026-05-13

---

## PATTERN MAPPING COMPLETE

**Phase:** 04 - data-modules
**Files classified:** 13
**Analogs found:** 13 / 13

### Coverage
- Files with exact analog: 6 (`ucr.py`, `uea.py`, `ett.py`, `modules/__init__.py`, `modules/classes/__init__.py`, `enums/data.py`)
- Files with role-match analog: 4 (`base.py`, `classification.py`, `forecasting.py`, `utils/common.py`)
- Files with partial analog: 3 (`electricity.py`, `weather.py`, `__init__.py` root)

### Key Patterns Identified
- All modules inherit from `pl.LightningDataModule` + `ABC` via three-tier hierarchy (base -> classification/forecasting -> concrete)
- `prepare_data()` validates paths and loads raw data; `setup()` applies scaling (D-09 deviation from rbspaper)
- Constructor params use `ScalingMethod`, `ClassificationSplittingStrategy`, `ForecastingMode` enums -- not strings
- File paths are `Path` type only, not `str | Path` (D-07)
- Dataloader methods use keyword-only params with `extra_args`, `mode`, `strict_batch_size` (D-14)
- `persistent_workers=True` guarded by `num_workers > 0` check
- `custom_collate_fn` bound via `functools.partial` for strict batch size enforcement
- UEA uses raw `scipy.io.arff.loadarff()` + internal `_process_stacked_data()`, NOT the `arff.py` utility (D-12)
- Forecasting modules use `TensorDataset` (deferred proper dataset classes, D-13)
- ETT uses explicit `variant` param for split boundaries, not filename detection (D-06)
- Property names: `sequence_length`, `num_features`, `num_classes` with `_seq_len`, `_num_features`, `_num_classes` internals (D-11)

### File Created
`/Users/skaf/VSCodeProjects/tsdatasets/.planning/phases/04-data-modules/04-PATTERNS.md`

### Ready for Planning
Pattern mapping complete. Planner can now reference analog patterns in PLAN.md files.
