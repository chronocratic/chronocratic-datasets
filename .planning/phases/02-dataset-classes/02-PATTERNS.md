# Phase 2: Dataset Classes - Pattern Map

**Mapped:** 2026-05-11
**Files analyzed:** 12 (6 new source, 3 modified exports, 3 test)
**Analogs found:** 12 / 12

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `src/tscollection/datasets/datasets/classes/fixed.py` | class (ABC hierarchy) | CRUD (read/iterable) | `_sources/rbspaper/src/rbspaper/data/datasets/abstract.py` (lines 43-199) | exact |
| `src/tscollection/datasets/datasets/classes/flexible.py` | class (ABC + concrete) | CRUD (read/iterable) | `_sources/rbspaper/src/rbspaper/data/datasets/abstract.py` (lines 235-421) | exact |
| `src/tscollection/datasets/datasets/classes/strategies.py` | class (strategy ABC) | CRUD (read/compute) | `_sources/rbspaper/src/rbspaper/data/datasets/strategies.py` | exact |
| `src/tscollection/datasets/datasets/ucr.py` | class (thin wrapper) | CRUD (read) | `_sources/rbspaper/src/rbspaper/data/datasets/ucr_dataset.py` | exact |
| `src/tscollection/datasets/datasets/uea.py` | class (thin wrapper) | CRUD (read) | `_sources/rbspaper/src/rbspaper/data/datasets/uea_dataset.py` | exact |
| `src/tscollection/datasets/datasets/ett.py` | class (thin wrapper) | CRUD (read) | `_sources/rbspaper/src/rbspaper/data/datasets/ett_dataset.py` | exact |
| `src/tscollection/datasets/datasets/classes/__init__.py` | config (exports) | N/A | `src/tscollection/datasets/enums/__init__.py` | exact |
| `src/tscollection/datasets/datasets/__init__.py` | config (exports) | N/A | `src/tscollection/datasets/enums/__init__.py` | exact |
| `src/tscollection/datasets/utils/__init__.py` | config (exports) | N/A | `src/tscollection/datasets/__init__.py` | exact |
| `tests/test_datasets.py` | test | CRUD (read/assert) | `tests/test_package.py` | role-match |
| `tests/conftest.py` | test config | N/A | No existing conftest; standard pytest | partial |
| `tests/fixtures/` | test data | N/A | No existing fixtures; standard pytest | none |

## Pattern Assignments

### `src/tscollection/datasets/datasets/classes/fixed.py` (class, ABC hierarchy)

**Analog:** `_sources/rbspaper/src/rbspaper/data/datasets/abstract.py`

This file contains the root `TimeSeriesDataset` ABC (shared base) plus the `FixedTimeSeriesDataset` hierarchy. The rbspaper source has all classes in one file (`abstract.py`); this port splits the fixed part here and the flexible part into `flexible.py`.

**Imports pattern** (rbspaper lines 7-30):
```python
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from tscollection.datasets.datasets.classes.strategies import (
    SequenceHandlingStrategy,
)
from tscollection.datasets.datasets.transformations import (
    convert_numpy_to_tensor,
    expand_data_dimensionality,
)
from tscollection.datasets.utils import compose, get_num_samples_from_ts
from tscollection.datasets.enums import TimeSeriesDatasetMode
```
Note: Import paths adapted from `src.rbspaper.*` to `tscollection.datasets.*`. Strategies import is only needed if `fixed.py` references flexible types (it does not, but `TimeSeriesDataset` is the base for both, so strategies are imported in `flexible.py` instead).

**Core ABC pattern -- TimeSeriesDataset** (rbspaper lines 43-122):
```python
class TimeSeriesDataset(Dataset[Any], ABC):
    """Abstract base for all time series datasets."""

    _get_sample_fun_map = {
        TimeSeriesDatasetMode.WITHOUT_LABELS: '_get_sample_1',
        TimeSeriesDatasetMode.WITH_LABELS: '_get_sample_2',
        TimeSeriesDatasetMode.FORECASTING: '_get_sample_3',
    }

    def __init__(
        self,
        data: Any,
        labels: Any,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None = None,
    ) -> None:
        super().__init__()
        self._data = data
        self._labels = labels
        self._mode = mode
        self._get_sample = getattr(self, self._get_sample_fun_map[mode])
        self._initiate_transformation_functionality(
            transformations_sequence or [], expand_dims_axis
        )

    @abstractmethod
    def _go_to_idx(self, idx: int) -> None: ...

    @abstractmethod
    def _get_current_data(self) -> np.ndarray: ...

    @abstractmethod
    def _get_current_label(self) -> np.ndarray | int | None: ...

    def _initiate_transformation_functionality(
        self,
        transformations_sequence: list[Callable] | tuple[Callable, ...],
        expand_dims_axis: int | None,
    ) -> None:
        sequence = list(transformations_sequence)
        if expand_dims_axis is not None:
            sequence.append(partial(expand_data_dimensionality, expand_dims_axis=expand_dims_axis))
        self._transform = compose(*sequence)

    def _get_sample_1(self) -> Any:
        return self._transform(self._get_current_data())

    def _get_sample_2(self) -> tuple[Any, Any]:
        sample = self._transform(self._get_current_data())
        label = self._get_current_label()
        return (sample, label)

    def _get_sample_3(self) -> tuple[Any, Any]:
        sample = self._transform(self._get_current_data())
        label = self._transform(self._get_current_label())
        return (sample, label)

    def __getitem__(self, index: int) -> Any:
        self._go_to_idx(index)
        return self._get_sample()

    def __len__(self) -> int:
        raise NotImplementedError
```

**Fixed dataset pattern** (rbspaper lines 125-165):
```python
class FixedTimeSeriesDataset(TimeSeriesDataset, ABC):
    """Dataset for fixed-length time series (e.g. UCR/UEA classification)."""

    def __init__(
        self,
        data: np.ndarray | pd.DataFrame,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None = None,
    ) -> None:
        super().__init__(...)
        self._n: int = 0

    def __len__(self) -> int:
        return len(self._data)

    def _go_to_idx(self, idx: int) -> None:
        self._n = idx

    def _get_current_label(self) -> int | None:
        if self._labels is None:
            return None
        return self._labels.iloc[self._n]
```

**Univariate fixed pattern** (rbspaper lines 167-198):
```python
class FixedTimeSeriesDatasetUnivariate(FixedTimeSeriesDataset, ABC):
    """Univariate classification dataset (UCR-style)."""

    def __init__(
        self,
        data: pd.DataFrame,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None = None,
    ) -> None:
        super().__init__(...)
        self._n = 0

    def _get_current_data(self) -> np.ndarray:
        return self._data.iloc[self._n].values
```

**Multivariate fixed pattern** (rbspaper lines 201-232):
```python
class FixedTimeSeriesDatasetMultivariate(FixedTimeSeriesDataset, ABC):
    """Multivariate classification dataset (UEA-style)."""

    def __init__(
        self,
        data: np.ndarray,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None = None,
    ) -> None:
        super().__init__(...)
        self._n = 0

    def _get_current_data(self) -> np.ndarray:
        return self._data[self._n]
```

**__all__ export** (rbspaper lines 32-40):
```python
__all__ = [
    'FixedTimeSeriesDataset',
    'FixedTimeSeriesDatasetMultivariate',
    'FixedTimeSeriesDatasetUnivariate',
    'TimeSeriesDataset',
]
```

**Key porting note:** `TimeSeriesDataset` must be in `fixed.py` because it is the common base for both fixed and flexible hierarchies. The `flexible.py` file will import `TimeSeriesDataset` from `fixed.py`. Also, `FixedTimeSeriesDataset` needs a `@property seq_len` added per DST-03 (does not exist in rbspaper source; see "Common Pitfalls #4" in RESEARCH.md).

---

### `src/tscollection/datasets/datasets/classes/flexible.py` (class, ABC + concrete)

**Analog:** `_sources/rbspaper/src/rbspaper/data/datasets/abstract.py` (lines 235-421)

**Imports pattern:**
```python
from __future__ import annotations

from abc import ABC
from bisect import bisect
from collections.abc import Callable
from itertools import accumulate
from typing import Any

import numpy as np

from tscollection.datasets.datasets.classes.fixed import TimeSeriesDataset
from tscollection.datasets.datasets.classes.strategies import (
    SequenceHandlingStrategy,
    SequenceHandlingStrategyMultipleFiles,
    SequenceHandlingStrategySingleFile,
)
from tscollection.datasets.datasets.transformations import convert_numpy_to_tensor
from tscollection.datasets.utils import get_num_samples_from_ts
from tscollection.datasets.enums import TimeSeriesDatasetMode
```

**Abstract flexible base** (rbspaper lines 235-284):
```python
class FlexibleTimeSeriesDataset(TimeSeriesDataset, ABC):
    """Abstract base for sliding-window datasets (forecasting)."""

    def __init__(
        self,
        data: list[np.ndarray] | np.ndarray,
        labels: list[np.ndarray] | np.ndarray | None,
        seq_len: int,
        step: int,
        mode: TimeSeriesDatasetMode,
        sequence_handling_strategy: SequenceHandlingStrategy,
        expand_dims_axis: int | None = 1,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None = (
            convert_numpy_to_tensor,
        ),
    ) -> None:
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence,
        )
        self._seq_len = seq_len
        self._step = step
        self._sequence_handling_strategy = sequence_handling_strategy
        self._n = 0
        self._num_sequences = self._get_num_sequences()

    def __len__(self) -> int:
        return self._num_sequences

    def _get_num_sequences(self) -> int:
        return self._sequence_handling_strategy.get_num_sequences(
            data=self._data, seq_len=self._seq_len, step=self._step
        )
```

**Single-file concrete** (rbspaper lines 286-339):
```python
class FlexibleTimeSeriesDatasetSingleFile(FlexibleTimeSeriesDataset):
    """Sliding-window dataset for a single continuous series."""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray | None,
        seq_len: int,
        step: int,
        mode: TimeSeriesDatasetMode,
        sequence_handling_strategy: SequenceHandlingStrategySingleFile,
        expand_dims_axis: int | None = 1,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None = (
            convert_numpy_to_tensor,
        ),
    ) -> None:
        super().__init__(...)
        self._num_sequences = self._get_num_sequences()

    def _get_num_samples(self) -> int:
        return get_num_samples_from_ts(self._data)

    def _go_to_idx(self, idx: int) -> None:
        if idx >= len(self):
            raise IndexError('Index out of range')
        self._n = idx

    def _get_current_label(self) -> np.ndarray | None:
        return self._sequence_handling_strategy.get_current_label(
            data=self._data, labels=self._labels, n=self._n, seq_len=self._seq_len
        )

    def _get_current_data(self) -> np.ndarray:
        return self._data[self._n : self._n + self._seq_len]
```

**Multi-file concrete** (rbspaper lines 342-421):
```python
class FlexibleTimeSeriesDatasetMultipleFiles(FlexibleTimeSeriesDataset):
    """Sliding-window dataset for multiple independent series."""

    def __init__(
        self,
        data: list[np.ndarray],
        labels: list[np.ndarray] | None,
        seq_len: int,
        step: int,
        mode: TimeSeriesDatasetMode,
        sequence_handling_strategy: SequenceHandlingStrategyMultipleFiles,
        expand_dims_axis: int | None = 1,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None = (
            convert_numpy_to_tensor,
        ),
    ) -> None:
        super().__init__(...)
        self._current_file = 0
        self._seq_len = seq_len
        self._step = step
        self._n = 0
        self._num_samples_per_file = self._get_num_samples_per_file()
        self._num_sequences_per_file = sequence_handling_strategy.get_num_sequences_per_file(
            data=self._data, seq_len=self._seq_len, step=self._step
        )
        self._accumulated_num_sequences_per_file = list(accumulate(self._num_sequences_per_file))

    def _get_num_samples_per_file(self) -> list[int]:
        return [get_num_samples_from_ts(ts) for ts in self._data]

    def _go_to_idx(self, idx: int) -> None:
        if idx >= len(self):
            raise IndexError('Index out of range')
        if idx in self._accumulated_num_sequences_per_file:
            self._current_file = self._accumulated_num_sequences_per_file.index(idx)
            self._n = 0
        else:
            file_num = bisect(self._accumulated_num_sequences_per_file, idx)
            self._current_file = file_num
            self._n = (
                idx - self._accumulated_num_sequences_per_file[file_num - 1]
                if file_num != 0
                else idx
            )

    def _get_current_label(self) -> np.ndarray | None:
        return self._sequence_handling_strategy.get_current_label(
            data=self._data,
            labels=self._labels,
            n=self._n,
            seq_len=self._seq_len,
            current_file=self._current_file,
        )

    def _get_current_data(self) -> np.ndarray:
        return self._data[self._current_file][self._n : self._n + self._seq_len]
```

**__all__ export:**
```python
__all__ = [
    'FlexibleTimeSeriesDataset',
    'FlexibleTimeSeriesDatasetMultipleFiles',
    'FlexibleTimeSeriesDatasetSingleFile',
]
```

---

### `src/tscollection/datasets/datasets/classes/strategies.py` (class, strategy ABC)

**Analog:** `_sources/rbspaper/src/rbspaper/data/datasets/strategies.py`

**Imports pattern** (rbspaper lines 8-15):
```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from tscollection.datasets.utils import get_num_samples_from_ts
```

**ABC pattern** (rbspaper lines 27-53):
```python
class SequenceHandlingStrategy(ABC):
    """Abstract strategy for computing sequence windows and labels."""

    @abstractmethod
    def get_num_sequences(self, data: np.ndarray, seq_len: int, step: int) -> int: ...

    @abstractmethod
    def get_current_label(
        self, data: np.ndarray, labels: np.ndarray | None, n: int, seq_len: int, **kwargs
    ) -> np.ndarray | None: ...


class SequenceHandlingStrategySingleFile(SequenceHandlingStrategy): ...

class SequenceHandlingStrategyMultipleFiles(SequenceHandlingStrategy):
    @abstractmethod
    def get_num_sequences_per_file(
        self, data: list[np.ndarray], seq_len: int, step: int
    ) -> list[int]: ...
```

**Forecasting strategy** (rbspaper lines 58-82):
```python
class ForecastingStrategySingleFile(SequenceHandlingStrategySingleFile):
    """Sliding-window strategy for single-series forecasting tasks."""

    def __init__(self, forecast_horizon: int) -> None:
        self._forecast_horizon = forecast_horizon

    def get_num_sequences(self, data: np.ndarray, seq_len: int, step: int) -> int:
        num_samples_ts = get_num_samples_from_ts(data)
        possible_steps = list(
            range(num_samples_ts - seq_len - self._forecast_horizon + 1, 0, -step)
        )
        possible_ends = [x + seq_len for x in possible_steps]
        valid_ends = [e for e in possible_ends if e + self._forecast_horizon <= num_samples_ts]
        return len(valid_ends)

    def get_current_label(
        self, data: np.ndarray, labels: np.ndarray | None, n: int, seq_len: int, **kwargs
    ) -> np.ndarray:
        return data[n + seq_len : n + seq_len + self._forecast_horizon]
```

**Classification single-file strategy** (rbspaper lines 85-99):
```python
class ClassificationStrategySingleFile(SequenceHandlingStrategySingleFile):
    """Sliding-window strategy for single-series classification."""

    def get_num_sequences(self, data: np.ndarray, seq_len: int, step: int) -> int:
        num_samples_ts = get_num_samples_from_ts(data)
        possible_steps = list(range(num_samples_ts - seq_len, 0, -step))
        possible_ends = [x + seq_len for x in possible_steps]
        return len([e for e in possible_ends if e < num_samples_ts])

    def get_current_label(
        self, data: np.ndarray, labels: np.ndarray | None, n: int, seq_len: int, **kwargs
    ) -> np.ndarray | None:
        if labels is None:
            return None
        return labels[n : n + seq_len]
```

**Classification multi-file strategy** (rbspaper lines 102-136):
```python
class ClassificationStrategyMultipleFiles(SequenceHandlingStrategyMultipleFiles):
    """Sliding-window strategy for multi-series classification."""

    def get_num_sequences_per_file(
        self, data: list[np.ndarray], seq_len: int, step: int
    ) -> list[int]:
        counts: list[int] = []
        for ts in data:
            num_samples_ts = get_num_samples_from_ts(ts)
            possible_steps = list(range(num_samples_ts - seq_len, 0, -step))
            possible_ends = [x + seq_len for x in possible_steps]
            counts.append(len([e for e in possible_ends if e < num_samples_ts]))
        return counts

    def get_num_sequences(
        self, data: np.ndarray | list[np.ndarray], seq_len: int, step: int
    ) -> int:
        data_list: list[np.ndarray] = data if isinstance(data, list) else [data]
        return sum(self.get_num_sequences_per_file(data_list, seq_len, step))

    def get_current_label(
        self,
        data: np.ndarray | list[np.ndarray],
        labels: np.ndarray | list[np.ndarray] | None,
        n: int,
        seq_len: int,
        **kwargs: Any,
    ) -> np.ndarray | None:
        if labels is None:
            return None
        current_file = kwargs.get('current_file', 0)
        if isinstance(labels, list):
            return labels[current_file][n : n + seq_len]
        return labels[n : n + seq_len]
```

**__all__ export** (rbspaper lines 17-24):
```python
__all__ = [
    'ClassificationStrategyMultipleFiles',
    'ClassificationStrategySingleFile',
    'ForecastingStrategySingleFile',
    'SequenceHandlingStrategy',
    'SequenceHandlingStrategyMultipleFiles',
    'SequenceHandlingStrategySingleFile',
]
```

---

### `src/tscollection/datasets/datasets/ucr.py` (class, thin wrapper)

**Analog:** `_sources/rbspaper/src/rbspaper/data/datasets/ucr_dataset.py`

This is a 14-line file. Copy the pattern verbatim, adapting imports.

**Full file pattern** (rbspaper lines 1-42):
```python
"""UCR univariate classification dataset."""

from __future__ import annotations

import pandas as pd

from tscollection.datasets.datasets.classes.fixed import FixedTimeSeriesDatasetUnivariate
from tscollection.datasets.datasets.transformations import convert_numpy_to_tensor
from tscollection.datasets.enums import TimeSeriesDatasetMode

__all__ = ['UCRClassificationUnivariateDataset']


class UCRClassificationUnivariateDataset(FixedTimeSeriesDatasetUnivariate):
    """PyTorch Dataset for UCR univariate classification."""

    def __init__(
        self,
        data: pd.DataFrame,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int = 1,
        transformations_sequence: tuple = (convert_numpy_to_tensor,),
    ) -> None:
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence,
        )
```

**Key pattern:** Thin wrapper that sets domain defaults (`expand_dims_axis=1`, `transformations_sequence=(convert_numpy_to_tensor,)`) and delegates everything to the ABC base. No new logic.

---

### `src/tscollection/datasets/datasets/uea.py` (class, thin wrapper)

**Analog:** `_sources/rbspaper/src/rbspaper/data/datasets/uea_dataset.py`

**Full file pattern** (rbspaper lines 1-42):
```python
"""UEA multivariate classification dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tscollection.datasets.datasets.classes.fixed import FixedTimeSeriesDatasetMultivariate
from tscollection.datasets.datasets.transformations import convert_numpy_to_tensor
from tscollection.datasets.enums import TimeSeriesDatasetMode

__all__ = ['UEAClassificationMultivariateDataset']


class UEAClassificationMultivariateDataset(FixedTimeSeriesDatasetMultivariate):
    """PyTorch Dataset for UEA multivariate classification."""

    def __init__(
        self,
        data: np.ndarray,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None = None,
        transformations_sequence: tuple = (convert_numpy_to_tensor,),
    ) -> None:
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence,
        )
```

**Key pattern:** Same as UCR but with `expand_dims_axis=None` (no expansion for multivariate 3D arrays) and accepts `np.ndarray` instead of `pd.DataFrame`.

---

### `src/tscollection/datasets/datasets/ett.py` (class, thin wrapper)

**Analog:** `_sources/rbspaper/src/rbspaper/data/datasets/ett_dataset.py`

**Full file pattern** (rbspaper lines 1-50):
```python
"""ETT (Electricity Transformer Temperature) forecasting dataset."""

from __future__ import annotations

import numpy as np

from tscollection.datasets.datasets.classes.flexible import FlexibleTimeSeriesDatasetSingleFile
from tscollection.datasets.datasets.classes.strategies import ForecastingStrategySingleFile
from tscollection.datasets.datasets.transformations import (
    convert_data_to_np_array,
    convert_numpy_to_tensor,
)
from tscollection.datasets.enums import TimeSeriesDatasetMode

__all__ = ['ETTDataset']


class ETTDataset(FlexibleTimeSeriesDatasetSingleFile):
    """PyTorch Dataset for ETT forecasting (ETTh1/ETTh2/ETTm1/ETTm2)."""

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        step: int,
        forecast_horizon: int,
        transformations_sequence: tuple = (convert_numpy_to_tensor, convert_data_to_np_array),
    ) -> None:
        super().__init__(
            data=data,
            labels=None,
            seq_len=seq_len,
            step=step,
            mode=TimeSeriesDatasetMode.FORECASTING,
            sequence_handling_strategy=ForecastingStrategySingleFile(
                forecast_horizon=forecast_horizon
            ),
            expand_dims_axis=None,
            transformations_sequence=transformations_sequence,
        )
```

**Key pattern:** Injects `ForecastingStrategySingleFile(forecast_horizon=...)` and sets `mode=TimeSeriesDatasetMode.FORECASTING`. Hardcodes `labels=None` since forecasting labels are derived from data by the strategy.

---

### `src/tscollection/datasets/datasets/classes/__init__.py` (config, exports)

**Analog:** `src/tscollection/datasets/enums/__init__.py` (current project file)

Current stub:
```python
"""Abstract base classes for time series datasets."""

__all__ = []  # Populated in Phase 2
```

Target pattern (copy from `enums/__init__.py` at lines 1-17):
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
    ClassificationStrategySingleFile,
    ForecastingStrategySingleFile,
    SequenceHandlingStrategy,
    SequenceHandlingStrategyMultipleFiles,
    SequenceHandlingStrategySingleFile,
)

__all__ = [
    'ClassificationStrategyMultipleFiles',
    'ClassificationStrategySingleFile',
    'FixedTimeSeriesDataset',
    'FixedTimeSeriesDatasetMultivariate',
    'FixedTimeSeriesDatasetUnivariate',
    'FlexibleTimeSeriesDataset',
    'FlexibleTimeSeriesDatasetMultipleFiles',
    'FlexibleTimeSeriesDatasetSingleFile',
    'ForecastingStrategySingleFile',
    'SequenceHandlingStrategy',
    'SequenceHandlingStrategyMultipleFiles',
    'SequenceHandlingStrategySingleFile',
    'TimeSeriesDataset',
]
```

**Key pattern:** Re-exports from submodules, alphabetically sorted, with `__all__` matching the imports. Same structure as `enums/__init__.py`.

---

### `src/tscollection/datasets/datasets/__init__.py` (config, exports)

**Analog:** `src/tscollection/datasets/enums/__init__.py` and `src/tscollection/datasets/__init__.py`

Current stub:
```python
"""Time series dataset classes (PyTorch Dataset)."""

__all__ = []  # Populated in Phase 2
```

Target pattern: Re-exports all concrete datasets and strategies for convenience:
```python
"""Time series dataset classes (PyTorch Dataset)."""

from tscollection.datasets.datasets.classes import (
    FixedTimeSeriesDataset,
    FixedTimeSeriesDatasetMultivariate,
    FixedTimeSeriesDatasetUnivariate,
    FlexibleTimeSeriesDataset,
    FlexibleTimeSeriesDatasetMultipleFiles,
    FlexibleTimeSeriesDatasetSingleFile,
    SequenceHandlingStrategy,
    ForecastingStrategySingleFile,
    ClassificationStrategySingleFile,
    ClassificationStrategyMultipleFiles,
)
from tscollection.datasets.datasets.ett import ETTDataset
from tscollection.datasets.datasets.ucr import UCRClassificationUnivariateDataset
from tscollection.datasets.datasets.uea import UEAClassificationMultivariateDataset

__all__ = [
    'ClassificationStrategyMultipleFiles',
    'ClassificationStrategySingleFile',
    'ETTDataset',
    'FixedTimeSeriesDataset',
    'FixedTimeSeriesDatasetMultivariate',
    'FixedTimeSeriesDatasetUnivariate',
    'FlexibleTimeSeriesDataset',
    'FlexibleTimeSeriesDatasetMultipleFiles',
    'FlexibleTimeSeriesDatasetSingleFile',
    'ForecastingStrategySingleFile',
    'SequenceHandlingStrategy',
    'UCRClassificationUnivariateDataset',
    'UEAClassificationMultivariateDataset',
]
```

---

### `src/tscollection/datasets/utils/__init__.py` (config, exports)

**Analog:** `src/tscollection/datasets/enums/__init__.py`

Current stub:
```python
"""Utility functions for data processing."""

__all__ = []  # Populated in Phase 2
```

Target pattern:
```python
"""Utility functions for data processing."""

from tscollection.datasets.datasets.transformations import (
    convert_data_to_np_array,
    convert_numpy_to_tensor,
    expand_data_dimensionality,
)
from tscollection.datasets.utils.common import (
    FunctionComposer,
    compose,
    get_num_samples_from_ts,
)

__all__ = [
    'FunctionComposer',
    'compose',
    'convert_data_to_np_array',
    'convert_numpy_to_tensor',
    'expand_data_dimensionality',
    'get_num_samples_from_ts',
]
```

Note: `transformations.py` will live under `datasets/datasets/` per the rbspaper source structure (it is imported by `abstract.py`), while `common.py` utilities (`compose`, `FunctionComposer`, `get_num_samples_from_ts`) live under `utils/`. The `utils/__init__.py` re-exports both for convenience.

---

### `tests/test_datasets.py` (test)

**Analog:** `tests/test_package.py`

**Test file structure pattern** (from `test_package.py` lines 1-77):
```python
"""Tests for dataset classes (DST-01 through DST-05)."""

# Constants at module level
PACKAGE_ROOT = pathlib.Path(__file__).parent.parent / 'src' / 'tscollection' / 'datasets'

# Plain function tests with docstrings referencing requirement IDs
def test_classification_yields_data_label():
    """DST-01: Classification dataset yields (data, label) pairs."""
    ...

def test_forecasting_yields_windows():
    """DST-02: Forecasting dataset yields sliding-window sequences."""
    ...

def test_fixed_seq_len_property():
    """DST-03: Fixed datasets expose seq_len as read-only property."""
    ...

def test_flexible_accepts_seq_len_step():
    """DST-04: Flexible datasets accept seq_len and step."""
    ...

def test_strategy_pattern():
    """DST-05: Strategy pattern decouples counting/labels."""
    ...
```

**Key patterns from test_package.py:**
- Docstrings reference requirement IDs (e.g., `"""DST-01: ..."""`)
- Plain `def test_...()` functions, no classes
- Direct imports inside test functions
- Simple `assert` statements (no mocks)
- Uses `pathlib.Path` for file system tests

**Test data pattern:** Per D-05, use synthetic numpy/pandas fixtures:
```python
import numpy as np
import pandas as pd
import torch

# Create synthetic DataFrame for classification tests
synthetic_df = pd.DataFrame(np.random.randn(10, 50))  # 10 samples, 50 timesteps
synthetic_labels = pd.Series([0, 1] * 5)  # 2 classes

# Create synthetic ndarray for forecasting tests
synthetic_forecast_data = np.random.randn(200, 7).astype(np.float32)  # 200 steps, 7 features
```

---

### `tests/conftest.py` (test config)

**Analog:** No existing conftest; follow standard pytest fixture patterns.

**Expected pattern:**
```python
"""Shared pytest fixtures for dataset tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_classification_df():
    """10 samples, 50 timesteps, float32."""
    return pd.DataFrame(np.random.randn(10, 50).astype(np.float32))


@pytest.fixture
def synthetic_classification_labels():
    """Binary labels for 10 samples."""
    return pd.Series([0, 1] * 5)


@pytest.fixture
def synthetic_forecast_data():
    """200 timesteps, 7 features (ETTh1-like)."""
    return np.random.randn(200, 7).astype(np.float32)


@pytest.fixture
def synthetic_multivariate_data():
    """5 samples, 30 timesteps, 4 features (UEA-like)."""
    return np.random.randn(5, 30, 4).astype(np.float32)
```

**Key pattern:** Fixtures return synthetic data that matches real dataset shapes. No file I/O. Fixture names are descriptive and include the data type (df, labels, data).

---

### `tests/fixtures/` (test data directory)

**Analog:** No existing fixtures directory.

Per D-05, this should contain 1-2 minimal real samples for format validation. This is a directory, not a Python file. Contents to be determined during implementation.

---

### `src/tscollection/datasets/datasets/transformations.py` (utility)

**Analog:** `_sources/rbspaper/src/rbspaper/data/datasets/transformations.py`

This is a new file (not listed in ROADMAP explicitly but referenced by D-03 and all dataset imports). It lives under `datasets/datasets/` to match the rbspaper source location.

**Full file pattern** (rbspaper lines 1-55):
```python
"""Data transformation helpers for PyTorch datasets."""

from __future__ import annotations

import numpy as np
import torch

__all__ = ['convert_data_to_np_array', 'convert_numpy_to_tensor', 'expand_data_dimensionality']


def convert_numpy_to_tensor(data: np.ndarray, dtype: str = 'float') -> torch.Tensor:
    dtype_map = {'float': torch.float, 'long': torch.long, 'int': torch.int, 'double': torch.double}
    return torch.from_numpy(data).to(dtype=dtype_map[dtype])


def convert_data_to_np_array(data: list | tuple, dtype: str = 'float') -> np.ndarray:
    dtype_map = {'float': np.float32, 'int': np.int32}
    return np.array(data).astype(dtype_map[dtype])


def expand_data_dimensionality(
    data: np.ndarray | torch.Tensor | list | tuple, expand_dims_axis: int
) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    return np.expand_dims(data, axis=expand_dims_axis)
```

---

### `src/tscollection/datasets/utils/common.py` (utility)

**Analog:** `_sources/rbspaper/src/rbspaper/data/utils/common.py`

This is a new file. Per D-03, port only `compose`, `FunctionComposer`, and `get_num_samples_from_ts`. Do NOT port `load_json`, `flatten_list`, `separate_target_feature_from_df`, `find_project_root`, `closest_power_of_2`, `cartesian_product_dict`, or `AccumulatingTimerCallback` (those are deferred to Phase 5 or out of scope).

**Imports pattern** (rbspaper lines 3-13):
```python
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
```

**Core utilities** (rbspaper lines 33-88):
```python
__all__ = ['FunctionComposer', 'compose', 'get_num_samples_from_ts']


def get_num_samples_from_ts(ts: np.ndarray) -> int:
    """Get number of samples from a time series."""
    return len(ts)


class FunctionComposer:
    """Composes a list of callables into a single callable."""

    def __init__(self, functions: list[Callable]) -> None:
        self.functions = [f for f in functions if f is not None]

    def __call__(self, data: Any) -> Any:
        result = data
        for f in self.functions:
            result = f(result)
        return result


def compose(*functions: Callable) -> Callable:
    """Compose multiple functions into a single callable."""
    return FunctionComposer(list(functions))
```

## Shared Patterns

### Import Path Convention
**Source:** All rbspaper source files + existing `tscollection.datasets` package
**Apply to:** All new files

rbspaper uses `src.rbspaper.*` as the import root. The new code MUST use `tscollection.datasets.*` as the root. Within the `datasets` subpackage, use relative imports for sibling modules:
```python
# CORRECT: relative import within datasets/datasets/
from tscollection.datasets.datasets.classes.fixed import FixedTimeSeriesDataset
# CORRECT: absolute import from enums
from tscollection.datasets.enums import TimeSeriesDatasetMode
# WRONG: leftover rbspaper path
from src.rbspaper.data.datasets.abstract import FixedTimeSeriesDataset
```

### `from __future__ import annotations` Pattern
**Source:** `_sources/rbspaper/src/rbspaper/data/datasets/abstract.py` (line 7)
**Apply to:** `fixed.py`, `flexible.py`, `strategies.py`, `transformations.py`, `common.py`, `ucr.py`, `uea.py`, `ett.py`

All dataset source files should include `from __future__ import annotations` as the first import (after the module docstring) to support forward references without circular import issues. This is the rbspaper convention and matches project guidance.

### `__all__` Export Pattern
**Source:** `src/tscollection/datasets/enums/__init__.py` (lines 11-17)
**Apply to:** All `__init__.py` files and all module files

Every module file declares `__all__` at module level with alphabetically sorted names. The `__init__.py` re-exports match the `__all__` of the submodules.

### Module Docstring Pattern
**Source:** `src/tscollection/datasets/enums/__init__.py` (line 1)
**Apply to:** All new module files

One-line docstring describing the module purpose. Examples:
- `"Abstract base classes for time series datasets."`
- `"Sequence handling strategies for flexible (sliding-window) datasets."`
- `"Data transformation helpers for PyTorch datasets."`

### Test Naming Convention
**Source:** `tests/test_package.py` (lines 20-77)
**Apply to:** `tests/test_datasets.py`

Function names start with `test_` and describe the requirement being verified. Docstrings start with the requirement ID:
```python
def test_classification_yields_data_label():
    """DST-01: Classification dataset yields (data, label) pairs."""
```

### Error Handling in Datasets
**Source:** `_sources/rbspaper/src/rbspaper/data/datasets/abstract.py` (lines 329-330, 396-397)
**Apply to:** `flexible.py` (`_go_to_idx` methods)

Raise `IndexError` with a string message for out-of-range access:
```python
if idx >= len(self):
    raise IndexError('Index out of range')
```

### Strategy Injection Pattern
**Source:** `_sources/rbspaper/src/rbspaper/data/datasets/ett_dataset.py` (lines 45-47)
**Apply to:** `ett.py` (and any future flexible dataset wrappers)

Forecasting datasets inject the strategy in `__init__`:
```python
sequence_handling_strategy=ForecastingStrategySingleFile(
    forecast_horizon=forecast_horizon
),
```

## No Analog Found

All 12 files have analogs. `tests/fixtures/` is a new data directory with no prior example, but this is purely file content (not Python code patterns). The planner should create minimal ARFF/CSV samples based on real UCR/UEA/ETT formats during implementation.

## Metadata

**Analog search scope:** `_sources/rbspaper/src/rbspaper/data/`, `src/tscollection/datasets/`, `tests/`
**Files scanned:** 14 (7 rbspaper source, 7 project files)
**Pattern extraction date:** 2026-05-11
