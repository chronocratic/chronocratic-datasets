"""Abstract PyTorch Dataset base classes for time series data.

Provides fixed-length and flexible (sliding-window) dataset hierarchies
for classification and forecasting tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from bisect import bisect
from collections.abc import Callable
from functools import partial
from itertools import accumulate
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.rbspaper.data.datasets.strategies import (
    SequenceHandlingStrategy,
    SequenceHandlingStrategyMultipleFiles,
    SequenceHandlingStrategySingleFile,
)
from src.rbspaper.data.datasets.transformations import (
    convert_numpy_to_tensor,
    expand_data_dimensionality,
)
from src.rbspaper.data.utils.common import compose, get_num_samples_from_ts
from src.rbspaper.enums.data_enums import TimeSeriesDatasetMode

__all__ = [
    'FixedTimeSeriesDataset',
    'FixedTimeSeriesDatasetMultivariate',
    'FixedTimeSeriesDatasetUnivariate',
    'FlexibleTimeSeriesDataset',
    'FlexibleTimeSeriesDatasetMultipleFiles',
    'FlexibleTimeSeriesDatasetSingleFile',
    'TimeSeriesDataset',
]


class TimeSeriesDataset(Dataset[Any], ABC):
    """Abstract base for all time series datasets.

    Supports three modes via mode-specific sample getters:
    - WITHOUT_LABELS (training)
    - WITH_LABELS (evaluation)
    - FORECASTING (input/target pairs)

    Args:
        data: Raw time series data.
        labels: Optional label array or Series.
        mode: Determines the sample signature.
        expand_dims_axis: Axis along which to expand data dimensions.
        transformations_sequence: Post-processing callables.
    """

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
    def _go_to_idx(self, idx: int) -> None:
        """Position internal cursor at index idx."""

    @abstractmethod
    def _get_current_data(self) -> np.ndarray:
        """Return data at current cursor position."""

    @abstractmethod
    def _get_current_label(self) -> np.ndarray | int | None:
        """Return label at current cursor position."""

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


class FixedTimeSeriesDataset(TimeSeriesDataset, ABC):
    """Dataset for fixed-length time series (e.g. UCR/UEA classification).

    Each sample is an independent time series of fixed length.

    Args:
        data: 2-D array or DataFrame of shape (samples, timesteps).
        labels: Optional label Series.
        mode: Dataset mode.
        expand_dims_axis: Dimension to expand.
        transformations_sequence: Post-processing callables.
    """

    def __init__(
        self,
        data: np.ndarray | pd.DataFrame,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None = None,
    ) -> None:
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence,
        )
        self._n: int = 0

    def __len__(self) -> int:
        return len(self._data)

    def _go_to_idx(self, idx: int) -> None:
        self._n = idx

    def _get_current_label(self) -> int | None:
        if self._labels is None:
            return None
        return self._labels.iloc[self._n]


class FixedTimeSeriesDatasetUnivariate(FixedTimeSeriesDataset, ABC):
    """Univariate classification dataset (UCR-style).

    Each row of the DataFrame is one time series.

    Args:
        data: DataFrame of shape (samples, timesteps).
        labels: Optional label Series.
        mode: Dataset mode.
        expand_dims_axis: Dimension to expand.
        transformations_sequence: Post-processing callables.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None = None,
    ) -> None:
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence,
        )
        self._n = 0

    def _get_current_data(self) -> np.ndarray:
        return self._data.iloc[self._n].values


class FixedTimeSeriesDatasetMultivariate(FixedTimeSeriesDataset, ABC):
    """Multivariate classification dataset (UEA-style).

    Each entry is a 3-D array (sample, timestep, feature).

    Args:
        data: 3-D numpy array of shape (samples, timesteps, features).
        labels: Optional label Series.
        mode: Dataset mode.
        expand_dims_axis: Dimension to expand.
        transformations_sequence: Post-processing callables.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None = None,
    ) -> None:
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence,
        )
        self._n = 0

    def _get_current_data(self) -> np.ndarray:
        return self._data[self._n]


class FlexibleTimeSeriesDataset(TimeSeriesDataset, ABC):
    """Abstract base for sliding-window datasets (forecasting).

    Creates fixed-length windows from a continuous time series.

    Args:
        data: Time series data (list of arrays or single array).
        labels: Optional label arrays.
        seq_len: Length of each sliding window.
        step: Step between consecutive windows.
        mode: Dataset mode.
        sequence_handling_strategy: Strategy for counting windows and labels.
        expand_dims_axis: Dimension to expand.
        transformations_sequence: Post-processing callables.
    """

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


class FlexibleTimeSeriesDatasetSingleFile(FlexibleTimeSeriesDataset):
    """Sliding-window dataset for a single continuous series.

    Args:
        data: 1-D or 2-D numpy array.
        labels: Optional label array.
        seq_len: Window length.
        step: Step between windows.
        mode: Dataset mode.
        sequence_handling_strategy: Label extraction strategy.
        expand_dims_axis: Dimension to expand.
        transformations_sequence: Post-processing callables.
    """

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
        super().__init__(
            data=data,
            labels=labels,
            seq_len=seq_len,
            step=step,
            mode=mode,
            sequence_handling_strategy=sequence_handling_strategy,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence,
        )
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


class FlexibleTimeSeriesDatasetMultipleFiles(FlexibleTimeSeriesDataset):
    """Sliding-window dataset for multiple independent series.

    Manages per-file sequence counts and maps a global index to
    (file_index, local_sequence_index).

    Args:
        data: List of numpy arrays.
        labels: Optional list of label arrays.
        seq_len: Window length.
        step: Step between windows.
        mode: Dataset mode.
        sequence_handling_strategy: Multi-file strategy.
        expand_dims_axis: Dimension to expand.
        transformations_sequence: Post-processing callables.
    """

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
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            seq_len=seq_len,
            step=step,
            sequence_handling_strategy=sequence_handling_strategy,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence,
        )
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
