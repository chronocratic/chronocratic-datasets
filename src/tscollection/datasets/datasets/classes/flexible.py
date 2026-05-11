"""Abstract base classes for flexible (sliding-window) time series datasets.

Provides sliding-window dataset hierarchies for forecasting and
windowed-classification tasks via the Strategy pattern.
"""

from __future__ import annotations

from abc import ABC
from bisect import bisect
from itertools import accumulate
from typing import TYPE_CHECKING

from tscollection.datasets.datasets.classes.fixed import TimeSeriesDataset
from tscollection.datasets.datasets.transformations import convert_numpy_to_tensor
from tscollection.datasets.utils import get_num_samples_from_ts

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from tscollection.datasets.datasets.classes.strategies import (
        SequenceHandlingStrategy,
        SequenceHandlingStrategyMultipleFiles,
        SequenceHandlingStrategySingleFile,
    )
    from tscollection.datasets.enums import TimeSeriesDatasetMode

__all__ = [
    'FlexibleTimeSeriesDataset',
    'FlexibleTimeSeriesDatasetMultipleFiles',
    'FlexibleTimeSeriesDatasetSingleFile',
]


class FlexibleTimeSeriesDataset(TimeSeriesDataset, ABC):
    """Abstract base for sliding-window datasets (forecasting).

    Creates fixed-length windows from a continuous time series using
    an injected :class:`SequenceHandlingStrategy`.

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

    _data: np.ndarray | list[np.ndarray]
    _labels: np.ndarray | list[np.ndarray] | None

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
        """Return the number of sliding-window sequences."""
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

    _data: np.ndarray
    _labels: np.ndarray | None

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
        # T-02-02-03: Bounds check
        if idx >= len(self):
            msg = 'Index out of range'
            raise IndexError(msg)
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
    (file_index, local_sequence_index) using ``bisect`` and
    ``itertools.accumulate``.

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

    _data: list[np.ndarray]
    _labels: list[np.ndarray] | None

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
        self._accumulated_num_sequences_per_file = list(
            accumulate(self._num_sequences_per_file)
        )

    def _get_num_samples_per_file(self) -> list[int]:
        return [get_num_samples_from_ts(ts) for ts in self._data]

    def _go_to_idx(self, idx: int) -> None:
        # T-02-02-03: Bounds check
        if idx >= len(self):
            msg = 'Index out of range'
            raise IndexError(msg)
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
