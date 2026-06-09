"""Abstract base classes for flexible (sliding-window) time series datasets.

Provides sliding-window dataset hierarchies for forecasting and
windowed-classification tasks via the Strategy pattern.
"""

from __future__ import annotations

from abc import ABC
from bisect import bisect
from itertools import accumulate
from typing import TYPE_CHECKING

from tscollection.datasets.datatypes._base.base import TimeSeriesDataset
from tscollection.datasets.utils import get_num_samples_from_ts
from tscollection.datasets.utils.transformations import convert_numpy_to_tensor

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from tscollection.datasets.datatypes._base.strategies import (
        SequenceHandlingStrategy,
        SequenceHandlingStrategyMultipleFiles,
        SequenceHandlingStrategySingleFile,
    )
    from tscollection.datasets.enums import TimeSeriesDatasetMode

__all__ = [
    'FlexibleTimeSeriesDataset',
    'FlexibleTimeSeriesDatasetMultipleFiles',
    'FlexibleTimeSeriesDatasetSingleFile',
    'FlexibleTimeSeriesDatasetSingleFileMultipleSeries',
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
        # Bounds check with negative index normalization
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
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
        self._num_samples_per_file = self._get_num_samples_per_file()
        self._num_sequences_per_file = sequence_handling_strategy.get_num_sequences_per_file(
            data=self._data, seq_len=self._seq_len, step=self._step
        )
        self._accumulated_num_sequences_per_file = list(accumulate(self._num_sequences_per_file))

    def _get_num_samples_per_file(self) -> list[int]:
        return [get_num_samples_from_ts(ts) for ts in self._data]

    def _go_to_idx(self, idx: int) -> None:
        # Bounds check with negative index normalization
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            msg = 'Index out of range'
            raise IndexError(msg)
        file_num = bisect(self._accumulated_num_sequences_per_file, idx)
        self._current_file = file_num
        self._n = (
            idx - self._accumulated_num_sequences_per_file[file_num - 1] if file_num != 0 else idx
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


class FlexibleTimeSeriesDatasetSingleFileMultipleSeries(FlexibleTimeSeriesDataset):
    """Sliding-window dataset for a single file containing multiple series.

    Handles 3D input arrays of shape (num_series, T, features) where each
    series is an independent time series. Uses ``bisect`` + ``accumulate``
    to map a global index to (series_index, window_index).

    Args:
        data: 3-D numpy array of shape (num_series, T, features).
        labels: Optional label array.
        seq_len: Window length.
        step: Step between windows.
        mode: Dataset mode.
        sequence_handling_strategy: Single-file label strategy.
        expand_dims_axis: Dimension to expand.
        transformations_sequence: Post-processing callables.
    """

    _data: np.ndarray

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray | None,
        seq_len: int,
        step: int,
        mode: TimeSeriesDatasetMode,
        sequence_handling_strategy: SequenceHandlingStrategySingleFile,
        expand_dims_axis: int | None = None,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None = (
            convert_numpy_to_tensor,
        ),
    ) -> None:
        if data.ndim != 3:
            msg = (
                f'Expected 3D array (num_series, T, features), '
                f'got {data.ndim}D with shape {data.shape}'
            )
            raise ValueError(msg)

        # Pre-compute per-series window counts before super().__init__() because
        # the parent calls _get_num_sequences() during initialization
        self._num_series = data.shape[0]
        self._num_sequences_per_series: list[int] = []
        for s in range(self._num_series):
            count = sequence_handling_strategy.get_num_sequences(
                data=data[s], seq_len=seq_len, step=step
            )
            self._num_sequences_per_series.append(count)
        self._accumulated_sequences = list(accumulate(self._num_sequences_per_series))

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

    def _get_num_sequences(self) -> int:
        """Return total windows across all series."""
        return self._accumulated_sequences[-1] if self._accumulated_sequences else 0

    def _go_to_idx(self, idx: int) -> None:
        """Map global index to (series, window) position."""
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            msg = 'Index out of range'
            raise IndexError(msg)
        series_num = bisect(self._accumulated_sequences, idx)
        self._current_series = series_num
        self._n = idx - self._accumulated_sequences[series_num - 1] if series_num != 0 else idx

    def _get_current_data(self) -> np.ndarray:
        """Return data window for current series position."""
        series_data = self._data[self._current_series]
        return series_data[self._n : self._n + self._seq_len]

    def _get_current_label(self) -> np.ndarray | None:
        """Return label for current series window."""
        series_data = self._data[self._current_series]
        return self._sequence_handling_strategy.get_current_label(
            data=series_data, labels=self._labels, n=self._n, seq_len=self._seq_len
        )
