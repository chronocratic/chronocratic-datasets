__all__ = [
    'FixedTimeSeriesDatasetMultivariate',
    'FixedTimeSeriesDatasetUnivariate',
    'FlexibleTimeSeriesDatasetMultipleFiles',
    'FlexibleTimeSeriesDatasetSingleFile',
]

from abc import ABC, abstractmethod
from bisect import bisect
from collections.abc import Callable
from functools import partial
from itertools import accumulate

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.autotsrc.enums import TimeSeriesDatasetMode
from src.autotsrc.utils import compose, get_num_samples_from_ts
from src.autotsrc.utils.transformations import convert_numpy_to_tensor, expand_data_dimensionality

from .strategies import (
    SequenceHandlingStrategy,
    SequenceHandlingStrategyMultipleFiles,
    SequenceHandlingStrategySingleFile,
)


class TimeSeriesDataset(Dataset, ABC):
    @property
    def _get_sample_fun_map(self) -> dict[TimeSeriesDatasetMode, Callable]:
        return {
            TimeSeriesDatasetMode.WITHOUT_LABELS: self._get_sample_1,
            TimeSeriesDatasetMode.WITH_LABELS: self._get_sample_2,
            TimeSeriesDatasetMode.FORECASTING: self._get_sample_3,
        }

    def __init__(
        self,
        data: np.ndarray | pd.DataFrame | list[np.ndarray],
        labels: pd.Series | pd.DataFrame | np.ndarray | list[np.ndarray] | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None,
    ) -> None:
        super().__init__()
        self._data = data
        self._labels = labels
        self._mode = mode
        self._get_sample = self._get_sample_fun_map[self._mode]
        self._initiate_transformation_functionality(transformations_sequence, expand_dims_axis)

    @abstractmethod
    def _go_to_idx(self, idx: int) -> None:
        pass

    @abstractmethod
    def _get_current_data(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_current_label(self) -> np.ndarray | int | None:
        pass

    def _initiate_transformation_functionality(
        self,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None,
        expand_dims_axis: int | None,
    ) -> None:
        if transformations_sequence is not None:
            transformations_sequence = [*transformations_sequence]
        else:
            transformations_sequence = []
        if expand_dims_axis:
            transformations_sequence.append(
                partial(expand_data_dimensionality, expand_dims_axis=expand_dims_axis)
            )
        self._transform = compose(*transformations_sequence)

    def _get_sample_1(self) -> np.ndarray:
        sample = self._get_current_data()
        return self._transform(sample)

    def _get_sample_2(self) -> tuple[np.ndarray, np.ndarray | int | None]:
        sample = self._get_current_data()
        label = self._get_current_label()
        return self._transform(sample), label

    def _get_sample_3(self) -> tuple:
        sample = self._get_current_data()
        label = self._get_current_label()
        return (self._transform(sample), self._transform(label))

    def __getitem__(self, index: int) -> np.ndarray | tuple:
        self._go_to_idx(index)
        sample = self._get_sample()
        return sample


class FixedTimeSeriesDataset(TimeSeriesDataset, ABC):
    def __init__(
        self,
        data: np.ndarray | pd.DataFrame,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None,
    ) -> None:
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence,
        )
        self._n = 0

    def __len__(self) -> int:
        return len(self._data)

    def _go_to_idx(self, idx: int) -> None:
        self._n = idx

    @abstractmethod
    def _get_current_data(self) -> np.ndarray:
        pass

    def _get_current_label(self) -> int | None:
        if isinstance(self._labels, pd.Series):
            return self._labels.iloc[self._n]
        if isinstance(self._labels, pd.DataFrame):
            return self._labels.iloc[self._n, 0]
        return None


class FixedTimeSeriesDatasetUnivariate(FixedTimeSeriesDataset, ABC):
    def __init__(
        self,
        data: pd.DataFrame,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None,
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
        if isinstance(self._data, pd.DataFrame):
            return self._data.iloc[self._n].to_numpy()
        return self._data[self._n]


class FixedTimeSeriesDatasetMultivariate(FixedTimeSeriesDataset, ABC):
    def __init__(
        self,
        data: np.ndarray | pd.DataFrame,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None,
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
        if isinstance(self._data, pd.DataFrame):
            return self._data.iloc[self._n].to_numpy()
        return self._data[self._n]


class FlexibleTimeSeriesDataset(TimeSeriesDataset, ABC):
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
        if self._data is None:
            return 0
        return get_num_samples_from_ts(self._data)

    def _go_to_idx(self, idx: int) -> None:
        if idx >= self.__len__():
            e = IndexError('Index out of range')
            raise e
        self._n = idx

    def _get_current_label(self) -> np.ndarray | None:
        if (not isinstance(self._data, np.ndarray)) or (
            self._labels is not None and not isinstance(self._labels, np.ndarray)
        ):
            exception = TypeError('FlexibleTimeSeriesDatasetSingleFile expects numpy array data.')
            raise exception
        return self._sequence_handling_strategy.get_current_label(
            data=self._data, labels=self._labels, n=self._n, seq_len=self._seq_len
        )

    def _get_current_data(self) -> np.ndarray:
        if not isinstance(self._data, np.ndarray):
            exception = TypeError('FlexibleTimeSeriesDatasetSingleFile expects numpy array data.')
            raise exception
        return np.array(self._data[self._n : self._n + self._seq_len])


class FlexibleTimeSeriesDatasetMultipleFiles(FlexibleTimeSeriesDataset):
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
        if not isinstance(self._data, list):
            exception = TypeError('FlexibleTimeSeriesDatasetMultipleFiles expects list data.')
            raise exception
        self._num_sequences_per_file = sequence_handling_strategy.get_num_sequences_per_file(
            data=self._data, seq_len=self._seq_len, step=self._step
        )
        self._accumulated_num_sequences_per_file = list(accumulate(self._num_sequences_per_file))

    def _get_num_samples_per_file(self) -> list[int]:
        if not isinstance(self._data, list):
            exception = TypeError('FlexibleTimeSeriesDatasetMultipleFiles expects list data.')
            raise exception
        return [get_num_samples_from_ts(ts) for ts in self._data]

    def _go_to_idx(self, idx: int) -> None:
        if idx >= self.__len__():
            e = IndexError('Index out of range')
            raise e
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
        if isinstance(self._labels, list):
            if not isinstance(self._data, list):
                exception = TypeError('FlexibleTimeSeriesDatasetMultipleFiles expects list data.')
                raise exception
            return self._sequence_handling_strategy.get_current_label(
                data=self._data,
                labels=self._labels,
                n=self._n,
                seq_len=self._seq_len,
                current_file=self._current_file,
            )
        return None

    def _get_current_data(self) -> np.ndarray:
        if not isinstance(self._data, list):
            exception = TypeError('FlexibleTimeSeriesDatasetMultipleFiles expects list data.')
            raise exception
        return self._data[self._current_file][self._n : self._n + self._seq_len]
