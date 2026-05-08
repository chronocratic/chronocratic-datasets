__all__ = [
    'ClassificationStrategyMultipleFiles',
    'ClassificationStrategySingleFile',
    'ForecastingStrategySingleFile',
    'SequenceHandlingStrategy',
    'SequenceHandlingStrategyMultipleFiles',
    'SequenceHandlingStrategySingleFile',
]

from abc import ABC, abstractmethod
from typing import cast

import numpy as np
import pandas as pd

from src.autotsrc.utils import get_num_samples_from_ts


class SequenceHandlingStrategy[DataT: (np.ndarray, list[np.ndarray])](ABC):
    @abstractmethod
    def get_num_sequences(self, data: DataT, seq_len: int, step: int) -> int:
        """
        Return the total number of valid sliding-window sequences in data.

        :param data: the time series data
        :param seq_len: length of each sequence window
        :param step: stride between consecutive windows
        :return: number of extractable sequences
        """

    @abstractmethod
    def get_current_label(
        self,
        data: DataT,
        labels: pd.Series | pd.DataFrame | np.ndarray | list[np.ndarray] | None,
        n: int,
        seq_len: int,
        **kwargs: object,
    ) -> np.ndarray | None:
        """
        Return the label for the sequence starting at position n, or None.

        :param data: the time series data
        :param labels: the label array, or None
        :param n: starting index of the current sequence
        :param seq_len: length of the sequence window
        :return: label array or None if labels are not available
        """


class SequenceHandlingStrategySingleFile(SequenceHandlingStrategy[np.ndarray]):
    @abstractmethod
    def get_num_sequences(self, data: np.ndarray, seq_len: int, step: int) -> int:
        """
        Return the total number of valid sequences in a single contiguous array.

        :param data: 1-D or 2-D time series array
        :param seq_len: length of each sequence window
        :param step: stride between consecutive windows
        :return: number of extractable sequences
        """

    @abstractmethod
    def get_current_label(
        self,
        data: np.ndarray,
        labels: pd.Series | pd.DataFrame | np.ndarray | list[np.ndarray] | None,
        n: int,
        seq_len: int,
        **kwargs: object,
    ) -> np.ndarray | None:
        """
        Return the label for the sequence starting at position n, or None.

        :param data: the time series array
        :param labels: the label array, or None
        :param n: starting index of the current sequence
        :param seq_len: length of the sequence window
        :return: label slice or None
        """


class SequenceHandlingStrategyMultipleFiles(SequenceHandlingStrategy[list[np.ndarray]]):
    @abstractmethod
    def get_num_sequences_per_file(
        self,
        data: pd.Series | pd.DataFrame | np.ndarray | list[np.ndarray],
        seq_len: int,
        step: int,
    ) -> list[int]:
        """
        Return the number of valid sequences for each individual file.

        :param data: list of per-file time series arrays
        :param seq_len: length of each sequence window
        :param step: stride between consecutive windows
        :return: list where element i is the sequence count for file i
        """

    @abstractmethod
    def get_num_sequences(self, data: list[np.ndarray], seq_len: int, step: int) -> int:
        """
        Return the total number of valid sequences across all files.

        :param data: list of per-file time series arrays
        :param seq_len: length of each sequence window
        :param step: stride between consecutive windows
        :return: sum of sequence counts across all files
        """

    def get_current_label(
        self,
        data: list[np.ndarray],
        labels: pd.Series | pd.DataFrame | np.ndarray | list[np.ndarray] | None,
        n: int,
        seq_len: int,
        **kwargs: object,
    ) -> np.ndarray | None:
        """
        Return the label for position n in the current file, or None.

        :param data: list of per-file time series arrays
        :param labels: list of per-file label arrays, or None
        :param n: local starting index within the current file
        :param seq_len: length of the sequence window
        :param kwargs: must contain ``current_file`` (int) — index of the active file
        :return: label slice or None
        """


class ForecastingStrategySingleFile(SequenceHandlingStrategySingleFile):
    def __init__(self, forecast_horizon: int) -> None:
        self._forecast_horizon = forecast_horizon

    def get_num_sequences(self, data: np.ndarray, seq_len: int, step: int) -> int:
        """
        Return the number of sequences for which a full forecast horizon fits within data.

        :param data: the time series array
        :param seq_len: length of the input sequence window
        :param step: stride between consecutive windows
        :return: count of valid (input, forecast) pairs
        """
        num_samples_ts = get_num_samples_from_ts(data)
        possible_steps = list(
            range(num_samples_ts - seq_len - self._forecast_horizon + 1, 0, -step)
        )
        possible_ends = [x + seq_len for x in possible_steps]
        valid_ends = list(
            filter(lambda x: x + self._forecast_horizon <= num_samples_ts, possible_ends)
        )
        return len(valid_ends)

    def get_current_label(
        self,
        data: np.ndarray,
        labels: pd.Series | pd.DataFrame | np.ndarray | list[np.ndarray] | None,  # noqa: ARG002
        n: int,
        seq_len: int,
        **_kwargs: object,
    ) -> np.ndarray | None:
        """
        Return the forecast target window immediately following the input sequence.

        :param data: the time series array
        :param labels: unused; the forecast target is derived from data
        :param n: starting index of the input sequence
        :param seq_len: length of the input sequence window
        :return: slice of data of length forecast_horizon starting at n + seq_len
        """
        return data[n + seq_len : n + seq_len + self._forecast_horizon]


class ClassificationStrategySingleFile(SequenceHandlingStrategySingleFile):
    def get_num_sequences(self, data: np.ndarray, seq_len: int, step: int) -> int:
        """
        Return the number of valid classification sequences (no forecast horizon).

        :param data: the time series array
        :param seq_len: length of each sequence window
        :param step: stride between consecutive windows
        :return: number of extractable sequences
        """
        num_samples_ts = get_num_samples_from_ts(data)
        possible_steps = list(range(num_samples_ts - seq_len, 0, -step))
        possible_ends = [x + seq_len for x in possible_steps]
        valid_ends = list(filter(lambda x: x < num_samples_ts, possible_ends))
        return len(valid_ends)

    def get_current_label(
        self,
        data: np.ndarray,  # noqa: ARG002
        labels: pd.Series | pd.DataFrame | np.ndarray | list[np.ndarray] | None,
        n: int,
        seq_len: int,
        **_kwargs: object,
    ) -> np.ndarray | None:
        """
        Return the label slice aligned with the current sequence window.

        :param data: the time series array (unused)
        :param labels: the label array, or None
        :param n: starting index of the current sequence
        :param seq_len: length of the sequence window
        :return: labels[n : n + seq_len], or None if labels is None
        """
        if labels is None:
            return None
        return np.array(labels[n : n + seq_len])


class ClassificationStrategyMultipleFiles(SequenceHandlingStrategyMultipleFiles):
    def get_num_sequences_per_file(
        self,
        data: pd.Series | pd.DataFrame | np.ndarray | list[np.ndarray],
        seq_len: int,
        step: int,
    ) -> list[int]:
        """
        Return the number of valid classification sequences for each file.

        :param data: list of per-file time series arrays
        :param seq_len: length of each sequence window
        :param step: stride between consecutive windows
        :return: list where element i is the sequence count for file i
        """
        num_sequences = []
        for ts in data:
            num_samples_ts = get_num_samples_from_ts(ts)
            possible_steps = list(range(num_samples_ts - seq_len, 0, -step))
            possible_ends = [x + seq_len for x in possible_steps]
            valid_ends = list(filter(lambda x: x < num_samples_ts, possible_ends))
            num_sequences.append(len(valid_ends))

        return num_sequences

    def get_num_sequences(self, data: list[np.ndarray], seq_len: int, step: int) -> int:
        """
        Return the total sequence count summed across all files.

        :param data: list of per-file time series arrays
        :param seq_len: length of each sequence window
        :param step: stride between consecutive windows
        :return: total number of extractable sequences
        """
        num_sequences_per_file = self.get_num_sequences_per_file(data, seq_len, step)
        return sum(num_sequences_per_file)

    def get_current_label(
        self,
        data: list[np.ndarray],  # noqa: ARG002
        labels: pd.Series | pd.DataFrame | np.ndarray | list[np.ndarray] | None,
        n: int,
        seq_len: int,
        **kwargs: object,
    ) -> np.ndarray | None:
        """
        Return the label slice for position n within the current file.

        :param data: list of per-file time series arrays (unused)
        :param labels: list of per-file label arrays, or None
        :param n: local starting index within the current file
        :param seq_len: length of the sequence window
        :param kwargs: must contain ``current_file`` (int) — index of the active file
        :return: labels[current_file][n : n + seq_len], or None if labels is None
        """
        if labels is None:
            return None
        current_file = cast('int', kwargs['current_file'])
        return labels[current_file][n : n + seq_len]
