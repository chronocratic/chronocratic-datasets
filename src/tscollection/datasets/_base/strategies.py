"""Sequence handling strategies for flexible (sliding-window) datasets.

Implements the Strategy pattern so that forecasting and classification
datasets can share the same base dataset class while differing in how
they count sequences and extract labels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

import numpy as np

from tscollection.datasets.utils import get_num_samples_from_ts

__all__ = [
    'ClassificationStrategyMultipleFiles',
    'ClassificationStrategySingleFile',
    'ForecastingStrategySingleFile',
    'SequenceHandlingStrategy',
    'SequenceHandlingStrategyMultipleFiles',
    'SequenceHandlingStrategySingleFile',
]


class SequenceHandlingStrategy(ABC):
    """Abstract strategy for computing sequence windows and labels.

    Subclasses implement ``get_num_sequences`` for counting valid sliding-window
    positions and ``get_current_label`` for extracting the label of the n-th
    sequence. Used by :class:`FlexibleTimeSeriesDataset` to decouple iteration
    logic from domain-specific windowing rules.
    """

    @abstractmethod
    def get_num_sequences(
        self, data: np.ndarray | list[np.ndarray], seq_len: int, step: int
    ) -> int:
        """Return total number of sliding-window sequences.

        Args:
            data: Input time series array (1-D or 2-D).
            seq_len: Length of each sequence window.
            step: Stride between consecutive windows.

        Returns:
            Number of valid sequence positions.
        """

    @abstractmethod
    def get_current_label(
        self,
        data: np.ndarray | list[np.ndarray],
        labels: np.ndarray | list[np.ndarray] | None,
        n: int,
        seq_len: int,
        **_kwargs: object,
    ) -> np.ndarray | None:
        """Return the label for the n-th sequence window.

        Args:
            data: Input time series array.
            labels: Label array, or ``None`` for label-less modes.
            n: Starting index of the current sequence window.
            seq_len: Length of the sequence window.
            **kwargs: Additional context (e.g., ``current_file``).

        Returns:
            Label slice or ``None`` if labels are unavailable.
        """


class SequenceHandlingStrategySingleFile(SequenceHandlingStrategy, ABC):
    """Abstract strategy for a single continuous time series.

    Provides no additional methods beyond :class:`SequenceHandlingStrategy`;
    serves as a narrowing base class for single-file implementations.
    """


class SequenceHandlingStrategyMultipleFiles(SequenceHandlingStrategy, ABC):
    """Abstract strategy for multiple independent time series.

    Adds ``get_num_sequences_per_file`` for computing per-file window counts,
    required when the dataset spans several separate arrays.
    """

    @abstractmethod
    def get_num_sequences_per_file(
        self, data: list[np.ndarray], seq_len: int, step: int
    ) -> list[int]:
        """Return per-file sequence counts.

        Args:
            data: List of per-file time series arrays.
            seq_len: Length of each sequence window.
            step: Stride between consecutive windows.

        Returns:
            List of integers — one count per file.
        """


# --------------------------------------------------------------------------- #
# Concrete strategies                                                          #
# --------------------------------------------------------------------------- #


class ForecastingStrategySingleFile(SequenceHandlingStrategySingleFile):
    """Sliding-window strategy for single-series forecasting tasks.

    The label is the segment of the time series immediately following the
    input window.

    Args:
        forecast_horizon: Number of time steps to predict after each window.
    """

    def __init__(self, forecast_horizon: int) -> None:
        self._forecast_horizon = forecast_horizon

    def get_num_sequences(
        self, data: np.ndarray | list[np.ndarray], seq_len: int, step: int
    ) -> int:
        """Return number of forecasting windows."""
        arr = cast('np.ndarray', data) if not isinstance(data, list) else data[0]
        num_samples_ts = get_num_samples_from_ts(arr)
        possible_steps = list(
            range(num_samples_ts - seq_len - self._forecast_horizon + 1, -1, -step)
        )
        possible_ends = [x + seq_len for x in possible_steps]
        valid_ends = [e for e in possible_ends if e + self._forecast_horizon <= num_samples_ts]
        return len(valid_ends)

    def get_current_label(
        self,
        data: np.ndarray | list[np.ndarray],
        labels: np.ndarray | list[np.ndarray] | None,
        n: int,
        seq_len: int,
        **_kwargs: object,
    ) -> np.ndarray:
        """Return forecast target segment after the input window."""
        _ = labels
        arr = cast('np.ndarray', data) if not isinstance(data, list) else data[0]
        return arr[n + seq_len : n + seq_len + self._forecast_horizon]


class ClassificationStrategySingleFile(SequenceHandlingStrategySingleFile):
    """Sliding-window strategy for single-series classification."""

    def get_num_sequences(
        self, data: np.ndarray | list[np.ndarray], seq_len: int, step: int
    ) -> int:
        """Return number of classification windows."""
        num_samples_ts = get_num_samples_from_ts(data)
        possible_steps = list(range(num_samples_ts - seq_len, -1, -step))
        possible_ends = [x + seq_len for x in possible_steps]
        return len([e for e in possible_ends if e <= num_samples_ts])

    def get_current_label(
        self,
        data: np.ndarray | list[np.ndarray],
        labels: np.ndarray | list[np.ndarray] | None,
        n: int,
        seq_len: int,
        **_kwargs: object,
    ) -> np.ndarray | None:
        """Return label slice for the classification window."""
        _ = data
        if labels is None:
            return None
        arr = cast('np.ndarray', labels) if not isinstance(labels, list) else labels[0]
        return arr[n : n + seq_len]


class ClassificationStrategyMultipleFiles(SequenceHandlingStrategyMultipleFiles):
    """Sliding-window strategy for multi-series classification.

    Handles a list of independent time series arrays, computing per-file
    sequence counts and extracting labels using a ``current_file`` index
    passed via ``kwargs``.
    """

    def get_num_sequences_per_file(
        self, data: list[np.ndarray], seq_len: int, step: int
    ) -> list[int]:
        """Return per-file classification window counts."""
        counts: list[int] = []
        for ts in data:
            num_samples_ts = get_num_samples_from_ts(ts)
            possible_steps = list(range(num_samples_ts - seq_len, -1, -step))
            possible_ends = [x + seq_len for x in possible_steps]
            counts.append(len([e for e in possible_ends if e <= num_samples_ts]))
        return counts

    def get_num_sequences(
        self, data: np.ndarray | list[np.ndarray], seq_len: int, step: int
    ) -> int:
        """Return total classification windows across all files."""
        data_list: list[np.ndarray] = (
            [data] if isinstance(data, np.ndarray) else data
        )
        return sum(self.get_num_sequences_per_file(data_list, seq_len, step))

    def get_current_label(
        self,
        data: np.ndarray | list[np.ndarray],
        labels: np.ndarray | list[np.ndarray] | None,
        n: int,
        seq_len: int,
        **_kwargs: object,
    ) -> np.ndarray | None:
        """Return label slice for multi-file classification window."""
        _ = data
        if labels is None:
            return None
        current_file: int = cast('int', _kwargs.get('current_file', 0))
        if isinstance(labels, list):
            return labels[current_file][n : n + seq_len]
        return labels[n : n + seq_len]
