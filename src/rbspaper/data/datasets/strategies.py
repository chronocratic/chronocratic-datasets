"""Sequence handling strategies for flexible (sliding-window) datasets.

Implements the Strategy pattern so that forecasting and classification
datasets can share the same base dataset class while differing in how
they count sequences and extract labels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.rbspaper.data.utils.common import get_num_samples_from_ts

__all__ = [
    'ClassificationStrategyMultipleFiles',
    'ClassificationStrategySingleFile',
    'ForecastingStrategySingleFile',
    'SequenceHandlingStrategy',
    'SequenceHandlingStrategyMultipleFiles',
    'SequenceHandlingStrategySingleFile',
]


class SequenceHandlingStrategy(ABC):
    """Abstract strategy for computing sequence windows and labels."""

    @abstractmethod
    def get_num_sequences(self, data: np.ndarray, seq_len: int, step: int) -> int:
        """Return total number of sliding-window sequences."""

    @abstractmethod
    def get_current_label(
        self, data: np.ndarray, labels: np.ndarray | None, n: int, seq_len: int, **kwargs
    ) -> np.ndarray | None:
        """Return the label for the n-th sequence window."""


class SequenceHandlingStrategySingleFile(SequenceHandlingStrategy):
    """Strategy for a single continuous time series."""


class SequenceHandlingStrategyMultipleFiles(SequenceHandlingStrategy):
    """Strategy for multiple independent time series."""

    @abstractmethod
    def get_num_sequences_per_file(
        self, data: list[np.ndarray], seq_len: int, step: int
    ) -> list[int]:
        """Return per-file sequence counts."""


# -- Concrete strategies --


class ForecastingStrategySingleFile(SequenceHandlingStrategySingleFile):
    """Sliding-window strategy for single-series forecasting tasks.

    The label is the segment immediately following the input window.

    Args:
        forecast_horizon: Number of steps to predict.
    """

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
        # data may be typed as np.ndarray but is really List[np.ndarray]
        data_list: list[np.ndarray] = data if isinstance(data, list) else [data]  # ty: ignore[invalid-assignment] isinstance narrowing on ndarray | list[ndarray] is a known ty limitation
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
