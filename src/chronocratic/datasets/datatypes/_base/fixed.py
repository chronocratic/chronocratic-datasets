"""Abstract base classes for fixed-length time series datasets.

Provides the ``TimeSeriesDataset`` root ABC and the
``FixedTimeSeriesDataset`` hierarchy (univariate and multivariate) for
classification tasks in which each sample is an independent, fixed-length
time series.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from chronocratic.datasets.datatypes._base.base import TimeSeriesDataset

if TYPE_CHECKING:
    from collections.abc import Callable

    from chronocratic.datasets.enums import TimeSeriesDatasetMode

MIN_DIM_FOR_SEQ_LEN: int = 2
"""Minimum array dimensions required to derive sequence length."""

__all__ = [
    "FixedTimeSeriesDataset",
    "FixedTimeSeriesDatasetMultivariate",
    "FixedTimeSeriesDatasetUnivariate",
]


class FixedTimeSeriesDataset(TimeSeriesDataset, ABC):
    """Dataset for fixed-length time series (e.g. UCR/UEA classification).

    Each sample is an independent time series of fixed length.

    Args:
        data: 2-D array or DataFrame of shape (samples, timesteps).
        labels: Optional label Series.
        mode: Dataset mode.
        expand_dims_axis: Dimension to expand.
        transformations_sequence: Post-processing callables.

    Raises:
        TypeError: If *data* is not a ``np.ndarray`` or ``pd.DataFrame``.
        ValueError: If *data* has fewer than 2 dimensions.
    """

    _data: np.ndarray | pd.DataFrame
    _labels: pd.Series | pd.DataFrame | np.ndarray | None

    def __init__(
        self,
        data: np.ndarray | pd.DataFrame,
        labels: pd.Series | pd.DataFrame | np.ndarray | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None = None,
    ) -> None:
        # Type-check data
        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            msg = f"data must be np.ndarray or pd.DataFrame, got {type(data).__name__}"
            raise TypeError(msg)
        # Validate minimum dimensions for seq_len
        if isinstance(data, np.ndarray) and data.ndim < MIN_DIM_FOR_SEQ_LEN:
            msg = f"data must have at least 2 dimensions for seq_len, got {data.ndim}D"
            raise ValueError(msg)
        if isinstance(data, pd.DataFrame) and data.shape[1] < 1:
            msg = "data DataFrame must have at least 1 column for seq_len"
            raise ValueError(msg)
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence,
        )
        self._n: int = 0

    @property
    def seq_len(self) -> int:
        """Return the sequence length of each sample (read-only).

        For ``pd.DataFrame`` inputs this is the number of columns
        (timesteps).  For ``np.ndarray`` inputs this is
        ``data.shape[1]``.

        Returns:
            The number of timesteps per sample.
        """
        if isinstance(self._data, np.ndarray):
            return self._data.shape[1]
        return len(self._data.iloc[0])

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._data)

    def _go_to_idx(self, idx: int) -> None:
        self._n = idx

    def _get_current_label(self) -> int | None:
        if self._labels is None:
            return None
        if isinstance(self._labels, (pd.Series, pd.DataFrame)):
            return self._labels.iloc[self._n]  # pyright: ignore[reportReturnType]
        return self._labels[self._n]


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

    _data: pd.DataFrame

    def __init__(
        self,
        data: pd.DataFrame,
        labels: pd.Series | pd.DataFrame | np.ndarray | None,
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
        return self._data.iloc[self._n].to_numpy(copy=True)


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

    _data: np.ndarray

    def __init__(
        self,
        data: np.ndarray,
        labels: pd.Series | pd.DataFrame | np.ndarray | None,
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
