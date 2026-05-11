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

MIN_DIM_FOR_SEQ_LEN: int = 2
"""Minimum array dimensions required to derive sequence length."""

__all__ = [
    'FixedTimeSeriesDataset',
    'FixedTimeSeriesDatasetMultivariate',
    'FixedTimeSeriesDatasetUnivariate',
    'TimeSeriesDataset',
]


class TimeSeriesDataset(Dataset[Any], ABC):
    """Abstract base for all time series datasets.

    Supports three modes via mode-specific sample getters:

    - ``WITHOUT_LABELS`` (training)
    - ``WITH_LABELS`` (evaluation)
    - ``FORECASTING`` (input/target pairs)

    Args:
        data: Raw time series data.
        labels: Optional label array or Series.
        mode: Determines the sample signature.
        expand_dims_axis: Axis along which to expand data dimensions.
        transformations_sequence: Post-processing callables.
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
        data: np.ndarray | list[np.ndarray] | pd.DataFrame,
        labels: np.ndarray | list[np.ndarray] | pd.Series | pd.DataFrame | None,
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
        """Position internal cursor at index *idx*."""

    @abstractmethod
    def _get_current_data(self) -> np.ndarray:
        """Return data at the current cursor position."""

    @abstractmethod
    def _get_current_label(self) -> np.ndarray | int | None:
        """Return the label at the current cursor position."""

    def _initiate_transformation_functionality(
        self,
        transformations_sequence: list[Callable] | tuple[Callable, ...],
        expand_dims_axis: int | None,
    ) -> None:
        """Build the composed transform pipeline.

        *expand_dims_axis* is appended after the user-provided
        *transformations_sequence*, so that dimension expansion always
        runs last.
        """
        sequence = list(transformations_sequence)
        if expand_dims_axis is not None:
            sequence.append(
                partial(expand_data_dimensionality, expand_dims_axis=expand_dims_axis)
            )
        self._transform = compose(*sequence)

    def _get_sample_1(self) -> object:
        """Return transformed data (WITHOUT_LABELS mode)."""
        return self._transform(self._get_current_data())

    def _get_sample_2(self) -> tuple[object, object]:
        """Return (transformed_data, label) (WITH_LABELS mode)."""
        sample = self._transform(self._get_current_data())
        label = self._get_current_label()
        return (sample, label)

    def _get_sample_3(self) -> tuple[object, object]:
        """Return (transformed_input, transformed_target) (FORECASTING mode)."""
        sample = self._transform(self._get_current_data())
        label = self._transform(self._get_current_label())
        return (sample, label)

    def __getitem__(self, index: int) -> object:
        """Return the sample at the given index."""
        self._go_to_idx(index)
        return self._get_sample()

    def __len__(self) -> int:
        """Return the number of samples."""
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

    Raises:
        TypeError: If *data* is not a ``np.ndarray`` or ``pd.DataFrame``.
        ValueError: If *data* has fewer than 2 dimensions.
    """

    _data: np.ndarray | pd.DataFrame
    _labels: pd.Series | pd.DataFrame | None

    def __init__(
        self,
        data: np.ndarray | pd.DataFrame,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None = None,
    ) -> None:
        # T-02-02-01: Type-check data
        if not isinstance(data, (np.ndarray, pd.DataFrame)):
            msg = f'data must be np.ndarray or pd.DataFrame, got {type(data).__name__}'
            raise TypeError(
                msg
            )
        # T-02-02-02: Validate minimum dimensions for seq_len
        if isinstance(data, np.ndarray) and data.ndim < MIN_DIM_FOR_SEQ_LEN:
            msg = f'data must have at least 2 dimensions for seq_len, got {data.ndim}D'
            raise ValueError(
                msg
            )
        if isinstance(data, pd.DataFrame) and data.shape[1] < 1:
            msg = 'data DataFrame must have at least 1 column for seq_len'
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

    _data: pd.DataFrame

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
        return self._data.iloc[self._n].to_numpy()


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
