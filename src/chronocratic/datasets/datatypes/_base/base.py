from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from chronocratic.datasets.enums import TimeSeriesDatasetMode
from chronocratic.datasets.utils import compose
from chronocratic.datasets.utils.transformations import expand_data_dimensionality


class TimeSeriesDataset(Dataset[Any], ABC):
    """Abstract base for all time series datasets.

    Supports three modes via mode-specific sample getters:

    - ``SAMPLE_ONLY`` (training)
    - ``SAMPLE_LABEL`` (evaluation)
    - ``INPUT_OUTPUT`` (input/target pairs)

    Args:
        data: Raw time series data.
        labels: Optional label array or Series.
        mode: Determines the sample signature.
        expand_dims_axis: Axis along which to expand data dimensions.
        transformations_sequence: Post-processing callables.
    """

    _get_sample_fun_map: ClassVar[dict[TimeSeriesDatasetMode, str]] = {
        TimeSeriesDatasetMode.SAMPLE_ONLY: "_get_sample_1",
        TimeSeriesDatasetMode.SAMPLE_LABEL: "_get_sample_2",
        TimeSeriesDatasetMode.INPUT_OUTPUT: "_get_sample_3",
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
            sequence.append(partial(expand_data_dimensionality, expand_dims_axis=expand_dims_axis))
        self._transform = compose(*sequence)

    def _get_sample_1(self) -> object:
        """Return transformed data (SAMPLE_ONLY mode)."""
        return self._transform(self._get_current_data())

    def _get_sample_2(self) -> tuple[object, object]:
        """Return (transformed_data, label) (SAMPLE_LABEL mode)."""
        sample = self._transform(self._get_current_data())
        label = self._get_current_label()
        return (sample, label)

    def _get_sample_3(self) -> tuple[object, object]:
        """Return (transformed_input, transformed_target) (INPUT_OUTPUT mode)."""
        sample = self._transform(self._get_current_data())
        label = self._get_current_label()
        if label is None:
            msg = "INPUT_OUTPUT mode requires labels; _get_current_label returned None"
            raise RuntimeError(msg)
        return (sample, self._transform(label))

    def __getitem__(self, index: int) -> object:
        """Return the sample at the given index."""
        self._go_to_idx(index)
        return self._get_sample()

    def __len__(self) -> int:
        """Return the number of samples."""
        raise NotImplementedError
