"""Electricity load forecasting dataset."""

from __future__ import annotations

import numpy as np

from src.rbspaper.data.datasets.abstract import FlexibleTimeSeriesDatasetSingleFile
from src.rbspaper.data.datasets.strategies import ForecastingStrategySingleFile
from src.rbspaper.data.datasets.transformations import (
    convert_data_to_np_array,
    convert_numpy_to_tensor,
)
from src.rbspaper.enums.data_enums import TimeSeriesDatasetMode

__all__ = ['ElectricityLoadDataset']


class ElectricityLoadDataset(FlexibleTimeSeriesDatasetSingleFile):
    """PyTorch Dataset for electricity load forecasting.

    Sliding-window dataset with forecast_horizon as the label target.

    Args:
        data: 2-D numpy array of shape (time, features).
        seq_len: Input window length.
        step: Step between consecutive windows.
        forecast_horizon: Number of future steps to predict.
        transformations_sequence: Post-processing callables.
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        step: int,
        forecast_horizon: int,
        transformations_sequence: tuple = (convert_numpy_to_tensor, convert_data_to_np_array),
    ) -> None:
        super().__init__(
            data=data,
            labels=None,
            seq_len=seq_len,
            step=step,
            mode=TimeSeriesDatasetMode.FORECASTING,
            sequence_handling_strategy=ForecastingStrategySingleFile(
                forecast_horizon=forecast_horizon
            ),
            expand_dims_axis=None,
            transformations_sequence=transformations_sequence,
        )
