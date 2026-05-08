__all__ = ['ETTDataset']

from collections.abc import Callable

import numpy as np

from src.autotsrc.datasets.classes.abstract import FlexibleTimeSeriesDatasetSingleFile
from src.autotsrc.datasets.classes.abstract.strategies import ForecastingStrategySingleFile
from src.autotsrc.enums import TimeSeriesDatasetMode
from src.autotsrc.utils.transformations import convert_data_to_np_array, convert_numpy_to_tensor


class ETTDataset(FlexibleTimeSeriesDatasetSingleFile):
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        step: int,
        forecast_horizon: int,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None = (
            convert_numpy_to_tensor,
            convert_data_to_np_array,
        ),
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
