"""Weather forecasting dataset.

Concrete dataset class for the Weather time series benchmark.
Data shape: (1, T, 22) post-transform, squeezed to (T, 22) for sliding.
T = 52696 hourly steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tscollection.datasets.datatypes._base.flexible import (
    FlexibleTimeSeriesDatasetSingleFile,
)
from tscollection.datasets.datatypes._base.strategies import (
    ClassificationStrategySingleFile,
    ForecastingStrategySingleFile,
)
from tscollection.datasets.enums import TimeSeriesDatasetMode
from tscollection.datasets.utils.transformations import convert_numpy_to_tensor

if TYPE_CHECKING:
    import numpy as np

__all__ = ['WeatherDataset']


class WeatherDataset(FlexibleTimeSeriesDatasetSingleFile):
    """PyTorch Dataset for Weather forecasting.

    Sliding-window dataset for the Weather benchmark (22 features,
    hourly granularity). Accepts pre-processed 2D data of shape
    (T, 22) where T = 52696 hourly steps.

    Args:
        data: 2-D numpy array of shape (T, 22).
        seq_len: Input window length.
        step: Step between consecutive windows.
        mode: Dataset mode string (e.g., 'sample_only', 'input_output').
        forecast_horizon: Number of future steps to predict.
        transformations_sequence: Post-processing callables.

    Raises:
        ValueError: If forecast_horizon is not positive.
    """

    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        step: int,
        mode: str,
        forecast_horizon: int,
        transformations_sequence: tuple = (convert_numpy_to_tensor,),
    ) -> None:
        if forecast_horizon <= 0:
            msg = f'forecast_horizon must be positive, got {forecast_horizon}'
            raise ValueError(msg)

        ts_mode = (
            TimeSeriesDatasetMode(mode)
            if isinstance(mode, str)
            else mode
        )

        strategy = (
            ForecastingStrategySingleFile(forecast_horizon=forecast_horizon)
            if ts_mode == TimeSeriesDatasetMode.INPUT_OUTPUT
            else ClassificationStrategySingleFile()
        )

        super().__init__(
            data=data,
            labels=None,
            seq_len=seq_len,
            step=step,
            mode=ts_mode,
            sequence_handling_strategy=strategy,
            expand_dims_axis=None,
            transformations_sequence=transformations_sequence,
        )
