"""Electricity forecasting dataset.

Concrete dataset class for the Electricity Load Diagrams benchmark.
Raw CSV shape: (27340, 371) — first column is MT_001 timestamp.
Post-transform: (370, 27340, 1) — 370 independent power clients.
Each client's consumption is treated as an independent time series.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tscollection.datasets.datatypes._base.flexible import (
    FlexibleTimeSeriesDatasetSingleFileMultipleSeries,
)
from tscollection.datasets.datatypes._base.strategies import (
    ClassificationStrategySingleFile,
    ForecastingStrategySingleFile,
)
from tscollection.datasets.enums import TimeSeriesDatasetMode
from tscollection.datasets.utils.transformations import convert_numpy_to_tensor

if TYPE_CHECKING:
    import numpy as np

__all__ = ['ElectricityDataset']


class ElectricityDataset(FlexibleTimeSeriesDatasetSingleFileMultipleSeries):
    """PyTorch Dataset for Electricity forecasting.

    Sliding-window dataset for the Electricity Load Diagrams benchmark.
    Handles 3D data of shape (num_clients, T, 1) where each client is
    an independent power consumption series. 370 clients total.

    Raw CSV: (27340, 371) with MT_001 timestamp column.
    Post-transform: (370, 27340, 1).

    Args:
        data: 3-D numpy array of shape (num_clients, T, 1).
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
        mode: TimeSeriesDatasetMode,
        forecast_horizon: int,
        transformations_sequence: tuple = (convert_numpy_to_tensor,),
    ) -> None:
        if forecast_horizon <= 0:
            msg = f'forecast_horizon must be positive, got {forecast_horizon}'
            raise ValueError(msg)

        strategy = (
            ForecastingStrategySingleFile(forecast_horizon=forecast_horizon)
            if mode == TimeSeriesDatasetMode.INPUT_OUTPUT
            else ClassificationStrategySingleFile()
        )

        super().__init__(
            data=data,
            labels=None,
            seq_len=seq_len,
            step=step,
            mode=mode,
            sequence_handling_strategy=strategy,
            expand_dims_axis=None,
            transformations_sequence=transformations_sequence,
        )
