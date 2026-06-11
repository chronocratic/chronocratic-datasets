"""ETT (Electricity Transformer Temperature) forecasting dataset.

Thin wrapper around FlexibleTimeSeriesDatasetSingleFile that injects
ForecastingStrategySingleFile and sets domain defaults for ETT-style
multivariate forecasting tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chronocratic.datasets.datatypes._base.flexible import FlexibleTimeSeriesDatasetSingleFile
from chronocratic.datasets.datatypes._base.strategies import ForecastingStrategySingleFile
from chronocratic.datasets.enums import TimeSeriesDatasetMode
from chronocratic.datasets.utils.transformations import convert_numpy_to_tensor

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

__all__ = ["ETTDataset"]


class ETTDataset(FlexibleTimeSeriesDatasetSingleFile):
    """PyTorch Dataset for ETT forecasting (ETTh1/ETTh2/ETTm1/ETTm2).

    Sliding-window dataset with forecast_horizon as the label target.
    Labels are derived from the data segment immediately following each
    input window (via ForecastingStrategySingleFile).

    Args:
        data: 2-D numpy array of shape (time, features).
        seq_len: Input window length.
        step: Step between consecutive windows.
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
        forecast_horizon: int,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.INPUT_OUTPUT,
        transformations_sequence: tuple[Callable, ...] = (convert_numpy_to_tensor,),
    ) -> None:
        # Validate forecast_horizon > 0
        if forecast_horizon <= 0:
            msg = f"forecast_horizon must be positive, got {forecast_horizon}"
            raise ValueError(msg)
        super().__init__(
            data=data,
            labels=None,
            seq_len=seq_len,
            step=step,
            mode=mode,
            sequence_handling_strategy=ForecastingStrategySingleFile(
                forecast_horizon=forecast_horizon
            ),
            expand_dims_axis=None,
            transformations_sequence=transformations_sequence,
        )
