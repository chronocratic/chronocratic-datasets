"""UCR univariate classification dataset."""

from __future__ import annotations

import pandas as pd

from src.rbspaper.data.datasets.abstract import FixedTimeSeriesDatasetUnivariate
from src.rbspaper.data.datasets.transformations import convert_numpy_to_tensor
from src.rbspaper.enums.data_enums import TimeSeriesDatasetMode

__all__ = ['UCRClassificationUnivariateDataset']


class UCRClassificationUnivariateDataset(FixedTimeSeriesDatasetUnivariate):
    """PyTorch Dataset for UCR univariate classification.

    Each row of the input DataFrame represents one time series sample.

    Args:
        data: DataFrame of shape (samples, timesteps).
        labels: Optional label Series.
        mode: Dataset mode (with/without labels).
        expand_dims_axis: Axis to expand dimensions on.
        transformations_sequence: Post-processing callables.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int = 1,
        transformations_sequence: tuple = (convert_numpy_to_tensor,),
    ) -> None:
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence,
        )
