"""UEA multivariate classification dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rbspaper.data.datasets.abstract import FixedTimeSeriesDatasetMultivariate
from src.rbspaper.data.datasets.transformations import convert_numpy_to_tensor
from src.rbspaper.enums.data_enums import TimeSeriesDatasetMode

__all__ = ['UEAClassificationMultivariateDataset']


class UEAClassificationMultivariateDataset(FixedTimeSeriesDatasetMultivariate):
    """PyTorch Dataset for UEA multivariate classification.

    Each sample is a 3-D array of shape (timesteps, features).

    Args:
        data: 3-D numpy array of shape (samples, timesteps, features).
        labels: Optional label Series.
        mode: Dataset mode.
        expand_dims_axis: Axis to expand (None for multivariate).
        transformations_sequence: Post-processing callables.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None = None,
        transformations_sequence: tuple = (convert_numpy_to_tensor,),
    ) -> None:
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence,
        )
