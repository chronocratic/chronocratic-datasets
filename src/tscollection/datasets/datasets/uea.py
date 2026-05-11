"""UEA multivariate classification dataset.

Thin wrapper around FixedTimeSeriesDatasetMultivariate that sets domain
defaults for UEA-style multivariate time series classification tasks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tscollection.datasets.datasets.classes.fixed import FixedTimeSeriesDatasetMultivariate
from tscollection.datasets.datasets.transformations import convert_numpy_to_tensor
from tscollection.datasets.enums import TimeSeriesDatasetMode

__all__ = ['UEAClassificationMultivariateDataset']


class UEAClassificationMultivariateDataset(FixedTimeSeriesDatasetMultivariate):
    """PyTorch Dataset for UEA multivariate classification.

    Each sample is a 3-D array of shape (timesteps, features).
    No dimension expansion by default (expand_dims_axis=None) since
    the multivariate shape is already fully specified.

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
