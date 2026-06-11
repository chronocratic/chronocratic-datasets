"""UCR univariate classification dataset.

Thin wrapper around FixedTimeSeriesDatasetUnivariate that sets domain
defaults for UCR-style time series classification tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chronocratic.datasets.datatypes._base.fixed import FixedTimeSeriesDatasetUnivariate
from chronocratic.datasets.utils.transformations import convert_numpy_to_tensor

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from chronocratic.datasets.enums import TimeSeriesDatasetMode

__all__ = ["UCRClassificationUnivariateDataset"]


class UCRClassificationUnivariateDataset(FixedTimeSeriesDatasetUnivariate):
    """PyTorch Dataset for UCR univariate classification.

    Each row of the input DataFrame represents one time series sample.
    Defaults to expanding dimensions along axis=1 (producing shape
    ``(1, timesteps)``) and converting numpy arrays to tensors.

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
        transformations_sequence: tuple[Callable, ...] = (convert_numpy_to_tensor,),
    ) -> None:
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence,
        )
