__all__ = ['UEAClassificationMultivariateDataset']

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.autotsrc.datasets.classes.abstract import FixedTimeSeriesDatasetMultivariate
from src.autotsrc.enums import TimeSeriesDatasetMode
from src.autotsrc.utils.transformations import convert_numpy_to_tensor


class UEAClassificationMultivariateDataset(FixedTimeSeriesDatasetMultivariate):
    def __init__(
        self,
        data: np.ndarray | pd.DataFrame,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int | None = None,
        transformations_sequence: list[Callable] | tuple[Callable, ...] | None = (
            convert_numpy_to_tensor,
        ),
    ) -> None:
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence,
        )
