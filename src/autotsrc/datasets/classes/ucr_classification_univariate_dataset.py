__all__ = ['UCRClassificationUnivariateDataset']

import pandas as pd

from src.autotsrc.datasets.classes.abstract import FixedTimeSeriesDatasetUnivariate
from src.autotsrc.enums import TimeSeriesDatasetMode
from src.autotsrc.utils.transformations import convert_data_to_np_array, convert_numpy_to_tensor


class UCRClassificationUnivariateDataset(FixedTimeSeriesDatasetUnivariate):
    def __init__(
        self,
        data: pd.DataFrame,
        labels: pd.Series | pd.DataFrame | None,
        mode: TimeSeriesDatasetMode,
        expand_dims_axis: int = 1,
        transformations_sequence: tuple = (convert_numpy_to_tensor, convert_data_to_np_array),
    ) -> None:
        super().__init__(
            data=data,
            labels=labels,
            mode=mode,
            expand_dims_axis=expand_dims_axis,
            transformations_sequence=transformations_sequence,
        )
