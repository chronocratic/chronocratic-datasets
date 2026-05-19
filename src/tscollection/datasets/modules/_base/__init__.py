"""Abstract base classes for time series data modules."""

from tscollection.datasets.modules._base.base import BaseTimeSeriesDataModule
from tscollection.datasets.modules._base.classification import (
    BaseClassificationTimeSeriesDataModule,
)
from tscollection.datasets.modules._base.forecasting import (
    BaseForecastingTimeSeriesDataModule,
)

__all__ = [
    'BaseClassificationTimeSeriesDataModule',
    'BaseForecastingTimeSeriesDataModule',
    'BaseTimeSeriesDataModule',
]
