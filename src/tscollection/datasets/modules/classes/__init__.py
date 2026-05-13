"""Abstract base classes for time series data modules."""

from tscollection.datasets.modules.classes.base import BaseTimeSeriesDataModule
from tscollection.datasets.modules.classes.classification import (
    BaseClassificationTimeSeriesDataModule,
)
from tscollection.datasets.modules.classes.forecasting import (
    BaseForecastingTimeSeriesDataModule,
)

__all__ = [
    'BaseClassificationTimeSeriesDataModule',
    'BaseForecastingTimeSeriesDataModule',
    'BaseTimeSeriesDataModule',
]
