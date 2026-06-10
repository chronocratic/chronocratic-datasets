"""Abstract base classes for time series data modules."""

from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule
from chronocratic.datasets.modules._base.classification import (
    BaseClassificationTimeSeriesDataModule,
)
from chronocratic.datasets.modules._base.forecasting import BaseForecastingTimeSeriesDataModule

__all__ = [
    'BaseClassificationTimeSeriesDataModule',
    'BaseForecastingTimeSeriesDataModule',
    'BaseTimeSeriesDataModule',
]
