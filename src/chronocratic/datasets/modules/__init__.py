"""LightningDataModule classes for time series datasets."""

from chronocratic.datasets.modules._base import (
    BaseClassificationTimeSeriesDataModule,
    BaseForecastingTimeSeriesDataModule,
    BaseTimeSeriesDataModule,
)
from chronocratic.datasets.modules.electricity import ElectricityLoadModule
from chronocratic.datasets.modules.ett import ETTDataModule
from chronocratic.datasets.modules.ucr import UCRClassificationDataModule
from chronocratic.datasets.modules.uea import UEAClassificationDataModule
from chronocratic.datasets.modules.weather import WeatherDataModule

__all__ = [
    'BaseClassificationTimeSeriesDataModule',
    'BaseForecastingTimeSeriesDataModule',
    'BaseTimeSeriesDataModule',
    'ETTDataModule',
    'ElectricityLoadModule',
    'UCRClassificationDataModule',
    'UEAClassificationDataModule',
    'WeatherModule',
]
