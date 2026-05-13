"""LightningDataModule classes for time series datasets."""

from tscollection.datasets.modules.classes import (
    BaseClassificationTimeSeriesDataModule,
    BaseForecastingTimeSeriesDataModule,
    BaseTimeSeriesDataModule,
)
from tscollection.datasets.modules.electricity import ElectricityLoadModule
from tscollection.datasets.modules.ett import ETTDataModule
from tscollection.datasets.modules.ucr import UCRClassificationDataModule
from tscollection.datasets.modules.uea import UEAClassificationDataModule
from tscollection.datasets.modules.weather import WeatherModule

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
