"""LightningDataModule classes for time series data."""

from src.rbspaper.data.modules.abstract import (
    BaseClassificationTimeSeriesDataModule,
    BaseForecastingTimeSeriesDataModule,
    BaseTimeSeriesDataModule,
)
from src.rbspaper.data.modules.electricity_load_datamodule import ElectricityLoadDataModule
from src.rbspaper.data.modules.ett_datamodule import ETTDataModule
from src.rbspaper.data.modules.ucr_datamodule import UCRTimeSeriesClassificationUnivariateDataModule
from src.rbspaper.data.modules.uea_datamodule import (
    UEATimeSeriesClassificationMultivariateDataModule,
)
from src.rbspaper.data.modules.weather_datamodule import WeatherDataModule

__all__ = [
    'BaseClassificationTimeSeriesDataModule',
    'BaseForecastingTimeSeriesDataModule',
    # abstract
    'BaseTimeSeriesDataModule',
    'ETTDataModule',
    'ElectricityLoadDataModule',
    # concrete
    'UCRTimeSeriesClassificationUnivariateDataModule',
    'UEATimeSeriesClassificationMultivariateDataModule',
    'WeatherDataModule',
]
