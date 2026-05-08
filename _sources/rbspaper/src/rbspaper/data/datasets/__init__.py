"""PyTorch Dataset classes for time series data."""

from src.rbspaper.data.datasets.abstract import (
    FixedTimeSeriesDataset,
    FixedTimeSeriesDatasetMultivariate,
    FixedTimeSeriesDatasetUnivariate,
    FlexibleTimeSeriesDataset,
    FlexibleTimeSeriesDatasetMultipleFiles,
    FlexibleTimeSeriesDatasetSingleFile,
    TimeSeriesDataset,
)
from src.rbspaper.data.datasets.electricity_load_dataset import ElectricityLoadDataset
from src.rbspaper.data.datasets.ett_dataset import ETTDataset
from src.rbspaper.data.datasets.strategies import (
    ClassificationStrategyMultipleFiles,
    ClassificationStrategySingleFile,
    ForecastingStrategySingleFile,
    SequenceHandlingStrategy,
    SequenceHandlingStrategyMultipleFiles,
    SequenceHandlingStrategySingleFile,
)
from src.rbspaper.data.datasets.transformations import (
    convert_data_to_np_array,
    convert_numpy_to_tensor,
    expand_data_dimensionality,
)
from src.rbspaper.data.datasets.ucr_dataset import UCRClassificationUnivariateDataset
from src.rbspaper.data.datasets.uea_dataset import UEAClassificationMultivariateDataset
from src.rbspaper.data.datasets.weather_dataset import WeatherDataset

__all__ = [
    'ClassificationStrategyMultipleFiles',
    'ClassificationStrategySingleFile',
    'ETTDataset',
    'ElectricityLoadDataset',
    'FixedTimeSeriesDataset',
    'FixedTimeSeriesDatasetMultivariate',
    'FixedTimeSeriesDatasetUnivariate',
    'FlexibleTimeSeriesDataset',
    'FlexibleTimeSeriesDatasetMultipleFiles',
    'FlexibleTimeSeriesDatasetSingleFile',
    'ForecastingStrategySingleFile',
    # strategies
    'SequenceHandlingStrategy',
    'SequenceHandlingStrategyMultipleFiles',
    'SequenceHandlingStrategySingleFile',
    # abstract
    'TimeSeriesDataset',
    # concrete datasets
    'UCRClassificationUnivariateDataset',
    'UEAClassificationMultivariateDataset',
    'WeatherDataset',
    'convert_data_to_np_array',
    # transformations
    'convert_numpy_to_tensor',
    'expand_data_dimensionality',
]
