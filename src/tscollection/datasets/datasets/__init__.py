"""Time series dataset classes (PyTorch Dataset)."""

from tscollection.datasets.datasets.classes import (
    ClassificationStrategyMultipleFiles,
    ClassificationStrategySingleFile,
    FixedTimeSeriesDataset,
    FixedTimeSeriesDatasetMultivariate,
    FixedTimeSeriesDatasetUnivariate,
    FlexibleTimeSeriesDataset,
    FlexibleTimeSeriesDatasetMultipleFiles,
    FlexibleTimeSeriesDatasetSingleFile,
    ForecastingStrategySingleFile,
    SequenceHandlingStrategy,
    SequenceHandlingStrategyMultipleFiles,
    SequenceHandlingStrategySingleFile,
    TimeSeriesDataset,
)
from tscollection.datasets.datasets.ett import ETTDataset
from tscollection.datasets.datasets.ucr import UCRClassificationUnivariateDataset
from tscollection.datasets.datasets.uea import UEAClassificationMultivariateDataset

__all__ = [
    'ClassificationStrategyMultipleFiles',
    'ClassificationStrategySingleFile',
    'ETTDataset',
    'FixedTimeSeriesDataset',
    'FixedTimeSeriesDatasetMultivariate',
    'FixedTimeSeriesDatasetUnivariate',
    'FlexibleTimeSeriesDataset',
    'FlexibleTimeSeriesDatasetMultipleFiles',
    'FlexibleTimeSeriesDatasetSingleFile',
    'ForecastingStrategySingleFile',
    'SequenceHandlingStrategy',
    'SequenceHandlingStrategyMultipleFiles',
    'SequenceHandlingStrategySingleFile',
    'TimeSeriesDataset',
    'UCRClassificationUnivariateDataset',
    'UEAClassificationMultivariateDataset',
]
