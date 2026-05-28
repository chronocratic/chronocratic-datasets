"""Abstract base classes for time series datasets."""

from tscollection.datasets._base.base import TimeSeriesDataset
from tscollection.datasets._base.fixed import (
    FixedTimeSeriesDataset,
    FixedTimeSeriesDatasetMultivariate,
    FixedTimeSeriesDatasetUnivariate,
)
from tscollection.datasets._base.flexible import (
    FlexibleTimeSeriesDataset,
    FlexibleTimeSeriesDatasetMultipleFiles,
    FlexibleTimeSeriesDatasetSingleFile,
)
from tscollection.datasets._base.strategies import (
    ClassificationStrategyMultipleFiles,
    ClassificationStrategySingleFile,
    ForecastingStrategySingleFile,
    SequenceHandlingStrategy,
    SequenceHandlingStrategyMultipleFiles,
    SequenceHandlingStrategySingleFile,
)

__all__ = [
    'ClassificationStrategyMultipleFiles',
    'ClassificationStrategySingleFile',
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
]
