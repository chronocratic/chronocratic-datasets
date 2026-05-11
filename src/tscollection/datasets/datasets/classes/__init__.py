"""Abstract base classes for time series datasets."""

from tscollection.datasets.datasets.classes.strategies import (
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
    'ForecastingStrategySingleFile',
    'SequenceHandlingStrategy',
    'SequenceHandlingStrategyMultipleFiles',
    'SequenceHandlingStrategySingleFile',
]
