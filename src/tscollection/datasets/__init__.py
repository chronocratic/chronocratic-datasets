"""tscollection.datasets -- Time series datasets for PyTorch Lightning."""

from __future__ import annotations

__version__ = '0.1.0'

from tscollection.datasets.enums import (
    ClassificationSplittingStrategy,
    DataForm,
    DatasetFamily,
    DistanceMetric,
    ForecastingMode,
    ScalingMethod,
    SplitMode,
    TimeSeriesDatasetMode,
)

__all__ = [
    'ClassificationSplittingStrategy',
    'DataForm',
    'DatasetFamily',
    'DistanceMetric',
    'ForecastingMode',
    'ScalingMethod',
    'SplitMode',
    'TimeSeriesDatasetMode',
    '__version__',
]
