"""tscollection.datasets -- Time series datasets for PyTorch Lightning."""

from __future__ import annotations

__version__ = '0.1.0'

from tscollection.datasets.enums import (
    DatasetFamily,
    DistanceMetric,
    ForecastingMode,
    ScalingMethod,
    SplitMode,
    SplittingStrategy,
    TimeSeriesDatasetMode,
)

__all__ = [
    'DatasetFamily',
    'DistanceMetric',
    'ForecastingMode',
    'ScalingMethod',
    'SplitMode',
    'SplittingStrategy',
    'TimeSeriesDatasetMode',
    '__version__',
]
