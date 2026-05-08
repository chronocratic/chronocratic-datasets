"""tsdatasets -- Zero-config time series datasets for PyTorch Lightning."""

from __future__ import annotations

__version__ = '0.1.0'

from tsdatasets.enums import (
    DistanceMetric,
    ForecastingMode,
    ScalingMethod,
    SplittingStrategy,
    TimeSeriesDatasetMode,
)

__all__ = [
    'DistanceMetric',
    'ForecastingMode',
    'ScalingMethod',
    'SplittingStrategy',
    'TimeSeriesDatasetMode',
    '__version__',
]
