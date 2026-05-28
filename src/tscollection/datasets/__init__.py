"""tscollection.datasets -- Time series datasets for PyTorch Lightning."""

from __future__ import annotations

__version__ = '0.1.0'

from tscollection.datasets.enums import (
    ClassificationSplitMode,
    DataForm,
    ForecastingMode,
    ForecastingSplitMode,
    ScalingMethod,
    TimeSeriesDatasetMode,
)

__all__ = [
    'ClassificationSplitMode',
    'DataForm',
    'ForecastingMode',
    'ForecastingSplitMode',
    'ScalingMethod',
    'TimeSeriesDatasetMode',
    '__version__',
]
