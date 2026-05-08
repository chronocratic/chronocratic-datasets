"""
Enumerations for the robust time series representations package.

This module provides enum classes used throughout the robust_time_series_representations
package for type-safe configuration and parameterization of time series analysis and
representation learning components.
"""

from .data_enums import (
    ForecastingTimeSeriesDatasetMode,
    TimeSeriesClassificationDatasetSplittingStrategy,
    TimeSeriesDatasetMode,
    TimeSeriesDistanceMetric,
)
from .general import TimeSeriesDownstreamTask

__all__ = [
    'ForecastingTimeSeriesDatasetMode',
    'TimeSeriesClassificationDatasetSplittingStrategy',
    'TimeSeriesDatasetMode',
    'TimeSeriesDistanceMetric',
    'TimeSeriesDownstreamTask',
]
