"""chronocratic.datasets.datatypes — Dataset type classes."""

from __future__ import annotations

from chronocratic.datasets.datatypes._base import (
    FixedTimeSeriesDatasetMultivariate,
    FixedTimeSeriesDatasetUnivariate,
    FlexibleTimeSeriesDatasetSingleFile,
    FlexibleTimeSeriesDatasetSingleFileMultipleSeries,
    TimeSeriesDataset,
)
from chronocratic.datasets.datatypes.electricity import ElectricityDataset
from chronocratic.datasets.datatypes.ett import ETTDataset
from chronocratic.datasets.datatypes.ucr import UCRClassificationUnivariateDataset
from chronocratic.datasets.datatypes.uea import UEAClassificationMultivariateDataset
from chronocratic.datasets.datatypes.weather import WeatherDataset

__all__ = [
    'ETTDataset',
    'ElectricityDataset',
    'FixedTimeSeriesDatasetMultivariate',
    'FixedTimeSeriesDatasetUnivariate',
    'FlexibleTimeSeriesDatasetSingleFile',
    'FlexibleTimeSeriesDatasetSingleFileMultipleSeries',
    'TimeSeriesDataset',
    'UCRClassificationUnivariateDataset',
    'UEAClassificationMultivariateDataset',
    'WeatherDataset',
]
