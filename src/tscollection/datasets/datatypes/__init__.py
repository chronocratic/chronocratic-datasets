"""tscollection.datasets.datatypes — Dataset type classes."""

from __future__ import annotations

from tscollection.datasets.datatypes._base import (
    FixedTimeSeriesDatasetMultivariate,
    FixedTimeSeriesDatasetUnivariate,
    FlexibleTimeSeriesDatasetSingleFile,
    FlexibleTimeSeriesDatasetSingleFileMultipleSeries,
    TimeSeriesDataset,
)
from tscollection.datasets.datatypes.electricity import ElectricityDataset
from tscollection.datasets.datatypes.ett import ETTDataset
from tscollection.datasets.datatypes.ucr import UCRClassificationUnivariateDataset
from tscollection.datasets.datatypes.uea import UEAClassificationMultivariateDataset
from tscollection.datasets.datatypes.weather import WeatherDataset

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
