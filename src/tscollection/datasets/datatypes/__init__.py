"""tscollection.datasets.datatypes — Dataset type classes."""

from __future__ import annotations

from tscollection.datasets.datatypes._base import (
    FixedTimeSeriesDatasetMultivariate,
    FixedTimeSeriesDatasetUnivariate,
    FlexibleTimeSeriesDatasetSingleFile,
    TimeSeriesDataset,
)
from tscollection.datasets.datatypes.ett import ETTDataset
from tscollection.datasets.datatypes.ucr import UCRClassificationUnivariateDataset
from tscollection.datasets.datatypes.uea import UEAClassificationMultivariateDataset

__all__ = [
    'ETTDataset',
    'FixedTimeSeriesDatasetMultivariate',
    'FixedTimeSeriesDatasetUnivariate',
    'FlexibleTimeSeriesDatasetSingleFile',
    'TimeSeriesDataset',
    'UCRClassificationUnivariateDataset',
    'UEAClassificationMultivariateDataset',
]
