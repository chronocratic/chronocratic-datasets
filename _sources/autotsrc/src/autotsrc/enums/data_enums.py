__all__ = [
    'ForecastingTimeSeriesDatasetMode',
    'TimeSeriesClassificationDatasetSplittingStrategy',
    'TimeSeriesDatasetMode',
    'TimeSeriesDistanceMetric',
]

from enum import Enum


class TimeSeriesDatasetMode(Enum):
    WITH_LABELS = 'with_labels'
    WITHOUT_LABELS = 'without_labels'
    FORECASTING = 'forecasting'


class TimeSeriesDistanceMetric(Enum):
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
    SOFT_DTW = 'soft_dtw'
    COSINE = 'cosine'


class TimeSeriesClassificationDatasetSplittingStrategy(Enum):
    AS_DEFINED = 'as_defined'
    MANUAL = 'manual'


class ForecastingTimeSeriesDatasetMode(Enum):
    UNIVARIATE = 'univariate'
    MULTIVARIATE = 'multivariate'
