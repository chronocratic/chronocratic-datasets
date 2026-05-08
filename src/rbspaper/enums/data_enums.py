__all__ = [
    'ForecastingTimeSeriesDatasetMode',
    'TimeSeriesClassificationDatasetSplittingStrategy',
    'TimeSeriesDatasetMode',
    'TimeSeriesDistanceMetric',
]

from enum import StrEnum


class TimeSeriesDatasetMode(StrEnum):
    WITH_LABELS = 'with_labels'
    WITHOUT_LABELS = 'without_labels'
    FORECASTING = 'forecasting'


class TimeSeriesClassificationDatasetSplittingStrategy(StrEnum):
    AS_DEFINED = 'as_defined'
    MANUAL = 'manual'


class TimeSeriesDistanceMetric(StrEnum):
    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
    SOFT_DTW = 'soft_dtw'
    COSINE = 'cosine'


class ForecastingTimeSeriesDatasetMode(StrEnum):
    UNIVARIATE = 'univariate'
    MULTIVARIATE = 'multivariate'
