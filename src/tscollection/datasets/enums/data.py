from enum import StrEnum


class TimeSeriesDatasetMode(StrEnum):
    """Mode for how the dataset yields samples."""

    WITH_LABELS = 'with_labels'
    WITHOUT_LABELS = 'without_labels'
    FORECASTING = 'forecasting'


class SplittingStrategy(StrEnum):
    """Strategy for train/test data splitting."""

    AS_DEFINED = 'as_defined'
    MANUAL = 'manual'


class ScalingMethod(StrEnum):
    """Method for data scaling."""

    NONE = 'none'
    MINMAX = 'minmax'
    STANDARD = 'standard'


class ForecastingMode(StrEnum):
    """Whether forecasting is univariate or multivariate."""

    UNIVARIATE = 'univariate'
    MULTIVARIATE = 'multivariate'


class DistanceMetric(StrEnum):
    """Distance metric for time series comparison."""

    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
    SOFT_DTW = 'soft_dtw'
    COSINE = 'cosine'
