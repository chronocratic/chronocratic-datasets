from enum import StrEnum


class TimeSeriesDatasetMode(StrEnum):
    """Mode for how the dataset yields samples."""

    WITH_LABELS = 'with_labels'
    WITHOUT_LABELS = 'without_labels'
    FORECASTING = 'forecasting'


class ClassificationSplittingStrategy(StrEnum):
    """Strategy for classification train/test data splitting."""

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


class DataForm(StrEnum):
    """Enum for the form (shape) of the data.

    Attributes:
        REGULAR: 2-D tabular data (samples x features).
        NESTED: 3-D array data (samples x timesteps x features).
        MULTI_FILES: List of 1-D arrays from multiple files.
    """

    REGULAR = 'regular'
    NESTED = 'nested'
    MULTI_FILES = 'multi_files'


class DistanceMetric(StrEnum):
    """Distance metric for time series comparison."""

    EUCLIDEAN = 'euclidean'
    MANHATTAN = 'manhattan'
    SOFT_DTW = 'soft_dtw'
    COSINE = 'cosine'


class DatasetFamily(StrEnum):
    """Dataset family identifier for registry classification."""

    UCR = 'ucr'
    UEA = 'uea'
    ETT = 'ett'
    ELECTRICITY = 'electricity'
    WEATHER = 'weather'
    EXCHANGE = 'exchange'
    TRAFFIC = 'traffic'
    ILLNESS = 'illness'


class SplitMode(StrEnum):
    """Mode for defining train/valid/test split boundaries."""

    INDEXED = 'indexed'
    FRACTIONAL = 'fractional'
