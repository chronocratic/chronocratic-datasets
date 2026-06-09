from enum import StrEnum


class TimeSeriesDatasetMode(StrEnum):
    """Mode for how the dataset yields samples."""

    SAMPLE_ONLY = 'sample_only'
    SAMPLE_LABEL = 'sample_label'
    INPUT_OUTPUT = 'input_output'


class ClassificationLoaderMode(StrEnum):
    """Loader-level mode for classification datasets."""

    SAMPLE_ONLY = 'sample_only'
    SAMPLE_LABEL = 'sample_label'


class ForecastingLoaderMode(StrEnum):
    """Loader-level mode for forecasting datasets."""

    RAW_SERIES = 'raw_series'
    INPUT_TARGET = 'input_target'
    INPUT_ONLY = 'input_only'


class ClassificationSplitMode(StrEnum):
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
