from enum import StrEnum


class TimeSeriesDatasetMode(StrEnum):
    """Mode for how the dataset yields samples.

    Attributes:
        SAMPLE_ONLY: Returns only the input data without labels.
        SAMPLE_LABEL: Returns the input data and its corresponding label.
        INPUT_OUTPUT: Returns separate input and output tensors for supervised learning.
    """

    SAMPLE_ONLY = "sample_only"
    SAMPLE_LABEL = "sample_label"
    INPUT_OUTPUT = "input_output"


class ClassificationLoaderMode(StrEnum):
    """Loader-level mode for classification datasets.

    Attributes:
        SAMPLE_ONLY: Returns only the input data without labels.
        SAMPLE_LABEL: Returns the input data and its corresponding label.
    """

    SAMPLE_ONLY = "sample_only"
    SAMPLE_LABEL = "sample_label"


class ForecastingLoaderMode(StrEnum):
    """Loader-level mode for forecasting datasets.

    Attributes:
        RAW_SERIES: Returns the full time series without splitting.
        INPUT_TARGET: Returns paired input and target tensors for supervised forecasting.
        INPUT_ONLY: Returns only the input portion of the series.
    """

    RAW_SERIES = "raw_series"
    INPUT_TARGET = "input_target"
    INPUT_ONLY = "input_only"


class ClassificationSplitMode(StrEnum):
    """Strategy for classification train/test data splitting.

    Attributes:
        AS_DEFINED: Uses the train/test split as defined in the original dataset.
        MANUAL: Allows manual specification of train/test split ratios.
    """

    AS_DEFINED = "as_defined"
    MANUAL = "manual"


class ScalingMethod(StrEnum):
    """Method for data scaling.

    Attributes:
        NONE: No scaling applied.
        MINMAX: Scales data to a specified range (default 0-1).
        STANDARD: Standardizes data to zero mean and unit variance.
    """

    NONE = "none"
    MINMAX = "minmax"
    STANDARD = "standard"


class ForecastingMode(StrEnum):
    """Whether forecasting is univariate or multivariate.

    Attributes:
        UNIVARIATE: Uses a single target variable per sample.
        MULTIVARIATE: Uses all available variables per sample.
    """

    UNIVARIATE = "univariate"
    MULTIVARIATE = "multivariate"


class DataForm(StrEnum):
    """Enum for the form (shape) of the data.

    Attributes:
        REGULAR: 2-D tabular data (samples x features).
        NESTED: 3-D array data (samples x timesteps x features).
        MULTI_FILES: List of 1-D arrays from multiple files.
    """

    REGULAR = "regular"
    NESTED = "nested"
    MULTI_FILES = "multi_files"


class DataPartition(StrEnum):
    """Data partition for train/validation/test splits.

    Attributes:
        TRAIN: Training data partition.
        VAL: Validation data partition.
        TEST: Test data partition.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
