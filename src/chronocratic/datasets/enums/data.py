from enum import StrEnum


class TimeSeriesDatasetMode(StrEnum):
    """Mode for how the dataset yields samples.

    Attributes:
        SAMPLE_ONLY: Returns only the input data without labels.
        SAMPLE_LABEL: Returns the input data and its corresponding label.
        INPUT_OUTPUT: Returns separate input and output tensors for supervised learning.
    """

    SAMPLE_ONLY = 'sample_only'
    SAMPLE_LABEL = 'sample_label'
    INPUT_OUTPUT = 'input_output'


class ClassificationLoaderMode(StrEnum):
    """Loader-level mode for classification datasets.

    Attributes:
        SAMPLE_ONLY: Returns only the input data without labels.
        SAMPLE_LABEL: Returns the input data and its corresponding label.
    """

    SAMPLE_ONLY = 'sample_only'
    SAMPLE_LABEL = 'sample_label'


class ForecastingLoaderMode(StrEnum):
    """Loader-level mode for forecasting datasets.

    Attributes:
        RAW_SERIES: Returns the full time series without splitting.
        INPUT_TARGET: Returns paired input and target tensors for supervised forecasting.
        INPUT_ONLY: Returns only the input portion of the series.
    """

    RAW_SERIES = 'raw_series'
    INPUT_TARGET = 'input_target'
    INPUT_ONLY = 'input_only'


class ClassificationSplitMode(StrEnum):
    """Strategy for classification train/test data splitting.

    Attributes:
        AS_DEFINED: Uses the train/test split as defined in the original dataset.
        MANUAL: Allows manual specification of train/test split ratios.
    """

    AS_DEFINED = 'as_defined'
    MANUAL = 'manual'


class ScalingMethod(StrEnum):
    """Method for data scaling.

    Attributes:
        NONE: No scaling applied.
        MINMAX: Scales data to a specified range (default 0-1).
        STANDARD: Standardizes data to zero mean and unit variance.
    """

    NONE = 'none'
    MINMAX = 'minmax'
    STANDARD = 'standard'


class ForecastingMode(StrEnum):
    """Whether forecasting is univariate or multivariate.

    Attributes:
        UNIVARIATE: Uses a single target variable per sample.
        MULTIVARIATE: Uses all available variables per sample.
    """

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


class DataPartition(StrEnum):
    """Data partition for train/validation/test splits.

    Attributes:
        TRAIN: Training data partition.
        VAL: Validation data partition.
        TEST: Test data partition.
    """

    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


def _parse_attributes(docstring):
    """Parse Google-style Attributes section into {name: description} dict."""
    attrs = {}
    if not docstring or 'Attributes:' not in docstring:
        return attrs
    in_attrs = False
    current_name = None
    current_desc = []
    for line in docstring.splitlines():
        if line.strip() == 'Attributes:':
            in_attrs = True
            continue
        if not in_attrs:
            continue
        if line and not line[0].isspace():
            in_attrs = False
            continue
        stripped = line.strip()
        if stripped and ':' in stripped and current_name is None:
            name, desc = stripped.split(':', 1)
            current_name = name.strip()
            desc = desc.strip()
            if desc:
                current_desc.append(desc)
        elif stripped:
            current_desc.append(stripped)
        if current_name and (not stripped or (stripped and ':' in stripped and stripped[0].isupper())):
            if current_name and current_desc:
                attrs[current_name] = ' '.join(current_desc)
                current_desc = []
            if stripped and ':' in stripped and stripped[0].isupper():
                continue
    if current_name and current_desc:
        attrs[current_name] = ' '.join(current_desc)
    return attrs


def _init_enum_docs():
    """Attach _attributes descriptions to enum classes."""
    for cls in [
        TimeSeriesDatasetMode,
        ClassificationLoaderMode,
        ForecastingLoaderMode,
        ClassificationSplitMode,
        ScalingMethod,
        ForecastingMode,
        DataForm,
        DataPartition,
    ]:
        cls._attributes = _parse_attributes(cls.__doc__)


_init_enum_docs()
