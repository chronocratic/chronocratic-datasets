"""UCR univariate classification dataset configurations.

Defines UCRConfig, a family-specific subclass of ClassificationConfig
for UCR archive datasets. UCR datasets use regular (flat) ARFF format
with a single univariate time series per sample.

Frozen instances for Coffee, ECG200, and FaceFour encode all intrinsic
dataset facts needed by downstream phases (download, modules, factory).
"""

from pydantic import Field, HttpUrl

from tscollection.datasets.config.base import (
    ArffFilePattern,
    ClassificationConfig,
    ClassificationFilePatterns,
)
from tscollection.datasets.enums.data import DatasetFamily

__all__ = ['UCR_COFFEE', 'UCR_ECG200', 'UCR_FACE_FOUR', 'UCRConfig']


# -- File pattern templates ------------------------------------------------

_UCR_FILE_PATTERNS = ClassificationFilePatterns(
    train=ArffFilePattern(arff='{dataset_name}_train.arff'),
    test=ArffFilePattern(arff='{dataset_name}_test.arff'),
)


# -- Config class ----------------------------------------------------------


class UCRConfig(ClassificationConfig):
    """Configuration for UCR univariate classification datasets.

    UCR datasets are single-channel time series stored as ARFF files.
    Each sample has a uniform length, and the archive provides separate
    train/test splits.

    Args:
        name: Dataset display name (e.g., 'Coffee').
        family: Always DatasetFamily.UCR (class default).
        url: Download URL for the dataset archive (.zip).
        sha256: SHA256 checksum of the downloaded archive, or None.
        num_classes: Number of distinct classification labels.
        target_col_name: ARFF column name for the class label.
        data_form: Always 'regular' for UCR (class default).
        file_patterns: Train/test ARFF filename templates.
        split_strategy: How to construct the train/test split.
        tasks: Tuple of supported task types.
    """

    family: DatasetFamily = DatasetFamily.UCR
    num_classes: int = Field(ge=1)
    data_form: str = 'regular'
    tasks: tuple[str, ...] = ('classification', 'representation')


# -- Dataset instances -----------------------------------------------------

UCR_COFFEE = UCRConfig(
    name='Coffee',
    url=HttpUrl('https://timeseriesclassification.com/Downloads/Coffee.zip'),
    sha256=None,
    num_classes=3,
    target_col_name='Class',
    file_patterns=_UCR_FILE_PATTERNS,
)

UCR_ECG200 = UCRConfig(
    name='ECG200',
    url=HttpUrl('https://timeseriesclassification.com/Downloads/ECG200.zip'),
    sha256=None,
    num_classes=5,
    target_col_name='Class',
    file_patterns=_UCR_FILE_PATTERNS,
)

UCR_FACE_FOUR = UCRConfig(
    name='FaceFour',
    url=HttpUrl('https://timeseriesclassification.com/Downloads/FaceFour.zip'),
    sha256=None,
    num_classes=4,
    target_col_name='Class',
    file_patterns=_UCR_FILE_PATTERNS,
)
