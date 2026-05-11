"""UEA multivariate classification dataset configurations.

Defines UEAConfig, a family-specific subclass of ClassificationConfig
for UEA archive datasets. UEA datasets use nested ARFF format with
multiple channels (multivariate time series) per sample.

Frozen instances for BasicMotions and AtrialFibrillation encode all
intrinsic dataset facts needed by downstream phases (download, modules,
factory).
"""

from pydantic import Field, HttpUrl

from tscollection.datasets.config.base import (
    ArffFilePattern,
    ClassificationConfig,
    ClassificationFilePatterns,
)
from tscollection.datasets.enums.data import DatasetFamily

__all__ = ['UEA_ATRIAL_FIBRILLATION', 'UEA_BASIC_MOTIONS', 'UEAConfig']


# -- File pattern templates ------------------------------------------------

_UEA_FILE_PATTERNS = ClassificationFilePatterns(
    train=ArffFilePattern(arff='{dataset_name}_train.arff'),
    test=ArffFilePattern(arff='{dataset_name}_test.arff'),
)


# -- Config class ----------------------------------------------------------


class UEAConfig(ClassificationConfig):
    """Configuration for UEA multivariate classification datasets.

    UEA datasets are multi-channel time series stored as nested ARFF
    files. Each sample has a uniform length across channels, and the
    archive provides separate train/test splits.

    Args:
        name: Dataset display name (e.g., 'BasicMotions').
        family: Always DatasetFamily.UEA (class default).
        url: Download URL for the dataset archive (.zip).
        sha256: SHA256 checksum of the downloaded archive, or None.
        num_classes: Number of distinct classification labels.
        target_col_name: ARFF column name for the class label.
        data_form: Always 'nested' for UEA (class default).
        file_patterns: Train/test ARFF filename templates.
        split_strategy: How to construct the train/test split.
        tasks: Tuple of supported task types.
    """

    family: DatasetFamily = DatasetFamily.UEA
    num_classes: int = Field(ge=1)
    data_form: str = 'nested'
    tasks: tuple[str, ...] = ('classification', 'representation')


# -- Dataset instances -----------------------------------------------------

UEA_BASIC_MOTIONS = UEAConfig(
    name='BasicMotions',
    url=HttpUrl('https://timeseriesclassification.com/Downloads/BasicMotions.zip'),
    sha256=None,
    num_classes=4,
    target_col_name='Class',
    file_patterns=_UEA_FILE_PATTERNS,
)

UEA_ATRIAL_FIBRILLATION = UEAConfig(
    name='AtrialFibrillation',
    url=HttpUrl('https://timeseriesclassification.com/Downloads/AtrialFibrillation.zip'),
    sha256=None,
    num_classes=2,
    target_col_name='Class',
    file_patterns=_UEA_FILE_PATTERNS,
)
