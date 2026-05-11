"""Tests for UEA multivariate classification configuration.

Verifies UEAConfig class inheritance, frozen behavior, correct field
values for BasicMotions and AtrialFibrillation instances, and the
data_form='nested' default.
"""

import pytest

from tscollection.datasets.config.uea import (
    UEAConfig,
    UEA_ATRIAL_FIBRILLATION,
    UEA_BASIC_MOTIONS,
)
from tscollection.datasets.enums import DatasetFamily, SplittingStrategy


# -- Class structure -------------------------------------------------------


def test_uea_config_inherits_classification_config() -> None:
    """UEAConfig must be a subclass of ClassificationConfig."""
    from tscollection.datasets.config.base import ClassificationConfig

    assert issubclass(UEAConfig, ClassificationConfig)


def test_uea_config_default_family() -> None:
    """UEAConfig must default to DatasetFamily.UEA."""
    from tscollection.datasets.config.base import (
        ArffFilePattern,
        ClassificationFilePatterns,
    )

    config = UEAConfig(
        name='TestDataset',
        url='https://example.com/test.zip',
        num_classes=2,
        target_col_name='Class',
        file_patterns=ClassificationFilePatterns(
            train=ArffFilePattern(arff='{dataset_name}_train.arff'),
            test=ArffFilePattern(arff='{dataset_name}_test.arff'),
        ),
    )
    assert config.family == DatasetFamily.UEA


def test_uea_config_data_form_default() -> None:
    """UEAConfig.data_form must default to 'nested' (computed value)."""
    from tscollection.datasets.config.base import (
        ArffFilePattern,
        ClassificationFilePatterns,
    )

    config = UEAConfig(
        name='TestDataset',
        url='https://example.com/test.zip',
        num_classes=2,
        target_col_name='Class',
        file_patterns=ClassificationFilePatterns(
            train=ArffFilePattern(arff='{dataset_name}_train.arff'),
            test=ArffFilePattern(arff='{dataset_name}_test.arff'),
        ),
    )
    assert config.data_form == 'nested'


# -- Frozen behavior -------------------------------------------------------


def test_uea_basic_motions_is_frozen() -> None:
    """UEA_BASIC_MOTIONS must be immutable."""
    with pytest.raises(ValueError, match='frozen'):
        UEA_BASIC_MOTIONS.name = 'Other'


def test_uea_atrial_fibrillation_is_frozen() -> None:
    """UEA_ATRIAL_FIBRILLATION must be immutable."""
    with pytest.raises(ValueError, match='frozen'):
        UEA_ATRIAL_FIBRILLATION.name = 'Other'


# -- UEA_BASIC_MOTIONS field values ----------------------------------------


def test_uea_basic_motions_name() -> None:
    """UEA_BASIC_MOTIONS.name must be 'BasicMotions'."""
    assert UEA_BASIC_MOTIONS.name == 'BasicMotions'


def test_uea_basic_motions_family() -> None:
    """UEA_BASIC_MOTIONS.family must be DatasetFamily.UEA."""
    assert UEA_BASIC_MOTIONS.family == DatasetFamily.UEA


def test_uea_basic_motions_num_classes() -> None:
    """UEA_BASIC_MOTIONS.num_classes must be 4."""
    assert UEA_BASIC_MOTIONS.num_classes == 4


def test_uea_basic_motions_target_col_name() -> None:
    """UEA_BASIC_MOTIONS.target_col_name must be 'Class'."""
    assert UEA_BASIC_MOTIONS.target_col_name == 'Class'


def test_uea_basic_motions_data_form() -> None:
    """UEA_BASIC_MOTIONS.data_form must be 'nested'."""
    assert UEA_BASIC_MOTIONS.data_form == 'nested'


def test_uea_basic_motions_tasks() -> None:
    """UEA_BASIC_MOTIONS.tasks must include classification and representation."""
    assert UEA_BASIC_MOTIONS.tasks == ('classification', 'representation')


def test_uea_basic_motions_split_strategy() -> None:
    """UEA_BASIC_MOTIONS.split_strategy must default to AS_DEFINED."""
    assert UEA_BASIC_MOTIONS.split_strategy == SplittingStrategy.AS_DEFINED


def test_uea_basic_motions_file_patterns() -> None:
    """UEA_BASIC_MOTIONS.file_patterns must use nested Pydantic models."""
    from tscollection.datasets.config.base import (
        ArffFilePattern,
        ClassificationFilePatterns,
    )

    assert isinstance(UEA_BASIC_MOTIONS.file_patterns, ClassificationFilePatterns)
    assert isinstance(UEA_BASIC_MOTIONS.file_patterns.train, ArffFilePattern)
    assert isinstance(UEA_BASIC_MOTIONS.file_patterns.test, ArffFilePattern)
    assert '{dataset_name}' in UEA_BASIC_MOTIONS.file_patterns.train.arff
    assert '{dataset_name}' in UEA_BASIC_MOTIONS.file_patterns.test.arff


def test_uea_basic_motions_url() -> None:
    """UEA_BASIC_MOTIONS.url must be a valid HTTPS URL."""
    assert str(UEA_BASIC_MOTIONS.url).startswith('https://')
    assert 'timeseriesclassification.com' in str(UEA_BASIC_MOTIONS.url)
    assert 'BasicMotions' in str(UEA_BASIC_MOTIONS.url)


# -- UEA_ATRIAL_FIBRILLATION field values ----------------------------------


def test_uea_atrial_fibrillation_name() -> None:
    """UEA_ATRIAL_FIBRILLATION.name must be 'AtrialFibrillation'."""
    assert UEA_ATRIAL_FIBRILLATION.name == 'AtrialFibrillation'


def test_uea_atrial_fibrillation_num_classes() -> None:
    """UEA_ATRIAL_FIBRILLATION.num_classes must be 2."""
    assert UEA_ATRIAL_FIBRILLATION.num_classes == 2


def test_uea_atrial_fibrillation_family() -> None:
    """UEA_ATRIAL_FIBRILLATION.family must be DatasetFamily.UEA."""
    assert UEA_ATRIAL_FIBRILLATION.family == DatasetFamily.UEA


def test_uea_atrial_fibrillation_data_form() -> None:
    """UEA_ATRIAL_FIBRILLATION.data_form must be 'nested'."""
    assert UEA_ATRIAL_FIBRILLATION.data_form == 'nested'


def test_uea_atrial_fibrillation_model_copy() -> None:
    """UEA_ATRIAL_FIBRILLATION.model_copy must return a new instance."""
    copy = UEA_ATRIAL_FIBRILLATION.model_copy(update={'name': 'AF2'})
    assert copy.name == 'AF2'
    assert copy.num_classes == 2
    assert UEA_ATRIAL_FIBRILLATION.name == 'AtrialFibrillation'
