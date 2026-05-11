"""Tests for UCR univariate classification configuration.

Verifies UCRConfig class inheritance, frozen behavior, correct field
values for Coffee, ECG200, and FaceFour instances, and the data_form
computed property.
"""

import pytest

from tscollection.datasets.config.ucr import (
    UCRConfig,
    UCR_COFFEE,
    UCR_ECG200,
    UCR_FACE_FOUR,
)
from tscollection.datasets.enums import DatasetFamily, SplittingStrategy


# -- Class structure -------------------------------------------------------


def test_ucr_config_inherits_classification_config() -> None:
    """UCRConfig must be a subclass of ClassificationConfig."""
    from tscollection.datasets.config.base import ClassificationConfig

    assert issubclass(UCRConfig, ClassificationConfig)


def test_ucr_config_default_family() -> None:
    """UCRConfig must default to DatasetFamily.UCR."""
    from tscollection.datasets.config.base import ClassificationFilePatterns
    from tscollection.datasets.config.base import ArffFilePattern

    config = UCRConfig(
        name='TestDataset',
        url='https://example.com/test.zip',
        num_classes=2,
        target_col_name='Class',
        data_form='regular',
        file_patterns=ClassificationFilePatterns(
            train=ArffFilePattern(arff='{dataset_name}_train.arff'),
            test=ArffFilePattern(arff='{dataset_name}_test.arff'),
        ),
    )
    assert config.family == DatasetFamily.UCR


def test_ucr_config_data_form_property() -> None:
    """UCRConfig.data_form must return 'regular' as a computed property."""
    from tscollection.datasets.config.base import ClassificationFilePatterns
    from tscollection.datasets.config.base import ArffFilePattern

    config = UCRConfig(
        name='TestDataset',
        url='https://example.com/test.zip',
        num_classes=2,
        target_col_name='Class',
        file_patterns=ClassificationFilePatterns(
            train=ArffFilePattern(arff='{dataset_name}_train.arff'),
            test=ArffFilePattern(arff='{dataset_name}_test.arff'),
        ),
    )
    assert config.data_form == 'regular'


# -- Frozen behavior -------------------------------------------------------


def test_ucr_coffee_is_frozen() -> None:
    """UCR_COFFEE must be immutable; field reassignment raises ValueError."""
    with pytest.raises(ValueError, match='frozen'):
        UCR_COFFEE.name = 'Other'


def test_ucr_ecg200_is_frozen() -> None:
    """UCR_ECG200 must be immutable; field reassignment raises ValueError."""
    with pytest.raises(ValueError, match='frozen'):
        UCR_ECG200.name = 'Other'


def test_ucr_face_four_is_frozen() -> None:
    """UCR_FACE_FOUR must be immutable; field reassignment raises ValueError."""
    with pytest.raises(ValueError, match='frozen'):
        UCR_FACE_FOUR.name = 'Other'


# -- UCR_COFFEE field values -----------------------------------------------


def test_ucr_coffee_name() -> None:
    """UCR_COFFEE.name must be 'Coffee'."""
    assert UCR_COFFEE.name == 'Coffee'


def test_ucr_coffee_family() -> None:
    """UCR_COFFEE.family must be DatasetFamily.UCR."""
    assert UCR_COFFEE.family == DatasetFamily.UCR


def test_ucr_coffee_num_classes() -> None:
    """UCR_COFFEE.num_classes must be 3."""
    assert UCR_COFFEE.num_classes == 3


def test_ucr_coffee_target_col_name() -> None:
    """UCR_COFFEE.target_col_name must be 'Class'."""
    assert UCR_COFFEE.target_col_name == 'Class'


def test_ucr_coffee_data_form() -> None:
    """UCR_COFFEE.data_form must be 'regular'."""
    assert UCR_COFFEE.data_form == 'regular'


def test_ucr_coffee_tasks() -> None:
    """UCR_COFFEE.tasks must include classification and representation."""
    assert UCR_COFFEE.tasks == ('classification', 'representation')


def test_ucr_coffee_split_strategy() -> None:
    """UCR_COFFEE.split_strategy must default to AS_DEFINED."""
    assert UCR_COFFEE.split_strategy == SplittingStrategy.AS_DEFINED


def test_ucr_coffee_file_patterns_train() -> None:
    """UCR_COFFEE.file_patterns.train.arff must contain {dataset_name}."""
    assert '{dataset_name}' in UCR_COFFEE.file_patterns.train.arff


def test_ucr_coffee_file_patterns_test() -> None:
    """UCR_COFFEE.file_patterns.test.arff must contain {dataset_name}."""
    assert '{dataset_name}' in UCR_COFFEE.file_patterns.test.arff


def test_ucr_coffee_url_is_valid_https() -> None:
    """UCR_COFFEE.url must be a valid HTTPS URL."""
    assert str(UCR_COFFEE.url).startswith('https://')
    assert 'timeseriesclassification.com' in str(UCR_COFFEE.url)
    assert 'Coffee' in str(UCR_COFFEE.url)


# -- UCR_ECG200 field values -----------------------------------------------


def test_ucr_ecg200_name() -> None:
    """UCR_ECG200.name must be 'ECG200'."""
    assert UCR_ECG200.name == 'ECG200'


def test_ucr_ecg200_num_classes() -> None:
    """UCR_ECG200.num_classes must be 5."""
    assert UCR_ECG200.num_classes == 5


def test_ucr_ecg200_family() -> None:
    """UCR_ECG200.family must be DatasetFamily.UCR."""
    assert UCR_ECG200.family == DatasetFamily.UCR


def test_ucr_ecg200_data_form() -> None:
    """UCR_ECG200.data_form must be 'regular'."""
    assert UCR_ECG200.data_form == 'regular'


# -- UCR_FACE_FOUR field values --------------------------------------------


def test_ucr_face_four_name() -> None:
    """UCR_FACE_FOUR.name must be 'FaceFour'."""
    assert UCR_FACE_FOUR.name == 'FaceFour'


def test_ucr_face_four_num_classes() -> None:
    """UCR_FACE_FOUR.num_classes must be 4."""
    assert UCR_FACE_FOUR.num_classes == 4


def test_ucr_face_four_family() -> None:
    """UCR_FACE_FOUR.family must be DatasetFamily.UCR."""
    assert UCR_FACE_FOUR.family == DatasetFamily.UCR


def test_ucr_face_four_data_form() -> None:
    """UCR_FACE_FOUR.data_form must be 'regular'."""
    assert UCR_FACE_FOUR.data_form == 'regular'


# -- model_copy behavior ---------------------------------------------------


def test_ucr_coffee_model_copy() -> None:
    """UCR_COFFEE.model_copy must return a new instance with updated fields."""
    copy = UCR_COFFEE.model_copy(update={'name': 'Coffee2'})
    assert copy.name == 'Coffee2'
    assert copy.num_classes == 3  # Original field unchanged
    assert UCR_COFFEE.name == 'Coffee'  # Original instance unchanged


def test_file_patterns_are_nested_pydantic_models() -> None:
    """file_patterns must use nested Pydantic models, not dicts."""
    from tscollection.datasets.config.base import ClassificationFilePatterns
    from tscollection.datasets.config.base import ArffFilePattern

    assert isinstance(UCR_COFFEE.file_patterns, ClassificationFilePatterns)
    assert isinstance(UCR_COFFEE.file_patterns.train, ArffFilePattern)
    assert isinstance(UCR_COFFEE.file_patterns.test, ArffFilePattern)
