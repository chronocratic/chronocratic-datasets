"""Tests for config base classes (CFG-01, CFG-02).

Verifies that DatasetConfig, ClassificationConfig, and ForecastingConfig
are correctly structured as frozen Pydantic models with proper validation.
"""

import pytest
from pydantic import ValidationError

from tscollection.datasets.config.base import (
    ArffFilePattern,
    ClassificationConfig,
    ClassificationFilePatterns,
    DatasetConfig,
    ForecastingConfig,
)
from tscollection.datasets.enums import DatasetFamily, SplitMode, SplittingStrategy


class TestArffFilePattern:
    """Tests for the ArffFilePattern nested model."""

    def test_create_frozen_instance(self) -> None:
        """CFG-01: ArffFilePattern creates a frozen instance."""
        pattern = ArffFilePattern(arff='{dataset_name}_train.arff')
        assert pattern.arff == '{dataset_name}_train.arff'

    def test_reassignment_raises(self) -> None:
        """CFG-01: ArffFilePattern reassignment raises ValueError."""
        pattern = ArffFilePattern(arff='{dataset_name}_train.arff')
        with pytest.raises(ValueError, match='frozen'):
            pattern.arff = 'other.arff'


class TestClassificationFilePatterns:
    """Tests for the ClassificationFilePatterns nested model."""

    def test_create_nested_instance(self) -> None:
        """CFG-01: ClassificationFilePatterns creates nested frozen instance."""
        patterns = ClassificationFilePatterns(
            train=ArffFilePattern(arff='{dataset_name}_train.arff'),
            test=ArffFilePattern(arff='{dataset_name}_test.arff'),
        )
        assert patterns.train.arff == '{dataset_name}_train.arff'
        assert patterns.test.arff == '{dataset_name}_test.arff'

    def test_deep_freeze(self) -> None:
        """CFG-01: Nested ArffFilePattern is also frozen."""
        patterns = ClassificationFilePatterns(
            train=ArffFilePattern(arff='{dataset_name}_train.arff'),
            test=ArffFilePattern(arff='{dataset_name}_test.arff'),
        )
        with pytest.raises(ValueError, match='frozen'):
            patterns.train.arff = 'other.arff'


class TestDatasetConfig:
    """Tests for the abstract DatasetConfig base class."""

    def test_cannot_instantiate_directly(self) -> None:
        """CFG-02: DatasetConfig is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DatasetConfig()  # type: ignore[misc,call-arg]

    def test_subclass_is_frozen(self) -> None:
        """CFG-01: Concrete subclass inherits frozen=True."""
        cfg = ClassificationConfig(
            name='TestDataset',
            family=DatasetFamily.UCR,
            url='https://example.com/test.zip',
            num_classes=2,
            data_form='regular',
            target_col_name='Class',
            file_patterns=ClassificationFilePatterns(
                train=ArffFilePattern(arff='{dataset_name}_train.arff'),
                test=ArffFilePattern(arff='{dataset_name}_test.arff'),
            ),
            tasks=('classification', 'representation'),
        )
        with pytest.raises(ValueError, match='frozen'):
            cfg.name = 'Other'

    def test_model_copy_returns_new_instance(self) -> None:
        """CFG-01: model_copy produces a new frozen instance with updated fields."""
        cfg = ClassificationConfig(
            name='TestDataset',
            family=DatasetFamily.UCR,
            url='https://example.com/test.zip',
            num_classes=2,
            data_form='regular',
            target_col_name='Class',
            file_patterns=ClassificationFilePatterns(
                train=ArffFilePattern(arff='{dataset_name}_train.arff'),
                test=ArffFilePattern(arff='{dataset_name}_test.arff'),
            ),
            tasks=('classification', 'representation'),
        )
        new_cfg = cfg.model_copy(update={'name': 'NewName'})
        assert new_cfg.name == 'NewName'
        assert cfg.name == 'TestDataset'  # Original unchanged

    def test_sha256_validator_accepts_valid(self) -> None:
        """CFG-01: sha256 field accepts None and valid 64-char hex strings."""
        valid_hash = 'a' * 64
        cfg = ForecastingConfig(
            name='Test',
            family=DatasetFamily.ETT,
            url='https://example.com/test.csv',
            sha256=valid_hash,
            split_mode=SplitMode.INDEXED,
            split_bounds=(8640, 11520, 14400),
            tasks=('forecasting',),
        )
        assert cfg.sha256 == valid_hash

    def test_sha256_validator_accepts_none(self) -> None:
        """CFG-01: sha256 field accepts None."""
        cfg = ForecastingConfig(
            name='Test',
            family=DatasetFamily.ETT,
            url='https://example.com/test.csv',
            sha256=None,
            split_mode=SplitMode.INDEXED,
            split_bounds=(8640, 11520, 14400),
            tasks=('forecasting',),
        )
        assert cfg.sha256 is None

    def test_sha256_validator_rejects_invalid(self) -> None:
        """CFG-01: sha256 field rejects non-64-char hex strings."""
        with pytest.raises(ValidationError, match='sha256'):
            ForecastingConfig(
                name='Test',
                family=DatasetFamily.ETT,
                url='https://example.com/test.csv',
                sha256='tooshort',
                split_mode=SplitMode.INDEXED,
                split_bounds=(8640, 11520, 14400),
                tasks=('forecasting',),
            )

    def test_httpurl_rejects_invalid(self) -> None:
        """CFG-01: HttpUrl rejects invalid URL strings."""
        with pytest.raises(ValidationError, match='url'):
            ForecastingConfig(
                name='Test',
                family=DatasetFamily.ETT,
                url='not-a-url',
                split_mode=SplitMode.INDEXED,
                split_bounds=(8640, 11520, 14400),
                tasks=('forecasting',),
            )

    def test_cache_key_property(self) -> None:
        """CFG-01: cache_key returns string derived from url and sha256."""
        cfg = ForecastingConfig(
            name='Test',
            family=DatasetFamily.ETT,
            url='https://example.com/test.csv',
            sha256=None,
            split_mode=SplitMode.INDEXED,
            split_bounds=(8640, 11520, 14400),
            tasks=('forecasting',),
        )
        assert cfg.cache_key == 'https://example.com/test.csv:unknown'

    def test_cache_key_with_sha256(self) -> None:
        """CFG-01: cache_key includes sha256 when present."""
        valid_hash = 'b' * 64
        cfg = ForecastingConfig(
            name='Test',
            family=DatasetFamily.ETT,
            url='https://example.com/test.csv',
            sha256=valid_hash,
            split_mode=SplitMode.INDEXED,
            split_bounds=(8640, 11520, 14400),
            tasks=('forecasting',),
        )
        assert cfg.cache_key == f'https://example.com/test.csv:{valid_hash}'


class TestClassificationConfig:
    """Tests for the ClassificationConfig intermediate class."""

    def test_requires_data_form(self) -> None:
        """CFG-02: ClassificationConfig raises error if data_form is None."""
        with pytest.raises(ValidationError, match='data_form'):
            ClassificationConfig(
                name='Test',
                family=DatasetFamily.UCR,
                url='https://example.com/test.zip',
                num_classes=2,
                data_form=None,
                target_col_name='Class',
                file_patterns=ClassificationFilePatterns(
                    train=ArffFilePattern(arff='{dataset_name}_train.arff'),
                    test=ArffFilePattern(arff='{dataset_name}_test.arff'),
                ),
                tasks=('classification',),
            )

    def test_default_split_strategy(self) -> None:
        """CFG-02: ClassificationConfig defaults split_strategy to AS_DEFINED."""
        cfg = ClassificationConfig(
            name='Test',
            family=DatasetFamily.UCR,
            url='https://example.com/test.zip',
            num_classes=2,
            data_form='regular',
            target_col_name='Class',
            file_patterns=ClassificationFilePatterns(
                train=ArffFilePattern(arff='{dataset_name}_train.arff'),
                test=ArffFilePattern(arff='{dataset_name}_test.arff'),
            ),
            tasks=('classification',),
        )
        assert cfg.split_strategy == SplittingStrategy.AS_DEFINED


class TestForecastingConfig:
    """Tests for the ForecastingConfig intermediate class."""

    def test_indexed_mode_valid(self) -> None:
        """CFG-02: ForecastingConfig accepts integer split_bounds with INDEXED mode."""
        cfg = ForecastingConfig(
            name='Test',
            family=DatasetFamily.ETT,
            url='https://example.com/test.csv',
            split_mode=SplitMode.INDEXED,
            split_bounds=(8640, 11520, 14400),
            tasks=('forecasting',),
        )
        assert cfg.split_mode == SplitMode.INDEXED
        assert cfg.split_bounds == (8640, 11520, 14400)

    def test_fractional_mode_valid(self) -> None:
        """CFG-02: ForecastingConfig accepts float split_bounds summing to 1.0."""
        cfg = ForecastingConfig(
            name='Test',
            family=DatasetFamily.ELECTRICITY,
            url='https://example.com/test.csv',
            split_mode=SplitMode.FRACTIONAL,
            split_bounds=(0.6, 0.2, 0.2),
            tasks=('forecasting',),
        )
        assert cfg.split_mode == SplitMode.FRACTIONAL
        assert cfg.split_bounds == (0.6, 0.2, 0.2)

    def test_fractional_bounds_must_sum_to_one(self) -> None:
        """CFG-02: Fractional split_bounds must sum to ~1.0."""
        with pytest.raises(ValidationError, match='sum'):
            ForecastingConfig(
                name='Test',
                family=DatasetFamily.ELECTRICITY,
                url='https://example.com/test.csv',
                split_mode=SplitMode.FRACTIONAL,
                split_bounds=(0.5, 0.3, 0.3),
                tasks=('forecasting',),
            )

    def test_bounds_length_must_be_three(self) -> None:
        """CFG-02: split_bounds must have exactly 3 elements."""
        with pytest.raises(ValidationError, match='3'):
            ForecastingConfig(
                name='Test',
                family=DatasetFamily.ETT,
                url='https://example.com/test.csv',
                split_mode=SplitMode.INDEXED,
                split_bounds=(8640, 11520),
                tasks=('forecasting',),
            )

    def test_default_seq_len_and_horizon(self) -> None:
        """CFG-02: default_seq_len=128 and default_horizon=96 are the defaults."""
        cfg = ForecastingConfig(
            name='Test',
            family=DatasetFamily.ETT,
            url='https://example.com/test.csv',
            split_mode=SplitMode.INDEXED,
            split_bounds=(8640, 11520, 14400),
            tasks=('forecasting',),
        )
        assert cfg.default_seq_len == 128
        assert cfg.default_horizon == 96
