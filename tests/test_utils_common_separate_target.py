"""Tests for ClassificationSplittingStrategy enum and separate_target_feature_from_df.

Verifies the enum rename (D-04) and the new utility function port
that are foundational for Phase 4 data modules.
"""

import numpy as np
import pandas as pd
import pytest


class TestClassificationSplittingStrategy:
    """Tests for the renamed ClassificationSplittingStrategy enum."""

    def test_as_defined_value(self) -> None:
        """ClassificationSplittingStrategy.AS_DEFINED equals 'as_defined'."""
        from tscollection.datasets.enums import ClassificationSplittingStrategy

        assert ClassificationSplittingStrategy.AS_DEFINED == 'as_defined'

    def test_manual_value(self) -> None:
        """ClassificationSplittingStrategy.MANUAL equals 'manual'."""
        from tscollection.datasets.enums import ClassificationSplittingStrategy

        assert ClassificationSplittingStrategy.MANUAL == 'manual'

    def test_import_from_root(self) -> None:
        """ClassificationSplittingStrategy is importable from tscollection.datasets."""
        from tscollection.datasets import ClassificationSplittingStrategy

        assert ClassificationSplittingStrategy.AS_DEFINED == 'as_defined'

    def test_old_name_does_not_exist(self) -> None:
        """SplittingStrategy should no longer be exported from enums."""
        import tscollection.datasets.enums as enums_module

        assert not hasattr(enums_module, 'SplittingStrategy')

    def test_old_name_not_in_root(self) -> None:
        """SplittingStrategy should no longer be exported from root package."""
        import tscollection.datasets as datasets_module

        assert not hasattr(datasets_module, 'SplittingStrategy')


class TestSeparateTargetFeatureFromDf:
    """Tests for the separate_target_feature_from_df utility."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame with numeric columns and a label column."""
        return pd.DataFrame(
            {
                'label': pd.Series(['A', 'B', 'A', 'C', 'B'], dtype='object'),
                'feat1': [1.0, 2.0, 3.0, 4.0, 5.0],
                'feat2': [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

    def test_import_from_utils(self) -> None:
        """separate_target_feature_from_df is importable from tscollection.datasets.utils."""
        from tscollection.datasets.utils import separate_target_feature_from_df

        assert callable(separate_target_feature_from_df)

    def test_separates_correctly(self, sample_df: pd.DataFrame) -> None:
        """Returns (features_df, target_series) with correct shapes."""
        from tscollection.datasets.utils import separate_target_feature_from_df

        features, target = separate_target_feature_from_df(sample_df, 'label')

        assert isinstance(features, pd.DataFrame)
        assert isinstance(target, pd.Series)
        assert list(features.columns) == ['feat1', 'feat2']
        assert list(target) == ['A', 'B', 'A', 'C', 'B']
        assert len(features) == 5
        assert len(target) == 5

    def test_feature_count_reduced(self, sample_df: pd.DataFrame) -> None:
        """Features DataFrame has one fewer column than original."""
        from tscollection.datasets.utils import separate_target_feature_from_df

        features, _ = separate_target_feature_from_df(sample_df, 'label')

        assert len(features.columns) == len(sample_df.columns) - 1

    def test_missing_column_raises_keyerror(self, sample_df: pd.DataFrame) -> None:
        """KeyError raised when target column doesn't exist (T-04-01-02 mitigation)."""
        from tscollection.datasets.utils import separate_target_feature_from_df

        with pytest.raises(KeyError):
            separate_target_feature_from_df(sample_df, 'nonexistent_column')

    def test_numeric_target(self) -> None:
        """Works with numeric target columns."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'target': [0, 1, 0]})
        from tscollection.datasets.utils import separate_target_feature_from_df

        features, target = separate_target_feature_from_df(df, 'target')

        assert list(features.columns) == ['a', 'b']
        assert list(target) == [0, 1, 0]
