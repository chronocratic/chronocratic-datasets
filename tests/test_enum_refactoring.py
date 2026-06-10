"""Tests for enum refactoring and mode mapping (Phase 08 Plan 01).

Verifies:
- TimeSeriesDatasetMode renamed values (SAMPLE_ONLY, SAMPLE_LABEL, INPUT_OUTPUT)
- New ClassificationLoaderMode enum
- New ForecastingLoaderMode enum
- Mode mapping dicts (CLASSIFICATION_LOADER_MAP, FORECASTING_LOADER_MAP)
- Enum exports from __init__.py
"""


class TestTimeSeriesDatasetModeRenamedValues:
    """Verify TimeSeriesDatasetMode enum values are renamed to task-agnostic semantics."""

    def test_sample_only_value(self):
        from chronocratic.datasets.enums import TimeSeriesDatasetMode

        assert TimeSeriesDatasetMode.SAMPLE_ONLY == 'sample_only'

    def test_sample_label_value(self):
        from chronocratic.datasets.enums import TimeSeriesDatasetMode

        assert TimeSeriesDatasetMode.SAMPLE_LABEL == 'sample_label'

    def test_input_output_value(self):
        from chronocratic.datasets.enums import TimeSeriesDatasetMode

        assert TimeSeriesDatasetMode.INPUT_OUTPUT == 'input_output'

    def test_old_values_removed(self):
        """Old enum values (WITH_LABELS, WITHOUT_LABELS, FORECASTING) must not exist."""
        from chronocratic.datasets.enums import TimeSeriesDatasetMode

        assert not hasattr(TimeSeriesDatasetMode, 'WITH_LABELS')
        assert not hasattr(TimeSeriesDatasetMode, 'WITHOUT_LABELS')
        assert not hasattr(TimeSeriesDatasetMode, 'FORECASTING')


class TestClassificationLoaderMode:
    """Verify ClassificationLoaderMode enum exists with correct values."""

    def test_sample_only_value(self):
        from chronocratic.datasets.enums import ClassificationLoaderMode

        assert ClassificationLoaderMode.SAMPLE_ONLY == 'sample_only'

    def test_sample_label_value(self):
        from chronocratic.datasets.enums import ClassificationLoaderMode

        assert ClassificationLoaderMode.SAMPLE_LABEL == 'sample_label'

    def test_importable_from_enums_package(self):
        from chronocratic.datasets.enums import ClassificationLoaderMode

        assert ClassificationLoaderMode is not None


class TestForecastingLoaderMode:
    """Verify ForecastingLoaderMode enum exists with correct values."""

    def test_raw_series_value(self):
        from chronocratic.datasets.enums import ForecastingLoaderMode

        assert ForecastingLoaderMode.RAW_SERIES == 'raw_series'

    def test_input_target_value(self):
        from chronocratic.datasets.enums import ForecastingLoaderMode

        assert ForecastingLoaderMode.INPUT_TARGET == 'input_target'

    def test_input_only_value(self):
        from chronocratic.datasets.enums import ForecastingLoaderMode

        assert ForecastingLoaderMode.INPUT_ONLY == 'input_only'

    def test_importable_from_enums_package(self):
        from chronocratic.datasets.enums import ForecastingLoaderMode

        assert ForecastingLoaderMode is not None


class TestModeMapping:
    """Verify loader-to-dataset mode mapping dictionaries."""

    def test_classification_loader_map_sample_only(self):
        from chronocratic.datasets.enums import (
            ClassificationLoaderMode,
            TimeSeriesDatasetMode,
        )
        from chronocratic.datasets.maps.loader_to_dataset import (
            CLASSIFICATION_LOADER_MAP,
        )

        result = CLASSIFICATION_LOADER_MAP[ClassificationLoaderMode.SAMPLE_ONLY]
        assert result == TimeSeriesDatasetMode.SAMPLE_ONLY

    def test_classification_loader_map_sample_label(self):
        from chronocratic.datasets.enums import (
            ClassificationLoaderMode,
            TimeSeriesDatasetMode,
        )
        from chronocratic.datasets.maps.loader_to_dataset import (
            CLASSIFICATION_LOADER_MAP,
        )

        result = CLASSIFICATION_LOADER_MAP[ClassificationLoaderMode.SAMPLE_LABEL]
        assert result == TimeSeriesDatasetMode.SAMPLE_LABEL

    def test_forecasting_loader_map_raw_series_is_none(self):
        from chronocratic.datasets.enums import ForecastingLoaderMode
        from chronocratic.datasets.maps.loader_to_dataset import (
            FORECASTING_LOADER_MAP,
        )

        assert FORECASTING_LOADER_MAP[ForecastingLoaderMode.RAW_SERIES] is None

    def test_forecasting_loader_map_input_target(self):
        from chronocratic.datasets.enums import (
            ForecastingLoaderMode,
            TimeSeriesDatasetMode,
        )
        from chronocratic.datasets.maps.loader_to_dataset import (
            FORECASTING_LOADER_MAP,
        )

        result = FORECASTING_LOADER_MAP[ForecastingLoaderMode.INPUT_TARGET]
        assert result == TimeSeriesDatasetMode.INPUT_OUTPUT

    def test_forecasting_loader_map_input_only(self):
        from chronocratic.datasets.enums import (
            ForecastingLoaderMode,
            TimeSeriesDatasetMode,
        )
        from chronocratic.datasets.maps.loader_to_dataset import (
            FORECASTING_LOADER_MAP,
        )

        result = FORECASTING_LOADER_MAP[ForecastingLoaderMode.INPUT_ONLY]
        assert result == TimeSeriesDatasetMode.SAMPLE_ONLY


class TestForecastingModeUnchanged:
    """Verify ForecastingMode enum was not modified (D-03)."""

    def test_univariate_unchanged(self):
        from chronocratic.datasets.enums import ForecastingMode

        assert ForecastingMode.UNIVARIATE == 'univariate'

    def test_multivariate_unchanged(self):
        from chronocratic.datasets.enums import ForecastingMode

        assert ForecastingMode.MULTIVARIATE == 'multivariate'


class TestBaseDatasetSampleMap:
    """Verify base dataset sample map uses new enum values."""

    def test_sample_only_mapping(self):
        from chronocratic.datasets.datatypes._base.base import (
            TimeSeriesDataset,
        )
        from chronocratic.datasets.enums import TimeSeriesDatasetMode

        sample_map = TimeSeriesDataset._get_sample_fun_map
        assert TimeSeriesDatasetMode.SAMPLE_ONLY in sample_map
        assert sample_map[TimeSeriesDatasetMode.SAMPLE_ONLY] == '_get_sample_1'

    def test_sample_label_mapping(self):
        from chronocratic.datasets.datatypes._base.base import (
            TimeSeriesDataset,
        )
        from chronocratic.datasets.enums import TimeSeriesDatasetMode

        sample_map = TimeSeriesDataset._get_sample_fun_map
        assert TimeSeriesDatasetMode.SAMPLE_LABEL in sample_map
        assert sample_map[TimeSeriesDatasetMode.SAMPLE_LABEL] == '_get_sample_2'

    def test_input_output_mapping(self):
        from chronocratic.datasets.datatypes._base.base import (
            TimeSeriesDataset,
        )
        from chronocratic.datasets.enums import TimeSeriesDatasetMode

        sample_map = TimeSeriesDataset._get_sample_fun_map
        assert TimeSeriesDatasetMode.INPUT_OUTPUT in sample_map
        assert sample_map[TimeSeriesDatasetMode.INPUT_OUTPUT] == '_get_sample_3'
