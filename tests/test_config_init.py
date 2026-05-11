"""Tests for the config package export chain (__init__.py, root exports)."""

from __future__ import annotations


class TestConfigInitAll:
    """Test config/__init__.py __all__ exports."""

    def test_all_11_instances_importable(self) -> None:
        """All 11 config instances must be importable from tscollection.datasets.config."""
        from tscollection.datasets.config import (
            ELECTRICITY_LOAD,
            ETT_H1,
            ETT_H2,
            ETT_M1,
            ETT_M2,
            UCR_COFFEE,
            UCR_ECG200,
            UCR_FACE_FOUR,
            UEA_ATRIAL_FIBRILLATION,
            UEA_BASIC_MOTIONS,
            WEATHER,
        )
        assert UCR_COFFEE.name == 'Coffee'
        assert UCR_ECG200.name == 'ECG200'
        assert UCR_FACE_FOUR.name == 'FaceFour'
        assert UEA_BASIC_MOTIONS.name == 'BasicMotions'
        assert UEA_ATRIAL_FIBRILLATION.name == 'AtrialFibrillation'
        assert ETT_H1.name == 'ETTh1'
        assert ETT_H2.name == 'ETTh2'
        assert ETT_M1.name == 'ETTm1'
        assert ETT_M2.name == 'ETTm2'
        assert ELECTRICITY_LOAD.name == 'electricity'
        assert WEATHER.name == 'weather'

    def test_factory_functions_importable(self) -> None:
        """Factory functions must be importable from tscollection.datasets.config."""
        from tscollection.datasets.config import (
            CONFIGS,
            get_config,
            list_configs,
        )
        assert callable(get_config)
        assert callable(list_configs)
        assert isinstance(CONFIGS, dict)
        assert len(CONFIGS) == 11

    def test_base_types_importable(self) -> None:
        """Base config types must be importable from tscollection.datasets.config."""
        from tscollection.datasets.config import (
            ClassificationConfig,
            DatasetConfig,
            ForecastingConfig,
        )
        assert ClassificationConfig is not None
        assert DatasetConfig is not None
        assert ForecastingConfig is not None

    def test_config_all_contains_expected_names(self) -> None:
        """config/__init__.py __all__ must contain all exported names."""
        import tscollection.datasets.config as cfg

        expected = {
            'UCR_COFFEE',
            'UCR_ECG200',
            'UCR_FACE_FOUR',
            'UEA_BASIC_MOTIONS',
            'UEA_ATRIAL_FIBRILLATION',
            'ETT_H1',
            'ETT_H2',
            'ETT_M1',
            'ETT_M2',
            'ELECTRICITY_LOAD',
            'WEATHER',
            'get_config',
            'list_configs',
            'CONFIGS',
            'DatasetConfig',
            'ClassificationConfig',
            'ForecastingConfig',
        }
        assert expected.issubset(set(cfg.__all__))


class TestRootPackageExports:
    """Test root __init__.py exports."""

    def test_dataset_family_importable(self) -> None:
        """DatasetFamily must be importable from tscollection.datasets."""
        from tscollection.datasets import DatasetFamily
        assert DatasetFamily.UCR is not None
        assert DatasetFamily.ETT is not None

    def test_split_mode_importable(self) -> None:
        """SplitMode must be importable from tscollection.datasets."""
        from tscollection.datasets import SplitMode
        assert SplitMode.INDEXED is not None
        assert SplitMode.FRACTIONAL is not None

    def test_existing_exports_still_work(self) -> None:
        """Previously exported enums must still be importable."""
        from tscollection.datasets import (
            DistanceMetric,
            ForecastingMode,
            ScalingMethod,
            SplittingStrategy,
            TimeSeriesDatasetMode,
        )
        assert DistanceMetric is not None
        assert ForecastingMode is not None
        assert ScalingMethod is not None
        assert SplittingStrategy is not None
        assert TimeSeriesDatasetMode is not None

    def test_full_export_chain(self) -> None:
        """Full chain: root -> config -> factory -> instances must work."""
        from tscollection.datasets import DatasetFamily
        from tscollection.datasets.config import UCR_COFFEE, get_config

        assert get_config(name='Coffee') is UCR_COFFEE
        assert UCR_COFFEE.family == DatasetFamily.UCR
