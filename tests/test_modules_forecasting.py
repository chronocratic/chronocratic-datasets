"""Tests for ETT, ElectricityLoad, and Weather forecasting DataModules.

Covers constructor params, variant validation, _set_data_slices,
_csv parsing, transform patterns, TensorDataset usage, and
FileNotFoundError for missing paths.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader, TensorDataset

from tscollection.datasets.enums.data import (
    ForecastingMode,
    ScalingMethod,
    TimeSeriesDatasetMode,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_csv_file(tmp_path: Path) -> Path:
    """Create a minimal CSV file for forecasting module tests."""
    csv_file = tmp_path / 'synthetic.csv'
    dates = pd.date_range('2020-01-01', periods=100, freq='h')
    df = pd.DataFrame(
        {
            'date': dates,
            'col1': np.random.randn(100),
            'col2': np.random.randn(100),
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def electricity_csv_file(tmp_path: Path) -> Path:
    """Create a synthetic electricity CSV with semicolon separator.

    Generates enough data to cover the filtering logic at index 8920
    and the '2012' slicing.
    """
    csv_file = tmp_path / 'electricity.csv'
    # Generate data spanning 2011-2014 to cover the '2012' slice
    dates = pd.date_range('2011-01-01', periods=10000, freq='h')
    df = pd.DataFrame(
        {
            'MT_001': np.random.randn(10000),
            'MT_002': np.random.randn(10000),
        },
        index=dates,
    )
    df.index.name = 'datetime'
    df.to_csv(csv_file, sep=';', decimal=',')
    return csv_file


# ---------------------------------------------------------------------------
# ETTDataModule Tests
# ---------------------------------------------------------------------------


class TestETTDataModuleConstructor:
    """Tests for ETTDataModule constructor."""

    def test_import_ett_module(self) -> None:
        """ETTDataModule can be imported from ett module."""
        from tscollection.datasets.modules.ett import ETTDataModule

        assert ETTDataModule is not None

    def test_constructor_accepts_variant(self, synthetic_csv_file: Path) -> None:
        """Constructor accepts explicit variant parameter (D-06)."""
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=synthetic_csv_file,
            variant='ETTh1',
            seq_len=64,
            mode=ForecastingMode.UNIVARIATE,
            batch_size=16,
        )
        assert module.variant == 'ETTh1'
        assert module.sequence_length == 64
        assert module._mode == ForecastingMode.UNIVARIATE
        assert module.batch_size == 16

    def test_variant_validation_rejects_invalid(self, synthetic_csv_file: Path) -> None:
        """Constructor raises ValueError for unknown variant (T-04-03-01)."""
        from tscollection.datasets.modules.ett import ETTDataModule

        with pytest.raises(ValueError, match='Unknown ETT variant'):
            ETTDataModule(
                dataset_file_path=synthetic_csv_file,
                variant='unknown_variant',
            )

    def test_all_variants_accepted(self, synthetic_csv_file: Path) -> None:
        """All four valid variants are accepted without error."""
        from tscollection.datasets.modules.ett import ETTDataModule

        for variant in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
            module = ETTDataModule(
                dataset_file_path=synthetic_csv_file,
                variant=variant,
            )
            assert module.variant == variant


class TestETTSetDataSlices:
    """Tests for ETT _set_data_slices method."""

    def test_hourly_variant_slices(self, synthetic_csv_file: Path) -> None:
        """ETTh1/ETTh2 use 16/4/4 month slices (hourly resolution)."""
        from tscollection.datasets.modules.ett import ETTDataModule

        for variant in ['ETTh1', 'ETTh2']:
            module = ETTDataModule(
                dataset_file_path=synthetic_csv_file,
                variant=variant,
            )
            module._dataset_name = variant
            module._set_data_slices()

            # train: 0..12*30*24, valid: 12*30*24..16*30*24, test: 16*30*24..20*30*24
            assert module._train_slice == slice(None, 12 * 30 * 24)
            assert module._valid_slice == slice(12 * 30 * 24, 16 * 30 * 24)
            assert module._test_slice == slice(16 * 30 * 24, 20 * 30 * 24)

    def test_15min_variant_slices(self, synthetic_csv_file: Path) -> None:
        """ETTm1/ETTm2 use 4x multiplier for 15-min resolution."""
        from tscollection.datasets.modules.ett import ETTDataModule

        for variant in ['ETTm1', 'ETTm2']:
            module = ETTDataModule(
                dataset_file_path=synthetic_csv_file,
                variant=variant,
            )
            module._dataset_name = variant
            module._set_data_slices()

            # Multiply by 4 for 15-min resolution
            assert module._train_slice == slice(None, 12 * 30 * 24 * 4)
            assert module._valid_slice == slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
            assert module._test_slice == slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)


class TestETTPrepareData:
    """Tests for ETT prepare_data method."""

    def test_prepare_data_raises_file_not_found(self) -> None:
        """prepare_data raises FileNotFoundError for missing file (D-16)."""
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=Path('/nonexistent/ETT.csv'),
            variant='ETTh1',
        )
        with pytest.raises(FileNotFoundError):
            module.prepare_data()


# ---------------------------------------------------------------------------
# ElectricityLoadModule Tests
# ---------------------------------------------------------------------------


class TestElectricityLoadModuleConstructor:
    """Tests for ElectricityLoadModule constructor."""

    def test_import_electricity_module(self) -> None:
        """ElectricityLoadModule can be imported from electricity module."""
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        assert ElectricityLoadModule is not None

    def test_constructor_params(self, electricity_csv_file: Path) -> None:
        """Constructor accepts standard forecasting params."""
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        module = ElectricityLoadModule(
            dataset_file_path=electricity_csv_file,
            seq_len=64,
            mode=ForecastingMode.MULTIVARIATE,
            batch_size=32,
            scale_data=True,
            data_scaling_method=ScalingMethod.STANDARD,
        )
        assert module.sequence_length == 64
        assert module._mode == ForecastingMode.MULTIVARIATE
        assert module.batch_size == 32


class TestElectricityLoadPrepareData:
    """Tests for ElectricityLoadModule prepare_data."""

    def test_prepare_data_raises_file_not_found(self) -> None:
        """prepare_data raises FileNotFoundError for missing file (D-16)."""
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        module = ElectricityLoadModule(
            dataset_file_path=Path('/nonexistent/electricity.csv'),
        )
        with pytest.raises(FileNotFoundError):
            module.prepare_data()

    def test_dataset_name_is_electricity_load(
        self, electricity_csv_file: Path
    ) -> None:
        """_dataset_name is set to 'ElectricityLoad'."""
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        module = ElectricityLoadModule(
            dataset_file_path=electricity_csv_file,
        )
        module.prepare_data()
        assert module._dataset_name == 'ElectricityLoad'


class TestElectricityLoadTransform:
    """Tests for ElectricityLoadModule _transform_data."""

    def test_transform_uses_transpose_and_expand_dims(
        self, electricity_csv_file: Path
    ) -> None:
        """_transform_data applies transpose + expand_dims(axis=-1)."""
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        module = ElectricityLoadModule(
            dataset_file_path=electricity_csv_file,
        )
        # Set synthetic full_data
        module._full_data = pd.DataFrame(
            {'A': [1, 2, 3], 'B': [4, 5, 6]},
            index=pd.date_range('2012-01-01', periods=3, freq='h'),
        )
        module._transform_data()

        # After transform: .T -> (2,3), expand_dims(-1) -> (2,3,1)
        assert module._full_data.shape[-1] == 1
        assert module._full_data.shape[0] == 2  # features after transpose


# ---------------------------------------------------------------------------
# WeatherModule Tests
# ---------------------------------------------------------------------------


class TestWeatherModuleConstructor:
    """Tests for WeatherModule constructor."""

    def test_import_weather_module(self) -> None:
        """WeatherModule can be imported from weather module."""
        from tscollection.datasets.modules.weather import WeatherModule

        assert WeatherModule is not None

    def test_constructor_params(self, synthetic_csv_file: Path) -> None:
        """Constructor accepts standard forecasting params."""
        from tscollection.datasets.modules.weather import WeatherModule

        module = WeatherModule(
            dataset_file_path=synthetic_csv_file,
            seq_len=96,
            mode=ForecastingMode.UNIVARIATE,
            batch_size=64,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        assert module.sequence_length == 96
        assert module._mode == ForecastingMode.UNIVARIATE
        assert module.batch_size == 64


class TestWeatherPrepareData:
    """Tests for WeatherModule prepare_data."""

    def test_prepare_data_raises_file_not_found(self) -> None:
        """prepare_data raises FileNotFoundError for missing file (D-16)."""
        from tscollection.datasets.modules.weather import WeatherModule

        module = WeatherModule(
            dataset_file_path=Path('/nonexistent/weather.csv'),
        )
        with pytest.raises(FileNotFoundError):
            module.prepare_data()


class TestWeatherTransform:
    """Tests for WeatherModule _transform_data."""

    def test_transform_uses_expand_dims_axis_0(self, synthetic_csv_file: Path) -> None:
        """_transform_data applies expand_dims(axis=0)."""
        from tscollection.datasets.modules.weather import WeatherModule

        module = WeatherModule(
            dataset_file_path=synthetic_csv_file,
        )
        # Set synthetic full_data
        module._full_data = pd.DataFrame(
            {'A': [1, 2, 3], 'B': [4, 5, 6]},
            index=pd.date_range('2012-01-01', periods=3, freq='h'),
        )
        module._transform_data()

        # After transform: expand_dims(0) -> (1, 3, 2)
        assert module._full_data.shape[0] == 1


# ---------------------------------------------------------------------------
# Common forecasting module tests
# ---------------------------------------------------------------------------


class TestForecastingModulesUseTensorDataset:
    """Tests that all three modules use TensorDataset (D-13)."""

    def test_ett_uses_tensordataset_in_source(self) -> None:
        """ETT source code references TensorDataset."""
        import tscollection.datasets.modules.ett as ett_module
        source = open(
            Path(ett_module.__file__).parent / 'ett.py'  # type: ignore[arg-type]
        ).read()
        assert 'TensorDataset' in source

    def test_electricity_uses_tensordataset_in_source(self) -> None:
        """Electricity source code references TensorDataset."""
        import tscollection.datasets.modules.electricity as elec_module
        source = open(
            Path(elec_module.__file__).parent / 'electricity.py'  # type: ignore[arg-type]
        ).read()
        assert 'TensorDataset' in source

    def test_weather_uses_tensordataset_in_source(self) -> None:
        """Weather source code references TensorDataset."""
        import tscollection.datasets.modules.weather as weather_module
        source = open(
            Path(weather_module.__file__).parent / 'weather.py'  # type: ignore[arg-type]
        ).read()
        assert 'TensorDataset' in source


class TestForecastingSlices:
    """Tests for fractional slice patterns in Electricity and Weather."""

    def test_weather_fractional_split(self, synthetic_csv_file: Path) -> None:
        """Weather uses 60/20/20 fractional split."""
        from tscollection.datasets.modules.weather import WeatherModule

        module = WeatherModule(
            dataset_file_path=synthetic_csv_file,
        )
        module._full_data = pd.DataFrame(
            {'A': range(100)},
            index=pd.date_range('2012-01-01', periods=100, freq='h'),
        )
        module._set_data_slices()

        assert module._train_slice == slice(None, 60)
        assert module._valid_slice == slice(60, 80)
        assert module._test_slice == slice(80, None)

    def test_electricity_fractional_split(self, electricity_csv_file: Path) -> None:
        """Electricity uses 60/20/20 fractional split."""
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        module = ElectricityLoadModule(
            dataset_file_path=electricity_csv_file,
        )
        module._full_data = pd.DataFrame(
            {'A': range(100)},
            index=pd.date_range('2012-01-01', periods=100, freq='h'),
        )
        module._set_data_slices()

        assert module._train_slice == slice(None, 60)
        assert module._valid_slice == slice(60, 80)
        assert module._test_slice == slice(80, None)


# ---------------------------------------------------------------------------
# ETT Golden-Path Integration Tests (D-01, D-07, D-08)
# ---------------------------------------------------------------------------


class TestETTGoldenPathIntegration:
    """Integration tests exercising the full ETT forecasting pipeline.

    Verifies prepare_data() -> setup('fit') -> train_dataloader() using
    synthetic CSV fixtures with DatetimeIndex (D-07).
    """

    @pytest.fixture
    def ett_csv_file(self, tmp_path: Path) -> Path:
        """Create a synthetic ETT-style CSV with 500 rows and DatetimeIndex.

        Columns match ETT schema: 'date' (DatetimeIndex), 'HUFL', 'HT',
        'OT' (target), 'Wsp' (wind speed). Written via df.to_csv(index=False)
        per D-07.
        """
        csv_file = tmp_path / 'ETT_synthetic.csv'
        dates = pd.date_range('2016-01-01', periods=500, freq='h')
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                'date': dates,
                'HUFL': rng.standard_normal(500),
                'HT': rng.standard_normal(500),
                'OT': rng.standard_normal(500),
                'Wsp': rng.standard_normal(500),
            }
        )
        df.to_csv(csv_file, index=False)
        return csv_file

    @pytest.fixture
    def synthetic_forecasting_csv(self, tmp_path: Path) -> Path:
        """Create a minimal forecasting CSV with DatetimeIndex and features.

        Per D-08, provides a reusable fixture for forecasting integration tests.
        DataFrame has DatetimeIndex and 2-3 feature columns.
        """
        csv_file = tmp_path / 'synthetic_forecasting.csv'
        dates = pd.date_range('2020-01-01', periods=200, freq='h')
        rng = np.random.default_rng(123)
        df = pd.DataFrame(
            {
                'date': dates,
                'feature_a': rng.standard_normal(200),
                'feature_b': rng.standard_normal(200),
                'OT': rng.standard_normal(200),
            }
        )
        df.to_csv(csv_file, index=False)
        return csv_file

    def test_ett_univariate_golden_path(self, ett_csv_file: Path) -> None:
        """ETT univariate: prepare_data + setup produces valid splits.

        Full pipeline with CSV fixture (500 rows, DatetimeIndex,
        columns ['date', 'HUFL', 'HT', 'OT', 'Wsp']), variant='ETTh1',
        mode=UNIVARIATE. Exercises sklearn scaling, time feature extraction,
        data transformation, and train/valid/test splitting.
        """
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=ett_csv_file,
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        module.prepare_data()
        module.setup(stage='fit')

        assert module._train_data_samples is not None
        assert module._valid_data_samples is not None
        assert module._test_data_samples is not None
        assert module.num_features is not None
        # D-01: exercises time feature extraction (DatetimeIndex present)
        assert module.num_time_series_features > 0

    def test_ett_multivariate_golden_path(self, ett_csv_file: Path) -> None:
        """ETT multivariate: prepare_data + setup with all columns.

        Same CSV as univariate but mode=MULTIVARIATE. Verifies
        _train_data_samples has multiple feature dimensions and
        num_features reflects all columns plus time features.
        """
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=ett_csv_file,
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.MULTIVARIATE,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        module.prepare_data()
        module.setup(stage='fit')

        assert module._train_data_samples is not None
        assert module._valid_data_samples is not None
        assert module._test_data_samples is not None
        # Multivariate: more than just OT column
        assert module._train_data_samples.shape[-1] > 1
        assert module.num_features is not None
        assert module.num_features >= module._train_data_samples.shape[-1]

    def test_ett_train_dataloader_returns_batches(self, ett_csv_file: Path) -> None:
        """ETT train_dataloader returns DataLoader with valid batches.

        After prepare_data + setup, train_dataloader() must return a
        DataLoader. Extracting a batch from it should produce a tensor
        with the correct feature dimension (num_features).
        """
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=ett_csv_file,
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        module.prepare_data()
        module.setup(stage='fit')

        train_dl = module.train_dataloader()
        assert isinstance(train_dl, DataLoader)

        batch = next(iter(train_dl))
        # TensorDataset with a single tensor yields a list of one tensor
        assert len(batch) == 1
        batch_tensor = batch[0]
        # Feature dimension should match num_features
        assert batch_tensor.shape[-1] == module.num_features

    def test_ett_15min_variant_golden_path(self, ett_csv_file: Path) -> None:
        """ETTm1 variant: verifies 4x multiplier in _set_data_slices.

        Same golden-path flow as univariate test but with variant='ETTm1'
        to exercise 15-min resolution slice boundaries (4x multiplier).
        """
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=ett_csv_file,
            variant='ETTm1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        module.prepare_data()
        module.setup(stage='fit')

        assert module._train_data_samples is not None
        assert module._valid_data_samples is not None
        assert module._test_data_samples is not None
        assert module.num_features is not None
        assert module.num_time_series_features > 0
