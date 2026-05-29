"""Tests for ETT, ElectricityLoad, and Weather forecasting DataModules.

Covers constructor params, variant validation, _set_data_slices,
_csv parsing, transform patterns, TensorDataset usage,
FileNotFoundError for missing paths, and lifecycle integration tests
(idempotency, prepare_data sentinel, finalize hook, dimensions, stage gating).
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from tscollection.datasets.enums.data import ForecastingMode, ScalingMethod

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_csv_file(tmp_path: Path) -> Path:
    """Create a minimal CSV file for forecasting module tests."""
    csv_file = tmp_path / 'synthetic.csv'
    dates = pd.date_range('2020-01-01', periods=100, freq='h')
    df = pd.DataFrame({'date': dates, 'col1': np.random.randn(100), 'col2': np.random.randn(100)})
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def electricity_csv_file(tmp_path: Path) -> Path:
    """Create a synthetic electricity CSV with semicolon separator.

    Generates enough data to cover the filtering logic and '2012' slicing.
    """
    csv_file = tmp_path / 'electricity.csv'
    # Generate data spanning 2011-2014 to cover the '2012:' slice
    dates = pd.date_range('2011-01-01', periods=10000, freq='h')
    df = pd.DataFrame(
        {'MT_001': np.random.randn(10000), 'MT_002': np.random.randn(10000)}, index=dates
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
        """Constructor accepts explicit variant parameter."""
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
        """Constructor raises ValueError for unknown variant."""
        from tscollection.datasets.modules.ett import ETTDataModule

        with pytest.raises(ValueError, match='Unknown ETT variant'):
            ETTDataModule(dataset_file_path=synthetic_csv_file, variant='unknown_variant')

    def test_all_variants_accepted(self, synthetic_csv_file: Path) -> None:
        """All four valid variants are accepted without error."""
        from tscollection.datasets.modules.ett import ETTDataModule

        for variant in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
            module = ETTDataModule(dataset_file_path=synthetic_csv_file, variant=variant)
            assert module.variant == variant


class TestETTSetDataSlices:
    """Tests for ETT _set_data_slices method."""

    def test_hourly_variant_slices(self, synthetic_csv_file: Path) -> None:
        """ETTh1/ETTh2 use 16/4/4 month slices (hourly resolution)."""
        from tscollection.datasets.modules.ett import ETTDataModule

        for variant in ['ETTh1', 'ETTh2']:
            module = ETTDataModule(dataset_file_path=synthetic_csv_file, variant=variant)
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
            module = ETTDataModule(dataset_file_path=synthetic_csv_file, variant=variant)
            module._dataset_name = variant
            module._set_data_slices()

            # Multiply by 4 for 15-min resolution
            assert module._train_slice == slice(None, 12 * 30 * 24 * 4)
            assert module._valid_slice == slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
            assert module._test_slice == slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)


class TestETTPrepareData:
    """Tests for ETT prepare_data method."""

    def test_prepare_data_raises_file_not_found(self) -> None:
        """prepare_data raises FileNotFoundError for missing file."""
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(dataset_file_path=Path('/nonexistent/ETT.csv'), variant='ETTh1')
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
        """prepare_data raises FileNotFoundError for missing file."""
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        module = ElectricityLoadModule(dataset_file_path=Path('/nonexistent/electricity.csv'))
        with pytest.raises(FileNotFoundError):
            module.prepare_data()

    def test_dataset_name_is_electricity_load(self, electricity_csv_file: Path) -> None:
        """_dataset_name is set to 'ElectricityLoad'."""
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        module = ElectricityLoadModule(dataset_file_path=electricity_csv_file)
        module.prepare_data()
        assert module._dataset_name == 'ElectricityLoad'


class TestElectricityLoadTransform:
    """Tests for ElectricityLoadModule _transform_data."""

    def test_transform_uses_transpose_and_expand_dims(self, electricity_csv_file: Path) -> None:
        """_transform_data applies transpose + expand_dims(axis=-1)."""
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        module = ElectricityLoadModule(dataset_file_path=electricity_csv_file)
        # Set synthetic full_data
        module._full_data = pd.DataFrame(
            {'A': [1, 2, 3], 'B': [4, 5, 6]}, index=pd.date_range('2012-01-01', periods=3, freq='h')
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
        """prepare_data raises FileNotFoundError for missing file."""
        from tscollection.datasets.modules.weather import WeatherModule

        module = WeatherModule(dataset_file_path=Path('/nonexistent/weather.csv'))
        with pytest.raises(FileNotFoundError):
            module.prepare_data()


class TestWeatherTransform:
    """Tests for WeatherModule _transform_data."""

    def test_transform_uses_expand_dims_axis_0(self, synthetic_csv_file: Path) -> None:
        """_transform_data applies expand_dims(axis=0)."""
        from tscollection.datasets.modules.weather import WeatherModule

        module = WeatherModule(dataset_file_path=synthetic_csv_file)
        # Set synthetic full_data
        module._full_data = pd.DataFrame(
            {'A': [1, 2, 3], 'B': [4, 5, 6]}, index=pd.date_range('2012-01-01', periods=3, freq='h')
        )
        module._transform_data()

        # After transform: expand_dims(0) -> (1, 3, 2)
        assert module._full_data.shape[0] == 1


# ---------------------------------------------------------------------------
# Common forecasting module tests
# ---------------------------------------------------------------------------


class TestForecastingModulesUseTensorDataset:
    """Tests that all three modules use TensorDataset."""

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

        module = WeatherModule(dataset_file_path=synthetic_csv_file)
        module._full_data = pd.DataFrame(
            {'A': range(100)}, index=pd.date_range('2012-01-01', periods=100, freq='h')
        )
        module._set_data_slices()

        assert module._train_slice == slice(None, 60)
        assert module._valid_slice == slice(60, 80)
        assert module._test_slice == slice(80, None)

    def test_electricity_fractional_split(self, electricity_csv_file: Path) -> None:
        """Electricity uses 60/20/20 fractional split."""
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        module = ElectricityLoadModule(dataset_file_path=electricity_csv_file)
        module._full_data = pd.DataFrame(
            {'A': range(100)}, index=pd.date_range('2012-01-01', periods=100, freq='h')
        )
        module._set_data_slices()

        assert module._train_slice == slice(None, 60)
        assert module._valid_slice == slice(60, 80)
        assert module._test_slice == slice(80, None)


# ---------------------------------------------------------------------------
# WeatherModule Integration Smoke Tests
# ---------------------------------------------------------------------------


class TestWeatherModuleIntegration:
    """Integration tests for WeatherModule dataloader pipeline.

    Verifies prepare_data() -> setup('fit') -> train_dataloader() using
    synthetic CSV fixtures with DatetimeIndex. Tests the fractional-split
    path (60/20/20) vs ETT's absolute-boundary splits.
    """

    @pytest.fixture
    def weather_csv_file(self, tmp_path: Path) -> Path:
        """Create a synthetic Weather CSV with 200 rows and DatetimeIndex.

        Columns: 'date' (DatetimeIndex), 'wbng', 'wbhh', 'wbat', 'sbfg'.
        Written via df.to_csv(index=False).
        """
        csv_path = tmp_path / 'weather.csv'
        dates = pd.date_range('2006-01-01', periods=200, freq='h')
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                'date': dates,
                'wbng': rng.standard_normal(200),
                'wbhh': rng.standard_normal(200),
                'wbat': rng.standard_normal(200),
                'sbfg': rng.standard_normal(200),
            }
        )
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_weather_golden_path_integration(self, weather_csv_file: Path) -> None:
        """Weather golden path: prepare_data + setup produces valid splits.

        Full pipeline with CSV fixture (200 rows, DatetimeIndex,
        columns ['date', 'wbng', 'wbhh', 'wbat', 'sbfg']),
        mode=UNIVARIATE. Exercises sklearn scaling, time feature
        extraction, data transformation, and train/valid/test splitting
        via the 60/20/20 fractional path.
        """
        from tscollection.datasets.modules.weather import WeatherModule

        module = WeatherModule(
            dataset_file_path=weather_csv_file,
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
        )
        module.prepare_data()
        module.setup(stage='fit')

        assert module._train_data_samples is not None
        assert module._valid_data_samples is not None
        assert module._test_data_samples is not None
        assert module.num_features is not None
        # DatetimeIndex present, so time features should be extracted
        assert module.num_time_series_features > 0

    def test_weather_fractional_split_bounds(self, weather_csv_file: Path) -> None:
        """Weather uses 60/20/20 fractional split.

        Verifies _train_slice == slice(None, 120), _valid_slice ==
        slice(120, 160), _test_slice == slice(160, None) for 200-row
        fixture. Confirms Weather's fractional split pattern differs
        from ETT's absolute month-boundary splits.
        """
        from tscollection.datasets.modules.weather import WeatherModule

        module = WeatherModule(dataset_file_path=weather_csv_file)
        module.prepare_data()
        module.setup(stage='fit')

        # 60/20/20 of 200 rows: train[:120], valid[120:160], test[160:]
        assert module._train_slice == slice(None, 120)
        assert module._valid_slice == slice(120, 160)
        assert module._test_slice == slice(160, None)

    def test_weather_dataloader_shapes(self, weather_csv_file: Path) -> None:
        """Weather dataloaders return DataLoader with correct batch shapes.

        After prepare_data + setup, train_dataloader() must return a
        DataLoader. Weather's expand_dims(axis=0) transform produces
        (1, samples, features) shape for _full_data, so _train_data_samples
        has shape (1, 120, total_features). DataLoader wraps this in
        TensorDataset.
        """
        from tscollection.datasets.modules.weather import WeatherModule

        module = WeatherModule(
            dataset_file_path=weather_csv_file,
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
        )
        module.prepare_data()
        module.setup(stage='fit')

        # Verify train dataloader
        train_dl = module.train_dataloader()
        assert isinstance(train_dl, DataLoader)
        batch = next(iter(train_dl))
        # TensorDataset with single tensor yields a list of one tensor
        assert len(batch) == 1
        batch_tensor = batch[0]
        # Feature dimension should match num_features
        assert batch_tensor.shape[-1] == module.num_features

        # Verify val dataloader
        val_dl = module.val_dataloader()
        assert isinstance(val_dl, DataLoader)
        val_batch = next(iter(val_dl))
        assert val_batch[0].shape[-1] == module.num_features

        # Verify test dataloader
        test_dl = module.test_dataloader()
        assert isinstance(test_dl, DataLoader)
        test_batch = next(iter(test_dl))
        assert test_batch[0].shape[-1] == module.num_features


# ---------------------------------------------------------------------------
# ETT Golden-Path Integration Tests
# ---------------------------------------------------------------------------


class TestETTGoldenPathIntegration:
    """Integration tests exercising the full ETT forecasting pipeline.

    Verifies prepare_data() -> setup('fit') -> train_dataloader() using
    synthetic CSV fixtures with DatetimeIndex.
    """

    @pytest.fixture
    def ett_csv_file(self, tmp_path: Path) -> Path:
        """Create a synthetic ETT-style CSV with 500 rows and DatetimeIndex.

        Columns match ETT schema: 'date' (DatetimeIndex), 'HUFL', 'HT',
        'OT' (target), 'Wsp' (wind speed). Written via df.to_csv(index=False).
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

        Provides a reusable fixture for forecasting integration tests.
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
        # exercises time feature extraction (DatetimeIndex present)
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


# ---------------------------------------------------------------------------
# Forecasting setup() Edge-Case Tests
# ---------------------------------------------------------------------------


class TestForecastingSetupEdgeCases:
    """Unit tests for forecasting setup() edge cases.

    Tests numpy _full_data (no DatetimeIndex), STANDARD scaling,
    and scale_data=False behavior.
    """

    def test_setup_numpy_full_data_skips_time_features(self) -> None:
        """setup() with numpy _full_data produces num_time_series_features == 0.

        Pre-populates module._full_data with a pure numpy array (no
        DatetimeIndex), sets slices, and calls setup(). Verifies that
        the no-DatetimeIndex branch (line 165-167 in forecasting.py)
        is hit.
        """
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=Path('/nonexistent/dummy.csv'),
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        # pure numpy, no DatetimeIndex
        rng = np.random.default_rng(42)
        module._full_data = rng.standard_normal((100, 5)).astype(np.float32)
        module._train_slice = slice(None, 60)
        module._valid_slice = slice(60, 80)
        module._test_slice = slice(80, None)

        module.setup(stage='fit')

        assert module.num_time_series_features == 0
        assert module._train_data_samples is not None
        assert module._train_data_samples.shape == (1, 60, 5)

    def test_setup_standard_scaling(self) -> None:
        """setup() with ScalingMethod.STANDARD uses StandardScaler.

        Verifies the StandardScaler branch (line 214-215 in
        forecasting.py) is exercised. Pre-populates numpy _full_data,
        sets slices, and calls setup().
        """
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=Path('/nonexistent/dummy.csv'),
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=True,
            data_scaling_method=ScalingMethod.STANDARD,
        )
        rng = np.random.default_rng(42)
        module._full_data = rng.standard_normal((100, 5)).astype(np.float32)
        module._train_slice = slice(None, 60)
        module._valid_slice = slice(60, 80)
        module._test_slice = slice(80, None)

        module.setup(stage='fit')

        assert module._train_data_samples is not None
        # No DatetimeIndex with numpy, so time features == 0
        assert module.num_time_series_features == 0

    def test_setup_scale_data_false(self) -> None:
        """setup() with scale_data=False completes without error.

        Verifies setup() respects the scale_data=False flag and produces
        valid data samples with unscaled values.
        """
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=Path('/nonexistent/dummy.csv'),
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=False,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        rng = np.random.default_rng(42)
        module._full_data = rng.standard_normal((100, 5)).astype(np.float32)
        module._train_slice = slice(None, 60)
        module._valid_slice = slice(60, 80)
        module._test_slice = slice(80, None)

        module.setup(stage='fit')

        assert module._train_data_samples is not None
        assert module._valid_data_samples is not None
        assert module._test_data_samples is not None


# ---------------------------------------------------------------------------
# ElectricityLoadModule Integration Smoke Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bug Fix Tests
# ---------------------------------------------------------------------------


class TestForecastingBugFixes:
    """Tests for scaler axis and scale_data flag."""

    def test_scaler_fits_train_only(self) -> None:
        """Scaler fits only on training time steps, not validation/test.

        Pre-populate numpy _full_data (100, 5) where validation rows (60-79)
        have values 100x larger than training rows (0-59). After setup, the
        scaled training data should reflect ONLY training statistics. If the
        scaler leaked validation data, the range would be wider (min would be
        near 0 for the full dataset, not just the train slice).
        """
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=Path('/nonexistent/dummy.csv'),
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        rng = np.random.default_rng(42)
        raw_data = rng.standard_normal((100, 5)).astype(np.float32)
        # Make validation rows (60-79) have large values
        raw_data[60:80] = raw_data[60:80] * 100.0 + 500.0
        module._full_data = raw_data
        module._train_slice = slice(None, 60)
        module._valid_slice = slice(60, 80)
        module._test_slice = slice(80, None)

        module.setup(stage='fit')

        # With MINMAX on train-only, train data should span [0, 1].
        # If scaler leaked validation data (which has values ~500+),
        # training data would be compressed near 0 (max << 1).
        train_min = module._train_data_samples.min()
        train_max = module._train_data_samples.max()
        assert train_min >= -1e-6, (
            f'Train min {train_min} is negative — scaler likely saw validation data'
        )
        assert train_max > 0.5, (
            f'Train max {train_max} is too low — scaler likely leaked validation '
            f'data (expected ~1.0 for MINMAX fitted on train only)'
        )

    def test_scale_data_false_preserves_values(self) -> None:
        """scale_data=False produces unscaled data identical to input.

        Pre-populate _full_data with known random values, copy to original,
        call setup. Assert values are unchanged after setup.
        """
        from tscollection.datasets.modules.ett import ETTDataModule

        rng = np.random.default_rng(42)
        original = rng.standard_normal((100, 5)).astype(np.float32)

        module = ETTDataModule(
            dataset_file_path=Path('/nonexistent/dummy.csv'),
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=False,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        module._full_data = original.copy()
        module._train_slice = slice(None, 60)
        module._valid_slice = slice(60, 80)
        module._test_slice = slice(80, None)

        module.setup(stage='fit')

        # _full_data after _transform_data has shape (1, 100, 5) due to
        # expand_dims(axis=0). Extract the actual data plane.
        transformed = module._full_data
        # Compare the data plane (squeeze axis 0 added by _transform_data)
        actual_data = transformed.squeeze(axis=0)
        assert actual_data.shape == (100, 5), f'Unexpected shape {actual_data.shape}'
        assert np.allclose(actual_data, original, atol=1e-6), (
            'Data was modified despite scale_data=False'
        )

    def test_scale_data_true_modifies_values(self) -> None:
        """scale_data=True actually transforms data values.

        Same setup as test_scale_data_false_preserves_values but with
        scale_data=True. Assert data values ARE different after setup.
        """
        from tscollection.datasets.modules.ett import ETTDataModule

        rng = np.random.default_rng(42)
        original = rng.standard_normal((100, 5)).astype(np.float32)

        module = ETTDataModule(
            dataset_file_path=Path('/nonexistent/dummy.csv'),
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        module._full_data = original.copy()
        module._train_slice = slice(None, 60)
        module._valid_slice = slice(60, 80)
        module._test_slice = slice(80, None)

        module.setup(stage='fit')

        transformed = module._full_data
        actual_data = transformed.squeeze(axis=0)
        assert not np.allclose(actual_data, original, atol=1e-6), (
            'Data was NOT modified despite scale_data=True'
        )


class TestElectricityBugFixes:
    """Tests for hardcoded iloc[8920] fix."""

    def test_prepare_data_small_dataset(self, tmp_path: Path) -> None:
        """ElectricityLoadModule.prepare_data() does not crash on small CSV.

        Create a synthetic electricity CSV with only 100 rows (semicolon-
        delimited, comma decimal, with MT_001 and MT_002 columns).
        Instantiate ElectricityLoadModule, call prepare_data().
        Assert no IndexError is raised and _full_data is set.
        """
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        csv_file = tmp_path / 'small_electricity.csv'
        dates = pd.date_range('2012-01-01', periods=100, freq='h')
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {'MT_001': rng.standard_normal(100), 'MT_002': rng.standard_normal(100)}, index=dates
        )
        df.index.name = 'datetime'
        df.to_csv(csv_file, sep=';', decimal=',')

        module = ElectricityLoadModule(dataset_file_path=csv_file, mode=ForecastingMode.UNIVARIATE)
        # Should NOT raise IndexError
        module.prepare_data()

        assert module._full_data is not None
        assert len(module._full_data) > 0


class TestElectricityModuleIntegration:
    """Integration tests for ElectricityLoadModule dataloader pipeline.

    Verifies prepare_data() -> setup('fit') -> train_dataloader() using
    the existing electricity_csv_file fixture (semicolon-delimited CSV,
    comma decimals, 10000 rows spanning 2011-2014). Tests the fractional-
    split path (60/20/20) with transpose + expand_dims(axis=-1) transform.
    """

    def test_electricity_golden_path_integration(self, electricity_csv_file: Path) -> None:
        """Electricity golden path: prepare_data + setup produces valid splits.

        Full pipeline with existing electricity CSV fixture (10000 rows,
        semicolon delimiter, comma decimal, columns MT_001/MT_002),
        mode=UNIVARIATE. Exercises CSV parsing, hourly resampling,
        column filtering, '2012:' slicing, sklearn scaling,
        time feature extraction, transpose + expand_dims transform, and
        60/20/20 fractional train/valid/test splitting.
        """
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        module = ElectricityLoadModule(
            dataset_file_path=electricity_csv_file,
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
        )
        module.prepare_data()
        module.setup(stage='fit')

        assert module._train_data_samples is not None
        assert module._valid_data_samples is not None
        assert module._test_data_samples is not None
        assert module.num_features is not None
        # Hardcoded dataset name
        assert module._dataset_name == 'ElectricityLoad'

    def test_electricity_transform_shape(self, electricity_csv_file: Path) -> None:
        """Electricity transform produces (features, samples, 1) pattern.

        After prepare_data + setup, _full_data should have shape
        (num_active_columns, samples, 1 + time_feature_dim) due to
        transpose + expand_dims(axis=-1) transform plus time feature
        concatenation. The trailing dimension reflects expand_dims
        (axis=-1) adding dimension 1, then time features appended
        to that dimension.
        """
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        module = ElectricityLoadModule(
            dataset_file_path=electricity_csv_file,
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
        )
        module.prepare_data()
        module.setup(stage='fit')

        # Trailing dimension >= 1 (expand_dims adds it, time features may enlarge)
        assert module._full_data.shape[-1] >= 1
        # First dimension = number of active columns after zero-column filter
        # For univariate mode, only MT_001 is selected, so shape[0] == 1
        assert module._full_data.shape[0] == 1
        # Second dimension = number of samples (from '2012:' onwards)
        assert module._full_data.shape[1] > 0


# ---------------------------------------------------------------------------
# Integration: Lifecycle Behavior Tests
# ---------------------------------------------------------------------------


class TestSetupIdempotency:
    """Verify setup('fit') called twice does not double-scale data.

    Uses attribute injection (numpy _full_data + slices) to bypass I/O,
    exercising the full MRO chain on each concrete forecasting module.
    """

    def test_ett_setup_fit_twice_no_double_scale(self) -> None:
        """ETT: setup('fit') called twice produces identical train samples."""
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=Path('/nonexistent/dummy.csv'),
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        rng = np.random.default_rng(42)
        module._full_data = rng.standard_normal((100, 5)).astype(np.float32)
        module._train_slice = slice(None, 60)
        module._valid_slice = slice(60, 80)
        module._test_slice = slice(80, None)

        module.setup(stage='fit')
        snapshot = module._train_data_samples.copy()
        module.setup(stage='fit')

        np.testing.assert_array_equal(snapshot, module._train_data_samples)

    def test_weather_setup_fit_twice_no_double_scale(self) -> None:
        """Weather: setup('fit') called twice produces identical train samples."""
        from tscollection.datasets.modules.weather import WeatherModule

        module = WeatherModule(
            dataset_file_path=Path('/nonexistent/dummy.csv'),
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        rng = np.random.default_rng(42)
        module._full_data = rng.standard_normal((100, 5)).astype(np.float32)
        module._train_slice = slice(None, 60)
        module._valid_slice = slice(60, 80)
        module._test_slice = slice(80, None)

        module.setup(stage='fit')
        snapshot = module._train_data_samples.copy()
        module.setup(stage='fit')

        np.testing.assert_array_equal(snapshot, module._train_data_samples)

    def test_electricity_setup_fit_twice_no_double_scale(self) -> None:
        """Electricity: setup('fit') called twice produces identical train samples."""
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        module = ElectricityLoadModule(
            dataset_file_path=Path('/nonexistent/dummy.csv'),
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        rng = np.random.default_rng(42)
        module._full_data = rng.standard_normal((100, 5)).astype(np.float32)
        module._train_slice = slice(None, 60)
        module._valid_slice = slice(60, 80)
        module._test_slice = slice(80, None)

        module.setup(stage='fit')
        snapshot = module._train_data_samples.copy()
        module.setup(stage='fit')

        np.testing.assert_array_equal(snapshot, module._train_data_samples)


class TestPrepareDataIdempotency:
    """Verify prepare_data() called twice runs I/O only once.

    Uses synthetic CSV fixtures and patches pd.read_csv to spy on
    the call count.
    """

    @pytest.fixture
    def ett_csv(self, tmp_path: Path) -> Path:
        """Create a minimal ETT-style CSV with 'date' and 'OT' columns."""
        csv_file = tmp_path / 'ett.csv'
        dates = pd.date_range('2016-01-01', periods=100, freq='h')
        df = pd.DataFrame(
            {
                'date': dates,
                'HUFL': np.random.default_rng(42).standard_normal(100),
                'OT': np.random.default_rng(43).standard_normal(100),
            }
        )
        df.to_csv(csv_file, index=False)
        return csv_file

    def test_ett_prepare_data_runs_io_once(self, ett_csv: Path) -> None:
        """ETT: prepare_data() twice → pd.read_csv called once."""
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=ett_csv,
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
        )

        with patch('pandas.read_csv', wraps=pd.read_csv) as spy:
            module.prepare_data()
            module.prepare_data()
            assert spy.call_count == 1

    def test_weather_prepare_data_runs_io_once(self, synthetic_csv_file: Path) -> None:
        """Weather: prepare_data() twice → pd.read_csv called once."""
        from tscollection.datasets.modules.weather import WeatherModule

        module = WeatherModule(
            dataset_file_path=synthetic_csv_file,
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.MULTIVARIATE,
        )

        with patch('pandas.read_csv', wraps=pd.read_csv) as spy:
            module.prepare_data()
            module.prepare_data()
            assert spy.call_count == 1

    def test_electricity_prepare_data_runs_io_once(self, electricity_csv_file: Path) -> None:
        """Electricity: prepare_data() twice → pd.read_csv called once."""
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        module = ElectricityLoadModule(
            dataset_file_path=electricity_csv_file,
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
        )

        with patch('pandas.read_csv', wraps=pd.read_csv) as spy:
            module.prepare_data()
            module.prepare_data()
            assert spy.call_count == 1


class TestFinalizePrepareData:
    """Verify slices are set during setup() after cache read.

    _finalize_prepare_data() is a no-op for forecasting modules.
    Slices are computed in setup() after raw data is available.
    """

    @pytest.fixture
    def ett_csv(self, tmp_path: Path) -> Path:
        """Create a minimal ETT-style CSV with 'date' and 'OT' columns."""
        csv_file = tmp_path / 'ett.csv'
        dates = pd.date_range('2016-01-01', periods=100, freq='h')
        df = pd.DataFrame(
            {
                'date': dates,
                'HUFL': np.random.default_rng(42).standard_normal(100),
                'OT': np.random.default_rng(43).standard_normal(100),
            }
        )
        df.to_csv(csv_file, index=False)
        return csv_file

    def test_ett_slices_set_after_setup(self, ett_csv: Path) -> None:
        """ETT: slices are populated by setup() after cache read."""
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=ett_csv,
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
        )
        module.prepare_data()
        module.setup(stage='fit')

        assert module._train_slice is not None
        assert module._valid_slice is not None
        assert module._test_slice is not None

    def test_weather_slices_set_after_setup(self, synthetic_csv_file: Path) -> None:
        """Weather: slices are populated by setup() after cache read."""
        from tscollection.datasets.modules.weather import WeatherModule

        module = WeatherModule(
            dataset_file_path=synthetic_csv_file,
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.MULTIVARIATE,
        )
        module.prepare_data()
        module.setup(stage='fit')

        assert module._train_slice is not None
        assert module._valid_slice is not None
        assert module._test_slice is not None

    def test_electricity_slices_set_after_setup(self, electricity_csv_file: Path) -> None:
        """Electricity: slices are populated by setup() after cache read."""
        from tscollection.datasets.modules.electricity import ElectricityLoadModule

        module = ElectricityLoadModule(
            dataset_file_path=electricity_csv_file,
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
        )
        module.prepare_data()
        module.setup(stage='fit')

        assert module._train_slice is not None
        assert module._valid_slice is not None
        assert module._test_slice is not None


class TestPrepareDimensions:
    """Verify prepare_dimensions() works pre-setup and post-setup.

    Tests the D4 invariant: dimensions computed from _full_data
    (pre-setup) agree with cached values (post-setup).
    """

    @pytest.fixture
    def ett_csv(self, tmp_path: Path) -> Path:
        """Create a minimal ETT-style CSV with 'date' and 'OT' columns."""
        csv_file = tmp_path / 'ett.csv'
        dates = pd.date_range('2016-01-01', periods=100, freq='h')
        df = pd.DataFrame(
            {
                'date': dates,
                'HUFL': np.random.default_rng(42).standard_normal(100),
                'OT': np.random.default_rng(43).standard_normal(100),
            }
        )
        df.to_csv(csv_file, index=False)
        return csv_file

    def test_pre_setup_dimensions(self, ett_csv: Path) -> None:
        """ETT: prepare_dimensions() after prepare_data() alone returns correct dims."""
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=ett_csv,
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
        )
        module.prepare_data()
        n_features, seq_len = module.prepare_dimensions()

        assert n_features is not None
        assert seq_len == 96

    def test_post_setup_dimensions(self, ett_csv: Path) -> None:
        """ETT: prepare_dimensions() after setup() returns cached values."""
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=ett_csv,
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
        )
        module.prepare_data()
        module.setup(stage='fit')
        n_features, seq_len = module.prepare_dimensions()

        assert n_features is not None
        assert seq_len == 96

    def test_pre_setup_matches_post_setup(self, ett_csv: Path) -> None:
        """ETT: dimensions agree whether computed pre-setup or post-setup (D4)."""
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=ett_csv,
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
        )
        module.prepare_data()
        pre_dims = module.prepare_dimensions()

        module.setup(stage='fit')
        post_dims = module.prepare_dimensions()

        assert pre_dims == post_dims


class TestSetupStageGating:
    """Verify stage-specific behavior in setup().

    Tests that setup caches fitted scalers and reuses them across
    stage calls. Tests that validate stage does not mutate data.
    """

    @pytest.fixture
    def ett_csv(self, tmp_path: Path) -> Path:
        """Create a minimal ETT-style CSV with 'date' and 'OT' columns."""
        csv_file = tmp_path / 'ett.csv'
        dates = pd.date_range('2016-01-01', periods=100, freq='h')
        df = pd.DataFrame(
            {
                'date': dates,
                'HUFL': np.random.default_rng(42).standard_normal(100),
                'OT': np.random.default_rng(43).standard_normal(100),
            }
        )
        df.to_csv(csv_file, index=False)
        return csv_file

    def test_fit_populates_scaler_cache(self, ett_csv: Path) -> None:
        """ETT: setup('fit') populates _data_scaler_cache."""
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=ett_csv,
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        module.prepare_data()
        module.setup(stage='fit')

        assert module._data_scaler_cache is not None

    def test_test_stage_reuses_scaler(self, ett_csv: Path) -> None:
        """ETT: setup('fit') then setup('test') reuses same scaler instance."""
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=ett_csv,
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        module.prepare_data()
        module.setup(stage='fit')
        scaler_id = id(module._data_scaler_cache)

        module.setup(stage='test')

        assert id(module._data_scaler_cache) == scaler_id

    def test_validate_does_not_mutate(self, ett_csv: Path) -> None:
        """ETT: setup('validate') does not mutate _train_data_samples."""
        from tscollection.datasets.modules.ett import ETTDataModule

        module = ETTDataModule(
            dataset_file_path=ett_csv,
            variant='ETTh1',
            seq_len=96,
            batch_size=16,
            mode=ForecastingMode.UNIVARIATE,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        module.prepare_data()
        module.setup(stage='fit')
        snapshot = module._train_data_samples.copy()

        module.setup(stage='validate')

        np.testing.assert_array_equal(snapshot, module._train_data_samples)
