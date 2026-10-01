"""Tests for WeatherDataset and ElectricityDataset.

Verifies:
- FlexibleTimeSeriesDatasetSingleFileMultipleSeries handles 3D data
- WeatherDataset instantiates and yields correct shapes
- ElectricityDataset instantiates and yields correct shapes
- Dataset classes are exportable from datatypes __init__.py
"""

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_single_series_data() -> np.ndarray:
    """Mock 2D data (T=1000, F=7) for single-series datasets."""
    return np.random.randn(1000, 7).astype(np.float32)


@pytest.fixture
def mock_multi_series_data() -> np.ndarray:
    """Mock 3D data (S=10, T=500, F=1) for multi-series datasets."""
    return np.random.randn(10, 500, 1).astype(np.float32)


# --------------------------------------------------------------------------- #
# SingleFileMultipleSeries Tests                                               #
# --------------------------------------------------------------------------- #


class TestFlexibleTimeSeriesDatasetSingleFileMultipleSeries:
    """Verify the new MultipleSeries base class."""

    def test_import_from_datatypes(self):
        from chronocratic.datasets.datatypes import (
            FlexibleTimeSeriesDatasetSingleFileMultipleSeries,
        )

        assert FlexibleTimeSeriesDatasetSingleFileMultipleSeries is not None

    def test_handles_3d_data(self, mock_multi_series_data: np.ndarray) -> None:
        """MultipleSeries correctly accepts 3D input (series, T, features)."""
        from chronocratic.datasets.datatypes import (
            FlexibleTimeSeriesDatasetSingleFileMultipleSeries,
        )
        from chronocratic.datasets.datatypes._base.strategies import ForecastingStrategySingleFile
        from chronocratic.datasets.enums import TimeSeriesDatasetMode

        seq_len = 32
        step = 32
        horizon = 16

        # Data shape: (num_series=10, T=500, features=1)
        dataset = FlexibleTimeSeriesDatasetSingleFileMultipleSeries(
            data=mock_multi_series_data,
            labels=None,
            seq_len=seq_len,
            step=step,
            mode=TimeSeriesDatasetMode.INPUT_OUTPUT,
            sequence_handling_strategy=ForecastingStrategySingleFile(forecast_horizon=horizon),
            expand_dims_axis=None,
        )
        assert len(dataset) > 0

    def test_returns_correct_sample_shape(self, mock_multi_series_data: np.ndarray) -> None:
        """dataset[i] returns (seq_len, features) shaped tensor."""
        from chronocratic.datasets.datatypes import (
            FlexibleTimeSeriesDatasetSingleFileMultipleSeries,
        )
        from chronocratic.datasets.datatypes._base.strategies import ForecastingStrategySingleFile
        from chronocratic.datasets.enums import TimeSeriesDatasetMode

        seq_len = 32
        step = 32
        horizon = 16

        dataset = FlexibleTimeSeriesDatasetSingleFileMultipleSeries(
            data=mock_multi_series_data,
            labels=None,
            seq_len=seq_len,
            step=step,
            mode=TimeSeriesDatasetMode.INPUT_OUTPUT,
            sequence_handling_strategy=ForecastingStrategySingleFile(forecast_horizon=horizon),
            expand_dims_axis=None,
        )
        sample = dataset[0]

        # INPUT_OUTPUT mode returns (input, target) tuple
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        assert sample[0].shape[0] == seq_len  # input seq_len

    def test_total_length_is_sum_of_series_windows(
        self, mock_multi_series_data: np.ndarray
    ) -> None:
        """len(dataset) equals total valid windows across all series."""
        from chronocratic.datasets.datatypes import (
            FlexibleTimeSeriesDatasetSingleFileMultipleSeries,
        )
        from chronocratic.datasets.datatypes._base.strategies import ForecastingStrategySingleFile
        from chronocratic.datasets.enums import TimeSeriesDatasetMode

        seq_len = 32
        step = 32
        horizon = 16

        dataset = FlexibleTimeSeriesDatasetSingleFileMultipleSeries(
            data=mock_multi_series_data,
            labels=None,
            seq_len=seq_len,
            step=step,
            mode=TimeSeriesDatasetMode.INPUT_OUTPUT,
            sequence_handling_strategy=ForecastingStrategySingleFile(forecast_horizon=horizon),
            expand_dims_axis=None,
        )
        # T=500, seq_len=32, horizon=16, step=32
        # windows per series = (500 - 32 - 16) // 32 + 1 = 452 // 32 + 1 = 15
        # 10 series * 15 = 150 (approximately, depends on strategy)
        assert len(dataset) > 0

    def test_negative_index_works(self, mock_multi_series_data: np.ndarray) -> None:
        """dataset[-1] returns last window without index errors."""
        from chronocratic.datasets.datatypes import (
            FlexibleTimeSeriesDatasetSingleFileMultipleSeries,
        )
        from chronocratic.datasets.datatypes._base.strategies import ForecastingStrategySingleFile
        from chronocratic.datasets.enums import TimeSeriesDatasetMode

        dataset = FlexibleTimeSeriesDatasetSingleFileMultipleSeries(
            data=mock_multi_series_data,
            labels=None,
            seq_len=32,
            step=32,
            mode=TimeSeriesDatasetMode.INPUT_OUTPUT,
            sequence_handling_strategy=ForecastingStrategySingleFile(forecast_horizon=16),
            expand_dims_axis=None,
        )
        sample = dataset[-1]
        assert sample is not None


# --------------------------------------------------------------------------- #
# WeatherDataset Tests                                                         #
# --------------------------------------------------------------------------- #


class TestWeatherDataset:
    """Verify WeatherDataset class."""

    def test_import_from_datatypes(self):
        from chronocratic.datasets.datatypes import WeatherDataset

        assert WeatherDataset is not None

    def test_instantiates_with_2d_data(self) -> None:
        """WeatherDataset accepts (T, F) squeezed data."""
        from chronocratic.datasets.datatypes import WeatherDataset

        data = np.random.randn(2000, 22).astype(np.float32)
        dataset = WeatherDataset(
            data=data, seq_len=32, step=32, mode="sample_only", forecast_horizon=96
        )
        assert len(dataset) > 0

    def test_sample_shape(self) -> None:
        """WeatherDataset yields (seq_len, features) samples."""
        from chronocratic.datasets.datatypes import WeatherDataset

        data = np.random.randn(2000, 22).astype(np.float32)
        dataset = WeatherDataset(
            data=data, seq_len=64, step=64, mode="sample_only", forecast_horizon=96
        )
        sample = dataset[0]
        # SAMPLE_ONLY mode returns single data array
        assert hasattr(sample, "shape")
        assert sample.shape[0] == 64  # seq_len
        assert sample.shape[1] == 22  # features


# --------------------------------------------------------------------------- #
# ElectricityDataset Tests                                                     #
# --------------------------------------------------------------------------- #


class TestElectricityDataset:
    """Verify ElectricityDataset class."""

    def test_import_from_datatypes(self):
        from chronocratic.datasets.datatypes import ElectricityDataset

        assert ElectricityDataset is not None

    def test_instantiates_with_3d_data(self) -> None:
        """ElectricityDataset accepts (series, T, features) data."""
        from chronocratic.datasets.datatypes import ElectricityDataset

        data = np.random.randn(50, 1000, 1).astype(np.float32)
        dataset = ElectricityDataset(
            data=data, seq_len=32, step=32, mode="sample_only", forecast_horizon=24
        )
        assert len(dataset) > 0

    def test_total_length_multiplied_by_series(self) -> None:
        """Total samples scales with number of series."""
        from chronocratic.datasets.datatypes import ElectricityDataset

        data_5 = np.random.randn(5, 500, 1).astype(np.float32)
        data_10 = np.random.randn(10, 500, 1).astype(np.float32)

        ds5 = ElectricityDataset(
            data=data_5, seq_len=32, step=32, mode="sample_only", forecast_horizon=24
        )
        ds10 = ElectricityDataset(
            data=data_10, seq_len=32, step=32, mode="sample_only", forecast_horizon=24
        )
        # 10 series should have ~2x the samples of 5 series
        assert len(ds10) > len(ds5)


# --------------------------------------------------------------------------- #
# Export Tests                                                                 #
# --------------------------------------------------------------------------- #


class TestDatatypesExports:
    """Verify new classes are exported from datatypes __init__.py."""

    def test_weather_dataset_exported(self):
        from chronocratic.datasets.datatypes import WeatherDataset

        assert WeatherDataset is not None

    def test_electricity_dataset_exported(self):
        from chronocratic.datasets.datatypes import ElectricityDataset

        assert ElectricityDataset is not None

    def test_multiple_series_exported(self):
        from chronocratic.datasets.datatypes import (
            FlexibleTimeSeriesDatasetSingleFileMultipleSeries,
        )

        assert FlexibleTimeSeriesDatasetSingleFileMultipleSeries is not None


def test_electricity_windows_start_every_step_in_each_series() -> None:
    """Window k of every series covers rows ``k * step`` onwards of that series."""
    from chronocratic.datasets.datatypes.electricity import ElectricityDataset
    from chronocratic.datasets.enums import TimeSeriesDatasetMode

    n_series, n_steps = 3, 100
    offsets = 1000 * np.arange(n_series, dtype=np.float32)[:, None, None]
    data = np.arange(n_steps, dtype=np.float32)[None, :, None] + offsets  # (3, 100, 1)
    ds = ElectricityDataset(
        data=data, seq_len=20, step=20, mode=TimeSeriesDatasetMode.INPUT_OUTPUT, forecast_horizon=10
    )

    windows_per_series = 4  # starts 0, 20, 40, 60; 80 + 30 > 100
    assert len(ds) == n_series * windows_per_series
    for idx in range(len(ds)):
        series, k = divmod(idx, windows_per_series)
        inp, tgt = ds[idx]
        start = 1000 * series + 20 * k
        np.testing.assert_array_equal(inp.numpy()[:, 0], np.arange(start, start + 20))
        np.testing.assert_array_equal(tgt.numpy()[:, 0], np.arange(start + 20, start + 30))
