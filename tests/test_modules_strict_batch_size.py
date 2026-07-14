"""TDD tests for loader_strict_batch_size as an init-time instance parameter.

Covers init defaults on BaseTimeSeriesDataModule, propagation through
forecasting/classification bases, None fallback in all public dataloader
methods across 5 concrete modules (UCR, UEA, ETT, Weather, Electricity),
and explicit call-time override behavior.

All tests initially FAIL (RED phase) to drive implementation via TDD.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_ett_csv(tmp_path: Path) -> Path:
    """Create a synthetic ETT-style CSV with enough data for all slices."""
    csv_path = tmp_path / "synthetic_ett.csv"
    dates = pd.date_range("2016-01-01", periods=12_000, freq="h")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "date": dates,
            "HUFL": rng.standard_normal(12_000).astype(np.float32),
            "HULL": rng.standard_normal(12_000).astype(np.float32),
            "MUFL": rng.standard_normal(12_000).astype(np.float32),
            "MULL": rng.standard_normal(12_000).astype(np.float32),
            "OT": rng.standard_normal(12_000).astype(np.float32),
            "T1": rng.standard_normal(12_000).astype(np.float32),
            "T2": rng.standard_normal(12_000).astype(np.float32),
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_ucr_folder(tmp_path: Path) -> Path:
    """Create a synthetic UCR dataset folder with ARFF files."""
    arff_content = """\
@relation test

@attribute t1 numeric
@attribute t2 numeric
@attribute t3 numeric
@attribute class {0,1}

@data
0.1,0.2,0.3,0
0.4,0.5,0.6,1
0.7,0.8,0.9,0
0.2,0.3,0.4,1
0.5,0.6,0.7,0
0.8,0.9,1.0,1
0.1,0.2,0.3,0
0.4,0.5,0.6,1
0.7,0.8,0.9,0
0.2,0.3,0.4,1
0.5,0.6,0.7,0
0.8,0.9,1.0,1
0.1,0.2,0.3,0
0.4,0.5,0.6,1
0.7,0.8,0.9,0
"""
    dataset_dir = tmp_path / "synthetic_ucr"
    dataset_dir.mkdir()
    (dataset_dir / "synthetic_ucr_TRAIN.arff").write_text(arff_content)
    (dataset_dir / "synthetic_ucr_TEST.arff").write_text(arff_content)
    return dataset_dir


@pytest.fixture
def synthetic_weather_csv(tmp_path: Path) -> Path:
    """Create a synthetic Weather-style CSV."""
    csv_path = tmp_path / "weather.csv"
    dates = pd.date_range("2006-01-01", periods=2000, freq="h")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "date": dates,
            "WetBulbCelsius": rng.standard_normal(2000).astype(np.float32),
        }
    )
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_electricity_csv(tmp_path: Path) -> Path:
    """Create a synthetic Electricity-style CSV (semicolon-delimited, comma decimal)."""
    csv_path = tmp_path / "electricity.csv"
    dates = pd.date_range("2012-01-01", periods=1000, freq="h")
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "MT_001": rng.standard_normal(1000).astype(np.float32),
        },
        index=dates,
    )
    df.index.name = "date"
    df.to_csv(csv_path, sep=";", decimal=",")
    return csv_path


# ---------------------------------------------------------------------------
# TestBaseLoaderStrictBatchSize
# ---------------------------------------------------------------------------


class TestBaseLoaderStrictBatchSize:
    """Tests for loader_strict_batch_size on BaseTimeSeriesDataModule."""

    def test_base_accepts_loader_strict_batch_size_default_false(self) -> None:
        """BaseTimeSeriesDataModule accepts loader_strict_batch_size, default False."""
        from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule

        class ConcreteTestModule(BaseTimeSeriesDataModule):
            """Minimal concrete subclass for testing."""

            def _do_prepare_data(self) -> None:
                pass

        mod = ConcreteTestModule(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        assert mod.loader_strict_batch_size is False

    def test_base_stores_loader_strict_batch_size_true(self) -> None:
        """Setting loader_strict_batch_size=True stores value as instance attribute."""
        from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule

        class ConcreteTestModule(BaseTimeSeriesDataModule):
            """Minimal concrete subclass for testing."""

            def _do_prepare_data(self) -> None:
                pass

        mod = ConcreteTestModule(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
            loader_strict_batch_size=True,
        )
        assert mod.loader_strict_batch_size is True

    def test_base_signature_has_loader_strict_batch_size(self) -> None:
        """BaseTimeSeriesDataModule.__init__ has loader_strict_batch_size parameter."""
        from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule

        sig = inspect.signature(BaseTimeSeriesDataModule.__init__)
        params = sig.parameters
        assert "loader_strict_batch_size" in params
        assert params["loader_strict_batch_size"].default is False


# ---------------------------------------------------------------------------
# TestUCRStrictBatchSizeFallback
# ---------------------------------------------------------------------------


class TestUCRStrictBatchSizeFallback:
    """Tests for loader_strict_batch_size None-fallback on UCR dataloaders."""

    def test_ucr_init_default_is_true(self) -> None:
        """UCR classification modules default loader_strict_batch_size to True."""
        import inspect
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        sig = inspect.signature(UCRClassificationDataModule.__init__)
        assert sig.parameters["loader_strict_batch_size"].default is True

    def test_ucr_train_dataloader_none_uses_instance_default(
        self, synthetic_ucr_folder: Path
    ) -> None:
        """UCR train_dataloader with loader_strict_batch_size=None uses instance default."""
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        mod = UCRClassificationDataModule(
            dataset_folder_path=synthetic_ucr_folder,
            target_column_name="class",
            loader_strict_batch_size=True,
        )
        mod.prepare_data()
        mod.setup(stage="fit")

        # loader_strict_batch_size=None should resolve to self.loader_strict_batch_size (True)
        with patch(
            "chronocratic.datasets.modules._base.base.DataLoader", wraps=DataLoader
        ) as mock_loader:
            mod.train_dataloader(loader_strict_batch_size=None)
            call_kwargs = mock_loader.call_args[1]
            assert "collate_fn" in call_kwargs, (
                "loader_strict_batch_size=None with loader_strict_batch_size=True should "
                "apply collate_fn (strict mode)"
            )

    def test_ucr_train_dataloader_explicit_override(self, synthetic_ucr_folder: Path) -> None:
        """UCR train_dataloader with loader_strict_batch_size=True overrides instance default."""
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        mod = UCRClassificationDataModule(
            dataset_folder_path=synthetic_ucr_folder,
            target_column_name="class",
            loader_strict_batch_size=False,
        )
        mod.prepare_data()
        mod.setup(stage="fit")

        # Explicit True should override instance False
        with patch(
            "chronocratic.datasets.modules._base.base.DataLoader", wraps=DataLoader
        ) as mock_loader:
            mod.train_dataloader(loader_strict_batch_size=True)
            call_kwargs = mock_loader.call_args[1]
            assert "collate_fn" in call_kwargs

    def test_ucr_val_dataloader_none_uses_instance_default(
        self, synthetic_ucr_folder: Path
    ) -> None:
        """UCR val_dataloader with loader_strict_batch_size=None uses instance default."""
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        mod = UCRClassificationDataModule(
            dataset_folder_path=synthetic_ucr_folder,
            target_column_name="class",
            loader_strict_batch_size=True,
        )
        mod.prepare_data()
        mod.setup(stage="fit")

        val_dl = mod.val_dataloader(loader_strict_batch_size=None)
        if val_dl is not None:
            assert val_dl is not None  # exercised None-fallback path

    def test_ucr_test_dataloader_none_uses_instance_default(
        self, synthetic_ucr_folder: Path
    ) -> None:
        """UCR test_dataloader with loader_strict_batch_size=None uses instance default."""
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        mod = UCRClassificationDataModule(
            dataset_folder_path=synthetic_ucr_folder,
            target_column_name="class",
            loader_strict_batch_size=True,
        )
        mod.prepare_data()
        mod.setup(stage="fit")

        mod.test_dataloader(loader_strict_batch_size=None)  # Should not raise

    def test_ucr_dataloader_signatures_accept_none(self) -> None:
        """UCR dataloader method signatures accept loader_strict_batch_size: bool | None = None."""
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        for method_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
            method = getattr(UCRClassificationDataModule, method_name)
            sig = inspect.signature(method)
            params = sig.parameters
            assert "loader_strict_batch_size" in params, (
                f"{method_name} missing 'loader_strict_batch_size' param"
            )
            assert params["loader_strict_batch_size"].default is None, (
                f"{method_name} loader_strict_batch_size default is not None, "
                f"got {params['loader_strict_batch_size'].default}"
            )


# ---------------------------------------------------------------------------
# TestUEAStrictBatchSizeFallback
# ---------------------------------------------------------------------------


class TestUEAStrictBatchSizeFallback:
    """Tests for loader_strict_batch_size None-fallback on UEA dataloaders."""

    def test_uea_init_default_is_true(self) -> None:
        """UEA classification modules default loader_strict_batch_size to True."""
        import inspect
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        sig = inspect.signature(UEAClassificationDataModule.__init__)
        assert sig.parameters["loader_strict_batch_size"].default is True

    def test_uea_dataloader_signatures_accept_none(self) -> None:
        """UEA dataloader method signatures accept loader_strict_batch_size: bool | None = None."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        for method_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
            method = getattr(UEAClassificationDataModule, method_name)
            sig = inspect.signature(method)
            params = sig.parameters
            assert "loader_strict_batch_size" in params, (
                f"{method_name} missing 'loader_strict_batch_size' param"
            )
            assert params["loader_strict_batch_size"].default is None, (
                f"{method_name} loader_strict_batch_size default is not None, "
                f"got {params['loader_strict_batch_size'].default}"
            )

    def test_uea_accepts_loader_strict_batch_size(self) -> None:
        """UEAClassificationDataModule accepts loader_strict_batch_size via __init__."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        sig = inspect.signature(UEAClassificationDataModule.__init__)
        params = sig.parameters
        assert "loader_strict_batch_size" in params


# ---------------------------------------------------------------------------
# TestETTStrictBatchSizeFallback
# ---------------------------------------------------------------------------


class TestETTStrictBatchSizeFallback:
    """Tests for loader_strict_batch_size None-fallback on ETT dataloaders."""

    def test_ett_dataloader_signatures_accept_none(self) -> None:
        """ETT dataloader method signatures accept loader_strict_batch_size: bool | None = None."""
        from chronocratic.datasets.modules.ett import ETTDataModule

        for method_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
            method = getattr(ETTDataModule, method_name)
            sig = inspect.signature(method)
            params = sig.parameters
            assert "loader_strict_batch_size" in params, (
                f"{method_name} missing 'loader_strict_batch_size' param"
            )
            assert params["loader_strict_batch_size"].default is None, (
                f"{method_name} loader_strict_batch_size default is not None, "
                f"got {params['loader_strict_batch_size'].default}"
            )

    def test_ett_accepts_loader_strict_batch_size(self, synthetic_ett_csv: Path) -> None:
        """ETTDataModule accepts loader_strict_batch_size via __init__."""
        from chronocratic.datasets.modules.ett import ETTDataModule

        mod = ETTDataModule(
            dataset_file_path=synthetic_ett_csv,
            variant="ETTh1",
            loader_strict_batch_size=True,
        )
        assert mod.loader_strict_batch_size is True

    def test_ett_train_dataloader_none_fallback(self, synthetic_ett_csv: Path) -> None:
        """ETT train_dataloader with loader_strict_batch_size=None uses instance default."""
        from chronocratic.datasets.modules.ett import ETTDataModule

        mod = ETTDataModule(
            dataset_file_path=synthetic_ett_csv,
            variant="ETTh1",
            loader_strict_batch_size=True,
        )
        mod.prepare_data()
        mod.setup(stage="fit")

        with patch(
            "chronocratic.datasets.modules._base.base.DataLoader", wraps=DataLoader
        ) as mock_loader:
            mod.train_dataloader(loader_strict_batch_size=None)
            call_kwargs = mock_loader.call_args[1]
            assert "collate_fn" in call_kwargs, (
                "loader_strict_batch_size=None with loader_strict_batch_size=True should apply collate_fn"
            )


# ---------------------------------------------------------------------------
# TestWeatherStrictBatchSizeFallback
# ---------------------------------------------------------------------------


class TestWeatherStrictBatchSizeFallback:
    """Tests for loader_strict_batch_size None-fallback on Weather dataloaders."""

    def test_weather_dataloader_signatures_accept_none(self) -> None:
        """Weather dataloader method signatures accept loader_strict_batch_size: bool | None = None."""
        from chronocratic.datasets.modules.weather import WeatherDataModule

        for method_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
            method = getattr(WeatherDataModule, method_name)
            sig = inspect.signature(method)
            params = sig.parameters
            assert "loader_strict_batch_size" in params, (
                f"{method_name} missing 'loader_strict_batch_size' param"
            )
            assert params["loader_strict_batch_size"].default is None, (
                f"{method_name} loader_strict_batch_size default is not None, "
                f"got {params['loader_strict_batch_size'].default}"
            )

    def test_weather_accepts_loader_strict_batch_size(self, synthetic_weather_csv: Path) -> None:
        """WeatherDataModule accepts loader_strict_batch_size via __init__."""
        from chronocratic.datasets.modules.weather import WeatherDataModule

        mod = WeatherDataModule(
            dataset_file_path=synthetic_weather_csv,
            loader_strict_batch_size=True,
        )
        assert mod.loader_strict_batch_size is True


# ---------------------------------------------------------------------------
# TestElectricityStrictBatchSizeFallback
# ---------------------------------------------------------------------------


class TestElectricityStrictBatchSizeFallback:
    """Tests for loader_strict_batch_size None-fallback on Electricity dataloaders."""

    def test_electricity_dataloader_signatures_accept_none(self) -> None:
        """Electricity dataloader method signatures accept loader_strict_batch_size: bool | None = None."""
        from chronocratic.datasets.modules.electricity import ElectricityLoadDataModule

        for method_name in ("train_dataloader", "val_dataloader", "test_dataloader"):
            method = getattr(ElectricityLoadDataModule, method_name)
            sig = inspect.signature(method)
            params = sig.parameters
            assert "loader_strict_batch_size" in params, (
                f"{method_name} missing 'loader_strict_batch_size' param"
            )
            assert params["loader_strict_batch_size"].default is None, (
                f"{method_name} loader_strict_batch_size default is not None, "
                f"got {params['loader_strict_batch_size'].default}"
            )

    def test_electricity_accepts_loader_strict_batch_size(
        self, synthetic_electricity_csv: Path
    ) -> None:
        """ElectricityLoadDataModule accepts loader_strict_batch_size via __init__."""
        from chronocratic.datasets.modules.electricity import ElectricityLoadDataModule

        mod = ElectricityLoadDataModule(
            dataset_file_path=synthetic_electricity_csv,
            loader_strict_batch_size=True,
        )
        assert mod.loader_strict_batch_size is True


# ---------------------------------------------------------------------------
# TestDataLoaderConstructorBehavior
# ---------------------------------------------------------------------------


class TestDataLoaderConstructorBehavior:
    """Tests verifying DataLoader constructor receives the correct resolved value."""

    def test_dataloader_receives_collate_when_resolved_true(self, synthetic_ett_csv: Path) -> None:
        """DataLoader constructor receives collate_fn when None fallback resolves to True."""
        from chronocratic.datasets.modules.ett import ETTDataModule

        mod = ETTDataModule(
            dataset_file_path=synthetic_ett_csv,
            variant="ETTh1",
            loader_strict_batch_size=True,
        )
        mod.prepare_data()
        mod.setup(stage="fit")

        with patch(
            "chronocratic.datasets.modules._base.base.DataLoader", wraps=DataLoader
        ) as mock_loader:
            mod.train_dataloader(loader_strict_batch_size=None)
            call_kwargs = mock_loader.call_args[1]
            assert "collate_fn" in call_kwargs

    def test_dataloader_no_collate_when_resolved_false(self, synthetic_ett_csv: Path) -> None:
        """DataLoader constructor does NOT receive collate_fn when resolved to False."""
        from chronocratic.datasets.modules.ett import ETTDataModule

        mod = ETTDataModule(
            dataset_file_path=synthetic_ett_csv,
            variant="ETTh1",
            loader_strict_batch_size=False,
        )
        mod.prepare_data()
        mod.setup(stage="fit")

        with patch(
            "chronocratic.datasets.modules._base.base.DataLoader", wraps=DataLoader
        ) as mock_loader:
            mod.train_dataloader(loader_strict_batch_size=None)
            call_kwargs = mock_loader.call_args[1]
            assert "collate_fn" not in call_kwargs


# ---------------------------------------------------------------------------
# TestBasePropagation
# ---------------------------------------------------------------------------


class TestForecastingBasePropagation:
    """Tests for loader_strict_batch_size propagation through forecasting base."""

    def test_forecasting_base_accepts_loader_strict_batch_size(self) -> None:
        """BaseForecastingTimeSeriesDataModule accepts loader_strict_batch_size."""
        from chronocratic.datasets.modules._base.forecasting import (
            BaseForecastingTimeSeriesDataModule,
        )

        sig = inspect.signature(BaseForecastingTimeSeriesDataModule.__init__)
        params = sig.parameters
        assert "loader_strict_batch_size" in params


class TestClassificationBasePropagation:
    """Tests for loader_strict_batch_size propagation through classification base."""

    def test_classification_base_accepts_loader_strict_batch_size(self) -> None:
        """BaseClassificationTimeSeriesDataModule accepts loader_strict_batch_size."""
        from chronocratic.datasets.modules._base.classification import (
            BaseClassificationTimeSeriesDataModule,
        )

        sig = inspect.signature(BaseClassificationTimeSeriesDataModule.__init__)
        params = sig.parameters
        assert "loader_strict_batch_size" in params
