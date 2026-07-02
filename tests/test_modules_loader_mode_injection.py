"""TDD tests for loader_mode injection in forecasting and classification datamodules.

Covers init defaults (D-01, D-02, D-09), property getter/setter (D-06, D-07),
None fallback (D-04), call-time override (D-10), classification parameter rename
(D-05), legacy rejection (D-13), and multi-module coverage (D-08, D-11, D-12, D-14).

All tests initially FAIL (RED phase) to drive implementation via TDD.
"""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from chronocratic.datasets.enums.data import (
    ClassificationLoaderMode,
    ForecastingLoaderMode,
)

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


class TestForecastingInitDefaults:
    """Tests for D-01: Forecasting module init sets loader_mode default to RAW_SERIES."""

    def test_ett_default_loader_mode(self, synthetic_ett_csv: Path) -> None:
        """ETTDataModule init sets loader_mode default to ForecastingLoaderMode.RAW_SERIES."""
        from chronocratic.datasets.modules.ett import ETTDataModule

        mod = ETTDataModule(dataset_file_path=synthetic_ett_csv, variant="ETTh1")
        assert mod.loader_mode == ForecastingLoaderMode.RAW_SERIES

    def test_ett_explicit_loader_mode(self, synthetic_ett_csv: Path) -> None:
        """ETTDataModule init accepts explicit loader_mode and stores it."""
        from chronocratic.datasets.modules.ett import ETTDataModule

        mod = ETTDataModule(
            dataset_file_path=synthetic_ett_csv,
            variant="ETTh1",
            loader_mode=ForecastingLoaderMode.INPUT_TARGET,
        )
        assert mod.loader_mode == ForecastingLoaderMode.INPUT_TARGET


class TestClassificationInitDefaults:
    """Tests for D-02: Classification module init sets loader_mode default to SAMPLE_LABEL."""

    def test_ucr_default_loader_mode(self, synthetic_ucr_folder: Path) -> None:
        """UCRClassificationDataModule init sets loader_mode default to SAMPLE_LABEL."""
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        mod = UCRClassificationDataModule(
            dataset_folder_path=synthetic_ucr_folder, target_column_name="class"
        )
        assert mod.loader_mode == ClassificationLoaderMode.SAMPLE_LABEL

    def test_ucr_explicit_loader_mode(self, synthetic_ucr_folder: Path) -> None:
        """UCRClassificationDataModule init accepts explicit loader_mode and stores it."""
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        mod = UCRClassificationDataModule(
            dataset_folder_path=synthetic_ucr_folder,
            target_column_name="class",
            loader_mode=ClassificationLoaderMode.SAMPLE_ONLY,
        )
        assert mod.loader_mode == ClassificationLoaderMode.SAMPLE_ONLY


class TestLoaderModePropertyGetter:
    """Tests for D-06: Base classes expose loader_mode as a property."""

    def test_forecasting_base_has_loader_mode_property(self, synthetic_ett_csv: Path) -> None:
        """BaseForecastingTimeSeriesDataModule has @property loader_mode."""
        from chronocratic.datasets.modules.ett import ETTDataModule

        mod = ETTDataModule(
            dataset_file_path=synthetic_ett_csv,
            variant="ETTh1",
            loader_mode=ForecastingLoaderMode.INPUT_ONLY,
        )
        # Property returns the stored loader_mode value
        assert mod.loader_mode == ForecastingLoaderMode.INPUT_ONLY

    def test_classification_base_has_loader_mode_property(self, synthetic_ucr_folder: Path) -> None:
        """BaseClassificationTimeSeriesDataModule has @property loader_mode."""
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        mod = UCRClassificationDataModule(
            dataset_folder_path=synthetic_ucr_folder,
            target_column_name="class",
            loader_mode=ClassificationLoaderMode.SAMPLE_ONLY,
        )
        assert mod.loader_mode == ClassificationLoaderMode.SAMPLE_ONLY


class TestLoaderModeSetterValidation:
    """Tests for D-07: loader_mode setter validates type and raises TypeError."""

    def test_forecasting_setter_accepts_valid_type(self, synthetic_ett_csv: Path) -> None:
        """Forecasting loader_mode setter accepts valid ForecastingLoaderMode."""
        from chronocratic.datasets.modules._base.forecasting import (
            BaseForecastingTimeSeriesDataModule,
        )
        from chronocratic.datasets.modules.ett import ETTDataModule

        # loader_mode must be a property (descriptor) on the base class
        assert isinstance(
            BaseForecastingTimeSeriesDataModule.__dict__["loader_mode"], property
        )
        mod = ETTDataModule(dataset_file_path=synthetic_ett_csv, variant="ETTh1")
        mod.loader_mode = ForecastingLoaderMode.INPUT_TARGET
        assert mod.loader_mode == ForecastingLoaderMode.INPUT_TARGET

    def test_forecasting_setter_rejects_classification_mode(
        self, synthetic_ett_csv: Path
    ) -> None:
        """Forecasting loader_mode setter raises TypeError for ClassificationLoaderMode."""
        from chronocratic.datasets.modules.ett import ETTDataModule

        mod = ETTDataModule(dataset_file_path=synthetic_ett_csv, variant="ETTh1")
        with pytest.raises(TypeError):
            mod.loader_mode = ClassificationLoaderMode.SAMPLE_LABEL  # type: ignore[assignment]

    def test_classification_setter_accepts_valid_type(self, synthetic_ucr_folder: Path) -> None:
        """Classification loader_mode setter accepts valid ClassificationLoaderMode."""
        from chronocratic.datasets.modules._base.classification import (
            BaseClassificationTimeSeriesDataModule,
        )
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        # loader_mode must be a property (descriptor) on the base class
        assert isinstance(
            BaseClassificationTimeSeriesDataModule.__dict__["loader_mode"], property
        )
        mod = UCRClassificationDataModule(
            dataset_folder_path=synthetic_ucr_folder, target_column_name="class"
        )
        mod.loader_mode = ClassificationLoaderMode.SAMPLE_ONLY
        assert mod.loader_mode == ClassificationLoaderMode.SAMPLE_ONLY

    def test_classification_setter_rejects_forecasting_mode(
        self, synthetic_ucr_folder: Path
    ) -> None:
        """Classification loader_mode setter raises TypeError for ForecastingLoaderMode."""
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        mod = UCRClassificationDataModule(
            dataset_folder_path=synthetic_ucr_folder, target_column_name="class"
        )
        with pytest.raises(TypeError):
            mod.loader_mode = ForecastingLoaderMode.RAW_SERIES  # type: ignore[assignment]

    def test_setter_rejects_string(self, synthetic_ett_csv: Path) -> None:
        """loader_mode setter raises TypeError for non-enum string."""
        from chronocratic.datasets.modules.ett import ETTDataModule

        mod = ETTDataModule(dataset_file_path=synthetic_ett_csv, variant="ETTh1")
        with pytest.raises(TypeError):
            mod.loader_mode = "raw_series"  # type: ignore[assignment]

    def test_setter_rejects_int(self, synthetic_ett_csv: Path) -> None:
        """loader_mode setter raises TypeError for non-enum int."""
        from chronocratic.datasets.modules.ett import ETTDataModule

        mod = ETTDataModule(dataset_file_path=synthetic_ett_csv, variant="ETTh1")
        with pytest.raises(TypeError):
            mod.loader_mode = 42  # type: ignore[assignment]


class TestDataLoaderNoneFallback:
    """Tests for D-04: Dataloader method with loader_mode=None falls back to self.loader_mode."""

    def test_forecasting_train_dataloader_none_fallback(
        self, synthetic_ett_csv: Path
    ) -> None:
        """Forecasting train_dataloader with loader_mode=None uses self.loader_mode."""
        from chronocratic.datasets.modules.ett import ETTDataModule

        mod = ETTDataModule(
            dataset_file_path=synthetic_ett_csv,
            variant="ETTh1",
            loader_mode=ForecastingLoaderMode.RAW_SERIES,
        )
        mod.prepare_data()
        mod.setup(stage="fit")
        # Calling with loader_mode=None should fall back to self.loader_mode
        dl = mod.train_dataloader(loader_mode=None)  # type: ignore[arg-type]
        assert isinstance(dl, DataLoader)

    def test_classification_train_dataloader_none_fallback(
        self, synthetic_ucr_folder: Path
    ) -> None:
        """Classification train_dataloader with loader_mode=None uses self.loader_mode."""
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        mod = UCRClassificationDataModule(
            dataset_folder_path=synthetic_ucr_folder,
            target_column_name="class",
            loader_mode=ClassificationLoaderMode.SAMPLE_LABEL,
        )
        mod.prepare_data()
        mod.setup(stage="fit")
        dl = mod.train_dataloader(loader_mode=None)  # type: ignore[arg-type]
        assert isinstance(dl, DataLoader)

    def test_forecasting_val_dataloader_none_fallback(
        self, synthetic_ett_csv: Path
    ) -> None:
        """Forecasting val_dataloader with loader_mode=None falls back to self.loader_mode."""
        from chronocratic.datasets.modules.ett import ETTDataModule

        mod = ETTDataModule(
            dataset_file_path=synthetic_ett_csv,
            variant="ETTh1",
            loader_mode=ForecastingLoaderMode.RAW_SERIES,
        )
        mod.prepare_data()
        mod.setup(stage="fit")
        dl = mod.val_dataloader(loader_mode=None)  # type: ignore[arg-type]
        # May be None if valid_size=0; otherwise a DataLoader
        assert dl is None or isinstance(dl, DataLoader)

    def test_forecasting_test_dataloader_none_fallback(
        self, synthetic_ett_csv: Path
    ) -> None:
        """Forecasting test_dataloader with loader_mode=None falls back to self.loader_mode."""
        from chronocratic.datasets.modules.ett import ETTDataModule

        mod = ETTDataModule(
            dataset_file_path=synthetic_ett_csv,
            variant="ETTh1",
            loader_mode=ForecastingLoaderMode.RAW_SERIES,
        )
        mod.prepare_data()
        mod.setup(stage="fit")
        dl = mod.test_dataloader(loader_mode=None)  # type: ignore[arg-type]
        assert isinstance(dl, DataLoader)

    def test_classification_val_dataloader_none_fallback(
        self, synthetic_ucr_folder: Path
    ) -> None:
        """Classification val_dataloader with loader_mode=None falls back to self.loader_mode."""
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        mod = UCRClassificationDataModule(
            dataset_folder_path=synthetic_ucr_folder,
            target_column_name="class",
            loader_mode=ClassificationLoaderMode.SAMPLE_LABEL,
        )
        mod.prepare_data()
        mod.setup(stage="fit")
        dl = mod.val_dataloader(loader_mode=None)  # type: ignore[arg-type]
        assert dl is None or isinstance(dl, DataLoader)


class TestDataLoaderExplicitOverride:
    """Tests for D-10: Dataloader method with explicit loader_mode overrides self.loader_mode."""

    def test_forecasting_train_dataloader_explicit_override(
        self, synthetic_ett_csv: Path
    ) -> None:
        """Forecasting train_dataloader with explicit loader_mode overrides self.loader_mode."""
        from chronocratic.datasets.modules.ett import ETTDataModule

        mod = ETTDataModule(
            dataset_file_path=synthetic_ett_csv,
            variant="ETTh1",
            loader_mode=ForecastingLoaderMode.INPUT_TARGET,
            forecast_horizon=96,
        )
        mod.prepare_data()
        mod.setup(stage="fit")
        # Override with RAW_SERIES at call time
        dl = mod.train_dataloader(loader_mode=ForecastingLoaderMode.RAW_SERIES)
        assert isinstance(dl, DataLoader)

    def test_classification_train_dataloader_explicit_override(
        self, synthetic_ucr_folder: Path
    ) -> None:
        """Classification train_dataloader with explicit loader_mode overrides self.loader_mode."""
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        mod = UCRClassificationDataModule(
            dataset_folder_path=synthetic_ucr_folder,
            target_column_name="class",
            loader_mode=ClassificationLoaderMode.SAMPLE_LABEL,
        )
        mod.prepare_data()
        mod.setup(stage="fit")
        # Override with SAMPLE_ONLY at call time
        dl = mod.train_dataloader(loader_mode=ClassificationLoaderMode.SAMPLE_ONLY)
        assert isinstance(dl, DataLoader)


class TestClassificationParamRename:
    """Tests for D-05, D-13: Classification dataloaders use loader_mode (not mode)."""

    def test_ucr_accepts_loader_mode_kwarg(self) -> None:
        """UCR train_dataloader accepts loader_mode= keyword (not mode=)."""
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        sig = inspect.signature(UCRClassificationDataModule.train_dataloader)
        params = sig.parameters
        assert "loader_mode" in params, (
            f"train_dataloader missing 'loader_mode' param; found: {list(params.keys())}"
        )

    def test_uea_accepts_loader_mode_kwarg(self) -> None:
        """UEA train_dataloader accepts loader_mode= keyword (not mode=)."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        sig = inspect.signature(UEAClassificationDataModule.train_dataloader)
        params = sig.parameters
        assert "loader_mode" in params, (
            f"train_dataloader missing 'loader_mode' param; found: {list(params.keys())}"
        )

    def test_ucr_rejects_mode_kwarg(self) -> None:
        """UCR train_dataloader rejects mode= keyword (raises TypeError)."""
        from chronocratic.datasets.modules.ucr import UCRClassificationDataModule

        # Verify the signature has no 'mode' parameter (only 'loader_mode')
        sig = inspect.signature(UCRClassificationDataModule.train_dataloader)
        params = sig.parameters
        assert "mode" not in params, (
            f"train_dataloader still has 'mode' param; found: {list(params.keys())}"
        )

    def test_uea_rejects_mode_kwarg(self) -> None:
        """UEA train_dataloader rejects mode= keyword (raises TypeError)."""
        from chronocratic.datasets.modules.uea import UEAClassificationDataModule

        # Verify the signature has no 'mode' parameter (only 'loader_mode')
        sig = inspect.signature(UEAClassificationDataModule.train_dataloader)
        params = sig.parameters
        assert "mode" not in params, (
            f"train_dataloader still has 'mode' param; found: {list(params.keys())}"
        )


class TestMultiModuleDefaults:
    """Tests for D-09, D-11, D-12: All concrete modules set correct loader_mode defaults."""

    def test_electricity_default_loader_mode(self, tmp_path: Path) -> None:
        """ElectricityLoadDataModule init default loader_mode is RAW_SERIES."""
        from chronocratic.datasets.modules.electricity import ElectricityLoadDataModule

        # Create a minimal CSV for Electricity (semicolon-delimited, comma decimal)
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

        mod = ElectricityLoadDataModule(dataset_file_path=csv_path)
        assert mod.loader_mode == ForecastingLoaderMode.RAW_SERIES

    def test_weather_default_loader_mode(self, tmp_path: Path) -> None:
        """WeatherDataModule init default loader_mode is RAW_SERIES."""
        from chronocratic.datasets.modules.weather import WeatherDataModule

        # Create a minimal CSV for Weather
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

        mod = WeatherDataModule(dataset_file_path=csv_path)
        assert mod.loader_mode == ForecastingLoaderMode.RAW_SERIES
