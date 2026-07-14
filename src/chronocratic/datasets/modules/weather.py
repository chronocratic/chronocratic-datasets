"""Weather forecasting LightningDataModule.

Reads the 7-year weather CSV and splits 60/20/20.

Uses TensorDataset for dataloaders.
Raises FileNotFoundError for missing paths.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from chronocratic.datasets.enums.data import (
    DataPartition,
    ForecastingLoaderMode,
    ForecastingMode,
    ScalingMethod,
    TimeSeriesDatasetMode,
)
from chronocratic.datasets.modules._base.forecasting import BaseForecastingTimeSeriesDataModule
from chronocratic.datasets.utils.cache import (
    atomic_save_metadata,
    atomic_save_npz,
    build_cache_key,
    CACHE_SCHEMA_VERSION,
)
from chronocratic.datasets.utils.features import TIME_FEATURE_COUNT

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["WeatherDataModule"]


class WeatherDataModule(BaseForecastingTimeSeriesDataModule):
    """LightningDataModule for weather forecasting.

    Reads CSV with standard format (comma-separated, period decimals),
    applies 60/20/20 fractional splits.

    .. rubric:: Data shape reference

    Weather is a single multivariate time series with 22 features.

    ==================  =================  ==================  ==================
    Dataset             Raw CSV Shape      Post-Transform      Notes
    ==================  =================  ==================  ==================
    Weather             (52696, 22)        (1, 52696, 22)      Hourly, 7 years
    ==================  =================  ==================  ==================

    For univariate mode, only the last column (``WetBulbCelsius``) is
    retained. The data transform uses ``expand_dims(axis=0)``, producing
    shape ``(1, T, F)``.

    Args:
        dataset_file_path: Path to the CSV file.
        seq_len: Input window length.
        mode: UNIVARIATE or MULTIVARIATE.
        batch_size: Batch size.
        valid_size: Validation fraction (unused, fixed 60/20/20).
        test_size: Test fraction (unused, fixed 60/20/20).
        shuffle: Whether to shuffle training data.
        scale_data: Whether to scale features.
        data_scaling_method: Scaling algorithm.
        data_scaling_range: Target min-max range.
        num_workers: DataLoader worker count.
        loader_mode: Per-init mode controlling dataloader output format.
            Defaults to ``ForecastingLoaderMode.RAW_SERIES``.
        loader_strict_batch_size: Instance-level default for strict batch
            size. Falls back from ``strict_batch_size=None`` in dataloader calls.
    """

    def __init__(
        self,
        *,
        dataset_file_path: Path,
        seq_len: int = 128,
        mode: ForecastingMode = ForecastingMode.UNIVARIATE,
        batch_size: int = 32,
        valid_size: float = 0.1,
        test_size: float = 0.5,
        shuffle: bool = False,
        scale_data: bool = True,
        data_scaling_method: ScalingMethod = ScalingMethod.MINMAX,
        data_scaling_range: tuple[float, float] = (0, 1),
        num_workers: int = 0,
        forecast_horizon: int = 96,
        step: int | None = None,
        loader_mode: ForecastingLoaderMode = ForecastingLoaderMode.RAW_SERIES,
        loader_strict_batch_size: bool = False,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            seq_len=seq_len,
            valid_size=valid_size,
            test_size=test_size,
            shuffle=shuffle,
            scale_data=scale_data,
            data_scaling_method=data_scaling_method,
            data_scaling_range=data_scaling_range,
            num_workers=num_workers,
            mode=mode,
            forecast_horizon=forecast_horizon,
            step=step,
            loader_mode=loader_mode,
            loader_strict_batch_size=loader_strict_batch_size,
        )
        self.dataset_file_path = dataset_file_path
        self._dataset_name = dataset_file_path.stem
        self._cache_key = build_cache_key(
            dataset_name=dataset_file_path.stem,
            params={
                "seq_len": seq_len,
                "mode": mode.value,
                "data_scaling_method": data_scaling_method.value,
                "data_scaling_range": list(data_scaling_range),
            },
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _set_data_slices(self) -> None:
        """Set 60/20/20 fractional train/valid/test splits."""
        if self._full_data_raw is None:
            msg = "_set_data_slices requires _full_data_raw. Ensure prepare_data() was called."
            raise RuntimeError(msg)
        num_samples = len(self._full_data_raw)
        self._train_slice = slice(None, int(0.6 * num_samples))
        self._valid_slice = slice(int(0.6 * num_samples), int(0.8 * num_samples))
        self._test_slice = slice(int(0.8 * num_samples), None)

    def _transform_data(self) -> None:
        """Transform ``_full_data_scaled`` using expand_dims(axis=0).

        Produces shape (1, samples, features).
        """
        if self._full_data_scaled is None:
            msg = "_transform_data requires _full_data_scaled. Ensure scaling completed."
            raise RuntimeError(msg)
        self._full_data_scaled = np.expand_dims(self._full_data_scaled, axis=0)

    # ------------------------------------------------------------------
    # Sliding dataset
    # ------------------------------------------------------------------

    def _build_sliding_dataset(
        self, data: np.ndarray, internal_mode: TimeSeriesDatasetMode, step: int, horizon: int
    ) -> Dataset:
        """Build sliding-window dataset for Weather.

        Weather data shape: (1, T, 22) post-transform. Squeeze axis 0
        to get (T, 22).

        Args:
            data: Partition data (1, T, 22).
            internal_mode: Mapped dataset mode.
            step: Stride between consecutive windows.
            horizon: Forecast horizon for label extraction.
        """
        from chronocratic.datasets.datatypes.weather import WeatherDataset

        assert self._seq_len is not None
        squeezed = data.squeeze(axis=0)
        return WeatherDataset(
            data=squeezed,
            seq_len=self._seq_len,
            step=step,
            mode=internal_mode,
            forecast_horizon=horizon,
        )

    # ------------------------------------------------------------------
    # Lightning lifecycle
    # ------------------------------------------------------------------

    def _do_prepare_data(self) -> None:
        """Validate file path, read CSV, and write cache.

        Reads the CSV, converts to numpy, and persists both the data
        (``.npz``) and metadata (``.json``) to the cache directory.
        ``_dataset_name`` is set from the filename stem.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
        """
        if not self.dataset_file_path.exists():
            msg = f"Dataset file not found: {self.dataset_file_path}"
            raise FileNotFoundError(msg)

        df = pd.read_csv(self.dataset_file_path, parse_dates=True, index_col="date")

        if self._mode == ForecastingMode.UNIVARIATE:
            df = df.iloc[:, -1:]  # Last column for univariate

        # Convert to numpy and persist to cache
        data = df.to_numpy().astype(np.float32)
        index_ns = df.index.astype(np.int64).to_numpy()

        cache_dir = self._resolve_cache_dir()
        cache_path = cache_dir / f"{self._cache_key}.npz"
        cache_dir.mkdir(parents=True, exist_ok=True)

        atomic_save_npz(cache_path, data=data, index=index_ns)

        # Store time index for reference
        self._time_index = pd.DatetimeIndex(df.index)

        # Compute 60/20/20 splits for metadata
        train_end = int(0.6 * len(data))
        valid_end = int(0.8 * len(data))
        splits = {
            "train": [0, train_end],
            "valid": [train_end, valid_end],
            "test": [valid_end, len(data)],
        }
        n_features = data.shape[1]
        if self.scale_data and self._time_index is not None:
            n_features += TIME_FEATURE_COUNT
        metadata = {
            "version": CACHE_SCHEMA_VERSION,
            "dataset_name": self._dataset_name,
            "n_features": n_features,
            "seq_len": self._seq_len,
            "splits": splits,
            "has_datetime_index": True,
            "data_scaling_method": self.data_scaling_method.value,
            "data_scaling_range": list(self.data_scaling_range),
        }
        atomic_save_metadata(cache_dir / f"{self._cache_key}_metadata.json", metadata)

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(
        self,
        *,
        loader_mode: ForecastingLoaderMode | None = None,
        shuffle: bool | None = None,
        strict_batch_size: bool | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader:
        """Build the training DataLoader.

        Args:
            loader_mode: Per-call mode controlling output format.
                Defaults to ``None``, which falls back to
                :attr:`loader_mode` set at init time.
                RAW_SERIES yields full series (existing behavior).
                INPUT_TARGET yields (input, target) sliding-window pairs.
                INPUT_ONLY yields input windows without targets.
            shuffle: Whether to shuffle. Defaults to :attr:`shuffle`.
            strict_batch_size: If True, pad the last batch. Defaults to
                ``None``, which falls back to :attr:`loader_strict_batch_size`.
            extra_args: Additional keyword arguments for DataLoader.

        Returns:
            Configured DataLoader for training.
        """
        effective_strict = (
            strict_batch_size if strict_batch_size is not None else self.loader_strict_batch_size
        )
        result = self._build_dataloader(
            data_partition=self._train_data_samples,
            partition=DataPartition.TRAIN,
            loader_mode=loader_mode,
            shuffle=shuffle,
            strict_batch_size=effective_strict,
            extra_args=extra_args,
        )
        assert result is not None  # _process_train_dataloader always returns DataLoader
        return result

    def val_dataloader(
        self,
        *,
        loader_mode: ForecastingLoaderMode | None = None,
        strict_batch_size: bool | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader | None:
        """Build the validation DataLoader."""
        effective_strict = (
            strict_batch_size if strict_batch_size is not None else self.loader_strict_batch_size
        )
        return self._build_dataloader(
            data_partition=self._valid_data_samples,
            partition=DataPartition.VAL,
            loader_mode=loader_mode,
            strict_batch_size=effective_strict,
            extra_args=extra_args,
        )

    def test_dataloader(
        self,
        *,
        loader_mode: ForecastingLoaderMode | None = None,
        strict_batch_size: bool | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader:
        """Build the test DataLoader."""
        effective_strict = (
            strict_batch_size if strict_batch_size is not None else self.loader_strict_batch_size
        )
        result = self._build_dataloader(
            data_partition=self._test_data_samples,
            partition=DataPartition.TEST,
            loader_mode=loader_mode,
            strict_batch_size=effective_strict,
            extra_args=extra_args,
        )
        assert result is not None  # _process_test_dataloader always returns DataLoader
        return result
