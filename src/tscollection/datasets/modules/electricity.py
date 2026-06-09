"""Electricity load forecasting LightningDataModule.

Reads the French electricity load CSV (semicolon-delimited, comma decimal),
resamples to hourly, and splits 60/20/20.

Uses TensorDataset for dataloaders.
Raises FileNotFoundError for missing paths.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from tscollection.datasets.enums.data import (
    ForecastingLoaderMode,
    ForecastingMode,
    ScalingMethod,
    TimeSeriesDatasetMode,
)
from tscollection.datasets.modules._base.forecasting import BaseForecastingTimeSeriesDataModule
from tscollection.datasets.utils.cache import (
    atomic_save_metadata,
    atomic_save_npz,
    build_cache_key,
    CACHE_SCHEMA_VERSION,
)
from tscollection.datasets.utils.features import TIME_FEATURE_COUNT

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ['ElectricityLoadModule']


class ElectricityLoadModule(BaseForecastingTimeSeriesDataModule):
    """LightningDataModule for electricity load forecasting.

    Reads semicolon-delimited CSV with comma decimals, resamples to
    hourly, and applies 60/20/20 fractional splits.

    The data transform uses transpose + expand_dims(axis=-1),
    producing shape (features, samples, 1).

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
        forecast_horizon: int = 24,
        step: int | None = None,
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
        )
        self.dataset_file_path = dataset_file_path
        self._dataset_name = 'ElectricityLoad'
        self._cache_key = build_cache_key(
            dataset_name='ElectricityLoad',
            params={
                'seq_len': seq_len,
                'mode': mode.value,
                'data_scaling_method': data_scaling_method.value,
                'data_scaling_range': list(data_scaling_range),
            },
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _set_data_slices(self) -> None:
        """Set 60/20/20 fractional train/valid/test splits."""
        if self._full_data_raw is None:
            msg = '_set_data_slices requires _full_data_raw. Ensure prepare_data() was called.'
            raise RuntimeError(msg)
        num_samples = len(self._full_data_raw)
        self._train_slice = slice(None, int(0.6 * num_samples))
        self._valid_slice = slice(int(0.6 * num_samples), int(0.8 * num_samples))
        self._test_slice = slice(int(0.8 * num_samples), None)

    def _transform_data(self) -> None:
        """Transform ``_full_data_scaled`` using transpose + expand_dims(axis=-1).

        Produces shape (features, samples, 1). Different from
        WeatherModule which uses expand_dims(axis=0).
        """
        if self._full_data_scaled is None:
            msg = '_transform_data requires _full_data_scaled. Ensure scaling completed.'
            raise RuntimeError(msg)
        self._full_data_scaled = self._full_data_scaled.T
        self._full_data_scaled = np.expand_dims(self._full_data_scaled, axis=-1)

    # ------------------------------------------------------------------
    # Sliding dataset
    # ------------------------------------------------------------------

    def _build_sliding_dataset(
        self,
        data: np.ndarray,
        internal_mode: TimeSeriesDatasetMode,
        step: int,
        horizon: int,
    ) -> Dataset:
        """Build sliding-window dataset for Electricity.

        Electricity data shape: (370, T, 1) post-transform.
        370 independent power clients, each treated as a series.

        Args:
            data: Partition data (370, T, 1).
            internal_mode: Mapped dataset mode.
            step: Stride between consecutive windows.
            horizon: Forecast horizon for label extraction.
        """
        from tscollection.datasets.datatypes.electricity import ElectricityDataset

        assert self._seq_len is not None
        return ElectricityDataset(
            data=data,
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

        Reads the semicolon-delimited CSV with comma decimals, resamples
        to hourly, and persists both the data (``.npz``) and metadata
        (``.json``) to the cache directory.
        ``_dataset_name`` is hardcoded to ``'ElectricityLoad'``.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
        """
        if not self.dataset_file_path.exists():
            msg = f'Dataset file not found: {self.dataset_file_path}'
            raise FileNotFoundError(msg)

        df = pd.read_csv(
            self.dataset_file_path, parse_dates=True, sep=';', decimal=',', index_col=[0]
        )
        df = df.resample('1h', closed='right').sum()
        df = df.loc[:, (df != 0).any(axis=0)]
        df.index = df.index.rename('date')
        df = df['2012':]

        if self._mode == ForecastingMode.UNIVARIATE:
            df = df[['MT_001']]

        # Convert to numpy and persist to cache
        data = df.to_numpy().astype(np.float32)
        index_ns = df.index.astype(np.int64).to_numpy()

        cache_dir = self._resolve_cache_dir()
        cache_path = cache_dir / f'{self._cache_key}.npz'
        cache_dir.mkdir(parents=True, exist_ok=True)

        atomic_save_npz(cache_path, data=data, index=index_ns)

        # Store time index for reference
        self._time_index = pd.DatetimeIndex(df.index)

        # Compute 60/20/20 splits for metadata
        train_end = int(0.6 * len(data))
        valid_end = int(0.8 * len(data))
        splits = {
            'train': [0, train_end],
            'valid': [train_end, valid_end],
            'test': [valid_end, len(data)],
        }
        n_features = data.shape[1]
        if self.scale_data and self._time_index is not None:
            n_features += TIME_FEATURE_COUNT
        metadata = {
            'version': CACHE_SCHEMA_VERSION,
            'dataset_name': self._dataset_name,
            'n_features': n_features,
            'seq_len': self._seq_len,
            'splits': splits,
            'has_datetime_index': True,
            'data_scaling_method': self.data_scaling_method.value,
            'data_scaling_range': list(self.data_scaling_range),
        }
        atomic_save_metadata(cache_dir / f'{self._cache_key}_metadata.json', metadata)

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(
        self,
        *,
        loader_mode: ForecastingLoaderMode = ForecastingLoaderMode.RAW_SERIES,
        shuffle: bool | None = None,
        strict_batch_size: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader:
        """Build the training DataLoader.

        Args:
            loader_mode: Per-call mode controlling output format.
                RAW_SERIES yields full series (existing behavior).
                INPUT_TARGET yields (input, target) sliding-window pairs.
                INPUT_ONLY yields input windows without targets.
            shuffle: Whether to shuffle. Defaults to :attr:`shuffle`.
            strict_batch_size: If True, pad the last batch.
            extra_args: Additional keyword arguments for DataLoader.

        Returns:
            Configured DataLoader for training.
        """
        result = self._build_dataloader(
            data_partition=self._train_data_samples,
            dataloader_fn=self._process_train_dataloader,
            loader_mode=loader_mode,
            shuffle=shuffle,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )
        assert result is not None  # train_dataloader always returns a DataLoader
        return result

    def val_dataloader(
        self,
        *,
        loader_mode: ForecastingLoaderMode = ForecastingLoaderMode.RAW_SERIES,
        strict_batch_size: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader | None:
        """Build the validation DataLoader."""
        return self._build_dataloader(
            data_partition=self._valid_data_samples,
            dataloader_fn=self._process_valid_dataloader,
            loader_mode=loader_mode,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )

    def test_dataloader(
        self,
        *,
        loader_mode: ForecastingLoaderMode = ForecastingLoaderMode.RAW_SERIES,
        strict_batch_size: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader:
        """Build the test DataLoader."""
        result = self._build_dataloader(
            data_partition=self._test_data_samples,
            dataloader_fn=self._process_test_dataloader,
            loader_mode=loader_mode,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )
        assert result is not None  # test_dataloader always returns a DataLoader
        return result
