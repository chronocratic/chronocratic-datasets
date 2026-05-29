"""Weather forecasting LightningDataModule.

Reads the 7-year weather CSV and splits 60/20/20.

Uses TensorDataset for dataloaders.
Raises FileNotFoundError for missing paths.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from tscollection.datasets.enums.data import ForecastingMode, ScalingMethod, TimeSeriesDatasetMode
from tscollection.datasets.modules._base.forecasting import BaseForecastingTimeSeriesDataModule
from tscollection.datasets.utils.cache import (
    atomic_save_metadata,
    atomic_save_npz,
    build_cache_key,
)
from tscollection.datasets.utils.features import TIME_FEATURE_COUNT

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ['WeatherModule']


class WeatherModule(BaseForecastingTimeSeriesDataModule):
    """LightningDataModule for weather forecasting.

    Reads CSV with standard format (comma-separated, period decimals),
    applies 60/20/20 fractional splits.

    The data transform uses expand_dims(axis=0), producing shape
    (1, samples, features). Different from ElectricityLoadModule which
    uses transpose + expand_dims(axis=-1).

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
        assert self._full_data_raw is not None, (
            '_full_data_raw was not set by prepare_data()'
        )
        num_samples = len(self._full_data_raw)
        self._train_slice = slice(None, int(0.6 * num_samples))
        self._valid_slice = slice(int(0.6 * num_samples), int(0.8 * num_samples))
        self._test_slice = slice(int(0.8 * num_samples), None)

    def _transform_data(self) -> None:
        """Transform ``_full_data_scaled`` using expand_dims(axis=0).

        Produces shape (1, samples, features). Different from
        ElectricityLoadModule which uses transpose + expand_dims(axis=-1).
        """
        assert self._full_data_scaled is not None
        self._full_data_scaled = np.expand_dims(self._full_data_scaled, axis=0)

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
            msg = f'Dataset file not found: {self.dataset_file_path}'
            raise FileNotFoundError(msg)

        df = pd.read_csv(self.dataset_file_path, parse_dates=True, index_col='date')

        if self._mode == ForecastingMode.UNIVARIATE:
            df = df.iloc[:, -1:]  # Last column for univariate

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
            "train": [0, train_end],
            "valid": [train_end, valid_end],
            "test": [valid_end, len(data)],
        }
        n_features = data.shape[1] + TIME_FEATURE_COUNT
        metadata = {
            "version": 1,
            "dataset_name": self._dataset_name,
            "n_features": n_features,
            "seq_len": self._seq_len,
            "splits": splits,
            "has_datetime_index": True,
            "data_scaling_method": self.data_scaling_method.value,
            "data_scaling_range": list(self.data_scaling_range),
        }
        atomic_save_metadata(
            cache_dir / f'{self._cache_key}_metadata.json', metadata
        )

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.FORECASTING,  # noqa: ARG002
        shuffle: bool | None = None,
        strict_batch_size: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader:
        """Build the training DataLoader.

        Args:
            mode: Dataset mode.
            shuffle: Whether to shuffle. Defaults to :attr:`shuffle`.
            strict_batch_size: If True, pad the last batch.
            extra_args: Additional keyword arguments for DataLoader.

        Returns:
            Configured DataLoader for training.
        """
        tensor = torch.from_numpy(self._train_data_samples).to(torch.float32)
        return self._process_train_dataloader(
            dataset_object=TensorDataset(tensor),
            shuffle=shuffle,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )

    def val_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.FORECASTING,  # noqa: ARG002
        strict_batch_size: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader | None:
        """Build the validation DataLoader.

        Returns ``None`` when :attr:`valid_size` is ``0.0``.

        Args:
            mode: Dataset mode.
            strict_batch_size: If True, pad the last batch.
            extra_args: Additional keyword arguments for DataLoader.

        Returns:
            Configured DataLoader for validation, or ``None``.
        """
        tensor = torch.from_numpy(self._valid_data_samples).to(torch.float32)
        return self._process_valid_dataloader(
            dataset_object=TensorDataset(tensor),
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )

    def test_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.FORECASTING,  # noqa: ARG002
        strict_batch_size: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader:
        """Build the test DataLoader.

        Args:
            mode: Dataset mode.
            strict_batch_size: If True, pad the last batch.
            extra_args: Additional keyword arguments for DataLoader.

        Returns:
            Configured DataLoader for testing.
        """
        tensor = torch.from_numpy(self._test_data_samples).to(torch.float32)
        return self._process_test_dataloader(
            dataset_object=TensorDataset(tensor),
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )
