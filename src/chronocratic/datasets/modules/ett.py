"""ETT (Electricity Transformer Temperature) LightningDataModule.

Supports ETTh1, ETTh2 (hourly) and ETTm1, ETTm2 (15-min) variants.
Uses standard 16-month / 4-month / 4-month splits.

Accepts explicit ``variant`` parameter (not filename auto-detection).
Uses TensorDataset for dataloaders.
Validates variant against known set.
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

__all__ = ["ETTDataModule"]

# Valid ETT variants
VALID_ETT_VARIANTS = frozenset({"ETTh1", "ETTh2", "ETTm1", "ETTm2"})


class ETTDataModule(BaseForecastingTimeSeriesDataModule):
    """LightningDataModule for ETT forecasting datasets.

    Supports ETTh1, ETTh2 (hourly) and ETTm1, ETTm2 (15-min). Uses
    standard 16-month / 4-month / 4-month splits based on variant.

    Accepts explicit ``variant`` parameter rather than
    auto-detecting from the filename.

    .. rubric:: Data shape reference

    ETT is a single multivariate time series. Raw CSV shape varies by
    variant: ETTh1/ETTh2 have 7 features, ETTm1/ETTm2 have 7 features.

    ==================  =================  ==================  ==================
    Variant             Raw CSV Shape      Post-Transform      Notes
    ==================  =================  ==================  ==================
    ETTh1, ETTh2        (17420, 7)         (1, 17420, 7)       Hourly, 12 months
    ETTm1, ETTm2        (69680, 7)         (1, 69680, 7)       15-min, 12 months
    ==================  =================  ==================  ==================

    For univariate mode, only the ``OT`` column is retained (shape
    becomes ``(1, T, 2)`` after adding time features).

    Args:
        dataset_file_path: Path to the CSV file.
        variant: ETT dataset variant (``"ETTh1"``, ``"ETTh2"``,
            ``"ETTm1"``, ``"ETTm2"``).
        seq_len: Input window length.
        mode: UNIVARIATE or MULTIVARIATE.
        batch_size: Batch size.
        valid_size: Validation fraction (unused, fixed by dataset).
        test_size: Test fraction (unused, fixed by dataset).
        shuffle: Whether to shuffle training data.
        scale_data: Whether to scale features.
        data_scaling_method: Scaling algorithm.
        data_scaling_range: Target min-max range.
        num_workers: DataLoader worker count.
        loader_mode: Per-init mode controlling dataloader output format.
            Defaults to ``ForecastingLoaderMode.RAW_SERIES``.
        loader_strict_batch_size: Instance-level default for strict batch
            size. Falls back from ``loader_strict_batch_size=None`` in dataloader calls.

    Raises:
        ValueError: If variant is not one of the four valid ETT variants.
    """

    def __init__(
        self,
        *,
        dataset_file_path: Path,
        variant: str,
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
        # Validate variant
        if variant not in VALID_ETT_VARIANTS:
            msg = f"Unknown ETT variant: {variant!r}. Must be one of {sorted(VALID_ETT_VARIANTS)}"
            raise ValueError(msg)
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
        self.variant = variant
        self._dataset_name = variant
        self._cache_key = build_cache_key(
            dataset_name=variant,
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
        """Set train/valid/test slices based on variant.

        ETTh1/ETTh2: 16/4/4 month split at hourly resolution.
        ETTm1/ETTm2: 16/4/4 month split at 15-min resolution (4x multiplier).
        """
        if self.variant in {"ETTh1", "ETTh2"}:
            self._train_slice = slice(None, 12 * 30 * 24)
            self._valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
            self._test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
        else:  # ETTm1, ETTm2
            self._train_slice = slice(None, 12 * 30 * 24 * 4)
            self._valid_slice = slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
            self._test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)

    def _transform_data(self) -> None:
        """Transform ``_full_data_scaled`` to shape (1, samples, features).

        Expands the first dimension of the already-scaled data array.
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
        """Build sliding-window dataset for ETT.

        ETT data shape: (1, T, F) post-transform. Squeeze axis 0 to
        get (T, F) for the single-file dataset.

        Args:
            data: Partition data (1, T, F).
            internal_mode: Mapped dataset mode.
            step: Stride between consecutive windows.
            horizon: Forecast horizon for label extraction.
        """
        from chronocratic.datasets.datatypes.ett import ETTDataset

        if self._seq_len is None:
            msg = "seq_len is not set. Ensure the datamodule was initialized with a seq_len value."
            raise RuntimeError(msg)
        squeezed = data.squeeze(axis=0)  # (1, T, F) -> (T, F)
        return ETTDataset(
            data=squeezed,
            seq_len=self._seq_len,
            step=step,
            forecast_horizon=horizon,
            mode=internal_mode,
        )

    # ------------------------------------------------------------------
    # Lightning lifecycle
    # ------------------------------------------------------------------

    def _do_prepare_data(self) -> None:
        """Validate file path, read CSV, and write cache.

        Reads the CSV, converts to numpy, and persists both the data
        (``.npz``) and metadata (``.json``) to the cache directory.
        ``_dataset_name`` is set from the variant (not from the filename).

        Raises:
            FileNotFoundError: If the CSV file does not exist.
        """
        if not self.dataset_file_path.exists():
            msg = f"Dataset file not found: {self.dataset_file_path}"
            raise FileNotFoundError(msg)

        df = pd.read_csv(self.dataset_file_path, parse_dates=True, index_col="date")

        if self._mode == ForecastingMode.UNIVARIATE:
            df = df[["OT"]]

        # Convert to numpy and persist to cache
        data = df.to_numpy().astype(np.float32)
        index_ns = df.index.astype(np.int64).to_numpy()

        cache_dir = self._resolve_cache_dir()
        cache_path = cache_dir / f"{self._cache_key}.npz"
        cache_dir.mkdir(parents=True, exist_ok=True)

        atomic_save_npz(cache_path, data=data, index=index_ns)

        # Store time index for reference (needed before metadata computation)
        self._time_index = pd.DatetimeIndex(df.index)

        # Compute variant-based splits for metadata
        self._set_data_slices()
        assert self._train_slice is not None
        assert self._valid_slice is not None
        assert self._test_slice is not None
        splits = {
            "train": [self._train_slice.start, self._train_slice.stop],
            "valid": [self._valid_slice.start, self._valid_slice.stop],
            "test": [self._test_slice.start, self._test_slice.stop],
        }
        # n_features includes time features only when scaling is enabled
        n_features = data.shape[1]
        if self.scale_data and self._time_index is not None:
            n_features += TIME_FEATURE_COUNT
        metadata = {
            "version": CACHE_SCHEMA_VERSION,
            "dataset_name": self.variant,
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
        loader_strict_batch_size: bool | None = None,
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
            loader_strict_batch_size: If True, pad the last batch. Defaults to
                ``None``, which falls back to :attr:`loader_strict_batch_size`.
            extra_args: Additional keyword arguments for DataLoader.

        Returns:
            Configured DataLoader for training.
        """
        effective_loader_strict = (
            loader_strict_batch_size
            if loader_strict_batch_size is not None
            else self.loader_strict_batch_size
        )
        result = self._build_dataloader(
            data_partition=self._train_data_samples,
            partition=DataPartition.TRAIN,
            loader_mode=loader_mode,
            shuffle=shuffle,
            loader_strict_batch_size=effective_loader_strict,
            extra_args=extra_args,
        )
        assert result is not None  # _process_train_dataloader always returns DataLoader
        return result

    def val_dataloader(
        self,
        *,
        loader_mode: ForecastingLoaderMode | None = None,
        loader_strict_batch_size: bool | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader | None:
        """Build the validation DataLoader.

        Returns ``None`` when :attr:`valid_size` is ``0.0``.

        Args:
            loader_mode: Per-call mode controlling output format.
            loader_strict_batch_size: If True, pad the last batch. Defaults to
                ``None``, which falls back to :attr:`loader_strict_batch_size`.
            extra_args: Additional keyword arguments for DataLoader.

        Returns:
            Configured DataLoader for validation, or ``None``.
        """
        effective_loader_strict = (
            loader_strict_batch_size
            if loader_strict_batch_size is not None
            else self.loader_strict_batch_size
        )
        return self._build_dataloader(
            data_partition=self._valid_data_samples,
            partition=DataPartition.VAL,
            loader_mode=loader_mode,
            loader_strict_batch_size=effective_loader_strict,
            extra_args=extra_args,
        )

    def test_dataloader(
        self,
        *,
        loader_mode: ForecastingLoaderMode | None = None,
        loader_strict_batch_size: bool | None = None,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader:
        """Build the test DataLoader.

        Args:
            loader_mode: Per-call mode controlling output format.
            loader_strict_batch_size: If True, pad the last batch. Defaults to
                ``None``, which falls back to :attr:`loader_strict_batch_size`.
            extra_args: Additional keyword arguments for DataLoader.

        Returns:
            Configured DataLoader for testing.
        """
        effective_loader_strict = (
            loader_strict_batch_size
            if loader_strict_batch_size is not None
            else self.loader_strict_batch_size
        )
        result = self._build_dataloader(
            data_partition=self._test_data_samples,
            partition=DataPartition.TEST,
            loader_mode=loader_mode,
            loader_strict_batch_size=effective_loader_strict,
            extra_args=extra_args,
        )
        assert result is not None  # _process_test_dataloader always returns DataLoader
        return result
