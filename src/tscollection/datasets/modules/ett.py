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
import torch
from torch.utils.data import DataLoader, TensorDataset

from tscollection.datasets.enums.data import ForecastingMode, ScalingMethod, TimeSeriesDatasetMode
from tscollection.datasets.modules._base.forecasting import BaseForecastingTimeSeriesDataModule

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ['ETTDataModule']

# Valid ETT variants
VALID_ETT_VARIANTS = frozenset({'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'})


class ETTDataModule(BaseForecastingTimeSeriesDataModule):
    """LightningDataModule for ETT forecasting datasets.

    Supports ETTh1, ETTh2 (hourly) and ETTm1, ETTm2 (15-min). Uses
    standard 16-month / 4-month / 4-month splits based on variant.

    Accepts explicit ``variant`` parameter rather than
    auto-detecting from the filename.

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

    Raises:
        ValueError: If variant is not one of the four valid ETT variants.
    """

    _full_data: pd.DataFrame | np.ndarray | None = None

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
    ) -> None:
        # Validate variant
        if variant not in VALID_ETT_VARIANTS:
            msg = f'Unknown ETT variant: {variant!r}. Must be one of {sorted(VALID_ETT_VARIANTS)}'
            raise ValueError(
                msg
            )
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
        self.variant = variant

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _set_data_slices(self) -> None:
        """Set train/valid/test slices based on variant.

        ETTh1/ETTh2: 16/4/4 month split at hourly resolution.
        ETTm1/ETTm2: 16/4/4 month split at 15-min resolution (4x multiplier).
        """
        if self.variant in {'ETTh1', 'ETTh2'}:
            self._train_slice = slice(None, 12 * 30 * 24)
            self._valid_slice = slice(12 * 30 * 24, 16 * 30 * 24)
            self._test_slice = slice(16 * 30 * 24, 20 * 30 * 24)
        else:  # ETTm1, ETTm2
            self._train_slice = slice(None, 12 * 30 * 24 * 4)
            self._valid_slice = slice(12 * 30 * 24 * 4, 16 * 30 * 24 * 4)
            self._test_slice = slice(16 * 30 * 24 * 4, 20 * 30 * 24 * 4)

    def _transform_data(self) -> None:
        """Transform ``_full_data`` to shape (1, samples, features).

        Converts DataFrame to numpy, then expands dimension on axis 0.
        """
        assert self._full_data is not None, '_full_data was not set by prepare_data()'
        if isinstance(self._full_data, pd.DataFrame):
            self._full_data = self._full_data.to_numpy()
        if isinstance(self._full_data, np.ndarray):
            self._full_data = np.expand_dims(self._full_data, axis=0)

    # ------------------------------------------------------------------
    # Lightning lifecycle
    # ------------------------------------------------------------------

    def _do_prepare_data(self) -> None:
        """Validate file path, read CSV, and prepare data.

        Raises ``FileNotFoundError`` if the CSV file does not
        exist. ``_dataset_name`` is set from the variant
        (not from the filename).
        """
        if not self.dataset_file_path.exists():
            msg = f'Dataset file not found: {self.dataset_file_path}'
            raise FileNotFoundError(msg)

        # _dataset_name from variant, not filename
        self._dataset_name = self.variant

        df = pd.read_csv(self.dataset_file_path, parse_dates=True, index_col='date')

        if self._mode == ForecastingMode.UNIVARIATE:
            df = df[['OT']]

        self._full_data = df

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.FORECASTING,
        shuffle: bool | None = None,
        strict_batch_size: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader:
        """Build the training DataLoader.

        Args:
            mode: Dataset mode (with/without labels, forecasting).
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
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.FORECASTING,
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
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.FORECASTING,
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
