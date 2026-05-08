"""Electricity load forecasting LightningDataModule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.rbspaper.data.modules.abstract import BaseForecastingTimeSeriesDataModule
from src.rbspaper.enums.data_enums import ForecastingTimeSeriesDatasetMode, TimeSeriesDatasetMode

__all__ = ['ElectricityLoadDataModule']


class ElectricityLoadDataModule(BaseForecastingTimeSeriesDataModule):
    """LightningDataModule for electricity load forecasting.

    Reads the 2012-2014 French electricity load CSV, resamples to
    hourly, and splits 60/20/20.

    Args:
        dataset_file_path: Path to the CSV file.
        seq_len: Input window length.
        mode: UNIVARIATE or MULTIVARIATE.
        test_size: Test fraction (unused, fixed 60/20/20).
        valid_size: Validation fraction (unused, fixed 60/20/20).
        batch_size: Batch size.
        shuffle: Whether to shuffle training data.
        scale_data: Whether to scale features.
        data_scaling_method: Scaling algorithm.
        data_scaling_range: Target min-max range.
        num_workers: DataLoader worker count.
    """

    _full_data: pd.DataFrame | np.ndarray | None = None

    def __init__(
        self,
        *,
        dataset_file_path: Path,
        seq_len: int = 128,
        mode: ForecastingTimeSeriesDatasetMode = ForecastingTimeSeriesDatasetMode.UNIVARIATE,
        test_size: float = 0.5,
        valid_size: float = 0.1,
        batch_size: int = 32,
        shuffle: bool = False,
        scale_data: bool = True,
        data_scaling_method: str = 'min_max',
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

    def _set_data_slices(self) -> None:
        if self._full_data is None:
            raise RuntimeError('_full_data was not set by prepare_data()')
        num_samples = len(self._full_data)
        self._train_slice = slice(None, int(0.6 * num_samples))
        self._valid_slice = slice(int(0.6 * num_samples), int(0.8 * num_samples))
        self._test_slice = slice(int(0.8 * num_samples), None)

    def _transform_data(self) -> None:
        if self._full_data is None:
            raise RuntimeError('_full_data was not set by prepare_data()')
        if isinstance(self._full_data, pd.DataFrame):
            self._full_data = self._full_data.to_numpy()
        if isinstance(self._full_data, np.ndarray):
            self._full_data = self._full_data.T
            self._full_data = np.expand_dims(self._full_data, axis=-1)

    def prepare_data(self) -> None:
        self._dataset_name = 'ElectricityLoad'
        df = pd.read_csv(
            self.dataset_file_path, parse_dates=True, sep=';', decimal=',', index_col=[0]
        )
        df = df.resample('1h', closed='right').sum()
        df = df.loc[:, df.cumsum(axis=0).iloc[8920] != 0]
        df.index = df.index.rename('date')
        df = df['2012':]

        if self._mode == ForecastingTimeSeriesDatasetMode.UNIVARIATE:
            df = df[['MT_001']]

        self._full_data = df
        self._post_prepare_data()

    def train_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.FORECASTING,
        shuffle: bool | None = None,
        strict_batch_size: bool = False,
        extra_args: dict | None = None,
    ) -> DataLoader:
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
        extra_args: dict | None = None,
    ) -> DataLoader | None:
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
        extra_args: dict | None = None,
    ) -> DataLoader:
        tensor = torch.from_numpy(self._test_data_samples).to(torch.float32)
        return self._process_test_dataloader(
            dataset_object=TensorDataset(tensor),
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )
