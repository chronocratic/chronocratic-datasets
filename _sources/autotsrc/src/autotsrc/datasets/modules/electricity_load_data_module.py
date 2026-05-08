__all__ = ['ElectricityLoadDataModule']

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.autotsrc.datasets.modules.abstract import BaseForecastingTimeSeriesDataModule
from src.autotsrc.enums import ForecastingTimeSeriesDatasetMode, TimeSeriesDatasetMode


class ElectricityLoadDataModule(BaseForecastingTimeSeriesDataModule):
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
        """Set train, validation, and test ranges based on fixed percentage splits."""
        if not isinstance(self._full_data, np.ndarray):
            self._full_data = np.array(self._full_data)
        num_samples = len(self._full_data)
        self._train_slice = slice(None, int(0.6 * num_samples))
        self._valid_slice = slice(int(0.6 * num_samples), int(0.8 * num_samples))
        self._test_slice = slice(int(0.8 * num_samples), None)

    def _transform_data(self) -> None:
        """Convert to channels-first forecasting tensor layout."""
        if not isinstance(self._full_data, np.ndarray):
            self._full_data = np.array(self._full_data)
        if self._full_data.ndim == 1:
            self._full_data = self._full_data[:, np.newaxis]
        self._full_data = self._full_data.T
        self._full_data = np.expand_dims(self._full_data, axis=-1)

    def prepare_data(self) -> None:
        """Load, clean, and prepare Electricity Load data for forecasting splits."""
        self._dataset_name = 'ElectricityLoad'

        df_data = pd.read_csv(
            self.dataset_file_path, parse_dates=True, sep=';', decimal=',', index_col=[0]
        )
        df_data = df_data.resample('1h', closed='right').sum()
        df_data = df_data.loc[
            :, df_data.cumsum(axis=0).iloc[8920] != 0
        ]  # filter out instances with missing values
        df_data.index = df_data.index.rename('date')
        df_data = df_data['2012':]

        if self._mode == ForecastingTimeSeriesDatasetMode.UNIVARIATE:
            df_data = df_data[['MT_001']]

        self._full_data = df_data

        self._post_prepare_data()

    def train_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.FORECASTING,  # noqa: ARG002
        shuffle: bool | None = None,
        strict_batch_size: bool = False,
        extra_args: dict | None = None,
    ) -> DataLoader:
        """Build the training dataloader for the prepared Electricity Load split."""
        data_samples_torch = torch.from_numpy(self._train_data_samples).to(torch.float32)
        dataset_object = TensorDataset(data_samples_torch)

        return self._process_train_dataloader(
            dataset_object=dataset_object,
            shuffle=shuffle,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )

    def val_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.FORECASTING,  # noqa: ARG002
        strict_batch_size: bool = False,
        extra_args: dict | None = None,
    ) -> DataLoader | None:
        """Build the validation dataloader for the prepared Electricity Load split."""
        data_samples_torch = torch.from_numpy(self._valid_data_samples).to(torch.float32)
        dataset_object = TensorDataset(data_samples_torch)

        return self._process_valid_dataloader(
            dataset_object=dataset_object,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )

    def test_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.FORECASTING,  # noqa: ARG002
        strict_batch_size: bool = False,
        extra_args: dict | None = None,
    ) -> DataLoader:
        """Build the test dataloader for the prepared Electricity Load split."""
        data_samples_torch = torch.from_numpy(self._test_data_samples).to(torch.float32)
        dataset_object = TensorDataset(data_samples_torch)

        return self._process_test_dataloader(
            dataset_object=dataset_object,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )
