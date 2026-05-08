"""Abstract LightningDataModule base classes for time series data.

Provides shared dataloader construction, scaling setup, and split
management for both classification and forecasting datasets.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader

from src.rbspaper.data.utils import (
    custom_collate_fn,
    extract_time_features,
    load_json,
    process_data_with_varying_sequence_lengths_single,
    separate_target_feature_from_df,
)
from src.rbspaper.data.utils.scaling import create_data_scaler
from src.rbspaper.enums.data_enums import (
    ForecastingTimeSeriesDatasetMode,
    TimeSeriesClassificationDatasetSplittingStrategy,
)

__all__ = [
    'BaseClassificationTimeSeriesDataModule',
    'BaseForecastingTimeSeriesDataModule',
    'BaseTimeSeriesDataModule',
]


class BaseTimeSeriesDataModule(pl.LightningDataModule, ABC):
    """Shared base for all time series LightningDataModules.

    Handles batch size, scaling, and dataloader construction.

    Args:
        batch_size: Batch size for dataloaders.
        seq_len: Sequence length (for forecasting).
        valid_size: Fraction of training data for validation.
        test_size: Fraction reserved as test set.
        shuffle: Whether to shuffle training dataloader.
        scale_data: Whether to apply scaling.
        data_scaling_method: 'min_max' or 'standardization'.
        data_scaling_range: Target range for min-max scaling.
        num_workers: DataLoader worker count.
        data_form: Data shape category for scaling.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int | None,
        valid_size: float,
        test_size: float,
        shuffle: bool,
        scale_data: bool,
        data_scaling_method: str,
        data_scaling_range: tuple[float, float],
        num_workers: int = 0,
        data_form: str = 'regular',
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self._seq_len = seq_len
        self.valid_size = valid_size
        self.test_size = test_size
        self.shuffle = shuffle
        self.scale_data = scale_data
        self.data_scaling_method = data_scaling_method
        self.data_scaling_range = data_scaling_range
        self.num_workers = num_workers
        self._data_form = data_form
        self._datatype_handling_functions_map: dict[str, Any] | None = None
        self._initiate_datatypes_handling_functions_map()
        self._dataset_name: str | None = None
        self._num_features: int | None = None
        self._train_data_samples: Any = None
        self._test_data_samples: Any = None
        self._valid_data_samples: Any = None
        self._dataset_class: Any = None

    @property
    def name(self) -> str | None:
        return self._dataset_name

    @property
    def n_features(self) -> int | None:
        return self._num_features

    @property
    def sequence_len(self) -> int | None:
        return self._seq_len

    @property
    def train_data_samples(self):
        return self._train_data_samples

    @property
    def test_data_samples(self):
        return self._test_data_samples

    @property
    def valid_data_samples(self):
        return self._valid_data_samples

    @property
    def all_data_samples(self) -> pd.DataFrame:
        return pd.concat(
            [self._train_data_samples, self._test_data_samples, self._valid_data_samples], axis=0
        )

    def _initiate_datatypes_handling_functions_map(self) -> None:
        self._datatype_handling_functions_map = defaultdict(lambda: lambda x: x, {})

    def _get_custom_collate_fn(self, desired_batch_size: int | None = None) -> Any:
        if desired_batch_size is None:
            desired_batch_size = self.batch_size
        return partial(custom_collate_fn, desired_batch_size=desired_batch_size)

    def setup(self, stage: str | None = None) -> None:
        scaler = create_data_scaler(
            scale=self.scale_data,
            scaling_range=self.data_scaling_range,
            scaling_method=self.data_scaling_method,
            data_form=self._data_form,
        )
        (self._train_data_samples, self._valid_data_samples, self._test_data_samples) = scaler(
            self._train_data_samples, self._valid_data_samples, self._test_data_samples
        )

    def _process_train_dataloader(
        self,
        *,
        dataset_object: Any,
        shuffle: bool | None = None,
        strict_batch_size: bool = False,
        extra_args: dict | None = None,
    ) -> DataLoader:
        if shuffle is None:
            shuffle = self.shuffle
        dataloader_args: dict[str, Any] = {
            'dataset': dataset_object,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'shuffle': shuffle,
            **(extra_args or {}),
        }
        if self.num_workers > 0:
            dataloader_args['persistent_workers'] = True
        if strict_batch_size:
            dataloader_args['collate_fn'] = self._get_custom_collate_fn()
        return DataLoader(**dataloader_args)

    def _process_test_dataloader(
        self,
        *,
        dataset_object: Any,
        strict_batch_size: bool = False,
        extra_args: dict | None = None,
    ) -> DataLoader:
        dataloader_args: dict[str, Any] = {
            'dataset': dataset_object,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'shuffle': False,
            **(extra_args or {}),
        }
        if self.num_workers > 0:
            dataloader_args['persistent_workers'] = True
        if strict_batch_size:
            dataloader_args['collate_fn'] = self._get_custom_collate_fn()
        return DataLoader(**dataloader_args)

    def _process_valid_dataloader(
        self,
        *,
        dataset_object: Any,
        strict_batch_size: bool = False,
        extra_args: dict | None = None,
    ) -> DataLoader | None:
        if self.valid_size == 0.0:
            return None
        return self._process_test_dataloader(
            dataset_object=dataset_object,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )


class BaseClassificationTimeSeriesDataModule(BaseTimeSeriesDataModule, ABC):
    """Base datamodule for classification datasets (UCR/UEA).

    Manages ARFF-based config, train/val/test splitting, target
    column extraction, and variable-length sequence handling.

    Args:
        dataset_config_path: Path to JSON config with column names and
            file patterns.
        batch_size: Batch size.
        valid_size: Fraction of training data for validation.
        shuffle: Whether to shuffle training data.
        scale_data: Whether to scale features.
        data_scaling_method: Scaling algorithm.
        data_form: Data shape category.
        data_scaling_range: Target min-max range.
        splitting_strategy: 'AS_DEFINED' or 'MANUAL'.
        test_size: Test set fraction for MANUAL splitting.
        num_workers: DataLoader worker count.
    """

    def __init__(
        self,
        *,
        dataset_config_path: Path,
        batch_size: int,
        valid_size: float,
        shuffle: bool,
        scale_data: bool,
        data_scaling_method: str,
        data_form: str,
        data_scaling_range: tuple[float, float],
        splitting_strategy: TimeSeriesClassificationDatasetSplittingStrategy = (
            TimeSeriesClassificationDatasetSplittingStrategy.AS_DEFINED
        ),
        test_size: float = 0.5,
        num_workers: int = 1,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            seq_len=None,
            valid_size=valid_size,
            test_size=test_size,
            shuffle=shuffle,
            scale_data=scale_data,
            data_scaling_method=data_scaling_method,
            data_scaling_range=data_scaling_range,
            data_form=data_form,
            num_workers=num_workers,
        )
        self.dataset_config = load_json(dataset_config_path)
        self.splitting_strategy = splitting_strategy
        self.test_size = test_size
        self.num_workers = num_workers
        self.target_column_name = self.dataset_config['main_config']['target_col_name']
        self._separate_target_feature = partial(
            separate_target_feature_from_df, target_feature_name=self.target_column_name
        )
        self._data_column_names: str | None = None
        self._num_classes: int | None = None
        self._train_data_labels: Any = None
        self._test_data_labels: Any = None
        self._valid_data_labels: Any = None

    @property
    def n_classes(self) -> int | None:
        return self._num_classes

    @property
    def train_data_labels(self):
        return self._train_data_labels

    @property
    def test_data_labels(self):
        return self._test_data_labels

    @property
    def valid_data_labels(self):
        return self._valid_data_labels

    @property
    def all_data_labels(self) -> pd.Series:
        return pd.concat(
            [self._train_data_labels, self._test_data_labels, self._valid_data_labels], axis=0
        )

    def _extract_data_column_names(self) -> None:
        self._data_column_names = [
            col for col in self._train_data_samples.columns if col != self.target_column_name
        ][0]

    def _process_data_with_varying_sequence_lengths(self) -> None:
        self._train_data_samples = process_data_with_varying_sequence_lengths_single(
            data=self._train_data_samples
        )
        self._valid_data_samples = process_data_with_varying_sequence_lengths_single(
            data=self._valid_data_samples
        )
        self._test_data_samples = process_data_with_varying_sequence_lengths_single(
            data=self._test_data_samples
        )


class BaseForecastingTimeSeriesDataModule(BaseTimeSeriesDataModule, ABC):
    """Base datamodule for forecasting datasets (ETT, electricity, weather).

    Manages time series slicing, feature extraction, and per-split
    scaling trained only on the training portion.

    Args:
        batch_size: Batch size.
        seq_len: Input window length.
        valid_size: Fraction for validation.
        test_size: Fraction for testing.
        shuffle: Whether to shuffle training data.
        scale_data: Whether to scale features.
        data_scaling_method: Scaling algorithm.
        data_scaling_range: Target min-max range.
        num_workers: DataLoader worker count.
        mode: UNIVARIATE or MULTIVARIATE forecasting.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int,
        valid_size: float,
        test_size: float,
        shuffle: bool,
        scale_data: bool,
        data_scaling_method: str,
        data_scaling_range: tuple[float, float],
        num_workers: int,
        mode: ForecastingTimeSeriesDatasetMode,
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
        )
        self._mode = mode
        self._train_slice: slice | None = None
        self._valid_slice: slice | None = None
        self._test_slice: slice | None = None
        self._full_data: np.ndarray | None = None
        self._num_time_series_features: int | None = None

    @property
    def train_slice(self) -> slice | None:
        return self._train_slice

    @property
    def valid_slice(self) -> slice | None:
        return self._valid_slice

    @property
    def test_slice(self) -> slice | None:
        return self._test_slice

    @property
    def num_time_series_features(self) -> int | None:
        return self._num_time_series_features

    @property
    def full_data(self) -> np.ndarray | None:
        return self._full_data

    @abstractmethod
    def _set_data_slices(self) -> None:
        """Define train/valid/test slice boundaries."""

    @abstractmethod
    def _transform_data(self) -> None:
        """Transform _full_data after scaling (e.g. reshape)."""

    def _prepare_data_scaler(self) -> StandardScaler | MinMaxScaler:
        if self.data_scaling_method == 'standardization':
            return StandardScaler()
        if self.data_scaling_method == 'min_max':
            return MinMaxScaler(feature_range=self.data_scaling_range)
        raise ValueError(f'Unsupported scaling method: {self.data_scaling_method}')

    def _split_data(self) -> None:
        assert self._full_data is not None
        assert self._train_slice is not None
        assert self._valid_slice is not None
        assert self._test_slice is not None
        train_data = self._full_data[:, self._train_slice]
        valid_data = self._full_data[:, self._valid_slice]
        test_data = self._full_data[:, self._test_slice]
        self._train_data_samples = train_data
        self._valid_data_samples = valid_data
        self._test_data_samples = test_data

    def setup(self, stage: str | None = None) -> None:
        assert self._full_data is not None
        assert self._train_slice is not None

        # Extract time features from DataFrame index if applicable
        if isinstance(self._full_data, pd.DataFrame):
            time_index = self._full_data.index
            full_array = self._full_data.to_numpy()
        else:
            time_index = None
            full_array = self._full_data

        if time_index is not None:
            time_series_features = extract_time_features(pd.DatetimeIndex(time_index))
            num_time_series_features = time_series_features.shape[-1]
        else:
            time_series_features = np.empty((0, 0))
            num_time_series_features = 0

        data_scaler = self._prepare_data_scaler()
        data_scaler.fit(full_array[:, self._train_slice])
        self._full_data = data_scaler.transform(full_array)
        self._transform_data()

        if num_time_series_features > 0:
            ts_feature_scaler = self._prepare_data_scaler()
            ts_feature_scaler.fit(time_series_features[:, self._train_slice])
            scaled_ts_features = ts_feature_scaler.transform(time_series_features)
            scaled_ts_features = np.expand_dims(scaled_ts_features, axis=0)
            assert self._full_data is not None
            repeated_ts = np.repeat(scaled_ts_features, self._full_data.shape[0], axis=0)
            self._full_data = np.concatenate([repeated_ts, self._full_data], axis=-1)

        self._num_time_series_features = num_time_series_features
        self._calculate_num_features()
        self._split_data()

    def _calculate_num_features(self) -> None:
        assert self._full_data is not None
        self._num_features = self._full_data.shape[-1]

    def _post_prepare_data(self) -> None:
        self._set_data_slices()
