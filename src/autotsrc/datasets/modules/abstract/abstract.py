__all__ = [
    'BaseClassificationTimeSeriesDataModule',
    'BaseForecastingTimeSeriesDataModule',
    'BaseTimeSeriesDataModule',
]

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.autotsrc.datasets.utils import (
    custom_collate_fn,
    extract_time_features,
    process_data_with_varying_sequence_lengths_single,
)
from src.autotsrc.enums import (
    ForecastingTimeSeriesDatasetMode,
    TimeSeriesClassificationDatasetSplittingStrategy,
)
from src.autotsrc.utils import load_json, separate_target_feature_from_df
from src.autotsrc.utils.data.strategies.scaling import create_data_scaler


class BaseTimeSeriesDataModule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int | None,
        valid_size: float,
        test_size: float,
        shuffle: bool,
        scale_data: bool,
        data_scaling_range: tuple[float, float],
        num_workers: int = 0,
        data_scaling_method: str = 'min_max',
        data_form: str = 'regular',
    ) -> None:
        """Initialize shared configuration for time-series data modules."""
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
        self._initiate_datatypes_handling_functions_map()
        self._dataset_name = None
        self._num_features = None
        self._train_data_samples = None
        self._test_data_samples = None
        self._valid_data_samples = None
        self._dataset_class = None

    @property
    def name(self) -> str | None:
        """Return the dataset name."""
        return self._dataset_name

    @property
    def n_features(self) -> int | None:
        """Return the number of input features."""
        return self._num_features

    @property
    def sequence_len(self) -> int | None:
        """Return the configured sequence length."""
        return self._seq_len

    @property
    def train_data_samples(self) -> pd.DataFrame | np.ndarray | None:
        """Return training split samples."""
        return self._train_data_samples

    @property
    def test_data_samples(self) -> pd.DataFrame | np.ndarray | None:
        """Return test split samples."""
        return self._test_data_samples

    @property
    def valid_data_samples(self) -> pd.DataFrame | np.ndarray | None:
        """Return validation split samples."""
        return self._valid_data_samples

    @property
    def all_data_samples(self) -> pd.DataFrame | np.ndarray | None:
        """Return all available sample splits concatenated row-wise."""
        data_samples = [
            data
            for data in [
                self._train_data_samples,
                self._test_data_samples,
                self._valid_data_samples,
            ]
            if data is not None
        ]
        return pd.concat(data_samples, axis=0)

    def _initiate_datatypes_handling_functions_map(self) -> None:
        self._datatype_handling_functions_map = defaultdict(lambda: lambda x: x, {})

    def _get_custom_collate_fn(
        self, desired_batch_size: int | None = None
    ) -> Callable[[list[object]], object]:
        if desired_batch_size is None:
            desired_batch_size = self.batch_size

        return partial(custom_collate_fn, desired_batch_size=desired_batch_size)

    def setup(self, stage: str | None = None) -> None:
        """
        Apply the configured data scaler to the train, validation, and test splits.

        :param stage: Lightning stage hint ('fit', 'test', etc.); unused but required by the
            interface
        """
        del stage
        scaler = create_data_scaler(
            scale=self.scale_data,
            scaling_range=self.data_scaling_range,
            scaling_method=self.data_scaling_method,
            data_form=self._data_form,
        )
        self._train_data_samples, self._valid_data_samples, self._test_data_samples = scaler(
            self._train_data_samples, self._valid_data_samples, self._test_data_samples
        )

    def _process_train_dataloader(
        self,
        *,
        dataset_object: Dataset[object],
        shuffle: bool | None = None,
        strict_batch_size: bool = False,
        extra_args: dict | None = None,
    ) -> DataLoader:
        """Build a training dataloader with optional strict batch collation."""
        if shuffle is None:
            shuffle = self.shuffle

        return DataLoader(
            dataset=dataset_object,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            **(extra_args or {}),
            **({'persistent_workers': True} if self.num_workers > 0 else {}),
            **({'collate_fn': self._get_custom_collate_fn()} if strict_batch_size else {}),
        )

    def _process_test_dataloader(
        self,
        *,
        dataset_object: Dataset[object],
        strict_batch_size: bool = False,
        extra_args: dict | None = None,
    ) -> DataLoader:
        """Build a deterministic evaluation dataloader for a dataset object."""
        return DataLoader(
            dataset=dataset_object,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            **(extra_args or {}),
            **({'persistent_workers': True} if self.num_workers > 0 else {}),
            **({'collate_fn': self._get_custom_collate_fn()} if strict_batch_size else {}),
        )

    def _process_valid_dataloader(
        self,
        *,
        dataset_object: Dataset[object],
        strict_batch_size: bool = False,
        extra_args: dict | None = None,
    ) -> DataLoader | None:
        """Build a validation dataloader unless validation split is disabled."""
        if self.valid_size == 0.0:
            return None

        return self._process_test_dataloader(
            dataset_object=dataset_object,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )


class BaseClassificationTimeSeriesDataModule(BaseTimeSeriesDataModule, ABC):
    def __init__(
        self,
        *,
        dataset_config_path: Path,
        batch_size: int,
        valid_size: float,  # percentage of the training set to use as validation set
        shuffle: bool,
        scale_data: bool,
        data_scaling_method: str,
        data_form: str,
        data_scaling_range: tuple[float, float],
        splitting_strategy: TimeSeriesClassificationDatasetSplittingStrategy = (
            TimeSeriesClassificationDatasetSplittingStrategy.AS_DEFINED
        ),
        test_size: float = 0.5,
        # percentage of the dataset to use as test set; only valid for MANUAL splitting
        num_workers: int = 1,
    ) -> None:
        """Initialize shared classification data module configuration and metadata."""
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
        self._separate_target_feature = partial(
            separate_target_feature_from_df,
            target_feature_name=self.dataset_config['main_config']['target_col_name'],
        )
        self.target_column_name = self.dataset_config['main_config']['target_col_name']
        self._data_column_names = None
        self._num_classes = None
        self._train_data_labels = None
        self._test_data_labels = None
        self._valid_data_labels = None

    @property
    def n_classes(self) -> int | None:
        """Return the number of classes."""
        return self._num_classes

    @property
    def train_data_labels(self) -> pd.Series | None:
        """Return training split labels."""
        return self._train_data_labels

    @property
    def test_data_labels(self) -> pd.Series | None:
        """Return test split labels."""
        return self._test_data_labels

    @property
    def valid_data_labels(self) -> pd.Series | None:
        """Return validation split labels."""
        return self._valid_data_labels

    @property
    def all_data_labels(self) -> pd.Series | None:
        """Return all available label splits concatenated row-wise."""
        data_samples = [
            data
            for data in [self._train_data_labels, self._test_data_labels, self._valid_data_labels]
            if data is not None
        ]
        return pd.concat(data_samples, axis=0)

    def _extract_data_column_names(self) -> None:
        """Extract the sole feature column name used by univariate datasets."""
        if isinstance(self._train_data_samples, pd.DataFrame):
            self._data_column_names = next(
                column_name
                for column_name in self._train_data_samples.columns
                if column_name != self.target_column_name
            )
        else:
            exception = TypeError('Expected self._train_data_samples to be a pandas DataFrame.')
            raise exception

    def _process_data_with_varying_sequence_lengths(self) -> None:
        """Center variable-length sequences in train, validation, and test splits."""
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
        """Initialize shared forecasting data module configuration and mode."""
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
        self._train_slice = None
        self._valid_slice = None
        self._test_slice = None
        self._full_data = None
        self._num_time_series_features = None

    @property
    def train_slice(self) -> slice | None:
        """Return the index slice for the training partition."""
        return self._train_slice

    @property
    def valid_slice(self) -> slice | None:
        """Return the index slice for the validation partition."""
        return self._valid_slice

    @property
    def test_slice(self) -> slice | None:
        """Return the index slice for the test partition."""
        return self._test_slice

    @property
    def num_time_series_features(self) -> int | None:
        """Return the number of engineered time features."""
        return self._num_time_series_features

    @property
    def full_data(self) -> pd.DataFrame | np.ndarray | None:
        """Return the full time-series matrix before split extraction."""
        return self._full_data

    @abstractmethod
    def _set_data_slices(self) -> None:
        """Derive train, validation, and test slices for the full series array."""
        raise NotImplementedError

    @abstractmethod
    def _transform_data(self) -> None:
        """Transform raw dataset representation into the model-ready array layout."""
        raise NotImplementedError

    def _prepare_data_scaler(self) -> StandardScaler | MinMaxScaler | None:
        if self.scale_data:
            if self.data_scaling_method == 'standardization':
                return StandardScaler()
            if self.data_scaling_method == 'min_max':
                return MinMaxScaler(feature_range=self.data_scaling_range)
        return None

    def _split_data(self) -> None:
        if self._full_data is None:
            exception = ValueError('full_data is not available for splitting.')
            raise exception
        train_data = self._full_data[:, self._train_slice]
        valid_data = self._full_data[:, self._valid_slice]
        test_data = self._full_data[:, self._test_slice]

        self._train_data_samples, self._valid_data_samples, self._test_data_samples = (
            train_data,
            valid_data,
            test_data,
        )

    def setup(self, stage: str | None = None) -> None:
        """Scale, transform, augment, and split forecasting data for the current stage."""
        del stage
        if self._full_data is not None:
            time_series_features = extract_time_features(self._full_data.index)
            num_time_series_features = time_series_features.shape[-1]

            data_scaler = self._prepare_data_scaler()
            if data_scaler is not None:
                data_scaler.fit(self._full_data[self._train_slice])
                self._full_data = data_scaler.transform(self._full_data)
            self._transform_data()

            if num_time_series_features > 0:
                time_series_features_scaler = self._prepare_data_scaler()
                if time_series_features_scaler is not None:
                    time_series_features_scaler.fit(time_series_features[:, self._train_slice])
                    scaled_time_series_features = time_series_features_scaler.transform(
                        time_series_features
                    )
                    scaled_time_series_features = np.expand_dims(
                        scaled_time_series_features, axis=0
                    )
                    repeated_time_series_features = np.repeat(
                        scaled_time_series_features, self._full_data.shape[0], axis=0
                    )
                    self._full_data = np.concatenate(
                        [repeated_time_series_features, self._full_data], axis=-1
                    )

            self._num_time_series_features = num_time_series_features
            self._calculate_num_features()
            self._split_data()

    def _calculate_num_features(self) -> None:
        self._num_features = self._full_data.shape[-1] if self._full_data is not None else 0

    def _post_prepare_data(self) -> None:
        self._set_data_slices()
