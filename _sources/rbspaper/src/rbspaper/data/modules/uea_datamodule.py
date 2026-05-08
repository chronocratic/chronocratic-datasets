"""UEA multivariate classification LightningDataModule."""

from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.rbspaper.data.datasets.uea_dataset import UEAClassificationMultivariateDataset
from src.rbspaper.data.modules.abstract import BaseClassificationTimeSeriesDataModule
from src.rbspaper.enums.data_enums import (
    TimeSeriesClassificationDatasetSplittingStrategy,
    TimeSeriesDatasetMode,
)

__all__ = ['UEATimeSeriesClassificationMultivariateDataModule']

if TYPE_CHECKING:
    from typing import Any


class UEATimeSeriesClassificationMultivariateDataModule(BaseClassificationTimeSeriesDataModule):
    """LightningDataModule for UEA multivariate classification datasets.

    Reads multi-dimensional ARFF files, encodes labels, and manages
    splits with variable-length handling.

    Args:
        dataset_folder_path: Path to the dataset ARFF directory.
        dataset_config_path: Path to JSON config file.
        batch_size: Batch size.
        valid_size: Fraction of training data for validation.
        shuffle: Whether to shuffle training data.
        scale_data: Whether to scale features.
        data_scaling_method: Scaling algorithm.
        data_scaling_range: Target min-max range.
        splitting_strategy: 'AS_DEFINED' or 'MANUAL'.
        test_size: Test set fraction for MANUAL splitting.
        num_workers: DataLoader worker count.
    """

    def __init__(
        self,
        *,
        dataset_folder_path: Path,
        dataset_config_path: Path,
        batch_size: int = 32,
        valid_size: float = 0.1,
        shuffle: bool = False,
        scale_data: bool = True,
        data_scaling_method: str = 'min_max',
        data_scaling_range: tuple[float, float] = (0, 1),
        splitting_strategy: TimeSeriesClassificationDatasetSplittingStrategy = (
            TimeSeriesClassificationDatasetSplittingStrategy.AS_DEFINED
        ),
        test_size: float = 0.5,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            dataset_config_path=dataset_config_path,
            batch_size=batch_size,
            valid_size=valid_size,
            shuffle=shuffle,
            scale_data=scale_data,
            data_scaling_method=data_scaling_method,
            data_scaling_range=data_scaling_range,
            splitting_strategy=splitting_strategy,
            test_size=test_size,
            data_form='nested',
            num_workers=num_workers,
        )
        self.dataset_folder_path = dataset_folder_path

    def _initiate_datatypes_handling_functions_map(self) -> None:
        self._datatype_handling_functions_map = defaultdict(
            lambda: lambda x: x,
            {
                'nominal': lambda x: x.str.decode('utf-8').astype('category').astype('int64'),
                'numeric': lambda x: x.astype('float64'),
            },
        )

    def _read_arff_data_file(self, file_path: Path) -> Any:
        data, meta = arff.loadarff(file_path)
        return data

    def _process_stacked_data(self, data: Any) -> tuple[np.ndarray, np.ndarray]:
        processed_data: list[np.ndarray] = []
        labels: list[str] = []
        for sample, label in data:
            sample_list = []
            for point in sample:
                point = point.tolist()
                point = [
                    float(d.decode('utf-8')) if isinstance(d, bytes) else float(d) for d in point
                ]
                sample_list.append(point)
            processed_data.append(np.array(sample_list))
            labels.append(label.decode('utf-8') if isinstance(label, bytes) else label)

        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)
        output_data = np.array(processed_data).astype(np.float32).swapaxes(1, 2)
        return output_data, np.array(encoded_labels)

    def prepare_data(self) -> None:
        self._dataset_name = self.dataset_folder_path.name
        config = self.dataset_config['main_config']['file_name_patterns']
        arff_train = Path(
            self.dataset_folder_path,
            config['train']['arff'].replace('{dataset_name}', self._dataset_name),
        )
        arff_test = Path(
            self.dataset_folder_path,
            config['test']['arff'].replace('{dataset_name}', self._dataset_name),
        )

        train_data = self._read_arff_data_file(arff_train)
        test_data = self._read_arff_data_file(arff_test)

        self._train_data_samples, self._train_data_labels = self._process_stacked_data(train_data)
        self._test_data_samples, self._test_data_labels = self._process_stacked_data(test_data)

        if self.splitting_strategy == TimeSeriesClassificationDatasetSplittingStrategy.MANUAL:
            full_samples = np.concatenate(
                [self._train_data_samples, self._test_data_samples], axis=0
            )
            full_labels = np.concatenate([self._train_data_labels, self._test_data_labels], axis=0)
            (
                self._train_data_samples,
                self._test_data_samples,
                self._train_data_labels,
                self._test_data_labels,
            ) = train_test_split(
                full_samples,
                full_labels,
                test_size=self.test_size,
                stratify=full_labels,
                random_state=42,
            )

        if self.valid_size > 0.0:
            label_counts = np.bincount(self._train_data_labels)
            valid_mask = np.isin(self._train_data_labels, np.where(label_counts > 1)[0])
            filtered_samples = self._train_data_samples[valid_mask]
            filtered_labels = self._train_data_labels[valid_mask]

            try:
                (
                    self._train_data_samples,
                    self._valid_data_samples,
                    self._train_data_labels,
                    self._valid_data_labels,
                ) = train_test_split(
                    filtered_samples,
                    filtered_labels,
                    test_size=self.valid_size,
                    stratify=filtered_labels,
                    random_state=42,
                )
            except ValueError as e:
                pattern = (
                    r'The test_size = \d+ should be'
                    r' greater or equal to the'
                    r' number of classes = \d+'
                )
                if re.match(pattern, str(e)):
                    test_size = len(set(filtered_labels))
                    (
                        self._train_data_samples,
                        self._valid_data_samples,
                        self._train_data_labels,
                        self._valid_data_labels,
                    ) = train_test_split(
                        filtered_samples,
                        filtered_labels,
                        test_size=test_size,
                        stratify=filtered_labels,
                        random_state=42,
                    )
                    logging.warning(
                        'Validation size adjusted to %d samples to cover all classes', test_size
                    )

        self._process_data_with_varying_sequence_lengths()

        self._train_data_labels = pd.Series(self._train_data_labels, dtype='category')
        self._test_data_labels = pd.Series(self._test_data_labels, dtype='category')
        self._valid_data_labels = (
            pd.Series(self._valid_data_labels, dtype='category') if self.valid_size > 0.0 else None
        )

        self._num_classes = len(self._train_data_labels.unique())
        self._seq_len, self._num_features = self._train_data_samples[0].shape

    def train_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.WITHOUT_LABELS,
        shuffle: bool | None = None,
        strict_batch_size: bool = True,
        extra_args: dict | None = None,
    ) -> DataLoader:
        dataset = UEAClassificationMultivariateDataset(
            data=self._train_data_samples, labels=self._train_data_labels, mode=mode
        )
        return self._process_train_dataloader(
            dataset_object=dataset,
            shuffle=shuffle,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )

    def val_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.WITHOUT_LABELS,
        strict_batch_size: bool = True,
        extra_args: dict | None = None,
    ) -> DataLoader | None:
        dataset = UEAClassificationMultivariateDataset(
            data=self._valid_data_samples, labels=self._valid_data_labels, mode=mode
        )
        return self._process_valid_dataloader(
            dataset_object=dataset, strict_batch_size=strict_batch_size, extra_args=extra_args
        )

    def test_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.WITHOUT_LABELS,
        strict_batch_size: bool = False,
        extra_args: dict | None = None,
    ) -> DataLoader:
        dataset = UEAClassificationMultivariateDataset(
            data=self._test_data_samples, labels=self._test_data_labels, mode=mode
        )
        return self._process_test_dataloader(
            dataset_object=dataset, strict_batch_size=strict_batch_size, extra_args=extra_args
        )
