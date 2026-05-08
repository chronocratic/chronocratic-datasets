__all__ = ['UEATimeSeriesClassificationMultivariateDataModule']

from collections import defaultdict
import logging
from pathlib import Path
import re
from typing import cast

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from src.autotsrc.datasets.classes import UEAClassificationMultivariateDataset
from src.autotsrc.datasets.modules.abstract import BaseClassificationTimeSeriesDataModule
from src.autotsrc.enums import (
    TimeSeriesClassificationDatasetSplittingStrategy,
    TimeSeriesDatasetMode,
)

logger = logging.getLogger(__name__)


class UEATimeSeriesClassificationMultivariateDataModule(BaseClassificationTimeSeriesDataModule):
    def __init__(
        self,
        *,
        dataset_folder_path: Path,
        dataset_config_path: Path,
        batch_size: int = 32,
        valid_size: float = 0.1,  # percentage of the training set to use as validation set
        shuffle: bool = False,
        scale_data: bool = True,
        data_scaling_method: str = 'min_max',
        data_scaling_range: tuple[float, float] = (0, 1),
        splitting_strategy: TimeSeriesClassificationDatasetSplittingStrategy = (
            TimeSeriesClassificationDatasetSplittingStrategy.AS_DEFINED
        ),
        test_size: float = 0.5,
        # percentage of the dataset to use as test set; only valid for MANUAL splitting
        num_workers: int = 0,
    ) -> None:
        """Initialize the UEA Multivariate Time Series Classification Data Module."""
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
        """
        Build the dtype-to-processing-function map used when reading ARFF files.

        Nominal columns are decoded from bytes and cast to integer category codes;
        numeric columns are cast to float64. Unknown types pass through unchanged.
        """
        datatype_handling_functions_map = {
            'nominal': lambda x: x.str.decode('utf-8').astype('category').astype('int64'),
            'numeric': lambda x: x.astype('float64'),
        }

        self._datatype_handling_functions_map = defaultdict(
            lambda: lambda x: x, datatype_handling_functions_map
        )

    def _read_arff_data_file(self, file_path: Path) -> NDArray:
        """
        Load an ARFF file and return the raw structured array.

        :param file_path: path to the ARFF file
        :return: structured NumPy array as returned by scipy.io.arff.loadarff
        """
        data, _ = arff.loadarff(file_path)
        return data

    def _process_stacked_data(self, data: NDArray) -> tuple[NDArray, NDArray]:
        """
        Convert a raw ARFF structured array into float32 sample and integer label arrays.

        Each sample is decoded from bytes where necessary, transposed to
        (time_steps, features) layout, and stacked into a single array.
        Labels are integer-encoded with sklearn's LabelEncoder.

        :param data: structured NumPy array with (sample, label) fields
        :return: tuple of (samples array of shape (N, time_steps, features),
        labels array of shape (N,))
        """
        processed_data = []
        labels = []
        for sample, label in data:
            processed_sample = []
            for data_point in sample:
                data_point_list = data_point.tolist()
                data_point_list = [
                    float(d.decode('utf-8')) if isinstance(d, bytes) else float(d)
                    for d in data_point_list
                ]
                processed_sample.append(data_point_list)
            processed_sample = np.array(processed_sample)
            label_utf8 = label.decode('utf-8')
            processed_data.append(processed_sample)
            labels.append(label_utf8)

        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        output_data = np.array(processed_data).astype(np.float32).swapaxes(1, 2)
        output_labels = np.array(labels)

        return output_data, output_labels

    def prepare_data(self) -> None:
        """Load, split, and preprocess UEA multivariate classification data."""
        self._dataset_name = self.dataset_folder_path.name
        arff_train_file_name = self.dataset_config['main_config']['file_name_patterns']['train'][
            'arff'
        ].replace('{dataset_name}', self._dataset_name)
        arff_test_file_name = self.dataset_config['main_config']['file_name_patterns']['test'][
            'arff'
        ].replace('{dataset_name}', self._dataset_name)
        arff_train_file_path = Path(self.dataset_folder_path, arff_train_file_name)
        arff_test_file_path = Path(self.dataset_folder_path, arff_test_file_name)

        train_data = self._read_arff_data_file(arff_train_file_path)
        test_data = self._read_arff_data_file(arff_test_file_path)

        self._train_data_samples, self._train_data_labels = self._process_stacked_data(train_data)
        self._test_data_samples, self._test_data_labels = self._process_stacked_data(test_data)

        if self.splitting_strategy == TimeSeriesClassificationDatasetSplittingStrategy.MANUAL:
            # merge the train and test data
            full_data_samples = np.concatenate(
                [self._train_data_samples, self._test_data_samples], axis=0
            )
            full_data_labels = np.concatenate(
                [self._train_data_labels, self._test_data_labels], axis=0
            )

            (
                self._train_data_samples,
                self._test_data_samples,
                self._train_data_labels,
                self._test_data_labels,
            ) = train_test_split(
                full_data_samples,
                full_data_labels,
                test_size=self.test_size,
                stratify=full_data_labels,
                random_state=42,
            )

        if self.valid_size > 0.0:
            # Stack samples and labels together for easier filtering
            full_data = list(zip(self._train_data_samples, self._train_data_labels, strict=True))

            # Filter out classes with only one instance
            label_counts = np.bincount(self._train_data_labels)
            valid_labels_mask = np.isin(self._train_data_labels, np.where(label_counts > 1)[0])

            filtered_data = [d for i, d in enumerate(full_data) if valid_labels_mask[i]]
            filtered_samples = np.array([d[0] for d in filtered_data])
            filtered_labels = np.array([d[1] for d in filtered_data])

            try:
                # Perform stratified split
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
                    r'The test_size = \d+ should be greater or equal to the number of classes = \d+'
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
                    logger.warning(
                        (
                            'Validation size was adjusted to include %s samples for dataset '
                            'to solve not having enough instances to cover all classes'
                        ),
                        test_size,
                    )

        self._process_data_with_varying_sequence_lengths()

        # convert labels to pandas series
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
        """Build the training dataloader for UEA multivariate classification."""
        dataset_object = UEAClassificationMultivariateDataset(
            data=cast('np.ndarray', self._train_data_samples),
            labels=cast('pd.Series', self._train_data_labels),
            mode=mode,
        )
        train_dataloader = self._process_train_dataloader(
            dataset_object=dataset_object,
            shuffle=shuffle,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )

        return train_dataloader

    def val_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.WITHOUT_LABELS,
        strict_batch_size: bool = True,
        extra_args: dict | None = None,
    ) -> DataLoader | None:
        """Build the validation dataloader for UEA multivariate classification."""
        dataset_object = UEAClassificationMultivariateDataset(
            data=cast('np.ndarray', self._valid_data_samples),
            labels=cast('pd.Series | None', self._valid_data_labels),
            mode=mode,
        )
        valid_dataloader = self._process_valid_dataloader(
            dataset_object=dataset_object,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )

        return valid_dataloader

    def test_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.WITHOUT_LABELS,
        strict_batch_size: bool = False,
        extra_args: dict | None = None,
    ) -> DataLoader:
        """Build the test dataloader for UEA multivariate classification."""
        dataset_object = UEAClassificationMultivariateDataset(
            data=cast('np.ndarray', self._test_data_samples),
            labels=cast('pd.Series', self._test_data_labels),
            mode=mode,
        )
        test_dataloader = self._process_test_dataloader(
            dataset_object=dataset_object,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )

        return test_dataloader
