"""UCR univariate classification LightningDataModule."""

from __future__ import annotations

from collections import defaultdict
import logging
from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.rbspaper.data.datasets.ucr_dataset import UCRClassificationUnivariateDataset
from src.rbspaper.data.modules.abstract import BaseClassificationTimeSeriesDataModule
from src.rbspaper.data.utils.arff import process_df_according_to_dtypes, read_arff_as_df
from src.rbspaper.enums.data_enums import (
    TimeSeriesClassificationDatasetSplittingStrategy,
    TimeSeriesDatasetMode,
)

__all__ = ['UCRTimeSeriesClassificationUnivariateDataModule']


class UCRTimeSeriesClassificationUnivariateDataModule(BaseClassificationTimeSeriesDataModule):
    """LightningDataModule for UCR univariate classification datasets.

    Reads train/test ARFF files, applies optional manual re-splitting,
    creates a validation split, and handles variable-length series.

    Args:
        dataset_folder_path: Path to the dataset ARFF directory.
        dataset_config_path: Path to JSON config file.
        batch_size: Batch size for dataloaders.
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
            data_form='regular',
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

    def _read_arff_file_as_df(self, file_path: Path) -> pd.DataFrame:
        df, meta = read_arff_as_df(file_path)
        df = process_df_according_to_dtypes(df, meta, self._datatype_handling_functions_map or {})
        return df

    def _clean_data_of_missing_values(self, df_data: pd.DataFrame) -> pd.DataFrame:
        df_data = df_data[df_data[self.target_column_name].notna()]
        feature_cols = [c for c in df_data.columns if c != self.target_column_name]
        df_data = df_data[~df_data[feature_cols].isna().all(axis=1)]
        return df_data

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

        train_data = self._read_arff_file_as_df(arff_train)
        test_data = self._read_arff_file_as_df(arff_test)
        train_data = self._clean_data_of_missing_values(train_data)
        test_data = self._clean_data_of_missing_values(test_data)

        if self.splitting_strategy == TimeSeriesClassificationDatasetSplittingStrategy.MANUAL:
            combined = pd.concat([train_data, test_data], axis=0, ignore_index=True)
            train_data, test_data = train_test_split(
                combined,
                test_size=self.test_size,
                stratify=combined[self.target_column_name],
                random_state=42,
            )

        self._train_data_samples, self._train_data_labels = self._separate_target_feature(
            train_data
        )
        self._test_data_samples, self._test_data_labels = self._separate_target_feature(test_data)

        self._num_classes = len(self._train_data_labels.unique())
        self._seq_len = len(self._train_data_samples.columns)
        self._num_features = 1
        self._extract_data_column_names()

        if self.valid_size > 0.0:
            data_df = self._train_data_samples.copy(deep=True)
            data_df['label'] = self._train_data_labels.copy(deep=True)
            filtered = data_df.groupby('label').filter(lambda x: len(x) > 1)
            X_filt = filtered.drop('label', axis=1)
            y_filt = filtered['label']
            try:
                (
                    self._train_data_samples,
                    self._valid_data_samples,
                    self._train_data_labels,
                    self._valid_data_labels,
                ) = train_test_split(
                    X_filt, y_filt, test_size=self.valid_size, stratify=y_filt, random_state=42
                )
            except ValueError as e:
                pattern = (
                    r'The test_size = \d+ should be'
                    r' greater or equal to the'
                    r' number of classes = \d+'
                )
                if re.match(pattern, str(e)):
                    test_size = len(set(y_filt))
                    (
                        self._train_data_samples,
                        self._valid_data_samples,
                        self._train_data_labels,
                        self._valid_data_labels,
                    ) = train_test_split(
                        X_filt, y_filt, test_size=test_size, stratify=y_filt, random_state=42
                    )
                    logging.warning(
                        "Validation size adjusted to %d for dataset '%s' to cover all classes",
                        test_size,
                        self._dataset_name,
                    )

        self._process_data_with_varying_sequence_lengths()

    def train_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.WITHOUT_LABELS,
        shuffle: bool | None = None,
        strict_batch_size: bool = True,
        extra_args: dict | None = None,
    ) -> DataLoader:
        dataset = UCRClassificationUnivariateDataset(
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
        dataset = UCRClassificationUnivariateDataset(
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
        dataset = UCRClassificationUnivariateDataset(
            data=self._test_data_samples, labels=self._test_data_labels, mode=mode
        )
        return self._process_test_dataloader(
            dataset_object=dataset, strict_batch_size=strict_batch_size, extra_args=extra_args
        )
