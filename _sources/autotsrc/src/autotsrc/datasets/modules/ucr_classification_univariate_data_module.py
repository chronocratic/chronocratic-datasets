__all__ = ['UCRTimeSeriesClassificationUnivariateDataModule']

from collections import defaultdict
import logging
from pathlib import Path
import re
from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.autotsrc.datasets.classes import UCRClassificationUnivariateDataset
from src.autotsrc.datasets.modules.abstract import BaseClassificationTimeSeriesDataModule
from src.autotsrc.enums import (
    TimeSeriesClassificationDatasetSplittingStrategy,
    TimeSeriesDatasetMode,
)
from src.autotsrc.utils.data.arff import process_df_according_to_dtypes, read_arff_as_df

logger = logging.getLogger(__name__)


class UCRTimeSeriesClassificationUnivariateDataModule(BaseClassificationTimeSeriesDataModule):
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
        datatype_handling_functions_map = {
            'nominal': lambda x: x.str.decode('utf-8').astype('category').astype('int64'),
            'numeric': lambda x: x.astype('float64'),
        }

        self._datatype_handling_functions_map = defaultdict(
            lambda: lambda x: x, datatype_handling_functions_map
        )

    def _read_arff_file_as_df(self, file_path: Path) -> pd.DataFrame:
        """
        Read an ARFF file and apply dtype-specific processing functions.

        :param file_path: path to the ARFF file
        :return: processed DataFrame
        """
        df, meta = read_arff_as_df(file_path)
        df = process_df_according_to_dtypes(df, meta, self._datatype_handling_functions_map)
        return df

    def _clean_data_of_missing_values(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with a missing label or with all feature values missing.

        :param df_data: raw DataFrame containing features and the target column
        :return: cleaned DataFrame
        """
        # Remove rows where label is None
        df_data = df_data[df_data[self.target_column_name].notna()]

        # Remove rows where all feature columns are None
        feature_columns_names_list = list(df_data.columns)
        feature_columns_names_list.remove(self.target_column_name)
        df_data = df_data[~df_data[feature_columns_names_list].isna().all(axis=1)]

        return df_data

    def prepare_data(self) -> None:
        """Load, clean, split, and preprocess UCR univariate classification data."""
        self._dataset_name = self.dataset_folder_path.name
        arff_train_file_name = self.dataset_config['main_config']['file_name_patterns']['train'][
            'arff'
        ].replace('{dataset_name}', self._dataset_name)
        arff_test_file_name = self.dataset_config['main_config']['file_name_patterns']['test'][
            'arff'
        ].replace('{dataset_name}', self._dataset_name)
        arff_train_file_path = Path(self.dataset_folder_path, arff_train_file_name)
        arff_test_file_path = Path(self.dataset_folder_path, arff_test_file_name)

        train_data = self._read_arff_file_as_df(arff_train_file_path)
        test_data = self._read_arff_file_as_df(arff_test_file_path)

        train_data = self._clean_data_of_missing_values(train_data)
        test_data = self._clean_data_of_missing_values(test_data)

        if self.splitting_strategy == TimeSeriesClassificationDatasetSplittingStrategy.MANUAL:
            # merge the train and test data
            combined_data = pd.concat([train_data, test_data], axis=0, ignore_index=True)
            train_data, test_data = train_test_split(
                combined_data,
                test_size=self.test_size,
                stratify=combined_data[self.target_column_name],
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
            # Convert to DataFrame for easier manipulation
            data_df = self._train_data_samples.copy(deep=True)
            data_df['label'] = self._train_data_labels.copy(deep=True)

            # Filter out classes with only one instance
            filtered_data = data_df.groupby('label').filter(lambda x: len(x) > 1)

            # Split the filtered data
            x_filtered = filtered_data.drop('label', axis=1)
            y_filtered = filtered_data['label']
            try:
                (
                    self._train_data_samples,
                    self._valid_data_samples,
                    self._train_data_labels,
                    self._valid_data_labels,
                ) = train_test_split(
                    x_filtered,
                    y_filtered,
                    test_size=self.valid_size,
                    stratify=y_filtered,
                    random_state=42,
                )
            except ValueError as e:
                pattern = (
                    r'The test_size = \d+ should be greater or equal to the number of classes = \d+'
                )
                if re.match(pattern, str(e)):
                    test_size = len(set(y_filtered))
                    (
                        self._train_data_samples,
                        self._valid_data_samples,
                        self._train_data_labels,
                        self._valid_data_labels,
                    ) = train_test_split(
                        x_filtered,
                        y_filtered,
                        test_size=test_size,
                        stratify=y_filtered,
                        random_state=42,
                    )
                    logger.warning(
                        (
                            'Validation size was adjusted to include %s for dataset "%s" '
                            'to solve not having enough instances to cover all classes'
                        ),
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
        """Build the training dataloader for UCR univariate classification."""
        dataset_object = UCRClassificationUnivariateDataset(
            data=cast('pd.DataFrame', self._train_data_samples),
            labels=self._train_data_labels,
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
        """Build the validation dataloader for UCR univariate classification."""
        dataset_object = UCRClassificationUnivariateDataset(
            data=cast('pd.DataFrame', self._valid_data_samples),
            labels=self._valid_data_labels,
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
        """Build the test dataloader for UCR univariate classification."""
        dataset_object = UCRClassificationUnivariateDataset(
            data=cast('pd.DataFrame', self._test_data_samples),
            labels=self._test_data_labels,
            mode=mode,
        )
        test_dataloader = self._process_test_dataloader(
            dataset_object=dataset_object,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )

        return test_dataloader
