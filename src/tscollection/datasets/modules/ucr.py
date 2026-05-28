"""UCR univariate classification LightningDataModule.

Reads train/test ARFF files, applies optional manual re-splitting,
creates a validation split, and handles variable-length series.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tscollection.datasets.ucr import UCRClassificationUnivariateDataset
from tscollection.datasets.enums.data import (
    ClassificationSplittingStrategy,
    DataForm,
    ScalingMethod,
    TimeSeriesDatasetMode,
)
from tscollection.datasets.modules._base.classification import (
    BaseClassificationTimeSeriesDataModule,
)
from tscollection.datasets.utils.arff import (
    process_df_according_to_dtypes,
    read_arff_as_df,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ['UCRClassificationDataModule']

logger = logging.getLogger(__name__)


class UCRClassificationDataModule(BaseClassificationTimeSeriesDataModule):
    """LightningDataModule for UCR univariate classification datasets.

    Reads train/test ARFF files, applies optional manual re-splitting,
    creates a validation split, and handles variable-length series.

    Per D-01, accepts ``dataset_folder_path`` (Path) and
    ``target_column_name`` as explicit constructor parameters.
    No JSON config files. ARFF file patterns are hardcoded:
    ``{dataset_name}_TRAIN.arff`` and ``{dataset_name}_TEST.arff``.

    Per D-02, ``data_form`` is hardcoded as ``DataForm.REGULAR``.

    Args:
        dataset_folder_path: Path to the dataset ARFF directory.
        target_column_name: Name of the target/label column in the ARFF files.
        batch_size: Batch size for dataloaders.
        valid_size: Fraction of training data for validation.
        shuffle: Whether to shuffle training data.
        scale_data: Whether to scale features.
        data_scaling_method: Scaling algorithm, typed as
            :class:`~tscollection.datasets.enums.data.ScalingMethod`.
        data_scaling_range: Target ``(min, max)`` range for
            :data:`ScalingMethod.MINMAX`.
        splitting_strategy: ``AS_DEFINED`` or ``MANUAL`` splitting,
            typed as
            :class:`~tscollection.datasets.enums.data.ClassificationSplittingStrategy`.
        test_size: Test set fraction for ``MANUAL`` splitting.
        num_workers: DataLoader worker count.
    """

    def __init__(
        self,
        *,
        dataset_folder_path: Path,
        target_column_name: str,
        batch_size: int = 32,
        valid_size: float = 0.1,
        shuffle: bool = False,
        scale_data: bool = True,
        data_scaling_method: ScalingMethod = ScalingMethod.MINMAX,
        data_scaling_range: tuple[float, float] = (0, 1),
        splitting_strategy: ClassificationSplittingStrategy = (
            ClassificationSplittingStrategy.AS_DEFINED
        ),
        test_size: float = 0.5,
        num_workers: int = 0,
    ) -> None:
        super().__init__(
            dataset_folder_path=dataset_folder_path,
            batch_size=batch_size,
            valid_size=valid_size,
            shuffle=shuffle,
            scale_data=scale_data,
            data_scaling_method=data_scaling_method,
            data_scaling_range=data_scaling_range,
            target_column_name=target_column_name,
            splitting_strategy=splitting_strategy,
            test_size=test_size,
            num_workers=num_workers,
            data_form=DataForm.REGULAR,
        )

    def _initiate_datatypes_handling_functions_map(self) -> None:
        """Initialize ARFF dtype handling map.

        Nominal columns are decoded from bytes to UTF-8, cast to
        category, then to int64. Numeric columns are cast to float64.
        """
        self._datatype_handling_functions_map = defaultdict(
            lambda: lambda x: x,
            {
                'nominal': lambda x: x.str.decode(
                    'utf-8'
                ).astype('category').astype('int64'),
                'numeric': lambda x: x.astype('float64'),
            },
        )

    def _read_arff_file_as_df(self, file_path: Path) -> pd.DataFrame:
        """Read an ARFF file into a processed pandas DataFrame.

        Args:
            file_path: Path to the ARFF file.

        Returns:
            DataFrame with properly typed columns.
        """
        df, meta = read_arff_as_df(file_path)
        df = process_df_according_to_dtypes(
            df, meta, self._datatype_handling_functions_map or {}  # type: ignore[arg-type]
        )
        return df

    def _clean_data_of_missing_values(self, df_data: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with missing target or all-missing features.

        Args:
            df_data: DataFrame containing target and feature columns.

        Returns:
            Cleaned DataFrame with missing values removed.
        """
        df_data = df_data[df_data[self.target_column_name].notna()]
        feature_cols = [
            c for c in df_data.columns if c != self.target_column_name
        ]
        df_data = df_data[~df_data[feature_cols].isna().all(axis=1)]
        return df_data

    # ------------------------------------------------------------------
    # Lightning lifecycle
    # ------------------------------------------------------------------

    def _do_prepare_data(self) -> None:
        """Validate paths, read ARFF files, split, and prepare data.

        Per D-16, raises ``FileNotFoundError`` if the dataset folder
        does not exist. Reads train/test ARFF files, applies optional
        manual re-splitting, creates validation split, and processes
        variable-length sequences.
        """
        # Validate folder exists (T-04-02-01)
        if not self.dataset_folder_path.exists():
            raise FileNotFoundError(
                f'Dataset folder not found: {self.dataset_folder_path}'
            )

        self._dataset_name = self.dataset_folder_path.name

        # Construct ARFF paths (D-01: hardcoded patterns)
        arff_train = self.dataset_folder_path / f'{self._dataset_name}_TRAIN.arff'
        arff_test = self.dataset_folder_path / f'{self._dataset_name}_TEST.arff'

        # Read and process ARFF files
        train_data = self._read_arff_file_as_df(arff_train)
        test_data = self._read_arff_file_as_df(arff_test)

        # Clean missing values
        train_data = self._clean_data_of_missing_values(train_data)
        test_data = self._clean_data_of_missing_values(test_data)

        # Apply splitting strategy
        if self.splitting_strategy == ClassificationSplittingStrategy.MANUAL:
            combined = pd.concat(
                [train_data, test_data], axis=0, ignore_index=True
            )
            train_data, test_data = train_test_split(
                combined,
                test_size=self.test_size,
                stratify=combined[self.target_column_name],
                random_state=42,
            )

        # Separate target features
        (
            self._train_data_samples,
            self._train_data_labels,
        ) = self._separate_target_feature(train_data)
        (
            self._test_data_samples,
            self._test_data_labels,
        ) = self._separate_target_feature(test_data)

        # Compute module state
        self._num_classes = len(self._train_data_labels.unique())
        self._seq_len = len(self._train_data_samples.columns)
        self._num_features = 1

        # Create validation split
        self._valid_data_labels = None
        self._valid_data_samples = None
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
                    X_filt,
                    y_filt,
                    test_size=self.valid_size,
                    stratify=y_filt,
                    random_state=42,
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
                        X_filt,
                        y_filt,
                        test_size=test_size,
                        stratify=y_filt,
                        random_state=42,
                    )
                    logger.warning(
                        "Validation size adjusted to %d for dataset '%s' "
                        'to cover all classes',
                        test_size,
                        self._dataset_name,
                    )

        # Variable-length processing
        self._process_data_with_varying_sequence_lengths()

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.WITHOUT_LABELS,
        shuffle: bool | None = None,
        strict_batch_size: bool = True,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader:
        """Build the training DataLoader.

        Args:
            mode: Dataset mode (with/without labels, forecasting).
            shuffle: Whether to shuffle. Defaults to :attr:`shuffle`.
            strict_batch_size: If True, pad the last batch via
                :func:`custom_collate_fn`.
            extra_args: Additional keyword arguments forwarded to
                the DataLoader constructor.

        Returns:
            Configured DataLoader for training.
        """
        dataset = UCRClassificationUnivariateDataset(
            data=self._train_data_samples,
            labels=self._train_data_labels,
            mode=mode,
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
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader | None:
        """Build the validation DataLoader.

        Returns ``None`` when :attr:`valid_size` is ``0.0``.

        Args:
            mode: Dataset mode (with/without labels, forecasting).
            strict_batch_size: If True, pad the last batch via
                :func:`custom_collate_fn`.
            extra_args: Additional keyword arguments forwarded to
                the DataLoader constructor.

        Returns:
            Configured DataLoader for validation, or ``None``.
        """
        if self._valid_data_samples is None or self._valid_data_labels is None:
            return None
        dataset = UCRClassificationUnivariateDataset(
            data=self._valid_data_samples,
            labels=self._valid_data_labels,
            mode=mode,
        )
        return self._process_valid_dataloader(
            dataset_object=dataset,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )

    def test_dataloader(
        self,
        *,
        mode: TimeSeriesDatasetMode = TimeSeriesDatasetMode.WITHOUT_LABELS,
        strict_batch_size: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader:
        """Build the test DataLoader.

        Args:
            mode: Dataset mode (with/without labels, forecasting).
            strict_batch_size: If True, pad the last batch via
                :func:`custom_collate_fn`.
            extra_args: Additional keyword arguments forwarded to
                the DataLoader constructor.

        Returns:
            Configured DataLoader for testing.
        """
        dataset = UCRClassificationUnivariateDataset(
            data=self._test_data_samples,
            labels=self._test_data_labels,
            mode=mode,
        )
        return self._process_test_dataloader(
            dataset_object=dataset,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )
