"""UEA multivariate classification LightningDataModule.

Reads nested ARFF files via scipy.io.arff.loadarff (D-12), encodes
labels with LabelEncoder, and manages splits with variable-length
handling.

Per D-02, ``data_form`` is hardcoded as ``DataForm.NESTED``.
Per D-12, uses raw scipy loading (not utils/arff.py).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from tscollection.datasets.uea import UEAClassificationMultivariateDataset
from tscollection.datasets.enums.data import (
    ClassificationSplittingStrategy,
    DataForm,
    ScalingMethod,
    TimeSeriesDatasetMode,
)
from tscollection.datasets.modules._base.classification import (
    BaseClassificationTimeSeriesDataModule,
)

if TYPE_CHECKING:
    pass

__all__ = ['UEAClassificationDataModule']

logger = logging.getLogger(__name__)


class UEAClassificationDataModule(BaseClassificationTimeSeriesDataModule):
    """LightningDataModule for UEA multivariate classification datasets.

    Reads multi-dimensional nested ARFF files using raw
    :func:`scipy.io.arff.loadarff` (D-12), decodes byte values,
    encodes labels with :class:`sklearn.preprocessing.LabelEncoder`,
    and manages splits with variable-length handling.

    Per D-02, ``data_form`` is hardcoded as ``DataForm.NESTED``.
    Per D-01, ARFF file patterns are hardcoded:
    ``{dataset_name}_TRAIN.arff`` and ``{dataset_name}_TEST.arff``.

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
            data_form=DataForm.NESTED,
        )

    # ------------------------------------------------------------------
    # ARFF reading
    # ------------------------------------------------------------------

    def _read_arff_data_file(self, file_path: Path) -> Any:
        """Read an ARFF file using scipy.io.arff.loadarff.

        Uses raw scipy loading (D-12), NOT the utils/arff.py helpers.
        Nested ARFF data doesn't fit the DataFrame-based approach.

        Args:
            file_path: Path to the ARFF file.

        Returns:
            Raw numpy structured array from scipy loadarff.
        """
        data, _meta = arff.loadarff(file_path)
        return data

    def _process_stacked_data(
        self, data: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process nested ARFF data into samples and encoded labels.

        Iterates over (sample, label) pairs. For each sample, iterates
        over points, decoding bytes to float. Builds numpy arrays and
        encodes labels with LabelEncoder. Swaps axes so shape is
        (samples, timesteps, features).

        Args:
            data: Raw numpy structured array from scipy.loadarff.

        Returns:
            Tuple of (samples_array, encoded_labels_array).
        """
        processed_data: list[np.ndarray] = []
        labels: list[str] = []

        for sample, label in data:
            sample_list = []
            for point in sample:
                point = point.tolist()
                point = [
                    float(d.decode('utf-8'))
                    if isinstance(d, bytes)
                    else float(d)
                    for d in point
                ]
                sample_list.append(point)
            processed_data.append(np.array(sample_list))

            label_str = (
                label.decode('utf-8') if isinstance(label, bytes) else label
            )
            labels.append(label_str)

        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)
        output_data = np.array(processed_data).astype(np.float32).swapaxes(1, 2)
        return output_data, np.array(encoded_labels)

    # ------------------------------------------------------------------
    # Lightning lifecycle
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Validate paths, read ARFF files, split, and prepare data.

        Per D-16, raises ``FileNotFoundError`` if the dataset folder
        does not exist. Reads train/test ARFF files via scipy.io.arff,
        applies optional manual re-splitting, creates validation split,
        and processes variable-length sequences.
        """
        # Validate folder exists (T-04-02-01, D-16)
        if not self.dataset_folder_path.exists():
            raise FileNotFoundError(
                f'Dataset folder not found: {self.dataset_folder_path}'
            )

        self._dataset_name = self.dataset_folder_path.name

        # Construct ARFF paths (D-01: hardcoded patterns)
        arff_train = self.dataset_folder_path / f'{self._dataset_name}_TRAIN.arff'
        arff_test = self.dataset_folder_path / f'{self._dataset_name}_TEST.arff'

        # Read and process ARFF files via scipy (D-12)
        train_data = self._read_arff_data_file(arff_train)
        test_data = self._read_arff_data_file(arff_test)

        self._train_data_samples, self._train_data_labels = self._process_stacked_data(
            train_data
        )
        self._test_data_samples, self._test_data_labels = self._process_stacked_data(
            test_data
        )

        # Apply splitting strategy
        if self.splitting_strategy == ClassificationSplittingStrategy.MANUAL:
            full_samples = np.concatenate(
                [self._train_data_samples, self._test_data_samples], axis=0
            )
            full_labels = np.concatenate(
                [self._train_data_labels, self._test_data_labels], axis=0
            )
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

        # Create validation split
        self._valid_data_labels = None
        self._valid_data_samples = None
        if self.valid_size > 0.0:
            label_counts = np.bincount(self._train_data_labels)
            valid_mask = np.isin(
                self._train_data_labels, np.where(label_counts > 1)[0]
            )
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
                    logger.warning(
                        'Validation size adjusted to %d samples to cover all classes',
                        test_size,
                    )

        # Variable-length processing
        self._process_data_with_varying_sequence_lengths()

        # Convert labels to pd.Series with category dtype
        self._train_data_labels = pd.Series(self._train_data_labels, dtype='category')
        self._test_data_labels = pd.Series(self._test_data_labels, dtype='category')
        if self.valid_size > 0.0 and self._valid_data_labels is not None:
            self._valid_data_labels = pd.Series(
                self._valid_data_labels, dtype='category'
            )

        # Compute module state
        self._num_classes = len(self._train_data_labels.unique())
        self._seq_len, self._num_features = self._train_data_samples[0].shape

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
        dataset = UEAClassificationMultivariateDataset(
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
        dataset = UEAClassificationMultivariateDataset(
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
        dataset = UEAClassificationMultivariateDataset(
            data=self._test_data_samples,
            labels=self._test_data_labels,
            mode=mode,
        )
        return self._process_test_dataloader(
            dataset_object=dataset,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )
