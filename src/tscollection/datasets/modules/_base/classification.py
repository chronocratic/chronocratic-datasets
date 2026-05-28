"""Base LightningDataModule for classification time series datasets.

Provides label handling, target column separation, splitting strategy
management, and variable-length sequence processing for classification
datasets (UCR, UEA).
"""

from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import Any, TYPE_CHECKING

from tscollection.datasets.enums.data import (
    ClassificationSplittingStrategy,
    DataForm,
    ScalingMethod,
)
from tscollection.datasets.modules._base.base import BaseTimeSeriesDataModule
from tscollection.datasets.utils.common import separate_target_feature_from_df
from tscollection.datasets.utils.general import process_data_with_varying_sequence_lengths_single

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from torch.utils.data import DataLoader


__all__ = ['BaseClassificationTimeSeriesDataModule']


class BaseClassificationTimeSeriesDataModule(BaseTimeSeriesDataModule):
    """Base LightningDataModule for classification time series datasets.

    Extends :class:`BaseTimeSeriesDataModule` with label handling,
    target column separation, and variable-length sequence processing.
    Used by UCR and UEA classification modules.

    The constructor accepts ``target_column_name`` as an explicit parameter,
    uses :class:`ClassificationSplittingStrategy` enum for splitting,
    and relies on the inherited ``setup()`` which calls
    ``create_data_scaler()``.

    Args:
        dataset_folder_path: Path to the dataset folder containing
            ARFF/CSV files.
        batch_size: Batch size for dataloaders.
        valid_size: Fraction of training data reserved for validation.
        shuffle: Whether to shuffle the training dataloader.
        scale_data: Whether to apply data scaling.
        data_scaling_method: Scaling algorithm, typed as
            :class:`~tscollection.datasets.enums.data.ScalingMethod`.
        data_scaling_range: Target ``(min, max)`` range for
            :data:`ScalingMethod.MINMAX`.
        target_column_name: Name of the target/label column in the data.
        splitting_strategy: How to split train/test data, typed as
            :class:`~tscollection.datasets.enums.data.ClassificationSplittingStrategy`.
        test_size: Fraction reserved as test set (used with
            :data:`ClassificationSplittingStrategy.MANUAL`).
        num_workers: Number of DataLoader worker processes.
    """

    def __init__(
        self,
        *,
        dataset_folder_path: Path,
        batch_size: int = 32,
        valid_size: float = 0.1,
        shuffle: bool = False,
        scale_data: bool = True,
        data_scaling_method: ScalingMethod = ScalingMethod.MINMAX,
        data_scaling_range: tuple[float, float] = (0, 1),
        target_column_name: str,
        splitting_strategy: ClassificationSplittingStrategy = (
            ClassificationSplittingStrategy.AS_DEFINED
        ),
        test_size: float = 0.5,
        num_workers: int = 0,
        data_form: DataForm = DataForm.REGULAR,
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
            num_workers=num_workers,
            data_form=data_form,
        )
        self.dataset_folder_path = dataset_folder_path
        self.target_column_name = target_column_name
        self.splitting_strategy = splitting_strategy
        self._separate_target_feature = partial(
            separate_target_feature_from_df, target_feature_name=self.target_column_name
        )
        self._data_column_names: str | None = None
        self._num_classes: int | None = None
        self._train_data_labels: Any = None
        self._test_data_labels: Any = None
        self._valid_data_labels: Any = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int | None:
        """Number of distinct classes (read-only)."""
        return self._num_classes

    @property
    def train_data_labels(self) -> Any:
        """Training data labels."""
        return self._train_data_labels

    @property
    def test_data_labels(self) -> Any:
        """Test data labels."""
        return self._test_data_labels

    @property
    def valid_data_labels(self) -> Any:
        """Validation data labels."""
        return self._valid_data_labels

    @property
    def all_data_labels(self) -> pd.Series:
        """Concatenation of all label splits."""
        import pandas as pd

        return pd.concat(
            [self._train_data_labels, self._test_data_labels, self._valid_data_labels], axis=0
        )

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    def _compute_dimensions(self) -> tuple[int | None, int | None]:
        """Compute dimensions from classification train data samples.

        Raises RuntimeError if prepare_data() was never called, as
        train samples are required to determine feature count and
        sequence length.

        Returns:
            Tuple of (n_features, sequence_len).

        Raises:
            RuntimeError: If _train_data_samples is None (prepare_data
                not yet called).
        """
        if self._train_data_samples is None:
            msg = 'prepare_dimensions() requires prepare_data() to have run first'
            raise RuntimeError(msg)
        return self._num_features, self._seq_len

    @abstractmethod
    def _do_prepare_data(self) -> None:
        """Validate file paths, read data, and split into train/val/test.

        Subclasses must implement this method to:
        1. Validate folder/file existence.
        2. Read data (e.g., ARFF files).
        3. Clean missing values.
        4. Apply splitting strategy (AS_DEFINED or MANUAL).
        5. Separate target features.
        6. Compute ``_num_classes``, ``_seq_len``, ``_num_features``.
        7. Create validation split.
        8. Call ``_process_data_with_varying_sequence_lengths()``.
        """

    @abstractmethod
    def train_dataloader(self, **kwargs: Any) -> DataLoader:
        """Return the training DataLoader."""

    @abstractmethod
    def val_dataloader(self, **kwargs: Any) -> DataLoader | None:
        """Return the validation DataLoader."""

    @abstractmethod
    def test_dataloader(self, **kwargs: Any) -> DataLoader:
        """Return the test DataLoader."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_data_with_varying_sequence_lengths(self) -> None:
        """Apply variable-length centering to all data splits.

        Processes training, validation, and test data samples using
        :func:`process_data_with_varying_sequence_lengths_single`.
        Skips validation split when ``_valid_data_samples`` is ``None``.
        """
        self._train_data_samples = process_data_with_varying_sequence_lengths_single(
            data=self._train_data_samples  # ty:ignore[invalid-argument-type]
        )
        if self._valid_data_samples is not None:
            self._valid_data_samples = process_data_with_varying_sequence_lengths_single(
                data=self._valid_data_samples
            )
        self._test_data_samples = process_data_with_varying_sequence_lengths_single(
            data=self._test_data_samples  # ty:ignore[invalid-argument-type]
        )
