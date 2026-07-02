"""Base LightningDataModule for classification time series datasets.

Provides label handling, target column separation, splitting strategy
management, variable-length sequence processing, and cache-aware setup
for classification datasets (UCR, UEA).
"""

from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING

import pandas as pd

from chronocratic.datasets.enums.data import (
    ClassificationLoaderMode,
    ClassificationSplitMode,
    DataForm,
    ScalingMethod,
)
from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule
from chronocratic.datasets.utils.common import separate_target_feature_from_df
from chronocratic.datasets.utils.general import process_data_with_varying_sequence_lengths_single

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from torch.utils.data import DataLoader


__all__ = ["BaseClassificationTimeSeriesDataModule"]


class BaseClassificationTimeSeriesDataModule(BaseTimeSeriesDataModule):
    """Base LightningDataModule for classification time series datasets.

    Extends :class:`BaseTimeSeriesDataModule` with label handling,
    target column separation, and variable-length sequence processing.
    Used by UCR and UEA classification modules.

    The constructor accepts ``target_column_name`` as an explicit parameter,
    uses :class:`ClassificationSplitMode` enum for splitting,
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
            :class:`~chronocratic.datasets.enums.data.ScalingMethod`.
        data_scaling_range: Target ``(min, max)`` range for
            :data:`ScalingMethod.MINMAX`.
        target_column_name: Name of the target/label column in the data.
        splitting_strategy: How to split train/test data, typed as
            :class:`~chronocratic.datasets.enums.data.ClassificationSplitMode`.
        test_size: Fraction reserved as test set (used with
            :data:`ClassificationSplitMode.MANUAL`).
        num_workers: Number of DataLoader worker processes.
        loader_mode: Loader mode for classification dataloaders, typed as
            :class:`~chronocratic.datasets.enums.data.ClassificationLoaderMode`.
            Defaults to :data:`ClassificationLoaderMode.SAMPLE_LABEL`.
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
        splitting_strategy: ClassificationSplitMode = (ClassificationSplitMode.AS_DEFINED),
        test_size: float = 0.5,
        num_workers: int = 0,
        data_form: DataForm = DataForm.REGULAR,
        loader_mode: ClassificationLoaderMode = ClassificationLoaderMode.SAMPLE_LABEL,
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
        self._cache_key: str | None = None
        self._data_column_names: str | None = None
        self._num_classes: int | None = None
        self._train_data_labels: pd.Series | None = None
        self._test_data_labels: pd.Series | None = None
        self._valid_data_labels: pd.Series | None = None
        self.loader_mode = loader_mode

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
        splits = [
            s
            for s in (self._train_data_labels, self._test_data_labels, self._valid_data_labels)
            if s is not None
        ]
        if not splits:
            msg = "No data loaded. Call prepare_data() and setup() first."
            raise RuntimeError(msg)
        return pd.concat(splits, axis=0)

    @property
    def loader_mode(self) -> ClassificationLoaderMode:
        """Loader mode for classification dataloaders."""
        return self._loader_mode

    @loader_mode.setter
    def loader_mode(self, value: ClassificationLoaderMode) -> None:
        """Set loader mode with type validation.

        Raises:
            TypeError: If value is not a ClassificationLoaderMode.
        """
        if not isinstance(value, ClassificationLoaderMode):
            msg = f"loader_mode must be ClassificationLoaderMode, got {type(value).__name__}"
            raise TypeError(msg)
        self._loader_mode = value

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------

    def _compute_dimensions(self) -> tuple[int | None, int | None]:
        """Fallback dimension computation when metadata is unavailable.

        Raises RuntimeError since classification relies on cached attrs
        or metadata.json for dimensions. This path should only be hit if
        neither ``prepare_data()`` nor cache metadata are available.

        Returns:
            Tuple of (n_features, sequence_len).

        Raises:
            RuntimeError: If dimensions cannot be resolved.
        """
        if self._num_features is not None:
            return self._num_features, self._seq_len
        msg = "prepare_dimensions() requires prepare_data() to have run first"
        raise RuntimeError(msg)

    def setup(self, stage: str | None = None) -> None:
        """Load cached splits and apply data scaling.

        Calls ``_load_cached_data()`` to populate in-memory data arrays
        from the cache written by ``prepare_data()``, then delegates to
        the base class for scaling.

        Args:
            stage: Lightning stage identifier. Defaults to ``None``
                (equivalent to ``"fit"``).

        Raises:
            ValueError: If stage is not one of
                ``{'fit', 'validate', 'test', 'predict', None}``.
        """
        if stage not in ("fit", "validate", "test", "predict", None):
            msg = f"Unknown stage: {stage!r}"
            raise ValueError(msg)
        self._load_cached_data()
        super().setup(stage=stage)

    def reset(self) -> None:
        """Reset classification state while preserving the cache key."""
        original_cache_key = self._cache_key
        super().reset()
        self._cache_key = original_cache_key

    @abstractmethod
    def _load_cached_data(self) -> None:
        """Load cached data splits from the npz cache file.

        Subclasses must implement this method to read the npz file
        written by ``_do_prepare_data()`` and populate
        ``_train_data_samples``, ``_test_data_samples``,
        ``_valid_data_samples``, ``_train_data_labels``,
        ``_test_data_labels``, and ``_valid_data_labels``.
        """

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
