"""UEA multivariate classification LightningDataModule.

Reads nested ARFF files via scipy.io.arff.loadarff, encodes
labels with LabelEncoder, and manages splits with variable-length
handling. Caches post-processed splits for DDP-safe setup().

``data_form`` is hardcoded as ``DataForm.NESTED``.
Uses raw scipy loading (not utils/arff.py).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from chronocratic.datasets.datatypes.uea import UEAClassificationMultivariateDataset
from chronocratic.datasets.enums.data import (
    ClassificationLoaderMode,
    ClassificationSplitMode,
    DataForm,
    ScalingMethod,
)
from chronocratic.datasets.maps.loader_to_dataset import CLASSIFICATION_LOADER_MAP
from chronocratic.datasets.modules._base.classification import (
    BaseClassificationTimeSeriesDataModule,
)
from chronocratic.datasets.utils.cache import (
    atomic_save_metadata,
    atomic_save_npz,
    build_cache_key,
    CACHE_SCHEMA_VERSION,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from torch.utils.data import DataLoader

__all__ = ["UEAClassificationDataModule"]

logger = logging.getLogger(__name__)


class UEAClassificationDataModule(BaseClassificationTimeSeriesDataModule):
    """LightningDataModule for UEA multivariate classification datasets.

    Reads multi-dimensional nested ARFF files using raw
    :func:`scipy.io.arff.loadarff`, decodes byte values,
    encodes labels with :class:`sklearn.preprocessing.LabelEncoder`,
    and manages splits with variable-length handling.

    ``data_form`` is hardcoded as ``DataForm.NESTED``.
    ARFF file patterns are hardcoded:
    ``{dataset_name}_TRAIN.arff`` and ``{dataset_name}_TEST.arff``.

    Args:
        dataset_folder_path: Path to the dataset ARFF directory.
        target_column_name: Name of the target/label column in the ARFF files.
        batch_size: Batch size for dataloaders.
        valid_size: Fraction of training data for validation.
        shuffle: Whether to shuffle training data.
        scale_data: Whether to scale features.
        data_scaling_method: Scaling algorithm, typed as
            :class:`~chronocratic.datasets.enums.data.ScalingMethod`.
        data_scaling_range: Target ``(min, max)`` range for
            :data:`ScalingMethod.MINMAX`.
        splitting_strategy: ``AS_DEFINED`` or ``MANUAL`` splitting,
            typed as
            :class:`~chronocratic.datasets.enums.data.ClassificationSplitMode`.
        test_size: Test set fraction for ``MANUAL`` splitting.
        num_workers: DataLoader worker count.
        loader_mode: Per-init mode controlling dataloader output format.
            Defaults to ``ClassificationLoaderMode.SAMPLE_LABEL``.
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
        splitting_strategy: ClassificationSplitMode = (ClassificationSplitMode.AS_DEFINED),
        test_size: float = 0.5,
        num_workers: int = 0,
        loader_mode: ClassificationLoaderMode = ClassificationLoaderMode.SAMPLE_LABEL,
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
            loader_mode=loader_mode,
        )
        self._dataset_name = dataset_folder_path.name
        self._cache_key = build_cache_key(
            dataset_name=dataset_folder_path.name,
            params={
                "splitting_strategy": splitting_strategy.value,
                "test_size": test_size,
                "valid_size": valid_size,
                "data_scaling_method": data_scaling_method.value,
            },
        )

    # ------------------------------------------------------------------
    # ARFF reading
    # ------------------------------------------------------------------

    def _read_arff_data_file(self, file_path: Path) -> Any:
        """Read an ARFF file using scipy.io.arff.loadarff.

        Uses raw scipy loading, NOT the utils/arff.py helpers.
        Nested ARFF data doesn't fit the DataFrame-based approach.

        Args:
            file_path: Path to the ARFF file.

        Returns:
            Raw numpy structured array from scipy loadarff.
        """
        data, _meta = arff.loadarff(file_path)
        return data

    def _process_stacked_data(self, data: Any) -> tuple[np.ndarray, np.ndarray]:
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
            for sample_point in sample:
                point_list = sample_point.tolist()
                point = [
                    (float(d.decode("utf-8")) if isinstance(d, bytes) else float(d))
                    for d in point_list
                ]
                sample_list.append(point)
            processed_data.append(np.array(sample_list))

            label_str = label.decode("utf-8") if isinstance(label, bytes) else label
            labels.append(label_str)

        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)
        output_data = np.array(processed_data).astype(np.float32).swapaxes(1, 2)
        return output_data, np.array(encoded_labels)

    # ------------------------------------------------------------------
    # Lightning lifecycle
    # ------------------------------------------------------------------

    def _do_prepare_data(self) -> None:
        """Validate paths, read ARFF files, split, and cache data.

        Raises ``FileNotFoundError`` if the dataset folder
        does not exist. Reads train/test ARFF files via scipy.io.arff,
        applies optional manual re-splitting, creates validation split,
        processes variable-length sequences, and writes cache
        (npz + metadata.json).
        """
        # Validate folder exists
        if not self.dataset_folder_path.exists():
            msg = f"Dataset folder not found: {self.dataset_folder_path}"
            raise FileNotFoundError(msg)

        # Construct ARFF paths
        arff_train = self.dataset_folder_path / f"{self._dataset_name}_TRAIN.arff"
        arff_test = self.dataset_folder_path / f"{self._dataset_name}_TEST.arff"

        # Read and process ARFF files via scipy
        train_data = self._read_arff_data_file(arff_train)
        test_data = self._read_arff_data_file(arff_test)

        self._train_data_samples, self._train_data_labels = self._process_stacked_data(train_data)
        self._test_data_samples, self._test_data_labels = self._process_stacked_data(test_data)

        # Apply splitting strategy
        if self.splitting_strategy == ClassificationSplitMode.MANUAL:
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

        # Create validation split
        self._valid_data_labels = None
        self._valid_data_samples = None
        if self.valid_size > 0.0:
            label_counts = np.bincount(self._train_data_labels)
            valid_mask = np.isin(self._train_data_labels, np.where(label_counts > 1)[0])
            filtered_samples = self._train_data_samples[valid_mask]
            filtered_labels = self._train_data_labels[valid_mask]

            dropped_count = len(self._train_data_samples) - len(filtered_samples)
            if dropped_count > 0:
                logger.warning(
                    "Dropped %d samples from singleton classes in dataset %s. "
                    "These classes will not be present in training data.",
                    dropped_count,
                    self._dataset_name,
                )

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
                    r"The test_size = \d+ should be"
                    r" greater or equal to the"
                    r" number of classes = \d+"
                )
                if re.match(pattern, str(e)):
                    num_classes = len(set(filtered_labels))
                    (
                        self._train_data_samples,
                        self._valid_data_samples,
                        self._train_data_labels,
                        self._valid_data_labels,
                    ) = train_test_split(
                        filtered_samples,
                        filtered_labels,
                        test_size=num_classes,
                        stratify=filtered_labels,
                        random_state=42,
                    )
                    logger.warning(
                        "Validation size adjusted to %d samples to cover all classes", num_classes
                    )

        # Variable-length processing
        self._process_data_with_varying_sequence_lengths()

        # Convert labels to pd.Series with category dtype
        self._train_data_labels = pd.Series(self._train_data_labels, dtype="category")
        self._test_data_labels = pd.Series(self._test_data_labels, dtype="category")
        if self.valid_size > 0.0 and self._valid_data_labels is not None:
            self._valid_data_labels = pd.Series(self._valid_data_labels, dtype="category")

        # Compute module state
        self._num_classes = len(self._train_data_labels.unique())
        self._seq_len, self._num_features = self._train_data_samples[0].shape

        # Write cache
        cache_dir = self._get_cache_dir()
        cache_path = cache_dir / f"{self._cache_key}.npz"

        valid_samples_arr = (
            self._valid_data_samples
            if self._valid_data_samples is not None
            else np.empty((0, 1, 1), dtype=self._train_data_samples.dtype)
        )

        atomic_save_npz(
            path=cache_path,
            train_samples=self._train_data_samples,
            train_labels=self._train_data_labels.to_numpy(),
            test_samples=self._test_data_samples,
            test_labels=self._test_data_labels.to_numpy(),
            valid_samples=valid_samples_arr,
            valid_labels=self._valid_data_labels.to_numpy()
            if self._valid_data_labels is not None
            else np.empty((0,), dtype=self._train_data_labels.to_numpy().dtype),
        )

        atomic_save_metadata(
            path=cache_dir / f"{self._cache_key}_metadata.json",
            data={
                "version": CACHE_SCHEMA_VERSION,
                "dataset_name": self._dataset_name,
                "n_features": self._num_features,
                "seq_len": self._seq_len,
                "has_datetime_index": False,
                "data_scaling_method": self.data_scaling_method.value,
                "data_scaling_range": self.data_scaling_range,
            },
        )

    def _load_cached_data(self) -> None:
        """Load cached data splits from the npz cache file."""
        if self._train_data_samples is not None:
            return

        cache_dir = self._get_cache_dir()
        cache_path = cache_dir / f"{self._cache_key}.npz"
        loaded = np.load(str(cache_path))

        self._train_data_samples = loaded["train_samples"]
        self._train_data_labels = pd.Series(loaded["train_labels"], dtype="category")
        self._test_data_samples = loaded["test_samples"]
        self._test_data_labels = pd.Series(loaded["test_labels"], dtype="category")

        valid_samples = loaded["valid_samples"]
        if valid_samples.size > 0:
            self._valid_data_samples = valid_samples
            self._valid_data_labels = pd.Series(loaded["valid_labels"], dtype="category")
        else:
            self._valid_data_samples = None
            self._valid_data_labels = None

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(
        self,
        *,
        mode: ClassificationLoaderMode = ClassificationLoaderMode.SAMPLE_LABEL,
        shuffle: bool | None = None,
        strict_batch_size: bool = True,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader:  # ty:ignore[invalid-method-override]
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
            data=self._train_data_samples,  # ty:ignore[invalid-argument-type]
            labels=self._train_data_labels,
            mode=CLASSIFICATION_LOADER_MAP[mode],
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
        mode: ClassificationLoaderMode = ClassificationLoaderMode.SAMPLE_LABEL,
        strict_batch_size: bool = True,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader | None:  # ty:ignore[invalid-method-override]
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
            data=self._valid_data_samples,  # ty:ignore[invalid-argument-type]
            labels=self._valid_data_labels,
            mode=CLASSIFICATION_LOADER_MAP[mode],
        )
        return self._process_valid_dataloader(
            dataset_object=dataset, strict_batch_size=strict_batch_size, extra_args=extra_args
        )

    def test_dataloader(
        self,
        *,
        mode: ClassificationLoaderMode = ClassificationLoaderMode.SAMPLE_LABEL,
        strict_batch_size: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader:  # ty:ignore[invalid-method-override]
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
            data=self._test_data_samples,  # ty:ignore[invalid-argument-type]
            labels=self._test_data_labels,
            mode=CLASSIFICATION_LOADER_MAP[mode],
        )
        return self._process_test_dataloader(
            dataset_object=dataset, strict_batch_size=strict_batch_size, extra_args=extra_args
        )
