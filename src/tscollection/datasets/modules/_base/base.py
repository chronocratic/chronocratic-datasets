"""Base LightningDataModule for time series data.

Provides shared dataloader construction, scaling setup, and property
definitions used by both classification and forecasting branches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Any

import lightning.pytorch as pl
import pandas as pd
from torch.utils.data import DataLoader

from tscollection.datasets.enums.data import DataForm, ScalingMethod
from tscollection.datasets.utils.general import custom_collate_fn
from tscollection.datasets.utils.scaling import create_data_scaler

__all__ = ['BaseTimeSeriesDataModule']


class BaseTimeSeriesDataModule(pl.LightningDataModule, ABC):
    """Shared base for all time series LightningDataModules.

    Handles batch size, scaling, and dataloader construction.
    Subclasses implement dataset-specific ``prepare_data()`` for
    file validation and data loading.

    Args:
        batch_size: Batch size for dataloaders.
        seq_len: Sequence length. ``None`` for classification
            (computed from data), int for forecasting (user-provided).
        valid_size: Fraction of training data reserved for validation.
        test_size: Fraction reserved as test set.
        shuffle: Whether to shuffle the training dataloader.
        scale_data: Whether to apply data scaling.
        data_scaling_method: Scaling algorithm, typed as
            :class:`~tscollection.datasets.enums.data.ScalingMethod`.
        data_scaling_range: Target ``(min, max)`` range for
            :data:`ScalingMethod.MINMAX`.
        num_workers: Number of DataLoader worker processes.
        data_form: Data shape category for scaling, typed as
            :class:`~tscollection.datasets.enums.data.DataForm`.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        seq_len: int | None,
        valid_size: float,
        test_size: float,
        shuffle: bool,
        scale_data: bool,
        data_scaling_method: ScalingMethod = ScalingMethod.MINMAX,
        data_scaling_range: tuple[float, float] = (0, 1),
        num_workers: int = 0,
        data_form: DataForm = DataForm.REGULAR,
    ) -> None:
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
        self._datatype_handling_functions_map: dict[str, Any] | None = None
        self._initiate_datatypes_handling_functions_map()
        self._dataset_name: str | None = None
        self._num_features: int | None = None
        self._train_data_samples: Any = None
        self._test_data_samples: Any = None
        self._valid_data_samples: Any = None
        self._dataset_class: Any = None
        self._setup_completed_stages: set[str | None] = set()
        self._prepare_data_called: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str | None:
        """Dataset name."""
        return self._dataset_name

    @property
    def sequence_length(self) -> int | None:
        """Sequence length (read-only)."""
        return self._seq_len

    @property
    def num_features(self) -> int | None:
        """Number of features (read-only)."""
        return self._num_features

    @property
    def train_data_samples(self) -> Any:
        """Training data samples."""
        return self._train_data_samples

    @property
    def test_data_samples(self) -> Any:
        """Test data samples."""
        return self._test_data_samples

    @property
    def valid_data_samples(self) -> Any:
        """Validation data samples."""
        return self._valid_data_samples

    @property
    def all_data_samples(self) -> pd.DataFrame:
        """Concatenation of all data splits."""
        return pd.concat(
            [
                self._train_data_samples,
                self._test_data_samples,
                self._valid_data_samples,
            ],
            axis=0,
        )

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    def prepare_data(self) -> None:
        """Validate file paths and perform lightweight checks.

        Concrete wrapper that drives the template:
        1. Check idempotency sentinel (skip if already called).
        2. Call ``_do_prepare_data()`` (abstract — subclass I/O).
        3. Call ``_finalize_prepare_data()`` (hook — no-op default,
           forecasting overrides to set slices).
        4. Set sentinel.

        Per D-09, ``prepare_data()`` does NOT load or split data —
        that happens in ``setup()``.
        """
        if self._prepare_data_called:
            return
        self._do_prepare_data()
        self._finalize_prepare_data()
        self._prepare_data_called = True

    # ------------------------------------------------------------------
    # Dimension API (A1, D4)
    # ------------------------------------------------------------------

    def prepare_dimensions(self) -> tuple[int | None, int | None]:
        """Return (n_features, sequence_len) populated by prepare_data().

        Caller must invoke prepare_data() first. setup() is NOT required.
        Safe to call before or after setup() — returns cached attrs once
        populated (D4 short-circuit).

        Returns:
            Tuple of (n_features, sequence_len). Values may be None if
            dimensions have not yet been computed.
        """
        if self._num_features is not None:
            return self._num_features, self._seq_len
        return self._compute_dimensions()

    def _compute_dimensions(self) -> tuple[int | None, int | None]:
        """Subclass hook to compute dims from current state.

        Default returns cached attrs. Classification overrides to raise
        RuntimeError if prepare_data was never called. Forecasting
        overrides to derive from _full_data.

        Returns:
            Tuple of (n_features, sequence_len).
        """
        return self._num_features, self._seq_len

    @abstractmethod
    def _do_prepare_data(self) -> None:
        """Subclass hook for the I/O portion of prepare_data.

        Concrete modules implement this to validate file paths,
        read data, and set module state (``_full_data``,
        ``_num_features``, ``_seq_len``, etc.).
        """
        ...

    def _finalize_prepare_data(self) -> None:
        """Hook called after ``_do_prepare_data()`` completes.

        Default is no-op. Forecasting base overrides this to call
        ``_set_data_slices()`` so slice population is driven by the
        base wrapper rather than individual concrete modules.
        """
        return

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        """Apply data scaling via :func:`create_data_scaler`.

        Per D-09 and D-10, the classification branch uses
        ``create_data_scaler()`` from utilities. The forecasting
        branch overrides this method entirely with sklearn
        direct scaling.

        Args:
            stage: Lightning stage identifier (``"fit"`` or ``"test"``).
        """
        if stage in self._setup_completed_stages or None in self._setup_completed_stages:
            return

        scaler = create_data_scaler(
            scale=self.scale_data,
            scaling_range=self.data_scaling_range,
            scaling_method=self.data_scaling_method,
            data_form=self._data_form,
        )
        (
            self._train_data_samples,
            self._valid_data_samples,
            self._test_data_samples,
        ) = scaler(
            self._train_data_samples,
            self._valid_data_samples,
            self._test_data_samples,
        )
        self._setup_completed_stages.add(stage)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initiate_datatypes_handling_functions_map(self) -> None:
        """Initialize the datatype handling functions map."""
        self._datatype_handling_functions_map = defaultdict(
            lambda: lambda x: x, {}
        )

    def _get_custom_collate_fn(
        self, desired_batch_size: int | None = None
    ) -> Any:
        """Return a collate function bound to the desired batch size.

        Args:
            desired_batch_size: Target batch size. Defaults to
                :attr:`batch_size`.

        Returns:
            A partially applied :func:`custom_collate_fn`.
        """
        if desired_batch_size is None:
            desired_batch_size = self.batch_size
        return partial(custom_collate_fn, desired_batch_size=desired_batch_size)

    # ------------------------------------------------------------------
    # Dataloader construction
    # ------------------------------------------------------------------

    def _process_train_dataloader(
        self,
        *,
        dataset_object: Any,
        shuffle: bool | None = None,
        strict_batch_size: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader:
        """Build the training DataLoader.

        Args:
            dataset_object: PyTorch Dataset instance.
            shuffle: Whether to shuffle. Defaults to :attr:`shuffle`.
            strict_batch_size: If True, pad the last batch via
                :func:`custom_collate_fn`.
            extra_args: Additional keyword arguments forwarded to
                the DataLoader constructor.

        Returns:
            Configured DataLoader for training.
        """
        if shuffle is None:
            shuffle = self.shuffle
        dataloader_args: dict[str, Any] = {
            'dataset': dataset_object,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'shuffle': shuffle,
            **(extra_args or {}),
        }
        if self.num_workers > 0:
            dataloader_args['persistent_workers'] = True
        if strict_batch_size:
            dataloader_args['collate_fn'] = self._get_custom_collate_fn()
        return DataLoader(**dataloader_args)

    def _process_test_dataloader(
        self,
        *,
        dataset_object: Any,
        strict_batch_size: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader:
        """Build the test DataLoader.

        Always uses ``shuffle=False``.

        Args:
            dataset_object: PyTorch Dataset instance.
            strict_batch_size: If True, pad the last batch via
                :func:`custom_collate_fn`.
            extra_args: Additional keyword arguments forwarded to
                the DataLoader constructor.

        Returns:
            Configured DataLoader for testing.
        """
        dataloader_args: dict[str, Any] = {
            'dataset': dataset_object,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'shuffle': False,
            **(extra_args or {}),
        }
        if self.num_workers > 0:
            dataloader_args['persistent_workers'] = True
        if strict_batch_size:
            dataloader_args['collate_fn'] = self._get_custom_collate_fn()
        return DataLoader(**dataloader_args)

    def _process_valid_dataloader(
        self,
        *,
        dataset_object: Any,
        strict_batch_size: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> DataLoader | None:
        """Build the validation DataLoader.

        Returns ``None`` when :attr:`valid_size` is ``0.0``.

        Args:
            dataset_object: PyTorch Dataset instance.
            strict_batch_size: If True, pad the last batch via
                :func:`custom_collate_fn`.
            extra_args: Additional keyword arguments forwarded to
                the DataLoader constructor.

        Returns:
            Configured DataLoader for validation, or ``None``.
        """
        if self.valid_size == 0.0:
            return None
        return self._process_test_dataloader(
            dataset_object=dataset_object,
            strict_batch_size=strict_batch_size,
            extra_args=extra_args,
        )
