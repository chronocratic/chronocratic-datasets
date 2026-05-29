"""Base LightningDataModule for time series data.

Provides shared dataloader construction, scaling setup, and property
definitions used by both classification and forecasting branches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from tscollection.datasets.enums.data import DataForm, ScalingMethod
from tscollection.datasets.utils.cache import load_metadata, resolve_cache_dir
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
        cache_dir: Custom cache directory. ``None`` uses the default
            ``~/.cache/tsdatasets/<dataset_name>``.
    """

    prepare_data_per_node: bool = True

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
        cache_dir: Path | None = None,
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
        self._cache_dir = cache_dir
        self._datatype_handling_functions_map: dict[str, object] | None = None
        self._initiate_datatypes_handling_functions_map()
        self._dataset_name: str | None = None
        self._num_features: int | None = None
        self._train_data_samples: np.ndarray | pd.DataFrame | None = None
        self._test_data_samples: np.ndarray | pd.DataFrame | None = None
        self._valid_data_samples: np.ndarray | pd.DataFrame | None = None
        self._dataset_class: type | None = None
        self._setup_completed_stages: set[str | None] = set()
        self._prepare_data_called: bool = False
        self._scaler_cache: Callable[..., tuple[Any, Any, Any]] | None = None
        # Cache-related typed attributes
        self._full_data_raw: np.ndarray | None = None
        self._time_index: pd.DatetimeIndex | None = None
        self._full_data_scaled: np.ndarray | None = None
        self._cache_key: str | None = None
        self._data_scaler_cache = None
        self._ts_feature_scaler_cache = None

    # ------------------------------------------------------------------
    # Cache directory resolution
    # ------------------------------------------------------------------

    def _get_cache_dir(self) -> Path:
        """Return the resolved cache directory for the current dataset name.

        Computes the cache path lazily so that it always uses the most
        recent value of ``_dataset_name`` (which is set by
        ``_do_prepare_data()``).  If ``_dataset_name`` is not yet set,
        a placeholder name ``default`` is used.

        Returns:
            Absolute path to the cache directory.
        """
        name = self._dataset_name or 'default'
        return resolve_cache_dir(
            cache_dir=self._cache_dir,
            dataset_name=name,
        )

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
    def train_data_samples(self) -> np.ndarray | pd.DataFrame | None:
        """Training data samples."""
        return self._train_data_samples

    @property
    def test_data_samples(self) -> np.ndarray | pd.DataFrame | None:
        """Test data samples."""
        return self._test_data_samples

    @property
    def valid_data_samples(self) -> np.ndarray | pd.DataFrame | None:
        """Validation data samples."""
        return self._valid_data_samples

    @property
    def all_data_samples(self) -> np.ndarray | pd.DataFrame:
        """Concatenation of all data splits."""
        if isinstance(self._train_data_samples, pd.DataFrame):
            return pd.concat(
                [
                    self._train_data_samples,  # type: ignore[arg-type]
                    self._test_data_samples,  # type: ignore[arg-type]
                    self._valid_data_samples,  # type: ignore[arg-type]
                ],
                axis=0,
            )  # ty:ignore[no-matching-overload]
        import numpy as np

        return np.concatenate(
            [
                self._train_data_samples,  # type: ignore[arg-type]
                self._test_data_samples,  # type: ignore[arg-type]
                self._valid_data_samples,  # type: ignore[arg-type]
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

        ``prepare_data()`` does NOT load or split data —
        that happens in ``setup()``.
        """
        if self._prepare_data_called:
            return
        self._do_prepare_data()
        self._finalize_prepare_data()
        self._prepare_data_called = True

    # ------------------------------------------------------------------
    # Dimension API
    # ------------------------------------------------------------------

    def prepare_dimensions(self) -> tuple[int | None, int | None]:
        """Return (n_features, sequence_len) from cached attrs or metadata.

        Short-circuits if ``_num_features`` is already populated (e.g. after
        ``setup()``).  Otherwise attempts to read ``metadata.json`` from the
        cache directory so that dimensions are available without loading any
        arrays — the DDP-safe flow.

        If ``_cache_key`` is not yet set or metadata cannot be found, falls
        back to ``_compute_dimensions()`` for backward compatibility with
        subclasses that set ``_num_features`` in ``_do_prepare_data()``.

        Returns:
            Tuple of (n_features, sequence_len). Values may be None if
            dimensions have not yet been computed.

        Raises:
            FileNotFoundError: If metadata file does not exist and
                ``_cache_key`` is set.
            ValueError: If metadata schema version does not match
                :data:`~tscollection.datasets.utils.cache.CACHE_SCHEMA_VERSION`.
        """
        if self._num_features is not None:
            return self._num_features, self._seq_len
        if self._cache_key is not None:
            meta_path = self._get_cache_dir() / f'{self._cache_key}_metadata.json'
            try:
                meta = load_metadata(meta_path)
            except FileNotFoundError:
                msg = (
                    f'Cache metadata not found: {meta_path}. '
                    f'Call prepare_data() first or delete stale cache.'
                )
                raise FileNotFoundError(msg) from None
            else:
                n_features = meta['n_features']
                seq_len = meta['seq_len']
                self._num_features = n_features
                return n_features, seq_len
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

    def setup(self, stage: str | None = None) -> None:
        """Apply data scaling via :func:`create_data_scaler`.

        The classification branch uses
        ``create_data_scaler()`` from utilities. The forecasting
        branch overrides this method entirely with sklearn
        direct scaling.

        Stage branching:
        - ``fit``/``None``: Fit scaler on all splits.
        - ``test``/``predict``: Scale test data only (reuse cached scaler).
        - ``validate``: No data mutation; mark stage as complete.

        Idempotency guard: Repeated calls for the same stage are
        no-ops via ``_setup_completed_stages`` sentinel.

        Args:
            stage: Lightning stage identifier. Defaults to ``None``
                (equivalent to ``"fit"``).

        Raises:
            ValueError: If stage is not one of
                ``{'fit', 'validate', 'test', 'predict', None}``.
        """
        if stage not in ('fit', 'validate', 'test', 'predict', None):
            msg = f'Unknown stage: {stage!r}'
            raise ValueError(msg)
        if stage in self._setup_completed_stages:
            return
        # fit and None are equivalent — both scale all three splits
        if stage in ('fit', None) and (
            'fit' in self._setup_completed_stages or None in self._setup_completed_stages
        ):
            return

        # validate: no data mutation
        if stage == 'validate':
            self._setup_completed_stages.add(stage)
            return

        # Build or reuse scaler cache
        if self._scaler_cache is None:
            self._scaler_cache = create_data_scaler(
                scale=self.scale_data,
                scaling_range=self.data_scaling_range,
                scaling_method=self.data_scaling_method,
                data_form=self._data_form,
            )

        if stage in ('fit', None):
            (self._train_data_samples, self._valid_data_samples, self._test_data_samples) = (
                self._scaler_cache(
                    self._train_data_samples, self._valid_data_samples, self._test_data_samples
                )
            )
        elif stage in ('test', 'predict'):
            _, _, self._test_data_samples = self._scaler_cache(
                self._train_data_samples, self._valid_data_samples, self._test_data_samples
            )

        self._setup_completed_stages.add(stage)

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear lifecycle sentinels to allow re-use of this DataModule.

        Resets the setup stage tracking and prepare_data sentinel so that
        subsequent calls to setup() or prepare_data() will re-execute
        their logic. Also clears all cache-related attributes
        (_full_data_raw, _time_index, _full_data_scaled, scaler caches,
        data samples, _cache_key) to prevent stale state across resets.

        Useful for hyperparameter sweeps or re-training scenarios that
        reuse the same DataModule instance.
        """
        self._setup_completed_stages.clear()
        self._prepare_data_called = False
        self._full_data_raw = None
        self._time_index = None
        self._full_data_scaled = None
        self._data_scaler_cache = None
        self._ts_feature_scaler_cache = None
        self._train_data_samples = None
        self._valid_data_samples = None
        self._test_data_samples = None
        self._cache_key = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initiate_datatypes_handling_functions_map(self) -> None:
        """Initialize the datatype handling functions map."""
        self._datatype_handling_functions_map = defaultdict(lambda: lambda x: x, {})

    def _get_custom_collate_fn(self, desired_batch_size: int | None = None) -> partial:
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
        dataset_object: Dataset[object],
        shuffle: bool | None = None,
        strict_batch_size: bool = False,
        extra_args: dict[str, object] | None = None,
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
        dataloader_args: dict[str, object] = {
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
        return DataLoader(**dataloader_args)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    def _process_test_dataloader(
        self,
        *,
        dataset_object: Dataset[object],
        strict_batch_size: bool = False,
        extra_args: dict[str, object] | None = None,
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
        dataloader_args: dict[str, object] = {
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
        return DataLoader(**dataloader_args)  # type: ignore[arg-type]  # ty:ignore[invalid-argument-type]

    def _process_valid_dataloader(
        self,
        *,
        dataset_object: Dataset[object],
        strict_batch_size: bool = False,
        extra_args: dict[str, object] | None = None,
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
