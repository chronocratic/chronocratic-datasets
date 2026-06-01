"""Base LightningDataModule for forecasting time series datasets.

Provides time slicing, sklearn-based scaling, time feature extraction,
and data transformation hooks for forecasting datasets (ETT, Electricity,
Weather).
"""

from __future__ import annotations

from abc import abstractmethod
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tscollection.datasets.enums.data import ForecastingMode, ScalingMethod
from tscollection.datasets.modules._base.base import BaseTimeSeriesDataModule
from tscollection.datasets.utils.cache import (
    load_scaler,
    resolve_cache_dir,
    save_scaler,
)
from tscollection.datasets.utils.features import TIME_FEATURE_COUNT

logger = getLogger(__name__)

__all__ = ['BaseForecastingTimeSeriesDataModule']


class BaseForecastingTimeSeriesDataModule(BaseTimeSeriesDataModule):
    """Base LightningDataModule for forecasting time series datasets.

    Extends :class:`BaseTimeSeriesDataModule` with dataset-intrinsic
    time slicing, sklearn-based scaling, and cyclical time
    feature extraction. Overrides ``setup()`` entirely to handle
    forecasting-specific scaling (fit on train slice only).

    Subclasses implement ``_set_data_slices()`` to define train/val/test
    boundaries.

    Args:
        batch_size: Batch size for dataloaders.
        seq_len: Input window length for sliding windows.
        valid_size: Fraction of data reserved for validation.
        test_size: Fraction reserved as test set.
        shuffle: Whether to shuffle the training dataloader.
        scale_data: Whether to apply data scaling.
        data_scaling_method: Scaling algorithm, typed as
            :class:`~tscollection.datasets.enums.data.ScalingMethod`.
        data_scaling_range: Target ``(min, max)`` range for
            :data:`ScalingMethod.MINMAX`.
        num_workers: Number of DataLoader worker processes.
        mode: Forecasting mode (univariate or multivariate), typed as
            :class:`~tscollection.datasets.enums.data.ForecastingMode`.
    """

    def __init__(
        self,
        *,
        batch_size: int = 32,
        seq_len: int = 128,
        valid_size: float = 0.1,
        test_size: float = 0.5,
        shuffle: bool = False,
        scale_data: bool = True,
        data_scaling_method: ScalingMethod = ScalingMethod.MINMAX,
        data_scaling_range: tuple[float, float] = (0, 1),
        num_workers: int = 0,
        mode: ForecastingMode = ForecastingMode.UNIVARIATE,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            seq_len=seq_len,
            valid_size=valid_size,
            test_size=test_size,
            shuffle=shuffle,
            scale_data=scale_data,
            data_scaling_method=data_scaling_method,
            data_scaling_range=data_scaling_range,
            num_workers=num_workers,
        )
        self._mode = mode
        self._train_slice: slice | None = None
        self._valid_slice: slice | None = None
        self._test_slice: slice | None = None
        self._full_data_raw: np.ndarray | None = None
        self._time_index: pd.DatetimeIndex | None = None
        self._full_data_scaled: np.ndarray | None = None
        self._num_time_series_features: int | None = None
        self._data_scaler_cache: MinMaxScaler | StandardScaler | None = None
        self._ts_feature_scaler_cache: MinMaxScaler | StandardScaler | None = None
        # Backward-compat: routes _full_data property to raw before scaling,
        # to scaled after. Concrete modules still use _full_data in _transform_data.
        self._scaling_done: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def train_slice(self) -> slice | None:
        """Training data slice boundaries."""
        return self._train_slice

    @property
    def valid_slice(self) -> slice | None:
        """Validation data slice boundaries."""
        return self._valid_slice

    @property
    def test_slice(self) -> slice | None:
        """Test data slice boundaries."""
        return self._test_slice

    @property
    def full_data(self) -> np.ndarray | None:
        """Full data array.

        Returns ``_full_data_scaled`` after setup has scaled data,
        otherwise ``_full_data_raw``.
        """
        if self._scaling_done:
            return self._full_data_scaled
        return self._full_data_raw

    @property
    def _full_data(self) -> np.ndarray | None:
        """Backward-compat accessor for concrete modules."""
        return self.full_data

    @_full_data.setter
    def _full_data(self, value: np.ndarray | None) -> None:
        """Route writes to raw (before scaling) or scaled (after).

        When a DataFrame is set (backward-compat path), extracts
        the DatetimeIndex into ``_time_index`` and stores values
        as numpy in ``_full_data_raw``.
        """
        if value is None:
            return
        if self._scaling_done:
            self._full_data_scaled = value
            return
        if isinstance(value, pd.DataFrame):
            if isinstance(value.index, pd.DatetimeIndex):
                self._time_index = value.index
            self._full_data_raw = value.to_numpy()
        else:
            self._full_data_raw = value

    @property
    def num_time_series_features(self) -> int | None:
        """Number of cyclical time features extracted."""
        return self._num_time_series_features

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def _set_data_slices(self) -> None:
        """Define train/valid/test slice boundaries."""

    @abstractmethod
    def _transform_data(self) -> None:
        """Transform ``_full_data`` after scaling.

        Subclasses must implement this method to apply dataset-specific
        transformations (e.g., reshape, transpose, expand dimensions).
        """

    # ------------------------------------------------------------------
    # Dimension API override
    # ------------------------------------------------------------------

    def _compute_dimensions(self) -> tuple[int | None, int | None]:
        """Compute dimensions from typed attributes without requiring setup().

        Returns:
            Tuple of (n_features, sequence_len). n_features may be None
            if raw data is not set.
        """
        if self._full_data_raw is None:
            return None, self._seq_len
        raw_cols = self._full_data_raw.shape[-1]
        has_time_features = self._time_index is not None and self.scale_data
        n_features = raw_cols + TIME_FEATURE_COUNT if has_time_features else raw_cols
        self._num_features = n_features
        return n_features, self._seq_len

    # ------------------------------------------------------------------
    # Setup -- overrides base
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Scale data, extract time features, and split into train/val/test.

        The forecasting branch uses sklearn scalers directly
        (not ``create_data_scaler()``) because forecasting data has a
        different shape (features x timesteps). Fits scaler on train
        slice only to prevent data leakage.

        Stage branching:
        - ``fit``/``None``: Fit scalers, transform data, split into slices.
        - ``test``/``predict``: Reuse cached fitted scalers to transform.
        - ``validate``: No data mutation; mark stage as complete.

        Idempotency guard: Repeated calls for the same stage are
        no-ops via ``_setup_completed_stages`` sentinel.

        When ``scale_data`` is False, scaling and time feature extraction
        are skipped entirely to preserve raw values.

        Args:
            stage: Lightning stage identifier.

        Raises:
            ValueError: If stage is not one of
                ``{'fit', 'validate', 'test', 'predict', None}``.
        """
        if stage not in ('fit', 'validate', 'test', 'predict', None):
            msg = f'Unknown stage: {stage!r}'
            raise ValueError(msg)
        if stage in self._setup_completed_stages:
            return
        # fit and None are equivalent -- skip if the other already ran
        if stage in ('fit', None) and (
            'fit' in self._setup_completed_stages or None in self._setup_completed_stages
        ):
            return

        # Load raw data from cache; fall back to _full_data_raw for
        # backward-compat with concrete modules not yet writing cache.
        cache_dir = self._resolve_cache_dir()
        if self._cache_key is not None:
            cache_path = cache_dir / f'{self._cache_key}.npz'
            try:
                loaded = np.load(str(cache_path))
                self._full_data_raw = loaded['data'].astype(np.float32)
                if 'index' in loaded:
                    self._time_index = pd.DatetimeIndex(loaded['index'])
                else:
                    self._time_index = None
            except FileNotFoundError:
                pass

        # Backward-compat: ensure slices are set after raw data is available.
        if self._train_slice is None:
            self._set_data_slices()

        assert self._full_data_raw is not None, (
            'Full data not set; call prepare_data() first'
        )
        assert self._train_slice is not None, (
            'Train slice not set; call _set_data_slices() first'
        )

        # validate: no data mutation, just mark stage complete
        if stage == 'validate':
            self._setup_completed_stages.add(stage)
            return

        full_array = self._full_data_raw

        # Time feature extraction from typed _time_index
        if self._time_index is not None:
            from tscollection.datasets.utils.features import extract_time_features

            time_series_features = extract_time_features(self._time_index)
            num_time_series_features = time_series_features.shape[-1]
        else:
            time_series_features = np.empty((0, 0))
            num_time_series_features = 0

        if self.scale_data:
            if stage in ('fit', None):
                data_scaler = self._prepare_data_scaler()
                data_scaler.fit(full_array[self._train_slice])
                self._data_scaler_cache = data_scaler
                self._save_scaler_to_cache(data_scaler, 'data')

                self._full_data_scaled = data_scaler.transform(full_array)
                self._scaling_done = True

                self._transform_data()

                if num_time_series_features > 0:
                    ts_feature_scaler = self._prepare_data_scaler()
                    ts_feature_scaler.fit(time_series_features[self._train_slice])
                    self._ts_feature_scaler_cache = ts_feature_scaler
                    self._save_scaler_to_cache(ts_feature_scaler, 'ts')
                    self._apply_ts_features(ts_feature_scaler, time_series_features)
                self._num_time_series_features = num_time_series_features
                self._calculate_num_features()
                self._split_data()
            elif stage in ('test', 'predict'):
                if self._data_scaler_cache is None:
                    self._data_scaler_cache = self._load_scaler_from_cache('data')

                if self._data_scaler_cache is not None and self._train_data_samples is None:
                    self._full_data_scaled = self._data_scaler_cache.transform(full_array)
                    self._scaling_done = True
                    self._transform_data()

                    if num_time_series_features > 0:
                        if self._ts_feature_scaler_cache is None:
                            self._ts_feature_scaler_cache = (
                                self._load_scaler_from_cache('ts')
                            )
                        if self._ts_feature_scaler_cache is not None:
                            self._apply_ts_features(
                                self._ts_feature_scaler_cache, time_series_features
                            )
                    self._num_time_series_features = num_time_series_features
                    self._calculate_num_features()
                    self._split_data()
                elif self._train_data_samples is not None:
                    pass
                else:
                    logger.warning(
                        'scale_data=True but no fitted scaler cache available. '
                        'Data will not be scaled. Call setup(stage="fit") first or '
                        'provide a pre-fitted _data_scaler_cache.'
                    )
                    self._scaling_done = True
                    self._full_data_scaled = full_array.copy()
                    self._transform_data()
                    self._calculate_num_features()
                    self._split_data()
        else:
            self._full_data_scaled = full_array.copy()
            self._scaling_done = True
            self._transform_data()
            self._num_time_series_features = num_time_series_features
            self._calculate_num_features()
            self._split_data()

        self._setup_completed_stages.add(stage)

    # ------------------------------------------------------------------
    # Time feature helper
    # ------------------------------------------------------------------

    def _apply_ts_features(
        self,
        ts_scaler: MinMaxScaler | StandardScaler,
        time_series_features: np.ndarray,
    ) -> None:
        """Scale, expand, repeat and concatenate time features into _full_data_scaled.

        Shared helper to avoid duplicating the expand_dims/repeat/concatenate
        pattern between the fit and test/predict branches of ``setup()``.
        """
        assert self._full_data_scaled is not None
        scaled = np.expand_dims(ts_scaler.transform(time_series_features), axis=0)
        repeated = np.repeat(scaled, self._full_data_scaled.shape[0], axis=0)
        self._full_data_scaled = np.concatenate(
            [repeated, self._full_data_scaled], axis=-1
        )

    # ------------------------------------------------------------------
    # Cache I/O helpers
    # ------------------------------------------------------------------

    def _resolve_cache_dir(self) -> Path:
        """Return the resolved cache directory for the current dataset."""
        name = self._dataset_name or 'default'
        return resolve_cache_dir(
            cache_dir=self._cache_dir,
            dataset_name=name,
        )

    def _save_scaler_to_cache(self, scaler: object, kind: str) -> None:
        """Save a fitted scaler to the cache directory.

        Relies on ``save_scaler()`` internal atomic handling for DDP race
        safety rather than a pre-check (which is a TOCTOU vulnerability).
        """
        if self._cache_key is None:
            return
        cache_dir = self._resolve_cache_dir()
        scaler_path = cache_dir / f'{self._cache_key}_{kind}_scaler.pt'
        save_scaler(scaler=scaler, path=scaler_path)

    def _load_scaler_from_cache(self, kind: str) -> MinMaxScaler | StandardScaler | None:
        """Load a persisted scaler from the cache directory.

        Returns:
            The loaded scaler, or None if the file does not exist.
        """
        if self._cache_key is None:
            return None
        cache_dir = self._resolve_cache_dir()
        scaler_path = cache_dir / f'{self._cache_key}_{kind}_scaler.pt'
        try:
            return load_scaler(path=scaler_path)
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_data_scaler(self) -> MinMaxScaler | StandardScaler:
        """Instantiate the appropriate sklearn scaler.

        Compares ``self.data_scaling_method`` against ``ScalingMethod``
        enum members, NOT string literals.

        Returns:
            A scaler instance ready for fitting.

        Raises:
            ValueError: If the scaling method is not supported
                by the forecasting branch.
        """
        if self.data_scaling_method == ScalingMethod.MINMAX:
            return MinMaxScaler(feature_range=self.data_scaling_range)
        if self.data_scaling_method == ScalingMethod.STANDARD:
            return StandardScaler()
        msg = f'Unsupported scaling method for forecasting: {self.data_scaling_method}'
        raise ValueError(msg)

    def _calculate_num_features(self) -> None:
        """Calculate number of features from scaled data shape."""
        assert self._full_data_scaled is not None
        self._num_features = self._full_data_scaled.shape[-1]

    def _split_data(self) -> None:
        """Slice ``_full_data_scaled`` into train/valid/test splits.

        Uses ``_train_slice``, ``_valid_slice``, and ``_test_slice``
        defined by ``_set_data_slices()``.

        Raises:
            RuntimeError: If required attributes are not set.
        """
        if self._full_data_scaled is None:
            msg = '_split_data requires _full_data_scaled. Ensure scaling completed.'
            raise RuntimeError(msg)
        if self._train_slice is None:
            msg = '_split_data requires _train_slice. Ensure _set_data_slices() was called.'
            raise RuntimeError(msg)
        if self._valid_slice is None:
            msg = '_split_data requires _valid_slice. Ensure _set_data_slices() was called.'
            raise RuntimeError(msg)
        if self._test_slice is None:
            msg = '_split_data requires _test_slice. Ensure _set_data_slices() was called.'
            raise RuntimeError(msg)

        train_data = self._full_data_scaled[:, self._train_slice]
        valid_data = self._full_data_scaled[:, self._valid_slice]
        test_data = self._full_data_scaled[:, self._test_slice]

        self._train_data_samples = train_data
        self._valid_data_samples = valid_data
        self._test_data_samples = test_data

    def _finalize_prepare_data(self) -> None:
        """No-op -- slice computation moved to ``setup()`` after cache read."""
        return

    def reset(self) -> None:
        """Reset forecasting state, including the scaling flag.

        Restores ``_cache_key`` from the original init params since it is
        deterministic and required for cache-based ``setup()``.
        """
        original_cache_key = self._cache_key
        super().reset()
        self._cache_key = original_cache_key
        self._scaling_done = False
        self._num_time_series_features = None
