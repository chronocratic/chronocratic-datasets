"""Base LightningDataModule for forecasting time series datasets.

Provides time slicing, sklearn-based scaling, time feature extraction,
and data transformation hooks for forecasting datasets (ETT, Electricity,
Weather).
"""

from __future__ import annotations

from abc import abstractmethod
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tscollection.datasets.enums.data import ForecastingMode, ForecastingSplitMode, ScalingMethod
from tscollection.datasets.modules._base.base import BaseTimeSeriesDataModule
from tscollection.datasets.utils.features import TIME_FEATURE_COUNT

__all__ = ['BaseForecastingTimeSeriesDataModule']


class BaseForecastingTimeSeriesDataModule(BaseTimeSeriesDataModule):
    """Base LightningDataModule for forecasting time series datasets.

    Extends :class:`BaseTimeSeriesDataModule` with dataset-intrinsic
    time slicing, sklearn-based scaling, and cyclical time
    feature extraction. Overrides ``setup()`` entirely to handle
    forecasting-specific scaling (fit on train slice only).

    Split boundaries are driven by ``split_mode``:
    - ``ForecastingSplitMode.INDEXED`` — subclasses override
      ``_set_data_slices()`` with absolute index positions.
    - ``ForecastingSplitMode.FRACTIONAL`` — base class computes
      slices from ``split_ratios`` (train_frac, valid_frac).

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
        split_mode: How to compute train/valid/test boundaries, typed as
            :class:`~tscollection.datasets.enums.data.ForecastingSplitMode`.
        split_ratios: ``(train_frac, valid_frac)`` fractions for
            :data:`ForecastingSplitMode.FRACTIONAL`. Test fraction
            is the remainder. Defaults to ``(0.6, 0.2)``.
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
        split_mode: ForecastingSplitMode = ForecastingSplitMode.FRACTIONAL,
        split_ratios: tuple[float, float] = (0.6, 0.2),
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
        self._split_mode = split_mode
        self._split_ratios = split_ratios
        self._train_slice: slice | None = None
        self._valid_slice: slice | None = None
        self._test_slice: slice | None = None
        self._full_data: np.ndarray | pd.DataFrame | None = None
        self._num_time_series_features: int | None = None
        self._data_scaler_cache: MinMaxScaler | StandardScaler | None = None
        self._ts_feature_scaler_cache: MinMaxScaler | StandardScaler | None = None

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
    def full_data(self) -> np.ndarray | pd.DataFrame | None:
        """Full (unsliced) data array or DataFrame."""
        return self._full_data

    @property
    def num_time_series_features(self) -> int | None:
        """Number of cyclical time features extracted."""
        return self._num_time_series_features

    @property
    def split_mode(self) -> ForecastingSplitMode:
        """Train/valid/test split strategy (INDEXED or FRACTIONAL)."""
        return self._split_mode

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    def _set_data_slices(self) -> None:
        """Define train/valid/test slice boundaries.

        Dispatches on ``split_mode``:
        - ``ForecastingSplitMode.FRACTIONAL``: computes slices from
          ``split_ratios`` (train_frac, valid_frac). Test is remainder.
        - ``ForecastingSplitMode.INDEXED``: raises ``NotImplementedError`` —
          subclasses must override with absolute index positions
          (e.g., ETT's 16/4/4 months).
        """
        if self._split_mode == ForecastingSplitMode.INDEXED:
            msg = (
                'INDEXED split requires overriding _set_data_slices(). '
                'Define _train_slice, _valid_slice, _test_slice.'
            )
            raise NotImplementedError(msg)
        assert self._full_data is not None, '_full_data was not set by prepare_data()'
        num_samples = len(self._full_data)
        train_frac, valid_frac = self._split_ratios
        train_end = int(train_frac * num_samples)
        valid_end = int((train_frac + valid_frac) * num_samples)
        self._train_slice = slice(None, train_end)
        self._valid_slice = slice(train_end, valid_end)
        self._test_slice = slice(valid_end, None)

    @abstractmethod
    def _transform_data(self) -> None:
        """Transform ``_full_data`` after scaling.

        Subclasses must implement this method to apply datasetspecific
        transformations (e.g., reshape, transpose, expand dimensions).
        """

    # ------------------------------------------------------------------
    # Dimension API override
    # ------------------------------------------------------------------

    def _compute_dimensions(self) -> tuple[int | None, int | None]:
        """Compute dimensions from _full_data without requiring setup().

        For DataFrame with DatetimeIndex, adds TIME_FEATURE_COUNT to raw
        column count. For numpy arrays, uses raw column count only.

        Returns:
            Tuple of (n_features, sequence_len). n_features may be None
            if _full_data is not set.
        """
        if self._full_data is None:
            return None, self._seq_len
        if isinstance(self._full_data, pd.DataFrame):
            raw_cols = self._full_data.shape[-1]
            n_features = raw_cols + TIME_FEATURE_COUNT
            self._num_features = n_features
            return n_features, self._seq_len
        # numpy path: no time features, no DatetimeIndex
        raw_cols = self._full_data.shape[-1]
        self._num_features = raw_cols
        return raw_cols, self._seq_len

    # ------------------------------------------------------------------
    # Setup — overrides base
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

        assert self._full_data is not None, 'Full data not set; call prepare_data() first'
        assert self._train_slice is not None, 'Train slice not set; call _set_data_slices() first'

        # validate: no data mutation, just mark stage complete
        if stage == 'validate':
            self._setup_completed_stages.add(stage)
            return

        # Extract time features from DataFrame index if applicable
        if isinstance(self._full_data, pd.DataFrame):
            time_index = self._full_data.index
            full_array = self._full_data.to_numpy()
        else:
            time_index = None
            full_array = self._full_data

        # Time feature extraction
        if time_index is not None:
            from tscollection.datasets.utils.features import extract_time_features

            time_series_features = extract_time_features(pd.DatetimeIndex(time_index))
            num_time_series_features = time_series_features.shape[-1]
        else:
            time_series_features = np.empty((0, 0))
            num_time_series_features = 0

        if self.scale_data:
            if stage in ('fit', None):
                # Fit scaler on train slice only, transform full data.
                # Before _transform_data(), full_array has shape (time_steps, features),
                # so time slicing is axis 0 (not axis 1).
                data_scaler = self._prepare_data_scaler()
                data_scaler.fit(full_array[self._train_slice])
                self._data_scaler_cache = data_scaler
                self._full_data = data_scaler.transform(full_array)

                # Apply module-specific transform
                self._transform_data()

                # Scale time features if present
                if num_time_series_features > 0:
                    ts_feature_scaler = self._prepare_data_scaler()
                    ts_feature_scaler.fit(time_series_features[self._train_slice])
                    self._ts_feature_scaler_cache = ts_feature_scaler
                    scaled_ts_features = ts_feature_scaler.transform(time_series_features)
                    scaled_ts_features = np.expand_dims(scaled_ts_features, axis=0)
                    assert self._full_data is not None
                    repeated_ts = np.repeat(scaled_ts_features, self._full_data.shape[0], axis=0)
                    self._full_data = np.concatenate([repeated_ts, self._full_data], axis=-1)
                self._num_time_series_features = num_time_series_features
                self._calculate_num_features()
                self._split_data()
            elif stage in ('test', 'predict'):
                # Reuse cached fitted scalers
                if self._data_scaler_cache is not None and self._train_data_samples is None:
                    # Standalone test/predict (fit hasn't run yet)
                    self._full_data = self._data_scaler_cache.transform(full_array)
                    self._transform_data()

                    if num_time_series_features > 0 and self._ts_feature_scaler_cache is not None:
                        scaled_ts_features = self._ts_feature_scaler_cache.transform(
                            time_series_features
                        )
                        scaled_ts_features = np.expand_dims(scaled_ts_features, axis=0)
                        assert self._full_data is not None
                        repeated_ts = np.repeat(
                            scaled_ts_features, self._full_data.shape[0], axis=0
                        )
                        self._full_data = np.concatenate([repeated_ts, self._full_data], axis=-1)
                    self._num_time_series_features = num_time_series_features
                    self._calculate_num_features()
                    self._split_data()
                elif self._train_data_samples is not None:
                    # fit already ran; data is already transformed and split.
                    # Do nothing — scaler is already cached.
                    pass
                else:
                    # No cached scaler and no prior fit; transform without scaling
                    logging.warning(
                        'scale_data=True but no fitted scaler cache available. '
                        'Data will not be scaled. Call setup(stage="fit") first or '
                        'provide a pre-fitted _data_scaler_cache.'
                    )
                    self._transform_data()
                    self._calculate_num_features()
                    self._split_data()
        else:
            # No scaling: apply module-specific transform on raw data.
            self._transform_data()
            self._num_time_series_features = num_time_series_features
            self._calculate_num_features()
            self._split_data()

        self._setup_completed_stages.add(stage)

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
        """Calculate number of features from full data shape."""
        assert self._full_data is not None
        self._num_features = self._full_data.shape[-1]

    def _split_data(self) -> None:
        """Slice ``_full_data`` into train/valid/test splits.

        Uses ``_train_slice``, ``_valid_slice``, and ``_test_slice``
        defined by ``_set_data_slices()``.
        """
        assert self._full_data is not None
        assert self._train_slice is not None
        assert self._valid_slice is not None
        assert self._test_slice is not None

        train_data = self._full_data[:, self._train_slice]
        valid_data = self._full_data[:, self._valid_slice]
        test_data = self._full_data[:, self._test_slice]

        self._train_data_samples = train_data
        self._valid_data_samples = valid_data
        self._test_data_samples = test_data

    def _finalize_prepare_data(self) -> None:
        """Hook called after ``_do_prepare_data()`` to set data slices.

        Overrides the base class no-op. The base wrapper drives this
        call, so concrete forecasting modules no longer invoke it
        manually.
        """
        self._set_data_slices()
