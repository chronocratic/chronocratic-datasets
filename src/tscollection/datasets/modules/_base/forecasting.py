"""Base LightningDataModule for forecasting time series datasets.

Provides time slicing, sklearn-based scaling, time feature extraction,
and data transformation hooks for forecasting datasets (ETT, Electricity,
Weather).
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tscollection.datasets.enums.data import ForecastingMode, ScalingMethod
from tscollection.datasets.modules._base.base import BaseTimeSeriesDataModule

__all__ = ['BaseForecastingTimeSeriesDataModule']


class BaseForecastingTimeSeriesDataModule(BaseTimeSeriesDataModule):
    """Base LightningDataModule for forecasting time series datasets.

    Extends :class:`BaseTimeSeriesDataModule` with dataset-intrinsic
    time slicing, sklearn-based scaling (D-10), and cyclical time
    feature extraction. Overrides ``setup()`` entirely to handle
    forecasting-specific scaling (fit on train slice only).

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

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def _set_data_slices(self) -> None:
        """Define train/valid/test slice boundaries.

        Subclasses must implement this method to set ``_train_slice``,
        ``_valid_slice``, and ``_test_slice`` based on the dataset's
        intrinsic split pattern (e.g., ETT: 16/4/4 months, Weather:
        60/20/20 fractional).
        """

    @abstractmethod
    def _transform_data(self) -> None:
        """Transform ``_full_data`` after scaling.

        Subclasses must implement this method to apply datasetspecific
        transformations (e.g., reshape, transpose, expand dimensions).
        """

    # ------------------------------------------------------------------
    # Setup — overrides base
    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        """Scale data, extract time features, and split into train/val/test.

        Per D-10, the forecasting branch uses sklearn scalers directly
        (not ``create_data_scaler()``) because forecasting data has a
        different shape (features × timesteps). Fits scaler on train
        slice only to prevent data leakage (T-04-02-04).

        Stage validation (D1): unknown stages raise ValueError.
        Sentinel guard (B1): already-completed stages are skipped.
        Cache logic (D2/B4): fitted sklearn scaler instances are cached
        as ``_data_scaler_cache`` and ``_ts_feature_scaler_cache``.
        Stage branching: fit/None runs full scaling + transform + split,
        test/predict reuses cached scalers via transform only,
        validate is a no-op for data mutation.

        When ``scale_data`` is False, scaling and time feature extraction
        are skipped entirely to preserve raw values.

        Args:
            stage: Lightning stage identifier. Defaults to ``None``.
                Allowed values: ``"fit"``, ``"validate"``, ``"test"``,
                ``"predict"``, or ``None``.

        Raises:
            ValueError: If ``stage`` is not one of the allowed values.
        """
        if stage not in ('fit', 'validate', 'test', 'predict', None):
            raise ValueError(f'Unknown stage: {stage!r}')
        if stage in self._setup_completed_stages or None in self._setup_completed_stages:
            return

        if stage == 'validate':
            # No data mutation for validate stage
            self._setup_completed_stages.add(stage)
            return

        assert self._full_data is not None, 'Full data not set; call prepare_data() first'
        assert self._train_slice is not None, 'Train slice not set; call _set_data_slices() first'

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

            time_series_features = extract_time_features(
                pd.DatetimeIndex(time_index)
            )
            num_time_series_features = time_series_features.shape[-1]
        else:
            time_series_features = np.empty((0, 0))
            num_time_series_features = 0

        if self.scale_data:
            if stage in ('fit', None):
                # Fit scalers on train slice, transform full data, cache them
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
                    scaled_ts_features = ts_feature_scaler.transform(
                        time_series_features
                    )
                    scaled_ts_features = np.expand_dims(
                        scaled_ts_features, axis=0
                    )
                    assert self._full_data is not None
                    repeated_ts = np.repeat(
                        scaled_ts_features, self._full_data.shape[0], axis=0
                    )
                    self._full_data = np.concatenate(
                        [repeated_ts, self._full_data], axis=-1
                    )
            else:
                # stage in ('test', 'predict'): reuse cached scalers
                assert (
                    self._data_scaler_cache is not None
                ), 'Data scaler not cached; call setup("fit") first'
                self._full_data = self._data_scaler_cache.transform(full_array)

                # Apply module-specific transform
                self._transform_data()

                # Reuse cached time feature scaler if applicable
                if num_time_series_features > 0 and self._ts_feature_scaler_cache is not None:
                    scaled_ts_features = self._ts_feature_scaler_cache.transform(
                        time_series_features
                    )
                    scaled_ts_features = np.expand_dims(
                        scaled_ts_features, axis=0
                    )
                    assert self._full_data is not None
                    repeated_ts = np.repeat(
                        scaled_ts_features, self._full_data.shape[0], axis=0
                    )
                    self._full_data = np.concatenate(
                        [repeated_ts, self._full_data], axis=-1
                    )
        else:
            # No scaling: apply module-specific transform on raw data.
            # Keep original _full_data (may be DataFrame with DatetimeIndex)
            # so _transform_data can convert it properly.
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
        enum members (D-03), NOT string literals.

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
        raise ValueError(
            f'Unsupported scaling method for forecasting: '
            f'{self.data_scaling_method}'
        )

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

    def _post_prepare_data(self) -> None:
        """Called at the end of ``prepare_data()`` to set data slices.

        Subclasses should invoke this after loading ``_full_data``.
        """
        self._set_data_slices()
