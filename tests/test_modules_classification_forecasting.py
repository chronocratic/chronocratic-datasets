"""Tests for classification and forecasting base DataModule classes."""

from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from chronocratic.datasets.enums.data import ForecastingMode, ScalingMethod
from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule
from chronocratic.datasets.utils.features import TIME_FEATURE_COUNT


@pytest.fixture
def concrete_forecasting_class():
    """Concrete implementation of BaseForecastingTimeSeriesDataModule for testing."""
    from chronocratic.datasets.modules._base.forecasting import BaseForecastingTimeSeriesDataModule

    class ConcreteForecasting(BaseForecastingTimeSeriesDataModule):
        def _do_prepare_data(self) -> None:
            pass

        def _set_data_slices(self) -> None:
            pass

        def _transform_data(self) -> None:
            pass

        def _build_sliding_dataset(self, data, internal_mode, step, horizon):
            raise NotImplementedError

    return ConcreteForecasting


class TestBaseClassificationTimeSeriesDataModule:
    """Tests for BaseClassificationTimeSeriesDataModule."""

    @pytest.fixture
    def module_class(self):
        """Lazy-import the module class to verify it exists."""
        from chronocratic.datasets.modules._base.classification import (
            BaseClassificationTimeSeriesDataModule,
        )

        return BaseClassificationTimeSeriesDataModule

    def test_is_subclass_of_base_and_abc(self, module_class: type) -> None:
        """BaseClassificationTimeSeriesDataModule inherits from base and ABC."""
        assert issubclass(module_class, BaseTimeSeriesDataModule)
        assert issubclass(module_class, ABC)

    def test_constructor_accepts_target_column_name(self, module_class: type) -> None:
        """Classification base constructor accepts target_column_name."""
        # ABC + abstract methods prevent direct instantiation;
        # verify the signature accepts the parameter instead.
        import inspect

        sig = inspect.signature(module_class.__init__)
        assert 'target_column_name' in sig.parameters

    def test_constructor_accepts_splitting_strategy(self, module_class: type) -> None:
        """Classification base constructor accepts splitting_strategy enum."""
        import inspect

        sig = inspect.signature(module_class.__init__)
        assert 'splitting_strategy' in sig.parameters

    def test_constructor_accepts_data_form(self, module_class: type) -> None:
        """Classification base constructor accepts data_form (even if set by subclass)."""
        import inspect

        inspect.signature(module_class.__init__)
        # data_form is inherited from base; classification base may or may not
        # re-expose it, but the parent signature should include it.
        base_sig = inspect.signature(BaseTimeSeriesDataModule.__init__)
        assert 'data_form' in base_sig.parameters

    def test_exposes_num_classes_property(self) -> None:
        """Classification base exposes num_classes property."""
        from chronocratic.datasets.modules._base.classification import (
            BaseClassificationTimeSeriesDataModule,
        )

        assert hasattr(BaseClassificationTimeSeriesDataModule, 'num_classes')
        assert isinstance(
            BaseClassificationTimeSeriesDataModule.__dict__.get('num_classes'), property
        )

    def test_has_separate_target_feature_partial(self) -> None:
        """Classification base has _separate_target_feature as partial function."""
        # We need a concrete implementation to test instantiation.
        # Verify the attribute exists in the class or is set via __init__.
        import inspect

        from chronocratic.datasets.modules._base.classification import (
            BaseClassificationTimeSeriesDataModule,
        )

        source = inspect.getsource(BaseClassificationTimeSeriesDataModule.__init__)
        assert '_separate_target_feature' in source
        assert 'partial' in source


class TestBaseForecastingTimeSeriesDataModule:
    """Tests for BaseForecastingTimeSeriesDataModule."""

    @pytest.fixture
    def module_class(self):
        """Lazy-import the module class to verify it exists."""
        from chronocratic.datasets.modules._base.forecasting import (
            BaseForecastingTimeSeriesDataModule,
        )

        return BaseForecastingTimeSeriesDataModule

    def test_is_subclass_of_base_and_abc(self, module_class: type) -> None:
        """BaseForecastingTimeSeriesDataModule is subclass of BaseTimeSeriesDataModule and ABC."""
        assert issubclass(module_class, BaseTimeSeriesDataModule)
        assert issubclass(module_class, ABC)

    def test_constructor_accepts_mode(self, module_class: type) -> None:
        """Forecasting base constructor accepts mode (ForecastingMode)."""
        import inspect

        sig = inspect.signature(module_class.__init__)
        assert 'mode' in sig.parameters

    def test_constructor_accepts_seq_len(self, module_class: type) -> None:
        """Forecasting base constructor accepts seq_len (int)."""
        import inspect

        sig = inspect.signature(module_class.__init__)
        assert 'seq_len' in sig.parameters

    def test_has_abstract_set_data_slices(self, module_class: type) -> None:
        """Forecasting base _set_data_slices is abstract."""
        assert hasattr(module_class, '_set_data_slices')
        assert getattr(module_class._set_data_slices, '__isabstractmethod__', False)

    def test_has_abstract_transform_data(self, module_class: type) -> None:
        """Forecasting base has abstract _transform_data method."""
        assert hasattr(module_class, '_transform_data')
        assert getattr(module_class._transform_data, '__isabstractmethod__', False)

    def test_prepare_data_scaler_minmax(self, concrete_forecasting_class: type) -> None:
        """_prepare_data_scaler returns MinMaxScaler for ScalingMethod.MINMAX."""
        mod = concrete_forecasting_class(
            batch_size=32,
            seq_len=128,
            valid_size=0.1,
            test_size=0.5,
            shuffle=False,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
            data_scaling_range=(0, 1),
            num_workers=0,
            mode=ForecastingMode.UNIVARIATE,
        )
        scaler = mod._prepare_data_scaler()
        assert isinstance(scaler, MinMaxScaler)

    def test_prepare_data_scaler_standard(self, concrete_forecasting_class: type) -> None:
        """_prepare_data_scaler returns StandardScaler for ScalingMethod.STANDARD."""
        mod = concrete_forecasting_class(
            batch_size=32,
            seq_len=128,
            valid_size=0.1,
            test_size=0.5,
            shuffle=False,
            scale_data=True,
            data_scaling_method=ScalingMethod.STANDARD,
            data_scaling_range=(0, 1),
            num_workers=0,
            mode=ForecastingMode.UNIVARIATE,
        )
        scaler = mod._prepare_data_scaler()
        assert isinstance(scaler, StandardScaler)

    def test_prepare_data_scaler_invalid_raises(self, concrete_forecasting_class: type) -> None:
        """__init__ raises ValueError for scale_data=True with ScalingMethod.NONE."""
        with pytest.raises(
            ValueError, match=r'scale_data=True is incompatible with ScalingMethod\.NONE'
        ):
            concrete_forecasting_class(
                batch_size=32,
                seq_len=128,
                valid_size=0.1,
                test_size=0.5,
                shuffle=False,
                scale_data=True,
                data_scaling_method=ScalingMethod.NONE,
                data_scaling_range=(0, 1),
                num_workers=0,
                mode=ForecastingMode.UNIVARIATE,
            )


class TestPrepareDimensions:
    """Tests for prepare_dimensions() API and _compute_dimensions() hook."""

    def test_base_prepare_dimensions_exists(self) -> None:
        """prepare_dimensions() exists on base and returns a 2-tuple."""
        from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule

        class ConcreteTestModule(BaseTimeSeriesDataModule):
            def _do_prepare_data(self) -> None:
                pass

        module = ConcreteTestModule(
            batch_size=32,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        assert hasattr(module, 'prepare_dimensions')
        result = module.prepare_dimensions()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_classification_raises_without_prepare_data(self) -> None:
        """Classification _compute_dimensions raises RuntimeError without prepare_data."""
        from chronocratic.datasets.modules._base.classification import (
            BaseClassificationTimeSeriesDataModule,
        )

        class ConcreteClassification(BaseClassificationTimeSeriesDataModule):
            def _do_prepare_data(self) -> None:
                pass

            def _load_cached_data(self) -> None:
                pass

            def train_dataloader(self, **kwargs):
                pass

            def val_dataloader(self, **kwargs):
                pass

            def test_dataloader(self, **kwargs):
                pass

        module = ConcreteClassification(
            dataset_folder_path=Path('/nonexistent'),
            batch_size=32,
            valid_size=0.2,
            test_size=0.2,
            shuffle=False,
            scale_data=False,
            target_column_name='label',
        )
        # _train_data_samples is None — prepare_data was never called
        with pytest.raises(RuntimeError, match=r'prepare_dimensions.*prepare_data'):
            module.prepare_dimensions()

    def test_forecasting_pre_setup_with_dataframe(self, concrete_forecasting_class: type) -> None:
        """Forecasting _compute_dimensions adds TIME_FEATURE_COUNT for DataFrame."""
        module = concrete_forecasting_class(
            batch_size=32,
            seq_len=96,
            valid_size=0.1,
            test_size=0.5,
            shuffle=False,
            scale_data=True,
            mode=ForecastingMode.UNIVARIATE,
        )
        # Inject raw data and typed time index
        dates = pd.date_range('2020-01-01', periods=100, freq='h')
        module._full_data_raw = (
            np.random.default_rng(42).standard_normal((100, 8)).astype(np.float32)
        )
        module._time_index = pd.DatetimeIndex(dates)
        n_features, seq_len = module.prepare_dimensions()
        assert n_features == 8 + TIME_FEATURE_COUNT
        assert seq_len == 96

    def test_forecasting_pre_setup_with_numpy(self, concrete_forecasting_class: type) -> None:
        """Forecasting _compute_dimensions does NOT add TIME_FEATURE_COUNT for numpy."""
        module = concrete_forecasting_class(
            batch_size=32,
            seq_len=96,
            valid_size=0.1,
            test_size=0.5,
            shuffle=False,
            scale_data=False,
            mode=ForecastingMode.UNIVARIATE,
        )
        # Inject raw numpy array (no DatetimeIndex, so no time features)
        module._full_data_raw = (
            np.random.default_rng(42).standard_normal((100, 6)).astype(np.float32)
        )
        n_features, seq_len = module.prepare_dimensions()
        assert n_features == 6  # No TIME_FEATURE_COUNT added
        assert seq_len == 96

    def test_post_setup_returns_cached(self, concrete_forecasting_class: type) -> None:
        """prepare_dimensions() returns cached _num_features when set."""
        module = concrete_forecasting_class(
            batch_size=32,
            seq_len=96,
            valid_size=0.1,
            test_size=0.5,
            shuffle=False,
            scale_data=False,
            mode=ForecastingMode.UNIVARIATE,
        )
        # Simulate post-setup: _num_features is already populated
        module._num_features = 42
        # Inject typed attributes that would compute a different value
        dates = pd.date_range('2020-01-01', periods=100, freq='h')
        module._full_data_raw = (
            np.random.default_rng(42).standard_normal((100, 8)).astype(np.float32)
        )
        module._time_index = pd.DatetimeIndex(dates)
        # Should return cached value (42), NOT compute from typed attrs
        n_features, seq_len = module.prepare_dimensions()
        assert n_features == 42
        assert seq_len == 96


class TestForecastingTypedAttrs:
    """Tests for typed data attributes in forecasting modules."""

    def test_has_typed_attributes(self, concrete_forecasting_class: type) -> None:
        """Forecasting base has _full_data_raw, _time_index, _full_data_scaled."""
        mod = concrete_forecasting_class(
            batch_size=32,
            seq_len=128,
            valid_size=0.1,
            test_size=0.5,
            shuffle=False,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
            data_scaling_range=(0, 1),
            num_workers=0,
            mode=ForecastingMode.UNIVARIATE,
        )
        assert hasattr(mod, '_full_data_raw')
        assert hasattr(mod, '_time_index')
        assert hasattr(mod, '_full_data_scaled')
        assert mod._full_data_raw is None
        assert mod._time_index is None
        assert mod._full_data_scaled is None

    def test_full_data_property_routes_to_raw_before_scaling(
        self, concrete_forecasting_class: type
    ) -> None:
        """full_data property returns _full_data_raw before scaling."""
        mod = concrete_forecasting_class(
            batch_size=32,
            seq_len=128,
            valid_size=0.1,
            test_size=0.5,
            shuffle=False,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
            data_scaling_range=(0, 1),
            num_workers=0,
            mode=ForecastingMode.UNIVARIATE,
        )
        raw = np.random.default_rng(42).standard_normal((100, 5)).astype(np.float32)
        mod._full_data_raw = raw
        assert mod.full_data is not None
        np.testing.assert_array_equal(mod.full_data, raw)
        np.testing.assert_array_equal(mod._full_data_raw, raw)

    def test_dataframe_injection_sets_time_index(self, concrete_forecasting_class: type) -> None:
        """Setting typed attributes with DataFrame data works correctly."""
        mod = concrete_forecasting_class(
            batch_size=32,
            seq_len=128,
            valid_size=0.1,
            test_size=0.5,
            shuffle=False,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
            data_scaling_range=(0, 1),
            num_workers=0,
            mode=ForecastingMode.UNIVARIATE,
        )
        dates = pd.date_range('2020-01-01', periods=100, freq='h')
        df = pd.DataFrame(
            np.random.default_rng(42).standard_normal((100, 5)).astype(np.float32), index=dates
        )
        mod._full_data_raw = df.to_numpy()
        mod._time_index = pd.DatetimeIndex(df.index)
        assert mod._time_index is not None
        assert isinstance(mod._time_index, pd.DatetimeIndex)
        assert len(mod._time_index) == 100
        np.testing.assert_array_equal(mod._full_data_raw, df.to_numpy())

    def test_cache_helpers_exist(self) -> None:
        """Forecasting base has cache helper methods."""
        from chronocratic.datasets.modules._base.forecasting import (
            BaseForecastingTimeSeriesDataModule,
        )

        assert hasattr(BaseForecastingTimeSeriesDataModule, '_resolve_cache_dir')
        assert hasattr(BaseForecastingTimeSeriesDataModule, '_save_scaler_to_cache')
        assert hasattr(BaseForecastingTimeSeriesDataModule, '_load_scaler_from_cache')

    def test_resolve_cache_dir_returns_path(self, concrete_forecasting_class: type) -> None:
        """_resolve_cache_dir returns a Path object."""
        mod = concrete_forecasting_class(
            batch_size=32,
            seq_len=128,
            valid_size=0.1,
            test_size=0.5,
            shuffle=False,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
            data_scaling_range=(0, 1),
            num_workers=0,
            mode=ForecastingMode.UNIVARIATE,
        )
        mod._dataset_name = 'TestDataset'
        result = mod._resolve_cache_dir()
        assert isinstance(result, Path)
        assert 'TestDataset' in str(result)

    def test_finalize_prepare_data_is_noop(self) -> None:
        """_finalize_prepare_data does not set slices for forecasting."""
        from chronocratic.datasets.modules._base.forecasting import (
            BaseForecastingTimeSeriesDataModule,
        )

        # This test uses a custom _set_data_slices implementation that actually
        # sets self._train_slice, so it cannot share the standard fixture.
        class ConcreteForecasting(BaseForecastingTimeSeriesDataModule):
            def _do_prepare_data(self) -> None:
                pass

            def _set_data_slices(self) -> None:
                self._train_slice = slice(0, 10)

            def _transform_data(self) -> None:
                pass

            def _build_sliding_dataset(self, data, internal_mode, step, horizon):
                raise NotImplementedError

        mod = ConcreteForecasting(
            batch_size=32,
            seq_len=128,
            valid_size=0.1,
            test_size=0.5,
            shuffle=False,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
            data_scaling_range=(0, 1),
            num_workers=0,
            mode=ForecastingMode.UNIVARIATE,
        )
        mod._finalize_prepare_data()
        assert mod._train_slice is None
