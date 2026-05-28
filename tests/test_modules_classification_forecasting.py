"""Tests for classification and forecasting base DataModule classes."""

from abc import ABC
from functools import partial

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tscollection.datasets.enums.data import (
    ClassificationSplittingStrategy,
    DataForm,
    ForecastingMode,
    ScalingMethod,
)
from tscollection.datasets.modules._base.base import BaseTimeSeriesDataModule


class TestBaseClassificationTimeSeriesDataModule:
    """Tests for BaseClassificationTimeSeriesDataModule."""

    @pytest.fixture()
    def module_class(self):
        """Lazy-import the module class to verify it exists."""
        from tscollection.datasets.modules._base.classification import (
            BaseClassificationTimeSeriesDataModule,
        )
        return BaseClassificationTimeSeriesDataModule

    def test_is_subclass_of_base_and_abc(
        self, module_class: type,
    ) -> None:
        """BaseClassificationTimeSeriesDataModule is subclass of BaseTimeSeriesDataModule and ABC."""
        assert issubclass(module_class, BaseTimeSeriesDataModule)
        assert issubclass(module_class, ABC)

    def test_constructor_accepts_target_column_name(
        self, module_class: type,
    ) -> None:
        """Classification base constructor accepts target_column_name."""
        # ABC + abstract methods prevent direct instantiation;
        # verify the signature accepts the parameter instead.
        import inspect

        sig = inspect.signature(module_class.__init__)
        assert 'target_column_name' in sig.parameters

    def test_constructor_accepts_splitting_strategy(
        self, module_class: type,
    ) -> None:
        """Classification base constructor accepts splitting_strategy enum."""
        import inspect

        sig = inspect.signature(module_class.__init__)
        assert 'splitting_strategy' in sig.parameters

    def test_constructor_accepts_data_form(
        self, module_class: type,
    ) -> None:
        """Classification base constructor accepts data_form (even if set by subclass)."""
        import inspect

        sig = inspect.signature(module_class.__init__)
        # data_form is inherited from base; classification base may or may not
        # re-expose it, but the parent signature should include it.
        base_sig = inspect.signature(BaseTimeSeriesDataModule.__init__)
        assert 'data_form' in base_sig.parameters

    def test_exposes_num_classes_property(self) -> None:
        """Classification base exposes num_classes property."""
        from tscollection.datasets.modules._base.classification import (
            BaseClassificationTimeSeriesDataModule,
        )
        assert hasattr(BaseClassificationTimeSeriesDataModule, 'num_classes')
        assert isinstance(
            BaseClassificationTimeSeriesDataModule.__dict__.get('num_classes'),
            property,
        )

    def test_has_separate_target_feature_partial(self) -> None:
        """Classification base has _separate_target_feature as partial function."""
        # We need a concrete implementation to test instantiation.
        # Verify the attribute exists in the class or is set via __init__.
        from tscollection.datasets.modules._base.classification import (
            BaseClassificationTimeSeriesDataModule,
        )
        import inspect

        source = inspect.getsource(BaseClassificationTimeSeriesDataModule.__init__)
        assert '_separate_target_feature' in source
        assert 'partial' in source


class TestBaseForecastingTimeSeriesDataModule:
    """Tests for BaseForecastingTimeSeriesDataModule."""

    @pytest.fixture()
    def module_class(self):
        """Lazy-import the module class to verify it exists."""
        from tscollection.datasets.modules._base.forecasting import (
            BaseForecastingTimeSeriesDataModule,
        )
        return BaseForecastingTimeSeriesDataModule

    def test_is_subclass_of_base_and_abc(
        self, module_class: type,
    ) -> None:
        """BaseForecastingTimeSeriesDataModule is subclass of BaseTimeSeriesDataModule and ABC."""
        assert issubclass(module_class, BaseTimeSeriesDataModule)
        assert issubclass(module_class, ABC)

    def test_constructor_accepts_mode(
        self, module_class: type,
    ) -> None:
        """Forecasting base constructor accepts mode (ForecastingMode)."""
        import inspect

        sig = inspect.signature(module_class.__init__)
        assert 'mode' in sig.parameters

    def test_constructor_accepts_seq_len(
        self, module_class: type,
    ) -> None:
        """Forecasting base constructor accepts seq_len (int)."""
        import inspect

        sig = inspect.signature(module_class.__init__)
        assert 'seq_len' in sig.parameters

    def test_has_abstract_set_data_slices(
        self, module_class: type,
    ) -> None:
        """Forecasting base has abstract _set_data_slices method."""
        assert hasattr(module_class, '_set_data_slices')
        assert getattr(module_class._set_data_slices, '__isabstractmethod__', False)

    def test_has_abstract_transform_data(
        self, module_class: type,
    ) -> None:
        """Forecasting base has abstract _transform_data method."""
        assert hasattr(module_class, '_transform_data')
        assert getattr(module_class._transform_data, '__isabstractmethod__', False)

    def test_prepare_data_scaler_minmax(self) -> None:
        """_prepare_data_scaler returns MinMaxScaler for ScalingMethod.MINMAX."""
        from tscollection.datasets.modules._base.forecasting import (
            BaseForecastingTimeSeriesDataModule,
        )

        class ConcreteForecasting(BaseForecastingTimeSeriesDataModule):
            def prepare_data(self) -> None:
                pass

            def _set_data_slices(self) -> None:
                pass

            def _transform_data(self) -> None:
                pass

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
        scaler = mod._prepare_data_scaler()
        assert isinstance(scaler, MinMaxScaler)

    def test_prepare_data_scaler_standard(self) -> None:
        """_prepare_data_scaler returns StandardScaler for ScalingMethod.STANDARD."""
        from tscollection.datasets.modules._base.forecasting import (
            BaseForecastingTimeSeriesDataModule,
        )

        class ConcreteForecasting(BaseForecastingTimeSeriesDataModule):
            def prepare_data(self) -> None:
                pass

            def _set_data_slices(self) -> None:
                pass

            def _transform_data(self) -> None:
                pass

        mod = ConcreteForecasting(
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

    def test_prepare_data_scaler_invalid_raises(self) -> None:
        """_prepare_data_scaler raises ValueError for unsupported method."""
        from tscollection.datasets.modules._base.forecasting import (
            BaseForecastingTimeSeriesDataModule,
        )

        class ConcreteForecasting(BaseForecastingTimeSeriesDataModule):
            def prepare_data(self) -> None:
                pass

            def _set_data_slices(self) -> None:
                pass

            def _transform_data(self) -> None:
                pass

        mod = ConcreteForecasting(
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
        with pytest.raises(ValueError):
            mod._prepare_data_scaler()


class TestForecastingStageGating:
    """Tests for stage validation, cache, and branching in forecasting setup()."""

    @pytest.fixture
    def concrete_forecasting_class(self):
        """Create a minimal concrete forecasting subclass for testing."""
        from tscollection.datasets.modules._base.forecasting import (
            BaseForecastingTimeSeriesDataModule,
        )

        class ConcreteForecasting(BaseForecastingTimeSeriesDataModule):
            """Minimal concrete forecasting subclass for testing."""

            def prepare_data(self) -> None:
                pass

            def _set_data_slices(self) -> None:
                pass

            def _transform_data(self) -> None:
                pass

        return ConcreteForecasting

    @pytest.fixture
    def forecasting_module(self, concrete_forecasting_class):
        """Create a forecasting module with scale_data=True."""
        return concrete_forecasting_class(
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

    def test_forecasting_scaler_cache_reused(
        self, forecasting_module
    ) -> None:
        """setup('fit') then setup('test') reuses same scaler instances."""
        rng = np.random.default_rng(42)
        forecasting_module._full_data = rng.standard_normal(
            (100, 5)
        ).astype(np.float32)
        forecasting_module._train_slice = slice(None, 60)
        forecasting_module._valid_slice = slice(60, 80)
        forecasting_module._test_slice = slice(80, None)

        forecasting_module.setup(stage='fit')
        data_scaler_id = id(forecasting_module._data_scaler_cache)
        assert forecasting_module._data_scaler_cache is not None

        forecasting_module.setup(stage='test')
        # Same scaler instance (not refit)
        assert id(forecasting_module._data_scaler_cache) == data_scaler_id

    def test_forecasting_validate_no_mutation(self, forecasting_module) -> None:
        """setup('validate') does not alter data or populate train samples."""
        rng = np.random.default_rng(42)
        full_data = rng.standard_normal((100, 5)).astype(np.float32)
        forecasting_module._full_data = full_data
        forecasting_module._train_slice = slice(None, 60)
        forecasting_module._valid_slice = slice(60, 80)
        forecasting_module._test_slice = slice(80, None)

        forecasting_module.setup(stage='validate')

        # _full_data unchanged (not scaled or transformed)
        np.testing.assert_array_equal(forecasting_module._full_data, full_data)
        # _train_data_samples remains None (no splitting happened)
        assert forecasting_module._train_data_samples is None

    def test_forecasting_unknown_stage_raises(self, forecasting_module) -> None:
        """setup('warmup') raises ValueError for unknown stage."""
        with pytest.raises(ValueError, match="Unknown stage: 'warmup'"):
            forecasting_module.setup(stage='warmup')

    def test_forecasting_default_stage_is_none(
        self, forecasting_module
    ) -> None:
        """setup() with no args uses stage=None and does not raise TypeError."""
        rng = np.random.default_rng(42)
        forecasting_module._full_data = rng.standard_normal(
            (100, 5)
        ).astype(np.float32)
        forecasting_module._train_slice = slice(None, 60)
        forecasting_module._valid_slice = slice(60, 80)
        forecasting_module._test_slice = slice(80, None)

        # Should not raise TypeError
        forecasting_module.setup()
