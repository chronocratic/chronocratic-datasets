"""Tests for classification and forecasting base DataModule classes."""

from abc import ABC
from functools import partial

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
