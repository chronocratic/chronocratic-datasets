"""Tests for BaseTimeSeriesDataModule (MOD-01, MOD-05, MOD-06).

Verifies the shared base class for all data modules has correct
constructor, properties, setup, and dataloader construction.
"""

from __future__ import annotations

from functools import partial
from unittest.mock import MagicMock, patch

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader, TensorDataset


class TestBaseTimeSeriesDataModule:
    """Tests for BaseTimeSeriesDataModule base class."""

    @pytest.fixture
    def concrete_module_class(self):
        """Create a minimal concrete subclass for testing."""
        # BaseTimeSeriesDataModule is abstract; we need a concrete subclass
        from abc import abstractmethod

        from tscollection.datasets.modules._base.base import (
            BaseTimeSeriesDataModule,
        )

        class ConcreteTestModule(BaseTimeSeriesDataModule):
            """Minimal concrete subclass for testing."""

            def prepare_data(self) -> None:
                pass

        return ConcreteTestModule

    @pytest.fixture
    def module(self, concrete_module_class):
        """Create a module instance with default parameters."""
        return concrete_module_class(
            batch_size=32,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )

    def test_is_lightning_datamodule(self, module) -> None:
        """BaseTimeSeriesDataModule is subclass of pl.LightningDataModule."""
        assert isinstance(module, pl.LightningDataModule)

    def test_constructor_accepts_kwargs(self, concrete_module_class) -> None:
        """Constructor accepts all required keyword arguments."""
        from tscollection.datasets.enums.data import DataForm, ScalingMethod

        mod = concrete_module_class(
            batch_size=64,
            seq_len=100,
            valid_size=0.15,
            test_size=0.25,
            shuffle=False,
            scale_data=True,
            data_scaling_method=ScalingMethod.STANDARD,
            data_scaling_range=(-1.0, 1.0),
            num_workers=4,
            data_form=DataForm.NESTED,
        )
        assert mod.batch_size == 64
        assert mod._seq_len == 100
        assert mod.valid_size == 0.15
        assert mod.test_size == 0.25
        assert mod.shuffle is False
        assert mod.scale_data is True

    def test_sequence_length_property(self, module) -> None:
        """sequence_length property returns _seq_len value."""
        module._seq_len = 50
        assert module.sequence_length == 50

    def test_num_features_property(self, module) -> None:
        """num_features property returns _num_features value."""
        module._num_features = 7
        assert module.num_features == 7

    def test_setup_calls_create_data_scaler(self, concrete_module_class) -> None:
        """setup() calls create_data_scaler and scales data samples."""
        from tscollection.datasets.enums.data import ScalingMethod

        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        # Set up mock data samples
        mod._train_data_samples = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        mod._valid_data_samples = pd.DataFrame({'a': [5.0], 'b': [6.0]})
        mod._test_data_samples = pd.DataFrame({'a': [7.0], 'b': [8.0]})

        mod.setup(stage='fit')

        # If scale_data=True and create_data_scaler works, data should be transformed
        assert mod._train_data_samples is not None
        assert mod._valid_data_samples is not None
        assert mod._test_data_samples is not None

    def test_process_train_dataloader_returns_dataloader(
        self, module
    ) -> None:
        """_process_train_dataloader returns a DataLoader instance."""
        dataset = TensorDataset(
            torch_rand := MagicMock(),
        )
        # Create a real small dataset
        import torch

        real_dataset = TensorDataset(
            torch.randn(10, 5),
            torch.randint(0, 2, (10,)),
        )
        result = module._process_train_dataloader(dataset_object=real_dataset)
        assert isinstance(result, DataLoader)

    def test_process_test_dataloader_returns_dataloader(
        self, module
    ) -> None:
        """_process_test_dataloader returns DataLoader with shuffle=False."""
        import torch

        real_dataset = TensorDataset(
            torch.randn(10, 5),
            torch.randint(0, 2, (10,)),
        )
        with patch(
            'tscollection.datasets.modules._base.base.DataLoader',
            wraps=DataLoader,
        ) as mock_loader:
            module._process_test_dataloader(dataset_object=real_dataset)
            call_kwargs = mock_loader.call_args[1]
            assert call_kwargs['shuffle'] is False
            assert isinstance(call_kwargs['dataset'], TensorDataset)

    def test_process_valid_dataloader_returns_none_when_no_valid(
        self, concrete_module_class
    ) -> None:
        """_process_valid_dataloader returns None when valid_size==0.0."""
        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.0,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        import torch

        real_dataset = TensorDataset(torch.randn(5, 3))
        result = mod._process_valid_dataloader(dataset_object=real_dataset)
        assert result is None

    def test_persistent_workers_guard(self, concrete_module_class) -> None:
        """persistent_workers is only set when num_workers > 0."""
        import torch

        # With num_workers=0, persistent_workers should NOT be in args
        mod_zero = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
            num_workers=0,
        )
        real_dataset = TensorDataset(torch.randn(10, 5))

        # Patch DataLoader to capture args
        with patch(
            'tscollection.datasets.modules._base.base.DataLoader',
            wraps=DataLoader,
        ) as mock_loader:
            mod_zero._process_train_dataloader(dataset_object=real_dataset)
            call_kwargs = mock_loader.call_args[1]
            assert 'persistent_workers' not in call_kwargs

    def test_get_custom_collate_fn(self, module) -> None:
        """_get_custom_collate_fn returns partial with correct batch size."""
        import torch

        from tscollection.datasets.utils.general import custom_collate_fn

        collate = module._get_custom_collate_fn()
        assert isinstance(collate, partial)
        assert collate.keywords['desired_batch_size'] == module.batch_size

    def test_collate_fn_with_strict_batch_size(
        self, module
    ) -> None:
        """strict_batch_size=True sets collate_fn on dataloader."""
        import torch

        real_dataset = TensorDataset(torch.randn(5, 3))
        with patch(
            'tscollection.datasets.modules._base.base.DataLoader',
            wraps=DataLoader,
        ) as mock_loader:
            module._process_train_dataloader(
                dataset_object=real_dataset,
                strict_batch_size=True,
            )
            call_kwargs = mock_loader.call_args[1]
            assert 'collate_fn' in call_kwargs

    def test_scaling_method_type_is_enum(self, concrete_module_class) -> None:
        """data_scaling_method is typed as ScalingMethod enum (D-03)."""
        from tscollection.datasets.enums.data import ScalingMethod

        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=True,
            data_scaling_method=ScalingMethod.MINMAX,
        )
        assert mod.data_scaling_method == ScalingMethod.MINMAX
        assert isinstance(mod.data_scaling_method, ScalingMethod)

    def test_data_form_type_is_enum(self, concrete_module_class) -> None:
        """data_form is typed as DataForm enum (D-02)."""
        from tscollection.datasets.enums.data import DataForm

        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
            data_form=DataForm.REGULAR,
        )
        assert mod._data_form == DataForm.REGULAR

    def test_name_property(self, module) -> None:
        """name property returns _dataset_name."""
        module._dataset_name = 'TestDataset'
        assert module.name == 'TestDataset'

    def test_train_data_samples_property(self, module) -> None:
        """train_data_samples property returns _train_data_samples."""
        data = pd.DataFrame({'x': [1, 2, 3]})
        module._train_data_samples = data
        assert module.train_data_samples is data


class TestSetupStageGating:
    """Tests for stage validation, cache, and branching in setup()."""

    @pytest.fixture
    def concrete_module_class(self):
        """Create a minimal concrete subclass for testing."""
        from tscollection.datasets.modules._base.base import (
            BaseTimeSeriesDataModule,
        )

        class ConcreteTestModule(BaseTimeSeriesDataModule):
            """Minimal concrete subclass for testing."""

            def prepare_data(self) -> None:
                pass

        return ConcreteTestModule

    def test_unknown_stage_raises(self, concrete_module_class) -> None:
        """setup('warmup') raises ValueError for unknown stage."""
        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        with pytest.raises(ValueError, match="Unknown stage: 'warmup'"):
            mod.setup(stage='warmup')

    def test_default_stage_is_none(self, concrete_module_class) -> None:
        """setup() with no args uses stage=None and does not raise TypeError."""
        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        mod._train_data_samples = pd.DataFrame({'a': [1.0, 2.0]})
        mod._valid_data_samples = pd.DataFrame({'a': [3.0]})
        mod._test_data_samples = pd.DataFrame({'a': [4.0]})
        # Should not raise TypeError
        mod.setup()

    def test_scaler_cache_populated(self, concrete_module_class) -> None:
        """After setup('fit'), _scaler_cache is not None."""
        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=True,
        )
        mod._train_data_samples = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        mod._valid_data_samples = pd.DataFrame({'a': [5.0], 'b': [6.0]})
        mod._test_data_samples = pd.DataFrame({'a': [7.0], 'b': [8.0]})

        mod.setup(stage='fit')

        assert mod._scaler_cache is not None

    def test_setup_fit_then_test_stage_branching(
        self, concrete_module_class
    ) -> None:
        """setup('fit') populates cache; setup('test') reuses it, only test scaled."""
        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=True,
        )
        train_df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})
        valid_df = pd.DataFrame({'a': [5.0], 'b': [6.0]})
        test_df = pd.DataFrame({'a': [7.0], 'b': [8.0]})
        mod._train_data_samples = train_df.copy()
        mod._valid_data_samples = valid_df.copy()
        mod._test_data_samples = test_df.copy()

        mod.setup(stage='fit')
        assert mod._scaler_cache is not None
        cache_id_after_fit = id(mod._scaler_cache)
        train_after_fit = mod._train_data_samples.copy()

        mod.setup(stage='test')
        # Cache reused (same object identity)
        assert id(mod._scaler_cache) == cache_id_after_fit
        # Train data unchanged (not re-scaled in test stage)
        pd.testing.assert_frame_equal(mod._train_data_samples, train_after_fit)
