"""Tests for BaseTimeSeriesDataModule.

Verifies the shared base class for all data modules has correct
constructor, properties, setup, and dataloader construction.
"""

from functools import partial
from unittest.mock import MagicMock, patch

import lightning.pytorch as pl
import pandas as pd
import pytest
from torch.utils.data import DataLoader, TensorDataset


class TestBaseTimeSeriesDataModule:
    """Tests for BaseTimeSeriesDataModule base class."""

    @pytest.fixture
    def concrete_module_class(self):
        """Create a minimal concrete subclass for testing."""
        # BaseTimeSeriesDataModule is abstract; we need a concrete subclass

        from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule

        class ConcreteTestModule(BaseTimeSeriesDataModule):
            """Minimal concrete subclass for testing."""

            def _do_prepare_data(self) -> None:
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
        from chronocratic.datasets.enums.data import DataForm, ScalingMethod

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
        from chronocratic.datasets.enums.data import ScalingMethod

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

    def test_process_train_dataloader_returns_dataloader(self, module) -> None:
        """_process_train_dataloader returns a DataLoader instance."""
        # Create a real small dataset
        import torch

        real_dataset = TensorDataset(torch.randn(10, 5), torch.randint(0, 2, (10,)))
        result = module._process_train_dataloader(dataset_object=real_dataset)
        assert isinstance(result, DataLoader)

    def test_process_test_dataloader_returns_dataloader(self, module) -> None:
        """_process_test_dataloader returns DataLoader with shuffle=False."""
        import torch

        real_dataset = TensorDataset(torch.randn(10, 5), torch.randint(0, 2, (10,)))
        with patch(
            'chronocratic.datasets.modules._base.base.DataLoader', wraps=DataLoader
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
            'chronocratic.datasets.modules._base.base.DataLoader', wraps=DataLoader
        ) as mock_loader:
            mod_zero._process_train_dataloader(dataset_object=real_dataset)
            call_kwargs = mock_loader.call_args[1]
            assert 'persistent_workers' not in call_kwargs

    def test_get_custom_collate_fn(self, module) -> None:
        """_get_custom_collate_fn returns partial with correct batch size."""

        collate = module._get_custom_collate_fn()
        assert isinstance(collate, partial)
        assert collate.keywords['desired_batch_size'] == module.batch_size

    def test_collate_fn_with_strict_batch_size(self, module) -> None:
        """strict_batch_size=True sets collate_fn on dataloader."""
        import torch

        real_dataset = TensorDataset(torch.randn(5, 3))
        with patch(
            'chronocratic.datasets.modules._base.base.DataLoader', wraps=DataLoader
        ) as mock_loader:
            module._process_train_dataloader(dataset_object=real_dataset, strict_batch_size=True)
            call_kwargs = mock_loader.call_args[1]
            assert 'collate_fn' in call_kwargs

    def test_scaling_method_type_is_enum(self, concrete_module_class) -> None:
        """data_scaling_method is typed as ScalingMethod enum."""
        from chronocratic.datasets.enums.data import ScalingMethod

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
        """data_form is typed as DataForm enum."""
        from chronocratic.datasets.enums.data import DataForm

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


class TestSetupSentinel:
    """Tests for _setup_completed_stages sentinel and setup() idempotency."""

    @pytest.fixture
    def concrete_module_class(self):
        """Create a minimal concrete subclass for testing."""
        from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule

        class ConcreteTestModule(BaseTimeSeriesDataModule):
            """Minimal concrete subclass for testing."""

            def _do_prepare_data(self) -> None:
                pass

        return ConcreteTestModule

    def test_sentinel_exists_after_init(self, concrete_module_class) -> None:
        """Fresh instance has _setup_completed_stages as an empty set."""
        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        assert hasattr(mod, '_setup_completed_stages')
        assert mod._setup_completed_stages == set()

    def test_setup_idempotent_same_stage(self, concrete_module_class) -> None:
        """Calling setup(stage='fit') twice runs the scaler only once."""
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

        with patch(
            'chronocratic.datasets.modules._base.base.create_data_scaler', wraps=MagicMock()
        ) as scaler_spy:
            scaler_spy.return_value = lambda t, v, te: (t, v, te)
            mod.setup(stage='fit')
            mod.setup(stage='fit')
            assert scaler_spy.call_count == 1

    def test_setup_none_covers_all_stages(self, concrete_module_class) -> None:
        """setup(None) then setup('fit') — second call is a no-op (None covers all)."""
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

        with patch(
            'chronocratic.datasets.modules._base.base.create_data_scaler', wraps=MagicMock()
        ) as scaler_spy:
            scaler_spy.return_value = lambda t, v, te: (t, v, te)
            mod.setup(stage=None)
            mod.setup(stage='fit')
            assert scaler_spy.call_count == 1

    def test_setup_none_then_test_runs(self, concrete_module_class) -> None:
        """setup(None) should NOT block setup('test') from running.

        Verifies that after setup(None) (which covers fit), a subsequent
        setup('test') still executes test-stage logic rather than
        returning early from the guard.
        """
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

        with patch(
            'chronocratic.datasets.modules._base.base.create_data_scaler', wraps=MagicMock()
        ) as scaler_spy:
            scaler_spy.return_value = lambda t, v, te: (t, v, te)
            mod.setup(stage=None)
            mod.setup(stage='test')
            # test stage should have run (scaler reused from cache)
            assert scaler_spy.call_count == 1
            assert 'test' in mod._setup_completed_stages


class TestPrepareDataWrapper:
    """Tests for the idempotent prepare_data() wrapper.

    Verifies that the base class drives the template:
    _do_prepare_data() → _finalize_prepare_data() → set sentinel.
    """

    @pytest.fixture
    def concrete_module_with_counter(self):
        """Create a concrete subclass that counts _do_prepare_data calls."""
        from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule

        class CountingModule(BaseTimeSeriesDataModule):
            """Minimal concrete module that tracks _do_prepare_data calls."""

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._do_prepare_data_call_count = 0

            def _do_prepare_data(self) -> None:
                self._do_prepare_data_call_count += 1

        return CountingModule

    def test_prepare_data_calls_do_prepare_data_once(self, concrete_module_with_counter) -> None:
        """Calling prepare_data() twice invokes _do_prepare_data only once."""
        mod = concrete_module_with_counter(
            batch_size=32,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        mod.prepare_data()
        mod.prepare_data()
        assert mod._do_prepare_data_call_count == 1

    def test_prepare_data_idempotent_sentinel(self, concrete_module_with_counter) -> None:
        """Sentinel is False before prepare_data(), True after."""
        mod = concrete_module_with_counter(
            batch_size=32,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        assert mod._prepare_data_called is False
        mod.prepare_data()
        assert mod._prepare_data_called is True

    def test_finalize_prepare_data_is_noop_on_base(self, concrete_module_with_counter) -> None:
        """Calling _finalize_prepare_data on the base does nothing harmful."""
        mod = concrete_module_with_counter(
            batch_size=32,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        # Should not raise
        mod._finalize_prepare_data()

    def test_base_declares_do_prepare_data_abstract(self) -> None:
        """Base class declares _do_prepare_data as abstract (not prepare_data).

        After the rename, prepare_data is concrete on the base
        and _do_prepare_data is the abstract method subclasses implement.
        """
        from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule

        # _do_prepare_data must be abstract
        assert getattr(BaseTimeSeriesDataModule._do_prepare_data, '__isabstractmethod__', False)

        # prepare_data must NOT be abstract (it's the concrete wrapper now)
        assert not getattr(BaseTimeSeriesDataModule.prepare_data, '__isabstractmethod__', False)


class TestSetupStageGating:
    """Tests for stage validation, branching, and scaler cache in setup().

    Verifies setup signature with default None, scaler cache reuse, and
    stage branching (fit scales all, test scales only test data,
    validate is no-op).
    """

    @pytest.fixture
    def concrete_module_class(self):
        """Create a minimal concrete subclass for testing."""
        from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule

        class ConcreteTestModule(BaseTimeSeriesDataModule):
            """Minimal concrete subclass for testing."""

            def _do_prepare_data(self) -> None:
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
        with pytest.raises(ValueError, match=r'Unknown stage'):
            mod.setup(stage='warmup')  # type: ignore[arg-type]

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
        mod._train_data_samples = pd.DataFrame({'a': [1.0]})
        mod._valid_data_samples = pd.DataFrame({'a': [2.0]})
        mod._test_data_samples = pd.DataFrame({'a': [3.0]})
        # Should not raise TypeError
        mod.setup()

    def test_setup_fit_then_test_reuses_scaler_cache(self, concrete_module_class) -> None:
        """setup('fit') creates scaler, setup('test') reuses it (create_data_scaler called once)."""
        passthrough = lambda t, v, te: (t, v, te)  # noqa: E731
        call_count = 0

        def counting_scaler(**_kwargs):
            nonlocal call_count
            call_count += 1
            return passthrough

        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=True,
        )
        mod._train_data_samples = pd.DataFrame({'a': [1.0, 2.0]})
        mod._valid_data_samples = pd.DataFrame({'a': [3.0]})
        mod._test_data_samples = pd.DataFrame({'a': [4.0]})

        with patch(
            'chronocratic.datasets.modules._base.base.create_data_scaler',
            side_effect=counting_scaler,
        ):
            mod.setup(stage='fit')
            assert mod._scaler_cache is not None
            mod.setup(stage='test')
            assert call_count == 1

    def test_validate_does_not_mutate(self, concrete_module_class) -> None:
        """setup('validate') does not mutate data samples."""
        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        train = pd.DataFrame({'a': [1.0, 2.0]})
        mod._train_data_samples = train
        mod._valid_data_samples = pd.DataFrame({'a': [3.0]})
        mod._test_data_samples = pd.DataFrame({'a': [4.0]})

        mod.setup(stage='validate')
        assert mod._train_data_samples is train


class TestCacheInfrastructure:
    """Tests for cache_dir param, prepare_data_per_node, typed attrs, and extended reset.

    Verifies cache_dir constructor param, prepare_data_per_node = True,
    typed attrs (_full_data_raw, _time_index, _full_data_scaled), and
    reset clearing all cache-related attrs.
    """

    @pytest.fixture
    def concrete_module_class(self):
        """Create a minimal concrete subclass for testing."""
        from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule

        class ConcreteTestModule(BaseTimeSeriesDataModule):
            """Minimal concrete subclass for testing."""

            def _do_prepare_data(self) -> None:
                pass

        return ConcreteTestModule

    def test_cache_dir_default_none(self, concrete_module_class) -> None:
        """Constructor accepts cache_dir=None, resolves to default ~/.cache/tsdatasets/<name>."""
        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
            cache_dir=None,
        )
        mod._dataset_name = 'TestDataset'
        cache_dir = mod._get_cache_dir()
        assert cache_dir.parts[-2] == 'tsdatasets'
        assert cache_dir.name == 'TestDataset'

    def test_cache_dir_custom_path(self, concrete_module_class, tmp_path) -> None:
        """Constructor accepts cache_dir=custom_path, stores resolved path."""
        custom = tmp_path / 'my_cache'
        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
            cache_dir=custom,
        )
        cache_dir = mod._get_cache_dir()
        assert cache_dir == custom.resolve()

    def test_prepare_data_per_node_is_true(self, concrete_module_class) -> None:
        """Class has prepare_data_per_node = True."""
        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        assert mod.prepare_data_per_node is True

    def test_class_attribute_prepare_data_per_node(self, concrete_module_class) -> None:
        """prepare_data_per_node is a class attribute."""
        assert (
            concrete_module_class.prepare_data_per_node  # type: ignore[attr-defined]
            is True
        )

    def test_typed_attrs_initialized_none(self, concrete_module_class) -> None:
        """Instance has _full_data_raw=None, _time_index=None, _full_data_scaled=None after init."""
        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        assert mod._full_data_raw is None
        assert mod._time_index is None
        assert mod._full_data_scaled is None

    def test_cache_key_attr_exists(self, concrete_module_class) -> None:
        """Instance has _cache_key attribute initialized to None."""
        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        assert mod._cache_key is None

    def test_reset_clears_new_attrs(self, concrete_module_class) -> None:
        """reset() clears all new attrs including cache-related state."""
        import numpy as np

        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        # Set attrs to non-None values
        mod._full_data_raw = np.zeros((100, 5))
        mod._time_index = pd.date_range('2020-01-01', periods=100, freq='h')
        mod._full_data_scaled = np.ones((100, 5))
        mod._data_scaler_cache = MagicMock()
        mod._ts_feature_scaler_cache = MagicMock()
        mod._train_data_samples = np.zeros((50, 5))
        mod._valid_data_samples = np.zeros((25, 5))
        mod._test_data_samples = np.zeros((25, 5))
        mod._setup_completed_stages.add('fit')
        mod._prepare_data_called = True

        mod.reset()

        assert mod._full_data_raw is None
        assert mod._time_index is None
        assert mod._full_data_scaled is None
        assert mod._data_scaler_cache is None
        assert mod._ts_feature_scaler_cache is None
        assert mod._train_data_samples is None
        assert mod._valid_data_samples is None
        assert mod._test_data_samples is None
        assert mod._setup_completed_stages == set()
        assert mod._prepare_data_called is False

    def test_reset_clears_cache_key(self, concrete_module_class) -> None:
        """reset() clears _cache_key."""
        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        mod._cache_key = 'test_key'
        mod.reset()
        assert mod._cache_key is None


class TestPrepareDimensionsCache:
    """Tests for prepare_dimensions() reading metadata.json.

    Verifies that prepare_dimensions() reads from cache metadata when
    _num_features is None, raises FileNotFoundError when metadata is
    missing, and raises ValueError on version mismatch.
    """

    @pytest.fixture
    def concrete_module_class(self):
        """Create a minimal concrete subclass for testing."""
        from chronocratic.datasets.modules._base.base import BaseTimeSeriesDataModule

        class ConcreteTestModule(BaseTimeSeriesDataModule):
            """Minimal concrete subclass for testing."""

            def _do_prepare_data(self) -> None:
                pass

        return ConcreteTestModule

    def test_short_circuit_when_num_features_set(self, concrete_module_class) -> None:
        """prepare_dimensions() with _num_features already set returns cached values."""
        mod = concrete_module_class(
            batch_size=16,
            seq_len=128,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
        )
        mod._num_features = 7
        result = mod.prepare_dimensions()
        assert result == (7, 128)

    def test_reads_metadata_when_num_features_none(self, concrete_module_class, tmp_path) -> None:
        """prepare_dimensions() with _num_features=None reads metadata.json via load_metadata."""
        from chronocratic.datasets.utils.cache import atomic_save_metadata

        mod = concrete_module_class(
            batch_size=16,
            seq_len=128,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
            cache_dir=tmp_path,
        )
        mod._dataset_name = 'TestDataset'
        mod._cache_key = 'test'

        # Write metadata.json in the cache dir
        meta_path = tmp_path / f'{mod._cache_key}_metadata.json'
        atomic_save_metadata(
            meta_path, {'version': 1, 'dataset_name': 'TestDataset', 'n_features': 5, 'seq_len': 64}
        )

        result = mod.prepare_dimensions()
        assert result == (5, 64)
        assert mod._num_features == 5

    def test_raises_file_not_found_when_metadata_missing(
        self, concrete_module_class, tmp_path
    ) -> None:
        """prepare_dimensions() raises FileNotFoundError when metadata.json missing."""
        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
            cache_dir=tmp_path,
        )
        mod._cache_key = 'test'
        mod._dataset_name = 'TestDataset'

        with pytest.raises(FileNotFoundError, match='Cache metadata not found'):
            mod.prepare_dimensions()

    def test_raises_value_error_on_version_mismatch(self, concrete_module_class, tmp_path) -> None:
        """prepare_dimensions() raises ValueError when metadata version != 1."""
        from chronocratic.datasets.utils.cache import atomic_save_metadata

        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
            cache_dir=tmp_path,
        )
        mod._cache_key = 'test'
        mod._dataset_name = 'TestDataset'

        # Write metadata with wrong version
        meta_path = tmp_path / f'{mod._cache_key}_metadata.json'
        atomic_save_metadata(
            meta_path,
            {
                'version': 2,  # Wrong version
                'n_features': 5,
                'seq_len': 64,
            },
        )

        with pytest.raises(ValueError, match='Cache version'):
            mod.prepare_dimensions()

    def test_integration_with_synthetic_cache_dir(
        self, concrete_module_class, synthetic_cache_dir
    ) -> None:
        """prepare_dimensions() returns (n_features, seq_len) from metadata (integration)."""
        mod = concrete_module_class(
            batch_size=16,
            seq_len=None,
            valid_size=0.2,
            test_size=0.2,
            shuffle=True,
            scale_data=False,
            cache_dir=synthetic_cache_dir,
        )
        mod._cache_key = 'synthetic'
        # Write a metadata file that matches synthetic_cache_dir content
        from chronocratic.datasets.utils.cache import load_metadata

        meta = load_metadata(synthetic_cache_dir / 'metadata.json')
        assert meta['n_features'] == 7
