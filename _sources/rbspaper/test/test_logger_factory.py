"""Tests for logger factory and W&B tracking helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.rbspaper.pipeline.loggers import (
    _flatten_dict,
    _find_wandb_logger,
    _log_config_to_wandb,
    _log_results_to_wandb,
    create_loggers,
)


class TestLoggerFactory:
    """Verify create_loggers builds correct logger tuples per mode (D-01, D-02, D-07)."""

    def test_online_creates_dual_loggers(self, tmp_path: Any) -> None:
        """Online mode should return both TensorBoardLogger and WandbLogger."""
        loggers = create_loggers(
            run_dir=tmp_path,
            run_name='test_run',
            tracking_mode='online',
            persist_artifacts=True,
        )
        assert len(loggers) == 2
        assert 'TensorBoardLogger' in str(type(loggers[0]))
        assert 'WandbLogger' in str(type(loggers[1]))

    def test_offline_creates_dual_loggers(self, tmp_path: Any) -> None:
        """Offline mode should also return both loggers."""
        loggers = create_loggers(
            run_dir=tmp_path,
            run_name='test_run',
            tracking_mode='offline',
            persist_artifacts=True,
        )
        assert len(loggers) == 2

    def test_disabled_skips_wandb(self, tmp_path: Any) -> None:
        """Disabled mode should return only TensorBoardLogger."""
        loggers = create_loggers(
            run_dir=tmp_path,
            run_name='test_run',
            tracking_mode='disabled',
            persist_artifacts=True,
        )
        assert len(loggers) == 1
        assert 'TensorBoardLogger' in str(type(loggers[0]))

    def test_no_persist_returns_empty(self, tmp_path: Any) -> None:
        """persist_artifacts=False should return an empty tuple regardless of mode."""
        loggers = create_loggers(
            run_dir=tmp_path,
            run_name='test_run',
            tracking_mode='online',
            persist_artifacts=False,
        )
        assert loggers == ()

    def test_wandb_logger_log_model_false(self, tmp_path: Any, monkeypatch: Any) -> None:
        """WandbLogger must be created with log_model=False (D-07)."""
        captured_kwargs: dict[str, Any] = {}

        def mock_wandb_init(self: Any, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        monkeypatch.setattr(
            'lightning.pytorch.loggers.WandbLogger.__init__',
            mock_wandb_init,
        )

        create_loggers(
            run_dir=tmp_path,
            run_name='test_run',
            tracking_mode='online',
            persist_artifacts=True,
        )
        assert captured_kwargs.get('log_model') is False

    def test_wandb_logger_project_name(self, tmp_path: Any, monkeypatch: Any) -> None:
        """WandbLogger must use project='rbspaper' (D-06)."""
        captured_kwargs: dict[str, Any] = {}

        def mock_wandb_init(self: Any, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

        monkeypatch.setattr(
            'lightning.pytorch.loggers.WandbLogger.__init__',
            mock_wandb_init,
        )

        create_loggers(
            run_dir=tmp_path,
            run_name='test_run',
            tracking_mode='online',
            persist_artifacts=True,
        )
        assert captured_kwargs.get('project') == 'rbspaper'

    def test_tensorboard_save_dir(self, tmp_path: Any) -> None:
        """TensorBoardLogger save_dir must contain the provided run_dir path."""
        loggers = create_loggers(
            run_dir=tmp_path,
            run_name='test_run',
            tracking_mode='online',
            persist_artifacts=True,
        )
        tb_logger = loggers[0]  # TensorBoardLogger is always first
        assert str(tmp_path) in str(tb_logger.root_dir)

    def test_wandb_import_error_graceful_fallback(self, tmp_path: Any, monkeypatch: Any) -> None:
        """Missing WandbLogger should fall back to TensorBoardLogger only.

        Removes WandbLogger from the lightning.pytorch.loggers module to
        simulate it being unavailable (e.g., wandb package not installed),
        causing the lazy import to raise ImportError.
        """
        import lightning.pytorch.loggers as lp_loggers

        # Delete WandbLogger from the module so the import raises ImportError
        monkeypatch.delattr(lp_loggers, 'WandbLogger')

        loggers = create_loggers(
            run_dir=tmp_path,
            run_name='test_run',
            tracking_mode='online',
            persist_artifacts=True,
        )
        assert len(loggers) == 1
        assert 'TensorBoardLogger' in str(type(loggers[0]))


class TestFlattenDict:
    """Verify _flatten_dict handles nested structures (D-04 utility)."""

    def test_flat_dict_unchanged(self) -> None:
        """A flat dictionary should remain unchanged."""
        result = _flatten_dict(d={'a': 1, 'b': 2})
        assert result == {'a': 1, 'b': 2}

    def test_nested_dict_flattened(self) -> None:
        """Nested dictionaries should be flattened with separator."""
        result = _flatten_dict(d={'a': {'b': 1, 'c': {'d': 2}}})
        assert result == {'a_b': 1, 'a_c_d': 2}

    def test_list_with_primitives(self) -> None:
        """Lists of primitives should produce index-based keys."""
        result = _flatten_dict(d={'metrics': [0.85, 0.92]})
        assert result == {'metrics_0': 0.85, 'metrics_1': 0.92}

    def test_list_with_dicts(self) -> None:
        """Lists of dicts should recurse with index prefix."""
        result = _flatten_dict(d={'results': [{'accuracy': 0.85}, {'accuracy': 0.92}]})
        assert result == {'results_0_accuracy': 0.85, 'results_1_accuracy': 0.92}

    def test_empty_dict(self) -> None:
        """An empty dictionary should return an empty result."""
        result = _flatten_dict(d={})
        assert result == {}

    def test_custom_separator(self) -> None:
        """A custom separator should be used for key joining."""
        result = _flatten_dict(d={'a': {'b': 1}}, separator='/')
        assert result == {'a/b': 1}

    def test_none_values_preserved(self) -> None:
        """None values in leaf nodes should be preserved after flattening."""
        result = _flatten_dict(d={'a': None, 'b': {'c': None}})
        assert result == {'a': None, 'b_c': None}

    def test_mixed_types(self) -> None:
        """Mixed types (str, int, bool) should be handled correctly."""
        result = _flatten_dict(d={'path': '/some/path', 'count': 42, 'flag': True})
        assert result == {'path': '/some/path', 'count': 42, 'flag': True}


class TestFindWandbLogger:
    """Verify _find_wandb_logger locates or skips WandbLogger instances."""

    def test_finds_wandb_in_tuple(self, tmp_path: Any, monkeypatch: Any) -> None:
        """_find_wandb_logger should return the WandbLogger from a dual tuple."""
        loggers = create_loggers(
            run_dir=tmp_path,
            run_name='test_run',
            tracking_mode='online',
            persist_artifacts=True,
        )
        found = _find_wandb_logger(loggers=loggers)
        assert found is not None
        assert type(found).__name__ == 'WandbLogger'

    def test_returns_none_for_tensorboard_only(self, tmp_path: Any) -> None:
        """_find_wandb_logger should return None when only TensorBoardLogger exists."""
        loggers = create_loggers(
            run_dir=tmp_path,
            run_name='test_run',
            tracking_mode='disabled',
            persist_artifacts=True,
        )
        result = _find_wandb_logger(loggers=loggers)
        assert result is None

    def test_returns_none_for_empty_tuple(self) -> None:
        """_find_wandb_logger should return None for an empty loggers tuple."""
        result = _find_wandb_logger(loggers=())
        assert result is None


class TestWandbConfigAndResultsLogging:
    """Verify W&B logging helpers (D-05, D-06) with mocked runs."""

    def test_log_config_noop_without_wandb(self) -> None:
        """_log_config_to_wandb should not crash when no WandbLogger is present."""
        _log_config_to_wandb(config_data={'key': 'value'}, loggers=())

    def test_log_results_noop_without_wandb(self) -> None:
        """_log_results_to_wandb should not crash when no WandbLogger is present."""
        _log_results_to_wandb(
            results_summary={'a': 1},
            timing={'train': 10.0},
            loggers=(),
        )

    def test_log_config_calls_experiment_config_update(self, monkeypatch: Any) -> None:
        """_log_config_to_wandb should call experiment.config.update()."""
        captured_config: dict[str, Any] = {}

        class MockConfig:
            def update(self, data: dict[str, Any]) -> None:
                captured_config.update(data)

        class MockRun:
            config = MockConfig()

        class MockWandbLogger:
            experiment = MockRun()

            def __init__(self) -> None:
                pass

        mock_instance = MockWandbLogger()

        monkeypatch.setattr(
            'src.rbspaper.pipeline.loggers._find_wandb_logger',
            lambda loggers=(): mock_instance,
        )

        _log_config_to_wandb(
            config_data={'model_name': 'ts2vec', 'seed': 42},
            loggers=(mock_instance,),
        )
        assert captured_config['model_name'] == 'ts2vec'
        assert captured_config['seed'] == 42

    def test_log_results_calls_run_log(self, monkeypatch: Any) -> None:
        """_log_results_to_wandb should call run.log() with flattened metrics."""
        logged_data: dict[str, Any] = {}

        class MockRun:
            def log(self, data: dict[str, Any]) -> None:
                logged_data.update(data)

            config = MagicMock()

        class MockWandbLogger:
            experiment = MockRun()

            def __init__(self) -> None:
                pass

        mock_instance = MockWandbLogger()

        monkeypatch.setattr(
            'src.rbspaper.pipeline.loggers._find_wandb_logger',
            lambda loggers=(): mock_instance,
        )

        # Patch the wandb import inside _log_results_to_wandb
        import sys as real_sys

        mock_wandb_module = MagicMock()
        # Make wandb.Table constructible
        mock_wandb_module.Table = MagicMock()

        original_modules = dict(real_sys.modules)
        real_sys.modules['wandb'] = mock_wandb_module

        try:
            _log_results_to_wandb(
                results_summary={'accuracy': 0.85},
                timing={'train': 10.0},
                loggers=(mock_instance,),
            )
            assert 'accuracy' in logged_data
            assert 'timing_train' in logged_data
        finally:
            # Restore original modules
            real_sys.modules.clear()
            real_sys.modules.update(original_modules)
