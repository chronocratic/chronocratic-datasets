"""Tests for pipeline state management (PipelineState, Builder, serialization)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import types

# -------- PipelineState tests --------


def test_pipeline_state_creates_instance() -> None:
    """PipelineState creates instance with required fields."""
    from src.rbspaper.pipeline.state import PipelineState

    state = PipelineState(
        completed={'train': []},
        config_hash='abc12345',
        started_at='2026-01-01T00:00:00+00:00',
        last_updated='2026-01-01T00:00:00+00:00',
    )
    assert state.completed == {'train': []}
    assert state.config_hash == 'abc12345'
    assert state.started_at == '2026-01-01T00:00:00+00:00'
    assert state.last_updated == '2026-01-01T00:00:00+00:00'


def test_pipeline_state_is_step_complete_global() -> None:
    """is_step_complete returns True for marked global steps."""
    from src.rbspaper.pipeline.state import PipelineState

    state = PipelineState(
        completed={'train': []},
        config_hash='abc',
        started_at='2026-01-01T00:00:00+00:00',
        last_updated='2026-01-01T00:00:00+00:00',
    )
    assert state.is_step_complete(step='train') is True


def test_pipeline_state_is_step_complete_per_task() -> None:
    """is_step_complete checks per-task granularity correctly."""
    from src.rbspaper.pipeline.state import PipelineState

    state = PipelineState(
        completed={'encoding': ['classification']},
        config_hash='abc',
        started_at='2026-01-01T00:00:00+00:00',
        last_updated='2026-01-01T00:00:00+00:00',
    )
    assert state.is_step_complete(step='encoding', task_name='classification') is True
    assert state.is_step_complete(step='encoding', task_name='forecasting') is False


def test_pipeline_state_is_step_complete_missing() -> None:
    """is_step_complete returns False for steps not in completed dict."""
    from src.rbspaper.pipeline.state import PipelineState

    state = PipelineState(
        completed={'train': []},
        config_hash='abc',
        started_at='2026-01-01T00:00:00+00:00',
        last_updated='2026-01-01T00:00:00+00:00',
    )
    assert state.is_step_complete(step='missing') is False


def test_pipeline_state_is_frozen() -> None:
    """PipelineState is frozen and raises on field assignment."""
    from src.rbspaper.pipeline.state import PipelineState

    state = PipelineState(
        completed={},
        config_hash='abc',
        started_at='2026-01-01T00:00:00+00:00',
        last_updated='2026-01-01T00:00:00+00:00',
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        state.completed = {'train': []}  # type: ignore[misc] # ty: ignore[invalid-assignment]


# -------- _PipelineStateBuilder tests --------


def test_builder_creates_with_config_hash() -> None:
    """Builder creates with config_hash and empty completed."""
    from src.rbspaper.pipeline.state import _PipelineStateBuilder

    builder = _PipelineStateBuilder(config_hash='abc')
    assert builder._config_hash == 'abc'
    assert builder._completed == {}


def test_builder_mark_complete_global() -> None:
    """mark_complete adds global step marker (empty list)."""
    from src.rbspaper.pipeline.state import _PipelineStateBuilder

    builder = _PipelineStateBuilder(config_hash='abc')
    builder.mark_complete(step='train')
    assert builder._completed == {'train': []}


def test_builder_mark_complete_per_task() -> None:
    """mark_complete adds per-task granularity marker."""
    from src.rbspaper.pipeline.state import _PipelineStateBuilder

    builder = _PipelineStateBuilder(config_hash='abc')
    builder.mark_complete(step='encoding', task_name='classification')
    assert builder._completed == {'encoding': ['classification']}


def test_builder_mark_complete_no_duplicate() -> None:
    """Second mark_complete for same step+task does not duplicate."""
    from src.rbspaper.pipeline.state import _PipelineStateBuilder

    builder = _PipelineStateBuilder(config_hash='abc')
    builder.mark_complete(step='encoding', task_name='classification')
    builder.mark_complete(step='encoding', task_name='classification')
    assert builder._completed == {'encoding': ['classification']}


def test_builder_build_returns_frozen_state() -> None:
    """build() returns a frozen PipelineState with populated fields."""
    from src.rbspaper.pipeline.state import _PipelineStateBuilder

    builder = _PipelineStateBuilder(config_hash='abc')
    builder.mark_complete(step='train')
    state = builder.build()

    assert state.completed == {'train': []}
    assert state.config_hash == 'abc'
    assert state.started_at  # ISO timestamp
    assert state.last_updated  # ISO timestamp


# -------- Serialization tests --------


def test_to_dict_returns_all_fields() -> None:
    """to_dict returns dict with all PipelineState fields."""
    from src.rbspaper.pipeline.state import PipelineState, to_dict

    state = PipelineState(
        completed={'train': []},
        config_hash='abc12345',
        started_at='2026-01-01T00:00:00+00:00',
        last_updated='2026-01-01T00:00:00+00:00',
    )
    result = to_dict(state=state)
    assert result == {
        'completed': {'train': []},
        'config_hash': 'abc12345',
        'started_at': '2026-01-01T00:00:00+00:00',
        'last_updated': '2026-01-01T00:00:00+00:00',
    }


def test_from_dict_round_trip() -> None:
    """from_dict restores PipelineState from to_dict output."""
    from src.rbspaper.pipeline.state import PipelineState, from_dict, to_dict

    original = PipelineState(
        completed={'train': [], 'encoding': ['classification']},
        config_hash='deadbeef',
        started_at='2026-01-01T00:00:00+00:00',
        last_updated='2026-01-01T00:00:00+00:00',
    )
    restored = from_dict(data=to_dict(state=original))
    assert restored.completed == original.completed
    assert restored.config_hash == original.config_hash
    assert restored.started_at == original.started_at
    assert restored.last_updated == original.last_updated


def test_from_dict_raises_key_error() -> None:
    """from_dict raises KeyError for missing required field."""
    from src.rbspaper.pipeline.state import from_dict

    with pytest.raises(KeyError):
        from_dict(data={'config_hash': 'abc'})


# -------- _atomic_write_json tests --------


def test_atomic_write_json_writes_file(tmp_path: Path) -> None:
    """_atomic_write_json creates the target file with valid JSON."""
    from src.rbspaper.pipeline.state import _atomic_write_json

    target = tmp_path / 'test.json'
    _atomic_write_json(path=target, data={'key': 'value'})
    assert target.exists()


def test_atomic_write_json_no_tmp_after(tmp_path: Path) -> None:
    """No .tmp file remains after successful _atomic_write_json."""
    from src.rbspaper.pipeline.state import _atomic_write_json

    target = tmp_path / 'test.json'
    _atomic_write_json(path=target, data={'key': 'value'})
    tmp_file = tmp_path / 'test.json.tmp'
    assert not tmp_file.exists()


def test_atomic_write_json_round_trip(tmp_path: Path) -> None:
    """File content matches serialized data via json.load round-trip."""
    import json

    from src.rbspaper.pipeline.state import _atomic_write_json

    target = tmp_path / 'test.json'
    data = {'completed': {'train': []}, 'config_hash': 'abc12345'}
    _atomic_write_json(path=target, data=data)  # ty: ignore[invalid-argument-type]
    with target.open(mode='r') as fh:
        loaded = json.load(fh)
    assert loaded == data


def test_atomic_write_json_creates_parent_dir(tmp_path: Path) -> None:
    """_atomic_write_json creates parent directories if missing."""
    from src.rbspaper.pipeline.state import _atomic_write_json

    target = tmp_path / 'deep' / 'nested' / 'dir' / 'state.json'
    _atomic_write_json(path=target, data={'key': 'value'})
    assert target.exists()
    assert (tmp_path / 'deep' / 'nested' / 'dir').is_dir()


def test_state_filename_constant() -> None:
    """STATE_FILENAME constant is defined."""
    from src.rbspaper.pipeline.state import STATE_FILENAME

    assert STATE_FILENAME == '.pipeline_state.json'


# -------- save/load pipeline state tests --------


def test_save_load_round_trip(tmp_path: Path) -> None:
    """save_pipeline_state + load_pipeline_state produces equivalent state."""
    from src.rbspaper.pipeline.state import PipelineState, load_pipeline_state, save_pipeline_state

    original = PipelineState(
        completed={'train': [], 'encoding': ['classification']},
        config_hash='deadbeef',
        started_at='2026-01-01T00:00:00+00:00',
        last_updated='2026-01-01T00:00:00+00:00',
    )
    state_path = tmp_path / '.pipeline_state.json'
    save_pipeline_state(state=original, path=state_path)
    restored = load_pipeline_state(path=state_path)

    assert restored.completed == original.completed
    assert restored.config_hash == original.config_hash
    assert restored.started_at == original.started_at
    assert restored.last_updated == original.last_updated


def test_load_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    """load_pipeline_state raises FileNotFoundError for missing file."""
    from src.rbspaper.pipeline.state import load_pipeline_state

    missing = tmp_path / 'nonexistent.json'
    with pytest.raises(FileNotFoundError):
        load_pipeline_state(path=missing)


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    """save_pipeline_state creates parent directories if they don't exist."""
    from src.rbspaper.pipeline.state import PipelineState, save_pipeline_state

    state = PipelineState(
        completed={},
        config_hash='abc',
        started_at='2026-01-01T00:00:00+00:00',
        last_updated='2026-01-01T00:00:00+00:00',
    )
    state_path = tmp_path / 'deep' / 'nested' / '.pipeline_state.json'
    save_pipeline_state(state=state, path=state_path)
    assert state_path.exists()


# -------- compute_config_hash tests --------


def test_compute_config_hash_deterministic() -> None:
    """compute_config_hash returns the same hash for identical inputs."""
    from src.rbspaper.pipeline.state import compute_config_hash

    hash1 = compute_config_hash(model_params={'a': 1}, seed=42)
    hash2 = compute_config_hash(model_params={'a': 1}, seed=42)
    assert hash1 == hash2


def test_compute_config_hash_different_params() -> None:
    """compute_config_hash returns a different hash for different params."""
    from src.rbspaper.pipeline.state import compute_config_hash

    hash1 = compute_config_hash(model_params={'a': 1}, seed=42)
    hash2 = compute_config_hash(model_params={'a': 2}, seed=42)
    assert hash1 != hash2


def test_compute_config_hash_different_seed() -> None:
    """compute_config_hash returns a different hash for a different seed."""
    from src.rbspaper.pipeline.state import compute_config_hash

    hash1 = compute_config_hash(model_params={'a': 1}, seed=42)
    hash2 = compute_config_hash(model_params={'a': 1}, seed=99)
    assert hash1 != hash2


def test_compute_config_hash_length() -> None:
    """compute_config_hash returns exactly 8 characters."""
    from src.rbspaper.pipeline.state import compute_config_hash

    result = compute_config_hash(model_params={'a': 1}, seed=42)
    assert len(result) == 8


def test_compute_config_hash_hex_only() -> None:
    """compute_config_hash returns only lowercase hex characters."""
    import re

    from src.rbspaper.pipeline.state import compute_config_hash

    result = compute_config_hash(model_params={'a': 1}, seed=42)
    assert re.fullmatch(r'[0-9a-f]{8}', result) is not None


# -------- build_hierarchical_run_name tests --------


def test_build_hierarchical_run_name() -> None:
    """build_hierarchical_run_name returns correct path format."""
    from src.rbspaper.pipeline.config import build_hierarchical_run_name

    name = build_hierarchical_run_name(
        experiment_id='ts2vec', short_hash='a1b2c3d4', seed=42, dataset_name='Coffee'
    )
    assert name == 'ts2vec/a1b2c3d4/seed_42/Coffee'


# -------- CLI argument tests --------


def _load_runner_module() -> types.ModuleType:
    """Load runners/py/runner.py via importlib (no __init__.py in runners/py/)."""
    import importlib.util

    runner_path = Path(__file__).resolve().parent.parent / 'runners' / 'py' / 'runner.py'
    spec = importlib.util.spec_from_file_location('runner_module', runner_path)
    if spec is None or spec.loader is None:
        msg = 'Failed to create spec for runner.py'
        raise RuntimeError(msg)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dataset_index_parsing() -> None:
    """_parse_args accepts --dataset_index without --dataset_name."""
    runner = _load_runner_module()
    _parse_args = runner._parse_args

    args = _parse_args(
        ['--experiment_id', 'ts2vec', '--dataset_index', '0', '--data_root', '/tmp/data']
    )
    assert args.dataset_index == 0
    assert args.dataset_name is None


def test_mutually_exclusive_args() -> None:
    """_resolve_dataset rejects both --dataset_index and --dataset_name."""
    import argparse

    runner = _load_runner_module()
    _resolve_dataset = runner._resolve_dataset

    # Build namespace with both dataset args set (argparse allows this)
    args = argparse.Namespace(dataset_name='Coffee', dataset_index=0)
    with pytest.raises(SystemExit):
        _resolve_dataset(args=args)


def test_logging_creates_file(tmp_path: Path) -> None:
    """setup_logging creates the log directory and pipeline.log file handler."""
    import logging

    runner = _load_runner_module()
    setup_logging = runner.setup_logging

    # Clear existing handlers so setup_logging doesn't short-circuit
    root = logging.getLogger()
    original_handlers = root.handlers.copy()
    root.handlers.clear()

    try:
        log_dir = tmp_path / 'logs'
        setup_logging(log_dir=log_dir)
    finally:
        # Restore original handlers to avoid side effects
        root.handlers.clear()
        for handler in original_handlers:
            root.addHandler(handler)

    # Verify log_dir was created and a FileHandler was added
    assert log_dir.exists()
    # The root logger should now have a FileHandler pointing to pipeline.log
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) >= 1
