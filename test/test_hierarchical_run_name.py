"""Tests for build_hierarchical_run_name function."""

from __future__ import annotations

import pytest


def test_build_hierarchical_run_name_returns_correct_format() -> None:
    """build_hierarchical_run_name produces experiment_id/short_hash/seed_N/dataset_name."""
    from src.rbspaper.pipeline.config import build_hierarchical_run_name

    result = build_hierarchical_run_name(
        experiment_id='ts2vec', short_hash='a1b2c3d4', seed=42, dataset_name='Coffee'
    )
    assert result == 'ts2vec/a1b2c3d4/seed_42/Coffee'


def test_build_hierarchical_run_name_different_seeds() -> None:
    """Different seeds produce different path segments."""
    from src.rbspaper.pipeline.config import build_hierarchical_run_name

    result_42 = build_hierarchical_run_name(
        experiment_id='ts2vec', short_hash='a1b2c3d4', seed=42, dataset_name='Coffee'
    )
    result_99 = build_hierarchical_run_name(
        experiment_id='ts2vec', short_hash='a1b2c3d4', seed=99, dataset_name='Coffee'
    )
    assert result_42 != result_99
    assert 'seed_42' in result_42
    assert 'seed_99' in result_99


def test_build_hierarchical_run_name_uses_forward_slashes() -> None:
    """Function uses forward slashes (Path-compatible segments)."""
    from src.rbspaper.pipeline.config import build_hierarchical_run_name

    result = build_hierarchical_run_name(
        experiment_id='exp', short_hash='hash', seed=1, dataset_name='data'
    )
    assert '\\' not in result
    segments = result.split('/')
    assert len(segments) == 4


def test_build_hierarchical_run_name_different_datasets() -> None:
    """Different datasets produce different path segments."""
    from src.rbspaper.pipeline.config import build_hierarchical_run_name

    result_a = build_hierarchical_run_name(
        experiment_id='ts2vec', short_hash='a1b2c3d4', seed=42, dataset_name='Coffee'
    )
    result_b = build_hierarchical_run_name(
        experiment_id='ts2vec', short_hash='a1b2c3d4', seed=42, dataset_name='ECG'
    )
    assert result_a != result_b
    assert result_a.endswith('/Coffee')
    assert result_b.endswith('/ECG')
