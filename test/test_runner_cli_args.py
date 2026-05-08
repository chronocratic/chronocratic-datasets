"""Tests for runner CLI argument parsing: --dataset_index, --force, HPC detection."""

from __future__ import annotations

from typing import Any

import pytest

from runners.py.runner import _parse_args, _resolve_tracking_mode, main


class TestDatasetIndexArg:
    """Verify --dataset_index is accepted by the parser."""

    def test_dataset_index_accepted(self) -> None:
        """--dataset_index 0 should parse without error."""
        args = _parse_args(
            ['--dataset_index', '0', '--experiment_id', 'ts2vec', '--data_root', '/fake/data']
        )
        assert args.dataset_index == 0

    def test_dataset_index_default_none(self) -> None:
        """--dataset_index should default to None when absent."""
        args = _parse_args(
            ['--dataset_name', 'Coffee', '--experiment_id', 'ts2vec', '--data_root', '/fake/data']
        )
        assert args.dataset_index is None


class TestForceArg:
    """Verify --force flag is recognized by the parser."""

    def test_force_false_by_default(self) -> None:
        """--force should be False when absent."""
        args = _parse_args(
            ['--dataset_name', 'Coffee', '--experiment_id', 'ts2vec', '--data_root', '/fake/data']
        )
        assert args.force is False

    def test_force_true_when_set(self) -> None:
        """--force should be True when present."""
        args = _parse_args(
            [
                '--dataset_name',
                'Coffee',
                '--experiment_id',
                'ts2vec',
                '--data_root',
                '/fake/data',
                '--force',
            ]
        )
        assert args.force is True


class TestMainValidation:
    """Verify main() enforces mutual exclusivity and dataset resolution."""

    def test_mutually_exclusive_args_raise(self) -> None:
        """--dataset_index and --dataset_name together must raise SystemExit."""
        with pytest.raises(SystemExit):
            main(
                [
                    '--dataset_index',
                    '0',
                    '--dataset_name',
                    'Coffee',
                    '--experiment_id',
                    'ts2vec',
                    '--data_root',
                    '/fake/data',
                ]
            )

    def test_neither_dataset_arg_raises(self) -> None:
        """Neither --dataset_name nor --dataset_index must raise SystemExit."""
        with pytest.raises(SystemExit):
            main(['--experiment_id', 'ts2vec', '--data_root', '/fake/data'])

    def test_out_of_range_index_raises(self) -> None:
        """dataset_index that exceeds the registry must raise SystemExit."""
        with pytest.raises(SystemExit):
            main(
                [
                    '--dataset_index',
                    '99999',
                    '--experiment_id',
                    'ts2vec',
                    '--data_root',
                    '/fake/data',
                ]
            )

    def test_negative_index_raises(self) -> None:
        """Negative dataset_index must raise SystemExit."""
        with pytest.raises(SystemExit):
            main(
                ['--dataset_index', '-1', '--experiment_id', 'ts2vec', '--data_root', '/fake/data']
            )


class TestTrackingModeArg:
    """Verify --tracking_mode is accepted by the parser."""

    def test_tracking_mode_default_none(self) -> None:
        """--tracking_mode should default to None when absent."""
        args = _parse_args(
            ['--dataset_name', 'Coffee', '--experiment_id', 'ts2vec', '--data_root', '/fake/data']
        )
        assert args.tracking_mode is None

    def test_tracking_mode_online(self) -> None:
        """--tracking_mode online should parse without error."""
        args = _parse_args(
            [
                '--dataset_name',
                'Coffee',
                '--experiment_id',
                'ts2vec',
                '--data_root',
                '/fake/data',
                '--tracking_mode',
                'online',
            ]
        )
        assert args.tracking_mode == 'online'

    def test_tracking_mode_offline(self) -> None:
        """--tracking_mode offline should parse without error."""
        args = _parse_args(
            [
                '--dataset_name',
                'Coffee',
                '--experiment_id',
                'ts2vec',
                '--data_root',
                '/fake/data',
                '--tracking_mode',
                'offline',
            ]
        )
        assert args.tracking_mode == 'offline'

    def test_tracking_mode_disabled(self) -> None:
        """--tracking_mode disabled should parse without error."""
        args = _parse_args(
            [
                '--dataset_name',
                'Coffee',
                '--experiment_id',
                'ts2vec',
                '--data_root',
                '/fake/data',
                '--tracking_mode',
                'disabled',
            ]
        )
        assert args.tracking_mode == 'disabled'

    def test_tracking_mode_invalid_raises(self) -> None:
        """--tracking_mode with invalid value must raise SystemExit."""
        with pytest.raises(SystemExit):
            _parse_args(
                [
                    '--dataset_name',
                    'Coffee',
                    '--experiment_id',
                    'ts2vec',
                    '--data_root',
                    '/fake/data',
                    '--tracking_mode',
                    'invalid',
                ]
            )


class TestHpcAutoDetection:
    """Verify _resolve_tracking_mode auto-detects HPC via SLURM_JOB_ID (D-03)."""

    def test_cli_mode_overrides_auto_detection(self, monkeypatch: Any) -> None:
        """Explicit --tracking_mode should bypass auto-detection."""
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        result = _resolve_tracking_mode(cli_mode='disabled')
        assert result == 'disabled'

    def test_hpc_detection_defaults_to_offline(self, monkeypatch: Any) -> None:
        """When SLURM_JOB_ID is set and --tracking_mode is None, mode should be offline."""
        monkeypatch.setenv('SLURM_JOB_ID', '12345')
        result = _resolve_tracking_mode(cli_mode=None)
        assert result == 'offline'

    def test_local_detection_defaults_to_online(self, monkeypatch: Any) -> None:
        """When SLURM_JOB_ID is absent and --tracking_mode is None, mode should be online."""
        monkeypatch.delenv('SLURM_JOB_ID', raising=False)
        result = _resolve_tracking_mode(cli_mode=None)
        assert result == 'online'

    def test_hpc_detection_uses_env_var_value(self, monkeypatch: Any) -> None:
        """Any non-empty SLURM_JOB_ID should trigger offline mode."""
        monkeypatch.setenv('SLURM_JOB_ID', '1')
        result = _resolve_tracking_mode(cli_mode=None)
        assert result == 'offline'
