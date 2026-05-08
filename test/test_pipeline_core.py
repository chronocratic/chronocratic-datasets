"""Smoke tests for pipeline orchestration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import lightning.pytorch as pl
import numpy as np
import pytest
from tenacity import RetryError
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.rbspaper.attacks.config import (
    AttackExecutionContext,
    AttackExecutionMetadata,
    FgsmAttackParameters,
)
from src.rbspaper.attacks.enums import AttackBackend, AttackMethod
from src.rbspaper.enums.general import TimeSeriesDownstreamTask
from src.rbspaper.pipeline.config import (
    AttackRepresentationBundle,
    AttackRunConfig,
    AttackScopeConfig,
    AttackScopePolicy,
    DataConfig,
    DatasetTaskProfile,
    DownstreamTaskConfig,
    ExperimentPipelineConfig,
    PartitionRepresentations,
    PipelineArtifactConfig,
    QueryBudgetConfig,
    RepresentationAnalysisConfig,
    RepresentationEncodingConfig,
    TaskRepresentationBundle,
    TrainingConfig,
)
from src.rbspaper.pipeline.core import (
    _load_analysis_from_disk,
    _load_attacked_representations,
    _load_clean_representations,
    _load_metrics_from_disk,
    _resolve_existing_checkpoint,
    _write_experiment_config,
    retry_step,
    run_experiment_pipeline,
)
from src.rbspaper.pipeline.state import (
    _PipelineStateBuilder,
)

if TYPE_CHECKING:
    from pathlib import Path

CASE_COUNT = 4
EXPECTED_CLASSIFICATION_METRICS = 2
EXPECTED_SHARED_INPUT_METRICS = 2  # clean + attacked, per task


class _TinyLightningModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=4, out_features=2)
        self.model_name = 'TinyModel'

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = torch.mean(inputs, dim=1)
        return self.linear(pooled)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        inputs, labels = batch
        logits = self(inputs)
        return nn.functional.cross_entropy(logits, labels)

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        del batch_idx
        inputs, labels = batch
        logits = self(inputs)
        return nn.functional.cross_entropy(logits, labels)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=1e-2)


class _TinyDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()
        self._train = _build_dataset(size=20)
        self._valid = _build_dataset(size=12)
        self._test = _build_dataset(size=10)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self._train, batch_size=8, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self._valid, batch_size=6, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self._test, batch_size=5, shuffle=False)


def _build_dataset(*, size: int) -> TensorDataset:
    features = torch.randn(size=(size, 16, 4), dtype=torch.float32)
    labels = torch.randint(low=0, high=2, size=(size,), dtype=torch.long)
    return TensorDataset(features, labels)


def _fake_encode_data(**kwargs: object) -> torch.Tensor:
    data = kwargs['data']
    if not isinstance(data, torch.Tensor):
        message = 'Expected tensor inputs for encoding stub.'
        raise TypeError(message)
    return torch.mean(data, dim=1)


def _fake_execute_attack(**kwargs: object) -> tuple[torch.Tensor, AttackExecutionMetadata]:
    inputs = kwargs['inputs']
    if not isinstance(inputs, torch.Tensor):
        message = 'Expected tensor inputs for attack stub.'
        raise TypeError(message)
    return (
        inputs + 0.05,
        AttackExecutionMetadata(
            attack=AttackMethod.FGSM,
            backend=AttackBackend.TORCHATTACKS,
            success_rate=1.0,
            mean_l2=0.1,
            mean_linf=0.05,
        ),
    )


def _fake_evaluate(**kwargs: object) -> dict[str, float]:
    del kwargs
    return {'score': 0.5}


def _minimal_no_analysis() -> RepresentationAnalysisConfig:
    return RepresentationAnalysisConfig(
        enable_linear_separability=False,
        enable_geometry=False,
        enable_shift=False,
        enable_low_dim_artifacts=False,
    )


# ======== Existing tests (updated for DataConfig) ========


def test_downstream_task_config_cartesian_product() -> None:
    """Validate cartesian product expansion for downstream task configs."""
    task = DownstreamTaskConfig(
        task_name='classification',
        hyperparams={'evaluation_protocol': ['svm', 'ridge'], 'alpha': [0.1, 1.0]},
    )
    cases = task.hyperparam_cases()
    assert len(cases) == CASE_COUNT


def test_pipeline_smoke_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a minimal pipeline pass with mocked attacks and evaluation."""
    monkeypatch.setattr('src.rbspaper.pipeline.core.encode_data', _fake_encode_data)
    monkeypatch.setattr('src.rbspaper.pipeline.core.execute_attack', _fake_execute_attack)
    monkeypatch.setattr('src.rbspaper.pipeline.core.evaluate', _fake_evaluate)

    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(
            AttackRunConfig(
                name='fgsm_small',
                parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.05),
                context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
            ),
        ),
        downstream_tasks=(
            DownstreamTaskConfig(
                task_name='classification', hyperparams={'evaluation_protocol': ['svm']}
            ),
        ),
        analysis=RepresentationAnalysisConfig(
            enable_linear_separability=False,
            enable_geometry=True,
            enable_shift=True,
            enable_low_dim_artifacts=False,
        ),
        artifacts=PipelineArtifactConfig(
            output_dir=tmp_path, run_name='smoke', persist_artifacts=True
        ),
        seed=42,
    )

    results = run_experiment_pipeline(config=config)

    assert 'classification' in results.clean_representations
    assert 'classification' in results.attacked_representations
    assert len(results.downstream_metrics['classification']) == EXPECTED_CLASSIFICATION_METRICS
    assert (tmp_path / 'smoke' / 'results_summary.json').exists()
    assert (tmp_path / 'smoke' / 'experiment_config.json').exists()
    assert (tmp_path / 'smoke' / '.pipeline_state.json').exists()


def test_pipeline_rejects_query_budget_for_white_box_attack(tmp_path: Path) -> None:
    """Reject query budget when using white-box attacks."""
    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(
            AttackRunConfig(
                name='fgsm_query_budget_invalid',
                parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.05),
                context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
                query_budget=QueryBudgetConfig(max_queries=1000),
            ),
        ),
        downstream_tasks=(
            DownstreamTaskConfig(
                task_name='classification', hyperparams={'evaluation_protocol': ['svm']}
            ),
        ),
        analysis=RepresentationAnalysisConfig(enable_linear_separability=False),
        artifacts=PipelineArtifactConfig(output_dir=tmp_path, run_name='query_budget_invalid'),
        seed=42,
    )

    with pytest.raises(ValueError, match='query budget'):
        run_experiment_pipeline(config=config)


# ======== New tests ========


def test_attack_scope_config_rejects_missing_anchor() -> None:
    """AttackScopeConfig must raise when SHARED_INPUT is set without anchor_task."""
    with pytest.raises(ValueError, match='anchor_task is required'):
        AttackScopeConfig(scope=AttackScopePolicy.SHARED_INPUT)


def test_dataset_task_profile_rejects_primary_not_in_allowed() -> None:
    """DatasetTaskProfile must raise when primary_task is not in allowed_eval_tasks."""
    with pytest.raises(ValueError, match='must be included in allowed_eval_tasks'):
        DatasetTaskProfile(
            primary_task=TimeSeriesDownstreamTask.FORECASTING,
            allowed_eval_tasks=frozenset({TimeSeriesDownstreamTask.CLASSIFICATION}),
        )


def test_preflight_rejects_task_not_in_dataset_profile(tmp_path: Path) -> None:
    """Preflight must reject downstream tasks that fall outside the dataset profile."""
    profile = DatasetTaskProfile(
        primary_task=TimeSeriesDownstreamTask.CLASSIFICATION,
        allowed_eval_tasks=frozenset({TimeSeriesDownstreamTask.CLASSIFICATION}),
    )
    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule(), profile=profile),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(
            AttackRunConfig(
                name='fgsm',
                parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.05),
                context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
            ),
        ),
        downstream_tasks=(
            DownstreamTaskConfig(task_name='classification'),
            DownstreamTaskConfig(task_name='forecasting'),  # not in profile
        ),
        analysis=_minimal_no_analysis(),
        artifacts=PipelineArtifactConfig(output_dir=tmp_path, run_name='profile_reject'),
        seed=42,
    )

    with pytest.raises(ValueError, match='allowed_eval_tasks'):
        run_experiment_pipeline(config=config)


def test_preflight_rejects_wrong_anchor_in_shared_input(tmp_path: Path) -> None:
    """Preflight must reject attacks whose context.task differs from anchor_task in SHARED_INPUT."""
    scope = AttackScopeConfig(
        scope=AttackScopePolicy.SHARED_INPUT, anchor_task=TimeSeriesDownstreamTask.CLASSIFICATION
    )
    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(
            AttackRunConfig(
                name='fgsm_wrong_task',
                parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.05),
                context=AttackExecutionContext(
                    task=TimeSeriesDownstreamTask.FORECASTING  # mismatch
                ),
            ),
        ),
        downstream_tasks=(DownstreamTaskConfig(task_name='classification'),),
        analysis=_minimal_no_analysis(),
        artifacts=PipelineArtifactConfig(output_dir=tmp_path, run_name='anchor_mismatch'),
        attack_scope=scope,
        seed=42,
    )

    with pytest.raises(ValueError, match='anchor_task'):
        run_experiment_pipeline(config=config)


def test_pipeline_shared_input_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """In SHARED_INPUT mode, execute_attack runs once but results cover all downstream tasks."""
    attack_call_count = 0

    def _counting_attack(**kwargs: object) -> tuple[torch.Tensor, AttackExecutionMetadata]:
        nonlocal attack_call_count
        attack_call_count += 1
        return _fake_execute_attack(**kwargs)

    monkeypatch.setattr('src.rbspaper.pipeline.core.encode_data', _fake_encode_data)
    monkeypatch.setattr('src.rbspaper.pipeline.core.execute_attack', _counting_attack)
    monkeypatch.setattr('src.rbspaper.pipeline.core.evaluate', _fake_evaluate)

    scope = AttackScopeConfig(
        scope=AttackScopePolicy.SHARED_INPUT, anchor_task=TimeSeriesDownstreamTask.CLASSIFICATION
    )
    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(
            AttackRunConfig(
                name='fgsm_shared',
                parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.05),
                context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
            ),
        ),
        downstream_tasks=(
            DownstreamTaskConfig(
                task_name='classification', hyperparams={'evaluation_protocol': ['svm']}
            ),
            DownstreamTaskConfig(task_name='clustering'),
        ),
        analysis=_minimal_no_analysis(),
        artifacts=PipelineArtifactConfig(
            output_dir=tmp_path, run_name='shared_input', persist_artifacts=False
        ),
        attack_scope=scope,
        seed=42,
    )

    results = run_experiment_pipeline(config=config)

    # Attack must run exactly once despite two downstream tasks
    assert attack_call_count == 1

    # Attacked representations must appear for both tasks
    assert 'fgsm_shared' in results.attacked_representations['classification']
    assert 'fgsm_shared' in results.attacked_representations['clustering']

    # Each task must have clean + attacked metrics
    assert len(results.downstream_metrics['classification']) == EXPECTED_SHARED_INPUT_METRICS
    assert len(results.downstream_metrics['clustering']) == EXPECTED_SHARED_INPUT_METRICS


# ======== experiment_config.json tests ========


def test_write_experiment_config_writes_json_with_required_keys(
    tmp_path: Path,
) -> None:
    """_write_experiment_config writes experiment_config.json with required metadata keys."""
    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(
            trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}
        ),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(
            AttackRunConfig(
                name='fgsm_test',
                parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.1),
                context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
            ),
        ),
        downstream_tasks=(
            DownstreamTaskConfig(task_name='classification'),
            DownstreamTaskConfig(task_name='clustering'),
        ),
        analysis=_minimal_no_analysis(),
        artifacts=PipelineArtifactConfig(output_dir=tmp_path, run_name='config_test'),
        seed=123,
    )
    run_dir = tmp_path / 'config_test'
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_experiment_config(config=config, run_dir=run_dir)

    config_file = run_dir / 'experiment_config.json'
    assert config_file.exists()

    with config_file.open('r') as f:
        data = json.load(f)

    assert 'model_name' in data
    assert 'seed' in data
    assert 'downstream_tasks' in data
    assert 'attack_names' in data


def test_write_experiment_config_file_location(tmp_path: Path) -> None:
    """_write_experiment_config creates file at run_dir/experiment_config.json."""
    scope = AttackScopeConfig(
        scope=AttackScopePolicy.TASK_CONDITIONED,
    )
    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(
            trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}
        ),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(
            AttackRunConfig(
                name='fgsm_loc',
                parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.1),
                context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
            ),
        ),
        downstream_tasks=(DownstreamTaskConfig(task_name='classification'),),
        analysis=_minimal_no_analysis(),
        artifacts=PipelineArtifactConfig(output_dir=tmp_path, run_name='loc_test'),
        seed=99,
        attack_scope=scope,
    )
    run_dir = tmp_path / 'loc_test'
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_experiment_config(config=config, run_dir=run_dir)

    expected = run_dir / 'experiment_config.json'
    assert expected.exists()


def test_write_experiment_config_model_name_fallback(tmp_path: Path) -> None:
    """_write_experiment_config falls back to class name when model_name attribute is absent."""
    class ModelWithoutName(pl.LightningModule):
        """Minimal model with no model_name attribute."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return x

        def training_step(self, batch, batch_idx: int) -> torch.Tensor:
            del batch, batch_idx
            return torch.tensor(0.0)

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.SGD(self.parameters(), lr=0.01)

    scope = AttackScopeConfig(
        scope=AttackScopePolicy.TASK_CONDITIONED,
    )
    config = ExperimentPipelineConfig(
        model=ModelWithoutName(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(
            trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}
        ),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(
            AttackRunConfig(
                name='fgsm_fb',
                parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.1),
                context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
            ),
        ),
        downstream_tasks=(DownstreamTaskConfig(task_name='classification'),),
        analysis=_minimal_no_analysis(),
        artifacts=PipelineArtifactConfig(output_dir=tmp_path, run_name='fb_test'),
        seed=42,
        attack_scope=scope,
    )
    run_dir = tmp_path / 'fb_test'
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_experiment_config(config=config, run_dir=run_dir)

    with (run_dir / 'experiment_config.json').open('r') as f:
        data = json.load(f)

    assert data['model_name'] == 'ModelWithoutName'


# ======== Resume Gate Tests (Plan 03-08) ========


def test_pipeline_accepts_previous_state_none_runs_all_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_experiment_pipeline with previous_state=None runs all steps (normal mode)."""
    train_called = False
    encode_called = False
    eval_called = False

    def _counting_train(**kwargs: object) -> Path | None:  # type: ignore[return]
        nonlocal train_called
        train_called = True
        return None

    def _counting_encode(**kwargs: object) -> torch.Tensor:
        nonlocal encode_called
        encode_called = True
        return _fake_encode_data(**kwargs)

    def _counting_evaluate(**kwargs: object) -> dict[str, float]:
        nonlocal eval_called
        eval_called = True
        return _fake_evaluate(**kwargs)

    monkeypatch.setattr('src.rbspaper.pipeline.core.encode_data', _counting_encode)
    monkeypatch.setattr('src.rbspaper.pipeline.core.execute_attack', _fake_execute_attack)
    monkeypatch.setattr('src.rbspaper.pipeline.core.evaluate', _counting_evaluate)
    monkeypatch.setattr('src.rbspaper.pipeline.core._train_model', _counting_train)

    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(
            AttackRunConfig(
                name='fgsm_small',
                parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.05),
                context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
            ),
        ),
        downstream_tasks=(
            DownstreamTaskConfig(
                task_name='classification', hyperparams={'evaluation_protocol': ['svm']}
            ),
        ),
        analysis=_minimal_no_analysis(),
        artifacts=PipelineArtifactConfig(
            output_dir=tmp_path, run_name='resume_none', persist_artifacts=True
        ),
        seed=42,
    )

    results = run_experiment_pipeline(config=config, previous_state=None)

    assert results is not None
    assert 'classification' in results.clean_representations


def test_resolve_existing_checkpoint_returns_path_when_exists(tmp_path: Path) -> None:
    """_resolve_existing_checkpoint returns the canonical path if the checkpoint exists."""
    checkpoints_dir = tmp_path / 'checkpoints'
    checkpoints_dir.mkdir(parents=True)
    checkpoint_file = checkpoints_dir / 'best.ckpt'  # matches TrainingConfig default
    checkpoint_file.write_text('fake checkpoint')

    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(),
        downstream_tasks=(DownstreamTaskConfig(task_name='classification'),),
        analysis=_minimal_no_analysis(),
        artifacts=PipelineArtifactConfig(output_dir=tmp_path, run_name='ckpt_test'),
        seed=42,
    )

    result = _resolve_existing_checkpoint(run_dir=tmp_path, config=config)
    assert result == checkpoint_file


def test_resolve_existing_checkpoint_returns_none_when_missing(tmp_path: Path) -> None:
    """_resolve_existing_checkpoint returns None when the checkpoint file doesn't exist."""
    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(),
        downstream_tasks=(DownstreamTaskConfig(task_name='classification'),),
        analysis=_minimal_no_analysis(),
        artifacts=PipelineArtifactConfig(output_dir=tmp_path, run_name='no_ckpt'),
        seed=42,
    )

    result = _resolve_existing_checkpoint(run_dir=tmp_path, config=config)
    assert result is None


# ======== Disk Load Helper Tests (Task 2, Plan 03-08) ========


def test_load_clean_representations_round_trip(tmp_path: Path) -> None:
    """_load_clean_representations restores data persisted by _persist_clean_representations."""
    from src.rbspaper.pipeline.core import (
        _persist_clean_representations,
    )

    run_dir = tmp_path / 'round_trip'
    features = {
        'train': np.random.rand(10, 8).astype(np.float32),
        'valid': np.random.rand(6, 8).astype(np.float32),
        'test': np.random.rand(5, 8).astype(np.float32),
    }
    labels = {
        'train': np.random.randint(0, 2, size=10).astype(np.int64),
        'valid': np.random.randint(0, 2, size=6).astype(np.int64),
        'test': np.random.randint(0, 2, size=5).astype(np.int64),
    }
    clean_bundle = TaskRepresentationBundle(
        train=PartitionRepresentations(features=features['train'], labels=labels['train']),
        valid=PartitionRepresentations(features=features['valid'], labels=labels['valid']),
        test=PartitionRepresentations(features=features['test'], labels=labels['test']),
    )

    _persist_clean_representations(
        run_dir=run_dir, task_name='classification', clean_bundle=clean_bundle
    )

    loaded = _load_clean_representations(
        run_dir=run_dir, task_name='classification'
    )

    np.testing.assert_array_equal(loaded.train.features, features['train'])
    np.testing.assert_array_equal(loaded.train.labels, labels['train'])
    np.testing.assert_array_equal(loaded.test.features, features['test'])
    np.testing.assert_array_equal(loaded.test.labels, labels['test'])


def test_load_metrics_from_disk_round_trip(tmp_path: Path) -> None:
    """_load_metrics_from_disk returns list matching saved task_metrics."""
    run_dir = tmp_path / 'metrics_test'
    metrics = [
        {'score': 0.5, 'attack': None, 'task': 'classification'},
        {'score': 0.3, 'attack': 'fgsm', 'task': 'classification'},
    ]

    metrics_path = run_dir / 'metrics'
    metrics_path.mkdir(parents=True, exist_ok=True)
    with (metrics_path / 'classification.json').open('w') as f:
        json.dump(metrics, f)

    loaded = _load_metrics_from_disk(run_dir=run_dir, task_name='classification')
    assert loaded == metrics
    assert len(loaded) == 2
    assert loaded[0]['attack'] is None
    assert loaded[1]['attack'] == 'fgsm'


def test_load_attacked_representations_round_trip(tmp_path: Path) -> None:
    """_load_attacked_representations restores data persisted by _persist_attacked_representations."""
    from src.rbspaper.pipeline.core import (
        _persist_attacked_representations,
    )

    run_dir = tmp_path / 'attacked_test'
    attack_features = np.random.rand(5, 8).astype(np.float32)
    attack_labels = np.random.randint(0, 2, size=5).astype(np.int64)
    metadata = {'success_rate': 1.0, 'mean_l2': 0.1}

    bundle = AttackRepresentationBundle(
        test=PartitionRepresentations(features=attack_features, labels=attack_labels),
        metadata=metadata,
    )

    _persist_attacked_representations(
        run_dir=run_dir,
        task_name='classification',
        attack_name='fgsm_small',
        attacked_bundle=bundle,
    )

    # Need attack configs for the load function
    attacks = (
        AttackRunConfig(
            name='fgsm_small',
            parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.05),
            context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
        ),
    )
    loaded = _load_attacked_representations(
        run_dir=run_dir, task_name='classification', attacks=attacks
    )

    assert 'fgsm_small' in loaded
    np.testing.assert_array_equal(loaded['fgsm_small'].test.features, attack_features)
    np.testing.assert_array_equal(loaded['fgsm_small'].test.labels, attack_labels)
    assert loaded['fgsm_small'].metadata['success_rate'] == 1.0


def test_load_analysis_from_disk_round_trip(tmp_path: Path) -> None:
    """_load_analysis_from_disk restores persisted analysis results."""
    run_dir = tmp_path / 'analysis_test'
    analysis_data = {
        'classification': {
            'clean_geometry': {'mean_dist': 1.5},
            'attacked_geometry': {'fgsm': {'mean_dist': 2.0}},
        }
    }

    analysis_path = run_dir / 'analysis'
    analysis_path.mkdir(parents=True, exist_ok=True)
    with (analysis_path / 'analysis.json').open('w') as f:
        json.dump(analysis_data, f)

    loaded = _load_analysis_from_disk(run_dir=run_dir)
    assert loaded == analysis_data
    assert 'classification' in loaded


def test_load_clean_representations_missing_file_raises(tmp_path: Path) -> None:
    """_load_clean_representations raises FileNotFoundError when NPZ is missing."""
    run_dir = tmp_path / 'missing_clean'
    with pytest.raises(FileNotFoundError, match='Missing persisted clean rep'):
        _load_clean_representations(run_dir=run_dir, task_name='classification')


def test_load_metrics_from_disk_missing_file_raises(tmp_path: Path) -> None:
    """_load_metrics_from_disk raises FileNotFoundError when JSON is missing."""
    run_dir = tmp_path / 'missing_metrics'
    with pytest.raises(FileNotFoundError, match='Missing persisted metrics'):
        _load_metrics_from_disk(run_dir=run_dir, task_name='classification')


def test_load_analysis_from_disk_missing_file_raises(tmp_path: Path) -> None:
    """_load_analysis_from_disk raises FileNotFoundError when JSON is missing."""
    run_dir = tmp_path / 'missing_analysis'
    with pytest.raises(FileNotFoundError, match='Missing persisted analysis'):
        _load_analysis_from_disk(run_dir=run_dir)


# ======== Resume Gate Wiring Tests (Task 3, Plan 03-08) ========


def _build_persisted_run_dir(
    *,
    base_dir: Path,
    run_name: str,
    task_name: str = 'classification',
) -> Path:
    """Helper to build a run_dir with all persisted artifacts for resume testing."""
    run_dir = base_dir / run_name
    # Persist clean representations
    clean_dir = run_dir / 'representations' / task_name / 'clean'
    clean_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'valid', 'test'):
        np.savez(
            file=str(clean_dir / f'{split}.npz'),
            features=np.zeros((5, 8)),
            labels=np.zeros(5, dtype=np.int64),
        )
    # Persist attacked representations
    attack_dir = run_dir / 'representations' / task_name / 'attacks' / 'fgsm_small'
    attack_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        file=str(attack_dir / 'test.npz'),
        features=np.zeros((5, 8)),
        labels=np.zeros(5, dtype=np.int64),
    )
    # Persist metrics
    metrics_dir = run_dir / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / f'{task_name}.json').open('w') as f:
        json.dump([{'score': 0.5, 'attack': None, 'task': task_name}], f)
    # Persist analysis
    analysis_dir = run_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    with (analysis_dir / 'analysis.json').open('w') as f:
        json.dump({task_name: {}}, f)
    # Persist checkpoint
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / 'best.ckpt').write_text('fake')
    return run_dir


def test_pipeline_skips_train_when_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When state marks train complete, _train_model must not be called."""
    train_called = False

    def _no_op_train(**kwargs: object) -> Path | None:  # type: ignore[return]
        nonlocal train_called
        train_called = True
        return None

    builder = _PipelineStateBuilder(config_hash='abc12345')
    builder.mark_complete(step='train')
    completed_state = builder.build()

    monkeypatch.setattr('src.rbspaper.pipeline.core.encode_data', _fake_encode_data)
    monkeypatch.setattr('src.rbspaper.pipeline.core.execute_attack', _fake_execute_attack)
    monkeypatch.setattr('src.rbspaper.pipeline.core.evaluate', _fake_evaluate)
    monkeypatch.setattr('src.rbspaper.pipeline.core._train_model', _no_op_train)

    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(
            AttackRunConfig(
                name='fgsm_small',
                parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.05),
                context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
            ),
        ),
        downstream_tasks=(
            DownstreamTaskConfig(
                task_name='classification', hyperparams={'evaluation_protocol': ['svm']}
            ),
        ),
        analysis=_minimal_no_analysis(),
        artifacts=PipelineArtifactConfig(
            output_dir=tmp_path, run_name='skip_train', persist_artifacts=True
        ),
        seed=42,
    )

    run_experiment_pipeline(config=config, previous_state=completed_state)
    assert not train_called, '_train_model should be skipped when state marks it complete'


def test_pipeline_skips_encoding_and_loads_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When state marks encoding complete, encode_data is not called; bundle loaded from disk."""
    encode_called = False

    def _counting_encode(**kwargs: object) -> torch.Tensor:
        nonlocal encode_called
        encode_called = True
        return _fake_encode_data(**kwargs)

    # Build persisted artifacts
    _build_persisted_run_dir(base_dir=tmp_path, run_name='skip_enc')

    builder = _PipelineStateBuilder(config_hash='abc12345')
    builder.mark_complete(step='train')
    builder.mark_complete(step='encoding', task_name='classification')
    builder.mark_complete(step='attacks', task_name='classification')
    completed_state = builder.build()

    monkeypatch.setattr('src.rbspaper.pipeline.core.encode_data', _counting_encode)
    monkeypatch.setattr('src.rbspaper.pipeline.core.execute_attack', _fake_execute_attack)
    monkeypatch.setattr('src.rbspaper.pipeline.core.evaluate', _fake_evaluate)

    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(
            AttackRunConfig(
                name='fgsm_small',
                parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.05),
                context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
            ),
        ),
        downstream_tasks=(
            DownstreamTaskConfig(
                task_name='classification', hyperparams={'evaluation_protocol': ['svm']}
            ),
        ),
        analysis=_minimal_no_analysis(),
        artifacts=PipelineArtifactConfig(
            output_dir=tmp_path, run_name='skip_enc', persist_artifacts=True
        ),
        seed=42,
    )

    run_experiment_pipeline(config=config, previous_state=completed_state)
    assert not encode_called, 'encode_data should be skipped when encoding is marked complete'


def test_pipeline_skips_evaluate_and_loads_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When state marks evaluate complete, evaluate() is not called; metrics loaded from disk."""
    eval_called = False

    def _counting_evaluate(**kwargs: object) -> dict[str, float]:
        nonlocal eval_called
        eval_called = True
        return _fake_evaluate(**kwargs)

    # Build persisted artifacts
    _build_persisted_run_dir(base_dir=tmp_path, run_name='skip_eval')

    builder = _PipelineStateBuilder(config_hash='abc12345')
    builder.mark_complete(step='train')
    builder.mark_complete(step='encoding', task_name='classification')
    builder.mark_complete(step='attacks', task_name='classification')
    builder.mark_complete(step='evaluate', task_name='classification')
    completed_state = builder.build()

    monkeypatch.setattr('src.rbspaper.pipeline.core.encode_data', _fake_encode_data)
    monkeypatch.setattr('src.rbspaper.pipeline.core.execute_attack', _fake_execute_attack)
    monkeypatch.setattr('src.rbspaper.pipeline.core.evaluate', _counting_evaluate)

    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(
            AttackRunConfig(
                name='fgsm_small',
                parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.05),
                context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
            ),
        ),
        downstream_tasks=(
            DownstreamTaskConfig(
                task_name='classification', hyperparams={'evaluation_protocol': ['svm']}
            ),
        ),
        analysis=_minimal_no_analysis(),
        artifacts=PipelineArtifactConfig(
            output_dir=tmp_path, run_name='skip_eval', persist_artifacts=True
        ),
        seed=42,
    )

    results = run_experiment_pipeline(config=config, previous_state=completed_state)
    assert not eval_called, 'evaluate should be skipped when state marks it complete'
    # Metrics should be loaded from disk
    assert results.downstream_metrics['classification'] == [{'score': 0.5, 'attack': None, 'task': 'classification'}]


def test_experiment_config_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """experiment_config.json exists after a pipeline run."""
    monkeypatch.setattr('src.rbspaper.pipeline.core.encode_data', _fake_encode_data)
    monkeypatch.setattr('src.rbspaper.pipeline.core.execute_attack', _fake_execute_attack)
    monkeypatch.setattr('src.rbspaper.pipeline.core.evaluate', _fake_evaluate)

    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(
            AttackRunConfig(
                name='fgsm_small',
                parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.05),
                context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
            ),
        ),
        downstream_tasks=(
            DownstreamTaskConfig(
                task_name='classification', hyperparams={'evaluation_protocol': ['svm']}
            ),
        ),
        analysis=_minimal_no_analysis(),
        artifacts=PipelineArtifactConfig(
            output_dir=tmp_path, run_name='config_artifact', persist_artifacts=True
        ),
        seed=42,
    )

    run_experiment_pipeline(config=config)
    assert (tmp_path / 'config_artifact' / 'experiment_config.json').exists()


def test_state_file_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """.pipeline_state.json exists after a pipeline completes."""
    monkeypatch.setattr('src.rbspaper.pipeline.core.encode_data', _fake_encode_data)
    monkeypatch.setattr('src.rbspaper.pipeline.core.execute_attack', _fake_execute_attack)
    monkeypatch.setattr('src.rbspaper.pipeline.core.evaluate', _fake_evaluate)

    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(
            AttackRunConfig(
                name='fgsm_small',
                parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.05),
                context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
            ),
        ),
        downstream_tasks=(
            DownstreamTaskConfig(
                task_name='classification', hyperparams={'evaluation_protocol': ['svm']}
            ),
        ),
        analysis=_minimal_no_analysis(),
        artifacts=PipelineArtifactConfig(
            output_dir=tmp_path, run_name='state_artifact', persist_artifacts=True
        ),
        seed=42,
    )

    run_experiment_pipeline(config=config)
    assert (tmp_path / 'state_artifact' / '.pipeline_state.json').exists()


def test_force_true_runs_all_steps_despite_previous_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force=True ignores previous_state and runs all steps."""
    train_called = False

    def _counting_train(**kwargs: object) -> Path | None:  # type: ignore[return]
        nonlocal train_called
        train_called = True
        return None

    builder = _PipelineStateBuilder(config_hash='abc12345')
    builder.mark_complete(step='train')
    builder.mark_complete(step='encoding', task_name='classification')
    completed_state = builder.build()

    monkeypatch.setattr('src.rbspaper.pipeline.core.encode_data', _fake_encode_data)
    monkeypatch.setattr('src.rbspaper.pipeline.core.execute_attack', _fake_execute_attack)
    monkeypatch.setattr('src.rbspaper.pipeline.core.evaluate', _fake_evaluate)
    monkeypatch.setattr('src.rbspaper.pipeline.core._train_model', _counting_train)

    config = ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(trainer_kwargs={'max_epochs': 1, 'enable_checkpointing': False}),
        encoding=RepresentationEncodingConfig(batch_size=16, num_workers=0),
        attacks=(
            AttackRunConfig(
                name='fgsm_small',
                parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.05),
                context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
            ),
        ),
        downstream_tasks=(
            DownstreamTaskConfig(
                task_name='classification', hyperparams={'evaluation_protocol': ['svm']}
            ),
        ),
        analysis=_minimal_no_analysis(),
        artifacts=PipelineArtifactConfig(
            output_dir=tmp_path, run_name='force_run', persist_artifacts=True
        ),
        seed=42,
    )

    results = run_experiment_pipeline(
        config=config,
        previous_state=completed_state,
        force=True,
    )
    assert train_called, 'force=True should run train even when state marks it complete'
    assert results is not None


# ======== Retry Step Tests (Plan 03-09) ========


def test_retry_step_retries_runtime_error_three_times() -> None:
    """retry_step calls a function 3 times on RuntimeError, then raises RetryError."""
    call_count = 0

    def _always_fails() -> None:
        nonlocal call_count
        call_count += 1
        msg = 'transient failure'
        raise RuntimeError(msg)

    wrapped = retry_step(_always_fails)
    with pytest.raises(RetryError) as exc_info:
        wrapped()

    # Verify the underlying exception is the RuntimeError
    assert isinstance(exc_info.value.last_attempt.exception(), RuntimeError)
    assert call_count == 3, 'Function should be called exactly 3 times (initial + 2 retries)'


def test_retry_step_succeeds_on_second_attempt() -> None:
    """retry_step passes when the function succeeds within the retry limit."""
    call_count = 0

    def _fails_once() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            msg = 'transient failure'
            raise RuntimeError(msg)
        return 'success'

    wrapped = retry_step(_fails_once)
    result = wrapped()

    assert result == 'success'
    assert call_count == 2, 'Function should succeed on the second call'


def test_retry_step_does_not_retry_value_error() -> None:
    """retry_step must NOT retry ValueError (logic errors propagate immediately)."""
    call_count = 0

    def _logic_error() -> None:
        nonlocal call_count
        call_count += 1
        msg = 'invalid configuration'
        raise ValueError(msg)

    wrapped = retry_step(_logic_error)
    with pytest.raises(ValueError, match='invalid configuration'):
        wrapped()

    assert call_count == 1, 'ValueError should not be retried'
