"""Tests for preflight compatibility filtering."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import lightning.pytorch as pl

from src.rbspaper.attacks.config import (
    AttackExecutionContext,
    FgsmAttackParameters,
    SpsaAttackParameters,
)
from src.rbspaper.attacks.enums import AttackBackend
from src.rbspaper.enums.general import TimeSeriesDownstreamTask
from src.rbspaper.pipeline.config import (
    AttackRunConfig,
    DataConfig,
    DownstreamTaskConfig,
    ExperimentPipelineConfig,
    PipelineArtifactConfig,
    RepresentationAnalysisConfig,
    RepresentationEncodingConfig,
    TrainingConfig,
)
from src.rbspaper.pipeline.core import _preflight_pipeline_config


class _TinyLightningModel(pl.LightningModule):
    """Minimal model for preflight testing."""

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

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=1e-2)


class _TinyDataModule(pl.LightningDataModule):
    """Minimal datamodule for preflight testing."""

    def __init__(self) -> None:
        super().__init__()
        features = torch.randn(size=(20, 16, 4), dtype=torch.float32)
        labels = torch.randint(low=0, high=2, size=(20,), dtype=torch.long)
        self._train = TensorDataset(features, labels)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self._train, batch_size=8, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self._train, batch_size=8)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self._train, batch_size=8)


def _build_preflight_config(
    *,
    attack_config: AttackRunConfig,
    tmp_path: Path,
    downstream_tasks: tuple[DownstreamTaskConfig, ...] | None = None,
) -> ExperimentPipelineConfig:
    """Build a minimal ExperimentPipelineConfig for preflight testing."""
    if downstream_tasks is None:
        downstream_tasks = (
            DownstreamTaskConfig(
                task_name='classification', hyperparams={'evaluation_protocol': ['svm']}
            ),
        )
    return ExperimentPipelineConfig(
        model=_TinyLightningModel(),
        data=DataConfig(data_module=_TinyDataModule()),  # ty: ignore[invalid-argument-type]
        training=TrainingConfig(trainer_kwargs={'max_epochs': 1}),
        encoding=RepresentationEncodingConfig(batch_size=16),
        attacks=(attack_config,),
        downstream_tasks=downstream_tasks,
        analysis=RepresentationAnalysisConfig(enable_geometry=False, enable_shift=False),
        artifacts=PipelineArtifactConfig(output_dir=tmp_path, run_name='preflight_test'),
        seed=42,
    )


def test_preflight_raises_on_unsupported_attack(tmp_path: Path) -> None:
    """When an attack is not supported for the task, preflight raises ValueError."""
    # SPSA + FORECASTING is not in SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK
    unsupported_attack = AttackRunConfig(
        name='spsa_forecast',
        parameters=SpsaAttackParameters(
            backend=AttackBackend.TORCHATTACKS, epsilon=0.1, nb_iter=10
        ),
        context=AttackExecutionContext(task=TimeSeriesDownstreamTask.FORECASTING),
    )

    config = _build_preflight_config(
        attack_config=unsupported_attack,
        tmp_path=tmp_path,
        downstream_tasks=(
            DownstreamTaskConfig(
                task_name='forecasting', hyperparams={'evaluation_protocol': ['ridge']}
            ),
        ),
    )

    try:
        _preflight_pipeline_config(config=config)
        msg = 'Expected ValueError for unsupported attack'
        raise AssertionError(msg)
    except ValueError:
        pass  # Expected


def test_preflight_passes_on_supported_attack(caplog, tmp_path: Path) -> None:
    """Supported attacks should pass preflight without warnings."""
    supported_attack = AttackRunConfig(
        name='fgsm_class',
        parameters=FgsmAttackParameters(backend=AttackBackend.TORCHATTACKS, epsilon=0.1),
        context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
    )

    config = _build_preflight_config(attack_config=supported_attack, tmp_path=tmp_path)

    with caplog.at_level(logging.WARNING):
        _preflight_pipeline_config(config=config)

    # No skipping warnings should appear
    skip_warnings = [r for r in caplog.records if 'Skipping attack' in r.message]
    assert len(skip_warnings) == 0, f'Unexpected warnings: {[w.message for w in skip_warnings]}'
