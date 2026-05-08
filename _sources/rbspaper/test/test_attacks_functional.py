"""Tests for function-based attack wrappers."""

from __future__ import annotations

import lightning as L
import torch
from torch import nn

from src.rbspaper.attacks.config import AttackExecutionContext
import src.rbspaper.attacks.functional as functional_attacks
from src.rbspaper.enums.general import TimeSeriesDownstreamTask


class _SimplePredictionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=4, out_features=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = torch.mean(input=inputs, dim=1)
        return self.linear(pooled)


class _LightningPredictionModel(L.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=4, out_features=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = torch.mean(input=inputs, dim=1)
        return self.linear(pooled)


def test_fgsm_wrapper_calls_backend(monkeypatch) -> None:
    inputs = torch.zeros(size=(4, 8, 4), dtype=torch.float32)
    model = _SimplePredictionModel()

    def _fake_run_attack_backend(**kwargs):
        assert kwargs['task'] == TimeSeriesDownstreamTask.CLASSIFICATION
        assert kwargs['supervision'] is not None
        assert kwargs['supervision'].dtype == torch.long
        del kwargs
        return inputs + 0.25

    monkeypatch.setattr(functional_attacks, 'run_attack_backend', _fake_run_attack_backend)

    adversarial, metadata = functional_attacks.fgsm_attack(
        inputs=inputs,
        supervision=None,
        model=model,
        context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
        return_metadata=True,
    )

    assert torch.allclose(adversarial, inputs + 0.25)
    assert metadata.mean_l2 > 0.0
    assert metadata.mean_linf > 0.0


def test_pgd_accepts_lightning_module(monkeypatch) -> None:
    inputs = torch.zeros(size=(5, 8, 4), dtype=torch.float32)
    supervision = torch.zeros(size=(5,), dtype=torch.long)
    model = _LightningPredictionModel()

    def _fake_run_attack_backend(**kwargs):
        del kwargs
        return inputs + 0.1

    monkeypatch.setattr(functional_attacks, 'run_attack_backend', _fake_run_attack_backend)

    adversarial = functional_attacks.pgd_attack(
        inputs=inputs,
        supervision=supervision,
        model=model,
        context=AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION),
        epsilon=0.1,
        alpha=0.02,
        steps=5,
    )

    assert torch.allclose(adversarial, inputs + 0.1)  # ty: ignore[invalid-argument-type]
