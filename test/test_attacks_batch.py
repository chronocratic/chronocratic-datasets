"""Tests for batched attack helper."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.rbspaper.attacks.batch import attack_dataset, batched_attack
from src.rbspaper.attacks.config import AttackExecutionMetadata
from src.rbspaper.attacks.enums import AttackBackend, AttackMethod


def test_batched_attack_without_metadata() -> None:
    inputs = torch.zeros(size=(8, 16, 1), dtype=torch.float32)
    supervision = torch.zeros(size=(8,), dtype=torch.long)

    def _attack_fn(
        *, inputs: torch.Tensor, supervision: torch.Tensor, return_metadata: bool = False
    ) -> torch.Tensor:
        del supervision, return_metadata
        return inputs + 1.0

    adversarial = batched_attack(
        attack_fn=_attack_fn,
        inputs=inputs,
        supervision=supervision,
        batch_size=3,
        return_metadata=False,
    )
    assert torch.allclose(adversarial, inputs + 1.0)  # ty: ignore[invalid-argument-type]


def test_batched_attack_with_metadata() -> None:
    inputs = torch.zeros(size=(6, 10, 1), dtype=torch.float32)
    supervision = torch.zeros(size=(6,), dtype=torch.long)

    def _attack_fn(
        *, inputs: torch.Tensor, supervision: torch.Tensor, return_metadata: bool = False
    ) -> tuple[torch.Tensor, AttackExecutionMetadata]:
        del supervision
        assert return_metadata
        adversarial = inputs + 0.5
        metadata = AttackExecutionMetadata(
            attack=AttackMethod.FGSM,
            backend=AttackBackend.TORCHATTACKS,
            success_rate=1.0,
            mean_l2=0.5,
            mean_linf=0.5,
            extras={},
        )
        return adversarial, metadata

    adversarial, metadata = batched_attack(
        attack_fn=_attack_fn,
        inputs=inputs,
        supervision=supervision,
        batch_size=2,
        return_metadata=True,
    )

    assert torch.allclose(adversarial, inputs + 0.5)
    assert metadata['num_chunks'] == 3
    assert metadata['success_rate'] == 1.0


def test_attack_dataset_returns_tensor_dataset() -> None:
    inputs = torch.zeros(size=(7, 12, 2), dtype=torch.float32)
    supervision = torch.arange(end=7, dtype=torch.long) % 2
    dataset = TensorDataset(inputs, supervision)

    def _attack_fn(
        *, inputs: torch.Tensor, supervision: torch.Tensor, return_metadata: bool = False
    ) -> torch.Tensor:
        del supervision, return_metadata
        return inputs + 2.0

    dataloader = DataLoader(dataset=dataset, batch_size=3, shuffle=False)
    attacked_dataset = attack_dataset(
        attack_fn=_attack_fn,
        dataloader=dataloader,  # ty: ignore[invalid-argument-type]
        parallel_workers=2,
    )
    attacked_inputs, attacked_supervision = attacked_dataset.tensors  # ty: ignore[unresolved-attribute]
    assert torch.allclose(attacked_inputs, inputs + 2.0)
    assert torch.equal(attacked_supervision, supervision)
