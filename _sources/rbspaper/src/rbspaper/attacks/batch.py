"""Batch and dataloader utilities for function-based adversarial attacks."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.rbspaper.attacks._common import metadata_to_dict
from src.rbspaper.attacks.config import AttackExecutionMetadata

if TYPE_CHECKING:
    from src.rbspaper.attacks.functional import AttackKwargValue

AttackFunction = Callable[..., object]


@dataclass
class _BatchPayload:
    index: int
    inputs: Tensor
    supervision: Tensor


def _run_attack_for_payload(
    *,
    payload: _BatchPayload,
    attack_fn: AttackFunction,
    return_metadata: bool,
    attack_kwargs: dict[str, AttackKwargValue],
) -> tuple[int, Tensor, Tensor, AttackExecutionMetadata | None]:
    attack_result = attack_fn(
        inputs=payload.inputs,
        supervision=payload.supervision,
        return_metadata=return_metadata,
        **attack_kwargs,
    )

    if return_metadata:
        if not isinstance(attack_result, tuple):
            msg = 'Attack function did not return metadata tuple'
            raise TypeError(msg)
        attacked_inputs, metadata = attack_result
        if not isinstance(attacked_inputs, Tensor):
            msg = 'Attack function returned invalid adversarial tensor'
            raise TypeError(msg)
        if not isinstance(metadata, AttackExecutionMetadata):
            msg = 'Attack function returned invalid metadata object'
            raise TypeError(msg)
        return payload.index, attacked_inputs, payload.supervision, metadata

    attacked_inputs = attack_result[0] if isinstance(attack_result, tuple) else attack_result
    if not isinstance(attacked_inputs, Tensor):
        msg = 'Attack function returned invalid adversarial tensor'
        raise TypeError(msg)
    return payload.index, attacked_inputs, payload.supervision, None


def _aggregate_metadata(
    *, batch_size: int, parallel_workers: int, metadata_chunks: list[AttackExecutionMetadata]
) -> dict[str, object]:
    success_rates = [item.success_rate for item in metadata_chunks if item.success_rate is not None]
    return {
        'batch_size': batch_size,
        'parallel_workers': parallel_workers,
        'num_chunks': len(metadata_chunks),
        'mean_l2': float(sum(item.mean_l2 for item in metadata_chunks) / len(metadata_chunks)),
        'mean_linf': float(sum(item.mean_linf for item in metadata_chunks) / len(metadata_chunks)),
        'success_rate': float(sum(success_rates) / len(success_rates)) if success_rates else None,
        'chunks': [metadata_to_dict(metadata=item) for item in metadata_chunks],
    }


def batched_attack(
    *,
    attack_fn: AttackFunction,
    inputs: Tensor,
    supervision: Tensor,
    batch_size: int,
    return_metadata: bool = False,
    **attack_kwargs: AttackKwargValue,
) -> Tensor | tuple[Tensor, dict[str, object]]:
    """Apply an attack function in chunks and return attacked inputs."""
    dataloader = DataLoader(
        dataset=TensorDataset(inputs, supervision),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    attacked = attack_dataset(
        attack_fn=attack_fn,
        dataloader=dataloader,  # ty: ignore[invalid-argument-type]
        parallel_workers=1,
        return_metadata=return_metadata,
        attack_kwargs=attack_kwargs,
    )

    if return_metadata:
        if not isinstance(attacked, tuple):
            msg = 'Expected tuple result when return_metadata=True'
            raise TypeError(msg)
        attacked_dataset, metadata = attacked
        attacked_inputs, _ = attacked_dataset.tensors
        return attacked_inputs, metadata

    if isinstance(attacked, tuple):
        msg = 'Expected TensorDataset result when return_metadata=False'
        raise TypeError(msg)
    attacked_inputs, _ = attacked.tensors
    return attacked_inputs


def attack_dataset(
    *,
    attack_fn: AttackFunction,
    dataloader: DataLoader,
    parallel_workers: int = 1,
    return_metadata: bool = False,
    attack_kwargs: dict[str, AttackKwargValue] | None = None,
) -> TensorDataset | tuple[TensorDataset, dict[str, object]]:
    """Generate an attacked dataset from a DataLoader.

    This function is intentionally DataLoader-only to align with LightningDataModule workflows.
    """
    if parallel_workers <= 0:
        msg = 'parallel_workers must be a positive integer'
        raise ValueError(msg)

    attack_kwargs = attack_kwargs or {}
    payloads = [
        _BatchPayload(index=index, inputs=batch_inputs, supervision=batch_supervision)
        for index, (batch_inputs, batch_supervision) in enumerate(dataloader)
    ]

    if parallel_workers > 1 and not torch.cuda.is_available():
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            results = list(
                executor.map(
                    lambda payload: _run_attack_for_payload(
                        payload=payload,
                        attack_fn=attack_fn,
                        return_metadata=return_metadata,
                        attack_kwargs=attack_kwargs,
                    ),
                    payloads,
                )
            )
    else:
        results = [
            _run_attack_for_payload(
                payload=payload,
                attack_fn=attack_fn,
                return_metadata=return_metadata,
                attack_kwargs=attack_kwargs,
            )
            for payload in payloads
        ]

    ordered = sorted(results, key=lambda value: value[0])
    attacked_inputs = torch.cat([item[1] for item in ordered], dim=0)
    attacked_supervision = torch.cat([item[2] for item in ordered], dim=0)
    attacked_dataset = TensorDataset(attacked_inputs, attacked_supervision)

    if not return_metadata:
        return attacked_dataset

    metadata_chunks = [item[3] for item in ordered if item[3] is not None]
    if not metadata_chunks:
        msg = 'return_metadata=True but no metadata chunks were produced'
        raise RuntimeError(msg)

    metadata = _aggregate_metadata(
        batch_size=getattr(dataloader, 'batch_size', 0) or 0,
        parallel_workers=parallel_workers,
        metadata_chunks=[item for item in metadata_chunks if item is not None],
    )
    return attacked_dataset, metadata
