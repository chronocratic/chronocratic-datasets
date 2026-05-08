"""Shared utilities for function-based adversarial attacks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import inspect
from typing import TYPE_CHECKING

import torch
from torch import nn, Tensor

from src.rbspaper.attacks.config import AttackExecutionMetadata

if TYPE_CHECKING:
    from src.rbspaper.attacks.enums import AttackBackend, AttackMethod


ModelLike = nn.Module | Callable[[Tensor], Tensor]


class _CallableModule(nn.Module):
    """Wrap callables and adapter forward methods into an nn.Module."""

    def __init__(self, *, forward_fn: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self._dummy = nn.Parameter(
            data=torch.zeros(size=(), dtype=torch.float32), requires_grad=True
        )
        self._forward_fn = forward_fn

    def forward(self, inputs: Tensor) -> Tensor:
        return self._forward_fn(inputs)


def build_attack_model(*, model: ModelLike) -> nn.Module:
    """Build a module accepted by attack backends.

    Args:
        model: Model-like object used to score adversarial candidates.

    Returns:
        A torch module with a single-input forward.
    """
    if isinstance(model, nn.Module):
        return model

    if callable(model):
        return _CallableModule(forward_fn=model)

    msg = 'model must be nn.Module (including LightningModule) or Callable[[Tensor], Tensor]'
    raise TypeError(msg)


def infer_class_count(*, supervision: Tensor, class_count: int | None) -> int:
    """Infer class count from class supervision when not provided."""
    if class_count is not None:
        return class_count

    if supervision.numel() == 0:
        msg = 'supervision is empty and class_count was not provided'
        raise ValueError(msg)

    return int(torch.max(input=supervision).item()) + 1


def call_with_supported_kwargs(
    *, target: Callable[..., object], kwargs: dict[str, object]
) -> object:
    """Call a function while filtering unsupported keyword arguments."""
    signature = inspect.signature(target)
    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }
    return target(**supported_kwargs)


def compute_attack_statistics(*, clean: Tensor, adversarial: Tensor) -> tuple[float, float]:
    """Return average L2 and Linf norms over batch perturbations."""
    delta = (adversarial - clean).detach().reshape(clean.shape[0], -1)
    l2_values = torch.linalg.vector_norm(delta, ord=2, dim=1)
    linf_values = torch.linalg.vector_norm(delta, ord=float('inf'), dim=1)
    return float(torch.mean(input=l2_values).item()), float(torch.mean(input=linf_values).item())


def maybe_return_metadata(
    *,
    attack: AttackMethod,
    backend: AttackBackend,
    clean: Tensor,
    adversarial: Tensor,
    return_metadata: bool,
    success_mask: Tensor | None = None,
    extras: dict[str, float | int | str | bool] | None = None,
) -> Tensor | tuple[Tensor, AttackExecutionMetadata]:
    """Return tensor only or tensor plus metadata based on flag."""
    if not return_metadata:
        return adversarial

    mean_l2, mean_linf = compute_attack_statistics(clean=clean, adversarial=adversarial)
    success_rate: float | None = None
    if success_mask is not None and success_mask.numel() > 0:
        success_rate = float(torch.mean(input=success_mask.to(dtype=torch.float32)).item())

    metadata = AttackExecutionMetadata(
        attack=attack,
        backend=backend,
        success_rate=success_rate,
        mean_l2=mean_l2,
        mean_linf=mean_linf,
        extras=extras or {},
    )
    return adversarial, metadata


def metadata_to_dict(*, metadata: AttackExecutionMetadata) -> dict[str, object]:
    """Serialize metadata dataclass to plain dictionary."""
    return asdict(obj=metadata)
