"""Function-first adversarial attack API for time-series tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from src.rbspaper.attacks._backend import run_attack_backend
from src.rbspaper.attacks._common import build_attack_model, maybe_return_metadata, ModelLike
from src.rbspaper.attacks.config import AttackExecutionContext
from src.rbspaper.attacks.enums import AttackBackend, AttackMethod, AttackSupervisionRequirement
from src.rbspaper.attacks.registry import (
    get_default_backend,
    get_supervision_requirement,
    validate_attack_support,
)
from src.rbspaper.enums.general import TimeSeriesDownstreamTask

if TYPE_CHECKING:
    from src.rbspaper.attacks.config import AttackExecutionMetadata

AttackKwargValue = float | int | str | bool | Tensor | None


def _resolve_context(*, context: AttackExecutionContext | None) -> AttackExecutionContext:
    """Resolve optional context to a default task-aware execution context."""
    if context is not None:
        return context

    return AttackExecutionContext(task=TimeSeriesDownstreamTask.CLASSIFICATION)


def _resolve_classification_supervision(
    *, inputs: Tensor, supervision: Tensor | None, model: torch.nn.Module
) -> Tensor:
    """Resolve class supervision, using model-predicted pseudo labels when missing."""
    if supervision is not None:
        return supervision.to(dtype=torch.long)

    with torch.no_grad():
        logits = model(inputs)

    if logits.ndim == 1:
        return (logits > 0).to(dtype=torch.long)

    return torch.argmax(input=logits, dim=1).to(dtype=torch.long)


def _resolve_supervision(
    *,
    inputs: Tensor,
    supervision: Tensor | None,
    model: torch.nn.Module,
    task: TimeSeriesDownstreamTask,
    supervision_requirement: AttackSupervisionRequirement,
) -> Tensor | None:
    """Resolve supervision tensor based on task and requirement policy."""
    if supervision_requirement == AttackSupervisionRequirement.NOT_USED:
        return None

    if task == TimeSeriesDownstreamTask.CLASSIFICATION:
        return _resolve_classification_supervision(
            inputs=inputs, supervision=supervision, model=model
        )

    if task == TimeSeriesDownstreamTask.FORECASTING:
        if supervision is None:
            msg = 'Forecasting attacks require supervision tensor for optimization'
            raise ValueError(msg)
        return supervision.to(dtype=inputs.dtype)

    msg = f'Unsupported downstream task: {task.value}'
    raise ValueError(msg)


def execute_attack(
    *,
    attack: AttackMethod,
    inputs: Tensor,
    supervision: Tensor | None,
    model: ModelLike,
    context: AttackExecutionContext | None = None,
    backend: AttackBackend | None = None,
    return_metadata: bool = False,
    **attack_kwargs: AttackKwargValue,
) -> Tensor | tuple[Tensor, AttackExecutionMetadata]:
    """Execute a named attack on a tensor batch.

    Args:
        attack: Canonical attack method identifier.
        inputs: Clean batch of time-series inputs.
        supervision: Optional supervision associated with inputs. Depending on
            task and attack policy this can represent class labels or forecast
            target values.
        model: Attack model (nn.Module or LightningModule) or callable returning logits.
        context: Task-aware execution context.
        backend: Selected backend. If omitted, task-specific default is used.
        return_metadata: Whether to return runtime attack metadata.
        **attack_kwargs: Backend attack-specific keyword arguments.

    Returns:
        Adversarial tensor, or tuple of tensor and metadata.
    """
    resolved_context = _resolve_context(context=context)
    selected_backend = backend or get_default_backend(attack=attack, task=resolved_context.task)
    validate_attack_support(
        attack=attack,
        task=resolved_context.task,
        backend=selected_backend,
        has_supervision=supervision is not None,
    )

    attack_model = build_attack_model(model=model)
    supervision_requirement = get_supervision_requirement(attack=attack, task=resolved_context.task)
    resolved_supervision = _resolve_supervision(
        inputs=inputs,
        supervision=supervision,
        model=attack_model,
        task=resolved_context.task,
        supervision_requirement=supervision_requirement,
    )

    class_count_raw = attack_kwargs.pop('class_count', None)
    class_count_from_kwargs = int(class_count_raw) if isinstance(class_count_raw, int) else None

    num_classes_raw = attack_kwargs.pop('num_classes', None)
    class_count_alias = int(num_classes_raw) if isinstance(num_classes_raw, int) else None

    clip_min_raw = attack_kwargs.pop('clip_min', None)
    clip_min = float(clip_min_raw) if isinstance(clip_min_raw, int | float) else None

    clip_max_raw = attack_kwargs.pop('clip_max', None)
    clip_max = float(clip_max_raw) if isinstance(clip_max_raw, int | float) else None

    class_count = resolved_context.class_count or class_count_from_kwargs or class_count_alias
    clip_min_value = (
        resolved_context.clip_min if resolved_context.clip_min is not None else clip_min
    )
    clip_max_value = (
        resolved_context.clip_max if resolved_context.clip_max is not None else clip_max
    )

    adversarial = run_attack_backend(
        backend=selected_backend,
        method=attack,
        task=resolved_context.task,
        model=attack_model,
        inputs=inputs,
        supervision=resolved_supervision,
        attack_kwargs=attack_kwargs,  # ty: ignore[invalid-argument-type]
        class_count=class_count,
        clip_values=(clip_min_value, clip_max_value)
        if clip_min_value is not None and clip_max_value is not None
        else None,
    )
    return maybe_return_metadata(
        attack=attack,
        backend=selected_backend,
        clean=inputs,
        adversarial=adversarial,
        return_metadata=return_metadata,
        extras={'model_type': type(attack_model).__name__, 'task': resolved_context.task.value},
    )


def fgsm_attack(
    *,
    inputs: Tensor,
    supervision: Tensor | None,
    model: ModelLike,
    epsilon: float = 8.0 / 255.0,
    context: AttackExecutionContext | None = None,
    backend: AttackBackend | None = None,
    return_metadata: bool = False,
    **kwargs: AttackKwargValue,
) -> Tensor | tuple[Tensor, AttackExecutionMetadata]:
    """Fast Gradient Sign Method."""
    return execute_attack(
        attack=AttackMethod.FGSM,
        inputs=inputs,
        supervision=supervision,
        model=model,
        context=context,
        backend=backend,
        return_metadata=return_metadata,
        eps=epsilon,
        epsilon=epsilon,
        **kwargs,
    )


def bim_attack(
    *,
    inputs: Tensor,
    supervision: Tensor | None,
    model: ModelLike,
    epsilon: float = 8.0 / 255.0,
    alpha: float = 2.0 / 255.0,
    steps: int = 10,
    context: AttackExecutionContext | None = None,
    backend: AttackBackend | None = None,
    return_metadata: bool = False,
    **kwargs: AttackKwargValue,
) -> Tensor | tuple[Tensor, AttackExecutionMetadata]:
    """Basic Iterative Method (I-FGSM)."""
    return execute_attack(
        attack=AttackMethod.BIM,
        inputs=inputs,
        supervision=supervision,
        model=model,
        context=context,
        backend=backend,
        return_metadata=return_metadata,
        eps=epsilon,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        **kwargs,
    )


def pgd_attack(
    *,
    inputs: Tensor,
    supervision: Tensor | None,
    model: ModelLike,
    epsilon: float = 8.0 / 255.0,
    alpha: float = 2.0 / 255.0,
    steps: int = 10,
    random_start: bool = True,
    context: AttackExecutionContext | None = None,
    backend: AttackBackend | None = None,
    return_metadata: bool = False,
    **kwargs: AttackKwargValue,
) -> Tensor | tuple[Tensor, AttackExecutionMetadata]:
    """Projected Gradient Descent attack."""
    return execute_attack(
        attack=AttackMethod.PGD,
        inputs=inputs,
        supervision=supervision,
        model=model,
        context=context,
        backend=backend,
        return_metadata=return_metadata,
        eps=epsilon,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        random_start=random_start,
        **kwargs,
    )


def deepfool_attack(
    *,
    inputs: Tensor,
    supervision: Tensor | None,
    model: ModelLike,
    steps: int = 50,
    overshoot: float = 0.02,
    context: AttackExecutionContext | None = None,
    backend: AttackBackend | None = None,
    return_metadata: bool = False,
    **kwargs: AttackKwargValue,
) -> Tensor | tuple[Tensor, AttackExecutionMetadata]:
    """DeepFool attack."""
    return execute_attack(
        attack=AttackMethod.DEEPFOOL,
        inputs=inputs,
        supervision=supervision,
        model=model,
        context=context,
        backend=backend,
        return_metadata=return_metadata,
        steps=steps,
        max_iter=steps,
        overshoot=overshoot,
        **kwargs,
    )


def cw_attack(
    *,
    inputs: Tensor,
    supervision: Tensor | None,
    model: ModelLike,
    c: float = 1.0,
    kappa: float = 0.0,
    learning_rate: float = 1e-2,
    steps: int = 1000,
    context: AttackExecutionContext | None = None,
    backend: AttackBackend | None = None,
    return_metadata: bool = False,
    **kwargs: AttackKwargValue,
) -> Tensor | tuple[Tensor, AttackExecutionMetadata]:
    """Carlini and Wagner attack."""
    return execute_attack(
        attack=AttackMethod.CW,
        inputs=inputs,
        supervision=supervision,
        model=model,
        context=context,
        backend=backend,
        return_metadata=return_metadata,
        c=c,
        kappa=kappa,
        lr=learning_rate,
        learning_rate=learning_rate,
        steps=steps,
        max_iter=steps,
        **kwargs,
    )


def lbfgs_attack(
    *,
    inputs: Tensor,
    supervision: Tensor | None,
    model: ModelLike,
    epsilon: float = 8.0 / 255.0,
    steps: int = 20,
    context: AttackExecutionContext | None = None,
    backend: AttackBackend | None = None,
    return_metadata: bool = False,
    **kwargs: AttackKwargValue,
) -> Tensor | tuple[Tensor, AttackExecutionMetadata]:
    """L-BFGS attack."""
    return execute_attack(
        attack=AttackMethod.LBFGS,
        inputs=inputs,
        supervision=supervision,
        model=model,
        context=context,
        backend=backend,
        return_metadata=return_metadata,
        eps=epsilon,
        epsilon=epsilon,
        steps=steps,
        max_iter=steps,
        **kwargs,
    )


def uap_attack(
    *,
    inputs: Tensor,
    supervision: Tensor | None,
    model: ModelLike,
    epsilon: float = 8.0 / 255.0,
    delta: float = 0.2,
    max_iter: int = 20,
    attacker: str = 'fgsm',
    context: AttackExecutionContext | None = None,
    backend: AttackBackend | None = None,
    return_metadata: bool = False,
    **kwargs: AttackKwargValue,
) -> Tensor | tuple[Tensor, AttackExecutionMetadata]:
    """Universal adversarial perturbation attack."""
    return execute_attack(
        attack=AttackMethod.UAP,
        inputs=inputs,
        supervision=supervision,
        model=model,
        context=context,
        backend=backend,
        return_metadata=return_metadata,
        eps=epsilon,
        epsilon=epsilon,
        delta=delta,
        max_iter=max_iter,
        attacker=attacker,
        **kwargs,
    )


def spsa_attack(
    *,
    inputs: Tensor,
    supervision: Tensor | None,
    model: ModelLike,
    epsilon: float = 8.0 / 255.0,
    delta: float = 0.01,
    learning_rate: float = 0.01,
    steps: int = 20,
    samples: int = 128,
    context: AttackExecutionContext | None = None,
    backend: AttackBackend | None = None,
    return_metadata: bool = False,
    **kwargs: AttackKwargValue,
) -> Tensor | tuple[Tensor, AttackExecutionMetadata]:
    """SPSA query-efficient attack."""
    return execute_attack(
        attack=AttackMethod.SPSA,
        inputs=inputs,
        supervision=supervision,
        model=model,
        context=context,
        backend=backend,
        return_metadata=return_metadata,
        eps=epsilon,
        epsilon=epsilon,
        delta=delta,
        learning_rate=learning_rate,
        steps=steps,
        nb_iter=steps,
        samples=samples,
        nb_sample=samples,
        **kwargs,
    )


def mi_fgsm_attack(
    *,
    inputs: Tensor,
    supervision: Tensor | None,
    model: ModelLike,
    epsilon: float = 8.0 / 255.0,
    alpha: float = 2.0 / 255.0,
    steps: int = 10,
    decay: float = 1.0,
    context: AttackExecutionContext | None = None,
    backend: AttackBackend | None = None,
    return_metadata: bool = False,
    **kwargs: AttackKwargValue,
) -> Tensor | tuple[Tensor, AttackExecutionMetadata]:
    """Momentum Iterative FGSM attack."""
    return execute_attack(
        attack=AttackMethod.MI_FGSM,
        inputs=inputs,
        supervision=supervision,
        model=model,
        context=context,
        backend=backend,
        return_metadata=return_metadata,
        eps=epsilon,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        decay=decay,
        **kwargs,
    )


def autoattack(
    *,
    inputs: Tensor,
    supervision: Tensor | None,
    model: ModelLike,
    epsilon: float = 8.0 / 255.0,
    context: AttackExecutionContext | None = None,
    backend: AttackBackend | None = None,
    return_metadata: bool = False,
    **kwargs: AttackKwargValue,
) -> Tensor | tuple[Tensor, AttackExecutionMetadata]:
    """AutoAttack ensemble wrapper."""
    return execute_attack(
        attack=AttackMethod.AUTOATTACK,
        inputs=inputs,
        supervision=supervision,
        model=model,
        context=context,
        backend=backend,
        return_metadata=return_metadata,
        eps=epsilon,
        epsilon=epsilon,
        **kwargs,
    )


def hopskipjump_attack(
    *,
    inputs: Tensor,
    supervision: Tensor | None,
    model: ModelLike,
    max_iter: int = 20,
    max_eval: int = 1000,
    init_eval: int = 100,
    context: AttackExecutionContext | None = None,
    backend: AttackBackend | None = None,
    return_metadata: bool = False,
    **kwargs: AttackKwargValue,
) -> Tensor | tuple[Tensor, AttackExecutionMetadata]:
    """HopSkipJump decision-based attack."""
    return execute_attack(
        attack=AttackMethod.HOPSKIPJUMP,
        inputs=inputs,
        supervision=supervision,
        model=model,
        context=context,
        backend=backend,
        return_metadata=return_metadata,
        max_iter=max_iter,
        max_eval=max_eval,
        init_eval=init_eval,
        **kwargs,
    )


def transfer_attack(
    *,
    attack: AttackMethod,
    inputs: Tensor,
    supervision: Tensor | None,
    surrogate_model: ModelLike,
    context: AttackExecutionContext | None = None,
    backend: AttackBackend | None = None,
    return_metadata: bool = False,
    **attack_kwargs: AttackKwargValue,
) -> Tensor | tuple[Tensor, AttackExecutionMetadata]:
    """Gray-box helper that crafts adversarial examples on a surrogate model."""
    return execute_attack(
        attack=attack,
        inputs=inputs,
        supervision=supervision,
        model=surrogate_model,
        context=context,
        backend=backend,
        return_metadata=return_metadata,
        **attack_kwargs,
    )
