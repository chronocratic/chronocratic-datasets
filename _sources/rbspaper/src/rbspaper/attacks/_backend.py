"""Backend dispatch for ART and Torchattacks attack implementations."""

from __future__ import annotations

from importlib import import_module
from typing import cast, Protocol, TYPE_CHECKING

import numpy as np
import torch
from torch import nn, Tensor

from src.rbspaper.attacks._common import call_with_supported_kwargs, infer_class_count
from src.rbspaper.attacks.enums import AttackBackend, AttackMethod
from src.rbspaper.enums.general import TimeSeriesDownstreamTask

if TYPE_CHECKING:
    from src.rbspaper.attacks.functional import AttackKwargValue


class _TorchAttackCallable(Protocol):
    def __call__(self, inputs: Tensor, supervision: Tensor) -> Tensor: ...


class _ArtAttackCallable(Protocol):
    def generate(self, *, x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray: ...


_TORCHATTACKS_CLASS_NAMES: dict[AttackMethod, tuple[str, ...]] = {
    AttackMethod.FGSM: ('FGSM',),
    AttackMethod.BIM: ('BIM',),
    AttackMethod.PGD: ('PGD',),
    AttackMethod.DEEPFOOL: ('DeepFool',),
    AttackMethod.CW: ('CW',),
    AttackMethod.LBFGS: ('LBFGS',),
    AttackMethod.SPSA: ('SPSA',),
    AttackMethod.MI_FGSM: ('MIFGSM', 'MI_FGSM'),
    AttackMethod.AUTOATTACK: ('AutoAttack',),
    AttackMethod.ONE_PIXEL: ('OnePixel', 'OnePixelAttack'),
    AttackMethod.JSMA: ('JSMA',),
}

_ART_CLASS_NAMES: dict[AttackMethod, tuple[str, ...]] = {
    AttackMethod.FGSM: ('FastGradientMethod',),
    AttackMethod.BIM: ('BasicIterativeMethod',),
    AttackMethod.PGD: ('ProjectedGradientDescentPyTorch', 'ProjectedGradientDescent'),
    AttackMethod.DEEPFOOL: ('DeepFool',),
    AttackMethod.CW: ('CarliniL2Method',),
    AttackMethod.LBFGS: ('LBFGS',),
    AttackMethod.UAP: ('UniversalPerturbation',),
    AttackMethod.SPSA: ('SPSA', 'SimultaneousPerturbation'),
    AttackMethod.MI_FGSM: ('MomentumIterativeMethod',),
    AttackMethod.AUTOATTACK: ('AutoAttack',),
    AttackMethod.HOPSKIPJUMP: ('HopSkipJump',),
    AttackMethod.BOUNDARY: ('BoundaryAttack',),
    AttackMethod.JSMA: ('SaliencyMapMethod',),
    AttackMethod.SIMBA: ('SimBA',),
    AttackMethod.EAD: ('ElasticNet',),
    AttackMethod.ONE_PIXEL: ('PixelAttack',),
}


def _resolve_class(*, module_name: str, class_names: tuple[str, ...]) -> type[object]:
    module = import_module(module_name)
    for class_name in class_names:
        if hasattr(module, class_name):
            return getattr(module, class_name)

    msg = f'None of {class_names!r} found in {module_name!r}'
    raise AttributeError(msg)


def _build_art_classification_estimator(
    *,
    model: nn.Module,
    inputs: Tensor,
    supervision: Tensor,
    class_count: int | None,
    clip_values: tuple[float, float] | None,
) -> object:
    from art.estimators.classification import (  # noqa: PLC0415
        PyTorchClassifier,
    )

    if clip_values is None:
        clip_values = (float(torch.min(input=inputs).item()), float(torch.max(input=inputs).item()))

    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not model_parameters:
        msg = 'Attack model has no trainable parameters; wrap callables using build_attack_model'
        raise ValueError(msg)

    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(params=model_parameters, lr=0.01),
        input_shape=tuple(inputs.shape[1:]),
        nb_classes=infer_class_count(supervision=supervision, class_count=class_count),
        clip_values=clip_values,
    )
    return classifier


def _build_art_forecasting_estimator(
    *, model: nn.Module, inputs: Tensor, clip_values: tuple[float, float] | None
) -> object:
    from art.estimators.regression import (  # noqa: PLC0415
        PyTorchRegressor,
    )

    if clip_values is None:
        clip_values = (float(torch.min(input=inputs).item()), float(torch.max(input=inputs).item()))

    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not model_parameters:
        msg = 'Attack model has no trainable parameters; wrap callables using build_attack_model'
        raise ValueError(msg)

    regressor = PyTorchRegressor(
        model=model,
        loss=nn.MSELoss(),
        optimizer=torch.optim.SGD(params=model_parameters, lr=0.01),
        input_shape=tuple(inputs.shape[1:]),
        clip_values=clip_values,
    )
    return regressor


def run_torchattacks(
    *,
    method: AttackMethod,
    model: nn.Module,
    inputs: Tensor,
    supervision: Tensor,
    attack_kwargs: dict[str, AttackKwargValue],
) -> Tensor:
    """Execute a Torchattacks method and return adversarial samples."""
    if method not in _TORCHATTACKS_CLASS_NAMES:
        msg = f'{method.value} is not registered for torchattacks backend'
        raise ValueError(msg)

    try:
        attack_class = _resolve_class(
            module_name='torchattacks', class_names=_TORCHATTACKS_CLASS_NAMES[method]
        )
    except (ImportError, AttributeError) as error:
        msg = (
            'Unable to resolve requested attack in torchattacks. '
            'Install the attacks group and verify package version compatibility.'
        )
        raise RuntimeError(msg) from error

    attack_object = call_with_supported_kwargs(
        target=attack_class, kwargs={'model': model, **attack_kwargs}
    )
    attack = cast('_TorchAttackCallable', attack_object)
    model.eval()
    with torch.enable_grad():
        adversarial = attack(inputs, supervision)
    return adversarial.detach()


def run_art(
    *,
    method: AttackMethod,
    task: TimeSeriesDownstreamTask,
    model: nn.Module,
    inputs: Tensor,
    supervision: Tensor | None,
    attack_kwargs: dict[str, AttackKwargValue],
    class_count: int | None = None,
    clip_values: tuple[float, float] | None = None,
) -> Tensor:
    """Execute an ART method and return adversarial samples."""
    if method not in _ART_CLASS_NAMES:
        msg = f'{method.value} is not registered for ART backend'
        raise ValueError(msg)

    try:
        attack_class = _resolve_class(
            module_name='art.attacks.evasion', class_names=_ART_CLASS_NAMES[method]
        )
    except (ImportError, AttributeError) as error:
        msg = (
            'Unable to resolve requested attack in ART. '
            'Install adversarial-robustness-toolbox and verify package version compatibility.'
        )
        raise RuntimeError(msg) from error

    if task == TimeSeriesDownstreamTask.CLASSIFICATION:
        if supervision is None:
            msg = 'Classification ART attacks require resolved supervision before backend execution'
            raise ValueError(msg)
        estimator = _build_art_classification_estimator(
            model=model,
            inputs=inputs,
            supervision=supervision,
            class_count=class_count,
            clip_values=clip_values,
        )
    elif task == TimeSeriesDownstreamTask.FORECASTING:
        estimator = _build_art_forecasting_estimator(
            model=model, inputs=inputs, clip_values=clip_values
        )
    else:
        msg = f'Unsupported task for ART backend: {task.value}'
        raise ValueError(msg)

    attack_object = call_with_supported_kwargs(
        target=attack_class,
        kwargs={'estimator': estimator, 'classifier': estimator, **attack_kwargs},
    )
    attack = cast('_ArtAttackCallable', attack_object)
    supervision_np = None
    if supervision is not None:
        supervision_np = supervision.detach().cpu().numpy()
    adversarial_np = call_with_supported_kwargs(
        target=attack.generate, kwargs={'x': inputs.detach().cpu().numpy(), 'y': supervision_np}
    )
    return torch.as_tensor(
        data=np.asarray(adversarial_np), dtype=inputs.dtype, device=inputs.device
    )


def run_attack_backend(
    *,
    backend: AttackBackend,
    method: AttackMethod,
    task: TimeSeriesDownstreamTask,
    model: nn.Module,
    inputs: Tensor,
    supervision: Tensor | None,
    attack_kwargs: dict[str, AttackKwargValue],
    class_count: int | None = None,
    clip_values: tuple[float, float] | None = None,
) -> Tensor:
    """Run attack method using selected backend."""
    if backend == AttackBackend.TORCHATTACKS:
        if supervision is None:
            msg = 'Torchattacks backend requires supervision tensor at execution time'
            raise ValueError(msg)
        return run_torchattacks(
            method=method,
            model=model,
            inputs=inputs,
            supervision=supervision,
            attack_kwargs=attack_kwargs,
        )

    if backend == AttackBackend.ART:
        return run_art(
            method=method,
            task=task,
            model=model,
            inputs=inputs,
            supervision=supervision,
            attack_kwargs=attack_kwargs,
            class_count=class_count,
            clip_values=clip_values,
        )

    msg = f'Unsupported backend: {backend.value}'
    raise ValueError(msg)
