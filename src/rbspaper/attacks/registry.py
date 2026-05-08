"""Registry helpers for available attacks, backends, and threat models."""

from __future__ import annotations

from src.rbspaper.attacks.enums import (
    AttackBackend,
    AttackFamily,
    AttackMethod,
    AttackSupervisionRequirement,
    AttackThreatModel,
)
from src.rbspaper.enums.general import TimeSeriesDownstreamTask

ATTACK_THREAT_MODEL: dict[AttackMethod, AttackThreatModel] = {
    AttackMethod.FGSM: AttackThreatModel.WHITE_BOX,
    AttackMethod.BIM: AttackThreatModel.WHITE_BOX,
    AttackMethod.PGD: AttackThreatModel.WHITE_BOX,
    AttackMethod.DEEPFOOL: AttackThreatModel.WHITE_BOX,
    AttackMethod.CW: AttackThreatModel.WHITE_BOX,
    AttackMethod.LBFGS: AttackThreatModel.WHITE_BOX,
    AttackMethod.MI_FGSM: AttackThreatModel.WHITE_BOX,
    AttackMethod.AUTOATTACK: AttackThreatModel.WHITE_BOX,
    AttackMethod.SPSA: AttackThreatModel.BLACK_BOX,
    AttackMethod.UAP: AttackThreatModel.BLACK_BOX,
    AttackMethod.HOPSKIPJUMP: AttackThreatModel.BLACK_BOX,
    AttackMethod.BOUNDARY: AttackThreatModel.BLACK_BOX,
    AttackMethod.JSMA: AttackThreatModel.WHITE_BOX,
    AttackMethod.ONE_PIXEL: AttackThreatModel.BLACK_BOX,
    AttackMethod.SIMBA: AttackThreatModel.BLACK_BOX,
    AttackMethod.EAD: AttackThreatModel.WHITE_BOX,
}

SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK: dict[
    tuple[TimeSeriesDownstreamTask, AttackMethod], set[AttackBackend]
] = {
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.FGSM): {
        AttackBackend.TORCHATTACKS,
        AttackBackend.ART,
    },
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.BIM): {
        AttackBackend.TORCHATTACKS,
        AttackBackend.ART,
    },
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.PGD): {
        AttackBackend.TORCHATTACKS,
        AttackBackend.ART,
    },
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.DEEPFOOL): {
        AttackBackend.TORCHATTACKS,
        AttackBackend.ART,
    },
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.CW): {
        AttackBackend.TORCHATTACKS,
        AttackBackend.ART,
    },
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.LBFGS): {
        AttackBackend.TORCHATTACKS,
        AttackBackend.ART,
    },
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.SPSA): {
        AttackBackend.TORCHATTACKS,
        AttackBackend.ART,
    },
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.UAP): {AttackBackend.ART},
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.MI_FGSM): {
        AttackBackend.TORCHATTACKS,
        AttackBackend.ART,
    },
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.AUTOATTACK): {
        AttackBackend.TORCHATTACKS,
        AttackBackend.ART,
    },
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.HOPSKIPJUMP): {AttackBackend.ART},
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.BOUNDARY): {AttackBackend.ART},
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.JSMA): {
        AttackBackend.TORCHATTACKS,
        AttackBackend.ART,
    },
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.ONE_PIXEL): {
        AttackBackend.TORCHATTACKS,
        AttackBackend.ART,
    },
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.SIMBA): {AttackBackend.ART},
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.EAD): {AttackBackend.ART},
    (TimeSeriesDownstreamTask.FORECASTING, AttackMethod.FGSM): {AttackBackend.ART},
    (TimeSeriesDownstreamTask.FORECASTING, AttackMethod.BIM): {AttackBackend.ART},
    (TimeSeriesDownstreamTask.FORECASTING, AttackMethod.PGD): {AttackBackend.ART},
}

DEFAULT_BACKEND_BY_TASK_AND_ATTACK: dict[
    tuple[TimeSeriesDownstreamTask, AttackMethod], AttackBackend
] = {
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.FGSM): AttackBackend.TORCHATTACKS,
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.BIM): AttackBackend.TORCHATTACKS,
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.PGD): AttackBackend.TORCHATTACKS,
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.DEEPFOOL): AttackBackend.TORCHATTACKS,
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.CW): AttackBackend.TORCHATTACKS,
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.LBFGS): AttackBackend.TORCHATTACKS,
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.SPSA): AttackBackend.TORCHATTACKS,
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.UAP): AttackBackend.ART,
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.MI_FGSM): AttackBackend.TORCHATTACKS,
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.AUTOATTACK): AttackBackend.TORCHATTACKS,
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.HOPSKIPJUMP): AttackBackend.ART,
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.BOUNDARY): AttackBackend.ART,
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.JSMA): AttackBackend.ART,
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.ONE_PIXEL): AttackBackend.TORCHATTACKS,
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.SIMBA): AttackBackend.ART,
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.EAD): AttackBackend.ART,
    (TimeSeriesDownstreamTask.FORECASTING, AttackMethod.FGSM): AttackBackend.ART,
    (TimeSeriesDownstreamTask.FORECASTING, AttackMethod.BIM): AttackBackend.ART,
    (TimeSeriesDownstreamTask.FORECASTING, AttackMethod.PGD): AttackBackend.ART,
}

SUPERVISION_REQUIREMENT_BY_TASK_AND_ATTACK: dict[
    tuple[TimeSeriesDownstreamTask, AttackMethod], AttackSupervisionRequirement
] = {
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.FGSM): (
        AttackSupervisionRequirement.OPTIONAL
    ),
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.BIM): (
        AttackSupervisionRequirement.OPTIONAL
    ),
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.PGD): (
        AttackSupervisionRequirement.OPTIONAL
    ),
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.DEEPFOOL): (
        AttackSupervisionRequirement.OPTIONAL
    ),
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.CW): (
        AttackSupervisionRequirement.REQUIRED
    ),
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.LBFGS): (
        AttackSupervisionRequirement.REQUIRED
    ),
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.SPSA): (
        AttackSupervisionRequirement.OPTIONAL
    ),
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.UAP): (
        AttackSupervisionRequirement.OPTIONAL
    ),
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.MI_FGSM): (
        AttackSupervisionRequirement.OPTIONAL
    ),
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.AUTOATTACK): (
        AttackSupervisionRequirement.REQUIRED
    ),
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.HOPSKIPJUMP): (
        AttackSupervisionRequirement.NOT_USED
    ),
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.BOUNDARY): (
        AttackSupervisionRequirement.NOT_USED
    ),
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.JSMA): (
        AttackSupervisionRequirement.REQUIRED
    ),
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.ONE_PIXEL): (
        AttackSupervisionRequirement.OPTIONAL
    ),
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.SIMBA): (
        AttackSupervisionRequirement.OPTIONAL
    ),
    (TimeSeriesDownstreamTask.CLASSIFICATION, AttackMethod.EAD): (
        AttackSupervisionRequirement.REQUIRED
    ),
    (TimeSeriesDownstreamTask.FORECASTING, AttackMethod.FGSM): (
        AttackSupervisionRequirement.REQUIRED
    ),
    (TimeSeriesDownstreamTask.FORECASTING, AttackMethod.BIM): (
        AttackSupervisionRequirement.REQUIRED
    ),
    (TimeSeriesDownstreamTask.FORECASTING, AttackMethod.PGD): (
        AttackSupervisionRequirement.REQUIRED
    ),
}


def get_default_backend(*, attack: AttackMethod, task: TimeSeriesDownstreamTask) -> AttackBackend:
    """Return default backend for attack and downstream task."""
    return DEFAULT_BACKEND_BY_TASK_AND_ATTACK[(task, attack)]


def get_threat_model(*, attack: AttackMethod) -> AttackThreatModel:
    """Return threat-model category for an attack method."""
    return ATTACK_THREAT_MODEL[attack]


def get_supervision_requirement(
    *, attack: AttackMethod, task: TimeSeriesDownstreamTask
) -> AttackSupervisionRequirement:
    """Return supervision requirement policy for a task and attack."""
    return SUPERVISION_REQUIREMENT_BY_TASK_AND_ATTACK[(task, attack)]


def is_backend_supported(
    *, attack: AttackMethod, task: TimeSeriesDownstreamTask, backend: AttackBackend
) -> bool:
    """Return whether an attack/backend combination is supported for a task."""
    return backend in SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK.get((task, attack), set())


def validate_attack_support(
    *,
    attack: AttackMethod,
    task: TimeSeriesDownstreamTask,
    backend: AttackBackend,
    has_supervision: bool,
) -> None:
    """Validate task/backend/supervision constraints before attack execution."""
    if not is_backend_supported(attack=attack, task=task, backend=backend):
        msg = (
            f'Attack {attack.value!r} with backend {backend.value!r} '
            f'is not supported for task {task.value!r}'
        )
        raise ValueError(msg)

    supervision_requirement = get_supervision_requirement(attack=attack, task=task)
    if supervision_requirement == AttackSupervisionRequirement.REQUIRED and not has_supervision:
        msg = (
            f'Attack {attack.value!r} requires supervision for '
            f'task {task.value!r} and backend {backend.value!r}'
        )
        raise ValueError(msg)


def list_supported_attacks(*, task: TimeSeriesDownstreamTask | None = None) -> list[AttackMethod]:
    """List attack methods available in the registry.

    Args:
        task: Optional task filter.
    """
    if task is None:
        methods = {method for _, method in DEFAULT_BACKEND_BY_TASK_AND_ATTACK}
        return sorted(methods, key=lambda value: value.value)

    methods = {
        method
        for supported_task, method in DEFAULT_BACKEND_BY_TASK_AND_ATTACK
        if supported_task == task
    }
    return sorted(methods, key=lambda value: value.value)


def group_methods_by_family() -> dict[AttackFamily, frozenset[AttackMethod]]:
    """Group all registered attack methods by their threat-model family.

    Returns:
        Dict mapping AttackFamily to the frozenset of AttackMethods in that family.
    """
    result: dict[AttackFamily, set[AttackMethod]] = {
        AttackFamily.WHITE_BOX: set(),
        AttackFamily.BLACK_BOX: set(),
    }
    for method, threat in ATTACK_THREAT_MODEL.items():
        if threat == AttackThreatModel.WHITE_BOX:
            result[AttackFamily.WHITE_BOX].add(method)
        elif threat == AttackThreatModel.BLACK_BOX:
            result[AttackFamily.BLACK_BOX].add(method)
        # GRAY_BOX methods are not grouped into a family (none currently registered)
    return {k: frozenset(v) for k, v in result.items()}
