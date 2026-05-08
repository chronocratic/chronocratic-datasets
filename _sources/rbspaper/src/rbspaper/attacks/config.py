"""Dataclasses for attack configuration and runtime metadata."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import override

from src.rbspaper.attacks.enums import AttackBackend, AttackMethod, AttackObjective
from src.rbspaper.enums.general import TimeSeriesDownstreamTask


@dataclass(frozen=True)
class AttackExecutionContext:
    """Context used to execute an attack in a task-aware manner.

    Attributes:
        task: Downstream task used to select validation and backend pathways.
        objective: Optimization objective for the selected attack.
        class_count: Optional explicit class count for classification attacks.
        clip_min: Optional lower clipping bound.
        clip_max: Optional upper clipping bound.
        horizon_start: Optional forecasting horizon start index (inclusive).
        horizon_end: Optional forecasting horizon end index (exclusive).
    """

    task: TimeSeriesDownstreamTask = TimeSeriesDownstreamTask.CLASSIFICATION
    objective: AttackObjective = AttackObjective.UNTARGETED
    class_count: int | None = None
    clip_min: float | None = None
    clip_max: float | None = None
    horizon_start: int | None = None
    horizon_end: int | None = None


@dataclass
class AttackParameters(ABC):
    """Base attack parameter object used for reproducible attack runs."""

    backend: AttackBackend
    clip_min: float | None = None
    clip_max: float | None = None

    @property
    @abstractmethod
    def attack_method(self) -> AttackMethod:
        """Return canonical attack method identifier."""


@dataclass
class FgsmAttackParameters(AttackParameters):
    """Configuration for FGSM."""

    epsilon: float = 8.0 / 255.0

    @property
    @override
    def attack_method(self) -> AttackMethod:
        return AttackMethod.FGSM


@dataclass
class BimAttackParameters(AttackParameters):
    """Configuration for BIM / I-FGSM."""

    epsilon: float = 8.0 / 255.0
    alpha: float = 2.0 / 255.0
    steps: int = 10

    @property
    @override
    def attack_method(self) -> AttackMethod:
        return AttackMethod.BIM


@dataclass
class PgdAttackParameters(AttackParameters):
    """Configuration for PGD."""

    epsilon: float = 8.0 / 255.0
    alpha: float = 2.0 / 255.0
    steps: int = 10
    random_start: bool = True

    @property
    @override
    def attack_method(self) -> AttackMethod:
        return AttackMethod.PGD


@dataclass
class DeepFoolAttackParameters(AttackParameters):
    """Configuration for DeepFool."""

    steps: int = 50
    overshoot: float = 0.02

    @property
    @override
    def attack_method(self) -> AttackMethod:
        return AttackMethod.DEEPFOOL


@dataclass
class CwAttackParameters(AttackParameters):
    """Configuration for Carlini and Wagner attack."""

    confidence: float = 0.0
    learning_rate: float = 1e-2
    steps: int = 1000
    c: float = 1.0
    kappa: float = 0.0

    @property
    @override
    def attack_method(self) -> AttackMethod:
        return AttackMethod.CW


@dataclass
class LbfgsAttackParameters(AttackParameters):
    """Configuration for L-BFGS attack."""

    epsilon: float = 8.0 / 255.0
    steps: int = 20

    @property
    @override
    def attack_method(self) -> AttackMethod:
        return AttackMethod.LBFGS


@dataclass
class SpsaAttackParameters(AttackParameters):
    """Configuration for SPSA attack."""

    epsilon: float = 8.0 / 255.0
    delta: float = 0.01
    learning_rate: float = 0.01
    nb_iter: int = 20
    nb_sample: int = 128

    @property
    @override
    def attack_method(self) -> AttackMethod:
        return AttackMethod.SPSA


@dataclass
class UapAttackParameters(AttackParameters):
    """Configuration for universal adversarial perturbation."""

    epsilon: float = 8.0 / 255.0
    delta: float = 0.2
    max_iter: int = 20
    attacker: str = 'fgsm'
    attacker_params: dict[str, float | int] = field(default_factory=dict)

    @property
    @override
    def attack_method(self) -> AttackMethod:
        return AttackMethod.UAP


@dataclass
class AttackExecutionMetadata:
    """Runtime information returned by attack functions when requested."""

    attack: AttackMethod
    backend: AttackBackend
    success_rate: float | None
    mean_l2: float
    mean_linf: float
    extras: dict[str, float | int | str | bool] = field(default_factory=dict)
