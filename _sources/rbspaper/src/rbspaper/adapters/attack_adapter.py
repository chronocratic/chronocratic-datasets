"""Attack adapter contract used by staged experiments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.rbspaper.adapters.model_adapter import ModelAdapter


class AttackAdapter(ABC):
    """Defines a backend-agnostic attack interface."""

    @property
    @abstractmethod
    def attack_name(self) -> str:
        """Return attack identifier used in reporting."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return backend name, for example ``torchattacks`` or ``art``."""

    @abstractmethod
    def generate(self, *, model: ModelAdapter, inputs: Any, targets: Any) -> Any:
        """Generate adversarial samples for provided inputs."""
