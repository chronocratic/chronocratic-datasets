"""Task adapter contract for downstream and representation evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class TaskAdapter(ABC):
    """Defines downstream task evaluation interface."""

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Return task identifier."""

    @abstractmethod
    def evaluate(
        self, *, clean_embeddings: Any, adversarial_embeddings: Any, targets: Any
    ) -> dict[str, float]:
        """Evaluate clean and adversarial representations for this task."""
