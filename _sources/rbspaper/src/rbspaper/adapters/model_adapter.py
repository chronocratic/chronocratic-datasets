"""Model adapter contract used by pipeline and attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    """Defines the model-facing API for the benchmark pipeline."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return a stable model identifier."""

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return embedding dimensionality produced by ``encode``."""

    @classmethod
    @abstractmethod
    def load_from_checkpoint(cls, *, checkpoint_path: str, **kwargs: Any) -> ModelAdapter:
        """Load an adapter-bound model from checkpoint."""

    @abstractmethod
    def encode(self, *, inputs: Any) -> Any:
        """Encode input time series into embeddings."""

    @abstractmethod
    def forward_for_attack(self, *, inputs: Any, task_name: str) -> Any:
        """Forward pass used by attack backends to compute objectives."""

    @abstractmethod
    def supports_task(self, *, task_name: str) -> bool:
        """Return whether model has required head/logic for a given task."""
