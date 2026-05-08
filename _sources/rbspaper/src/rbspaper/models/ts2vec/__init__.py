"""TS2Vec model for hierarchical time series representation learning."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.rbspaper.models.ts2vec.model import TS2Vec

__all__ = ['TS2Vec']


def __getattr__(name: str) -> object:
    """Lazy import to avoid circular dependency with augmentation factories."""
    if name == 'TS2Vec':
        from src.rbspaper.models.ts2vec.model import TS2Vec as _TS2Vec  # noqa: PLC0415

        return _TS2Vec
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
