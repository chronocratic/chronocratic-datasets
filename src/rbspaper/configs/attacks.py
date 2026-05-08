"""Backward-compatible re-exports for attack configuration dataclasses."""

from src.rbspaper.attacks.config import (
    AttackExecutionMetadata,
    AttackParameters,
    BimAttackParameters,
    CwAttackParameters,
    DeepFoolAttackParameters,
    FgsmAttackParameters,
    LbfgsAttackParameters,
    PgdAttackParameters,
    SpsaAttackParameters,
    UapAttackParameters,
)

__all__ = [
    'AttackExecutionMetadata',
    'AttackParameters',
    'BimAttackParameters',
    'CwAttackParameters',
    'DeepFoolAttackParameters',
    'FgsmAttackParameters',
    'LbfgsAttackParameters',
    'PgdAttackParameters',
    'SpsaAttackParameters',
    'UapAttackParameters',
]
