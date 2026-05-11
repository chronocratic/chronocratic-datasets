"""Factory registry for dataset configurations.

Provides a static CONFIGS dictionary mapping dataset names to frozen
Pydantic configuration instances, along with lookup functions
(`get_config`, `list_configs`) for runtime access.

Per D-02 (both module-level constants + registry dict) and RESEARCH.md
Pattern 4 (Module-Level Constants + Registry Dict), all config instances
are explicitly imported rather than auto-discovered, ensuring IDE support
and fast startup.
"""

from __future__ import annotations

from tscollection.datasets.config.base import DatasetConfig
from tscollection.datasets.config.electricity import ELECTRICITY_LOAD
from tscollection.datasets.config.ett import (
    ETT_H1,
    ETT_H2,
    ETT_M1,
    ETT_M2,
)
from tscollection.datasets.config.ucr import (
    UCR_COFFEE,
    UCR_ECG200,
    UCR_FACE_FOUR,
)
from tscollection.datasets.config.uea import (
    UEA_ATRIAL_FIBRILLATION,
    UEA_BASIC_MOTIONS,
)
from tscollection.datasets.config.weather import WEATHER
from tscollection.datasets.enums.data import DatasetFamily

__all__ = ['CONFIGS', 'get_config', 'list_configs']

# ----------------------------------------------------------------------- #
# Registry                                                                 #
# ----------------------------------------------------------------------- #

_ALL_CONFIGS: tuple[DatasetConfig, ...] = (
    UCR_COFFEE,
    UCR_ECG200,
    UCR_FACE_FOUR,
    UEA_BASIC_MOTIONS,
    UEA_ATRIAL_FIBRILLATION,
    ETT_H1,
    ETT_H2,
    ETT_M1,
    ETT_M2,
    ELECTRICITY_LOAD,
    WEATHER,
)

CONFIGS: dict[str, DatasetConfig] = {cfg.name: cfg for cfg in _ALL_CONFIGS}


# ----------------------------------------------------------------------- #
# Lookup functions                                                         #
# ----------------------------------------------------------------------- #


def get_config(*, name: str) -> DatasetConfig:
    """Return the configuration for a dataset by name.

    Looks up the dataset name in the CONFIGS registry and returns the
    corresponding frozen Pydantic configuration instance.

    Args:
        name: Human-readable dataset name (e.g., 'Coffee', 'ETTh1').

    Returns:
        The frozen configuration instance matching the dataset name.

    Raises:
        KeyError: If no configuration exists for the given name.
    """
    try:
        return CONFIGS[name]
    except KeyError:
        raise KeyError(
            f'unknown dataset: {name!r}. '
            f'Available: {sorted(CONFIGS.keys())}'
        ) from None


def list_configs(*, family: DatasetFamily | None = None) -> list[DatasetConfig]:
    """Return all registered configurations, optionally filtered by family.

    When no family is specified, returns all configurations in the registry.
    When a family is given, returns only configurations matching that family.

    Args:
        family: Dataset family to filter by. If None, returns all configs.

    Returns:
        A list of frozen configuration instances matching the filter.
    """
    if family is None:
        return list(CONFIGS.values())
    return [cfg for cfg in CONFIGS.values() if cfg.family == family]
