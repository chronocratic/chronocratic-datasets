"""Static dataset registry used by initial benchmark scaffolding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetMetadata:
    """Dataset metadata used by experiment selection and validation."""

    name: str
    family: str
    tasks: tuple[str, ...]


_TASKS_CLASSIFICATION = ('classification', 'representation')
_TASKS_FORECASTING = ('forecasting', 'representation')

DATASET_REGISTRY: tuple[DatasetMetadata, ...] = (
    DatasetMetadata(name='Coffee', family='ucr', tasks=_TASKS_CLASSIFICATION),
    DatasetMetadata(name='ECG200', family='ucr', tasks=_TASKS_CLASSIFICATION),
    DatasetMetadata(name='FaceFour', family='ucr', tasks=_TASKS_CLASSIFICATION),
    DatasetMetadata(name='BasicMotions', family='uea', tasks=_TASKS_CLASSIFICATION),
    DatasetMetadata(name='AtrialFibrillation', family='uea', tasks=_TASKS_CLASSIFICATION),
    DatasetMetadata(name='ETTh1', family='ett', tasks=_TASKS_FORECASTING),
    DatasetMetadata(name='electricity', family='electricity', tasks=_TASKS_FORECASTING),
    DatasetMetadata(name='weather', family='weather', tasks=_TASKS_FORECASTING),
    DatasetMetadata(name='exchange_rate', family='exchange', tasks=_TASKS_FORECASTING),
    DatasetMetadata(name='traffic', family='traffic', tasks=_TASKS_FORECASTING),
    DatasetMetadata(name='illness', family='illness', tasks=_TASKS_FORECASTING),
)


def list_dataset_names(*, registry: tuple[DatasetMetadata, ...] = DATASET_REGISTRY) -> list[str]:
    """Return all dataset names from the registry."""
    return [dataset.name for dataset in registry]


def list_datasets_for_families(
    *, families: tuple[str, ...], registry: tuple[DatasetMetadata, ...] = DATASET_REGISTRY
) -> list[DatasetMetadata]:
    """Return datasets belonging to requested families.

    Args:
        families: Dataset families to keep.
        registry: Source registry.

    Returns:
        Ordered list of matching datasets.
    """
    allowed_families = set(families)
    return [dataset for dataset in registry if dataset.family in allowed_families]


def get_dataset_metadata(
    *, dataset_name: str, registry: tuple[DatasetMetadata, ...] = DATASET_REGISTRY
) -> DatasetMetadata:
    """Return dataset metadata for a dataset name."""
    for dataset in registry:
        if dataset.name == dataset_name:
            return dataset
    raise KeyError(f'unknown dataset: {dataset_name}')
