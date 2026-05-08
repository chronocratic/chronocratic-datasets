"""Helpers to build dataset task profiles from the registry."""

from __future__ import annotations

from src.rbspaper.data.registry import get_dataset_metadata
from src.rbspaper.enums.general import TimeSeriesDownstreamTask
from src.rbspaper.pipeline.config import DatasetTaskProfile

_TASK_MAP: dict[str, TimeSeriesDownstreamTask] = {
    'classification': TimeSeriesDownstreamTask.CLASSIFICATION,
    'forecasting': TimeSeriesDownstreamTask.FORECASTING,
    'clustering': TimeSeriesDownstreamTask.CLUSTERING,
}


def build_dataset_task_profile(*, dataset_name: str) -> DatasetTaskProfile | None:
    """Build a task profile from registry metadata.

    Reads the dataset metadata from the static registry and maps the
    registered task strings to enum values.

    Args:
        dataset_name: Registered dataset name.

    Returns:
        A dataset task profile, or None if the dataset has no
        non-representation tasks (e.g., unknown task strings).

    Raises:
        KeyError: If the dataset is not in the registry.
        ValueError: If a registered task string is unrecognized.
    """
    meta = get_dataset_metadata(dataset_name=dataset_name)

    # Filter out "representation" — it's not a downstream task, just a flag
    downstream_tasks = tuple(t for t in meta.tasks if t != 'representation')

    if not downstream_tasks:
        return None

    primary = _TASK_MAP[downstream_tasks[0]]
    allowed = frozenset(_TASK_MAP[t] for t in downstream_tasks)

    return DatasetTaskProfile(primary_task=primary, allowed_eval_tasks=allowed)
