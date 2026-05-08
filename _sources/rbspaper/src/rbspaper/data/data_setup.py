"""Data setup: resolves dataset names to datamodules with downstream tasks.

Acts as the single entry point for datamodule creation. Uses the
static registry (`rbspaper.data.registry`) as the source of truth
for dataset families.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.rbspaper.data.preparation import (
    get_ett_datamodule_with_tasks,
    get_ucr_classification_univariate_datamodule_with_tasks,
    get_uea_classification_multivariate_datamodule_with_tasks,
)
from src.rbspaper.data.registry import DATASET_REGISTRY, get_dataset_metadata, list_dataset_names
from src.rbspaper.enums.data_enums import (
    ForecastingTimeSeriesDatasetMode,
    TimeSeriesClassificationDatasetSplittingStrategy,
)

__all__ = [
    'get_all_datasets',
    'get_classification_datasets',
    'get_datamodule_with_downstream_tasks',
    'get_datasets_names',
    'get_forecasting_datasets',
]


# ---------------------------------------------------------------------------
# Dataset family helpers
# ---------------------------------------------------------------------------

_FAMILY_TO_TASK_MAP: dict[str, str] = {
    'ucr': 'classification',
    'uea': 'classification',
    'ett': 'forecasting',
    'electricity': 'forecasting',
    'weather': 'forecasting',
    'exchange': 'forecasting',
    'traffic': 'forecasting',
    'illness': 'forecasting',
}

CLASSIFICATION_FAMILIES = frozenset({'ucr', 'uea'})
FORECASTING_FAMILIES = frozenset(
    {'ett', 'electricity', 'weather', 'exchange', 'traffic', 'illness'}
)


def get_classification_datasets() -> list[str]:
    """Return all classification dataset names from the registry."""
    return [d.name for d in DATASET_REGISTRY if d.family in CLASSIFICATION_FAMILIES]


def get_forecasting_datasets() -> list[str]:
    """Return all forecasting dataset names from the registry."""
    return [d.name for d in DATASET_REGISTRY if d.family in FORECASTING_FAMILIES]


def get_all_datasets(form: str = 'list') -> set[str] | list[str]:
    """Return every dataset name from the registry.

    Args:
        form: Return type - 'set' or 'list'.

    Returns:
        Collection of dataset names.
    """
    names = list_dataset_names()
    if form == 'set':
        return set(names)
    return names


def get_datasets_names(family: str, form: str = 'set') -> set[str] | list[str]:
    """Return dataset names for a specific family.

    Args:
        family: Family identifier (e.g. 'ucr', 'ett').
        form: Return type - 'set' or 'list'.

    Returns:
        Collection of dataset names.
    """
    from src.rbspaper.data.registry import list_datasets_for_families

    datasets = list_datasets_for_families(families=(family,))
    names = [d.name for d in datasets]
    if form == 'set':
        return set(names)
    return names


# ---------------------------------------------------------------------------
# Datamodule parameter builders
# ---------------------------------------------------------------------------


def _get_global_datamodule_defaults(num_workers: int = 0) -> dict[str, Any]:
    """Shared defaults for all datamodules."""
    return {
        'scale_data': True,
        'data_scaling_method': 'min_max',
        'data_scaling_range': (0, 1),
        'valid_size': 0.25,
        'test_size': 0.5,
        'shuffle': True,
        'num_workers': num_workers,
    }


def _get_classification_datamodule_params(
    *, dataset_config_path: Path, num_workers: int = 0, batch_size: int = 16
) -> dict[str, Any]:
    """Build parameter dict for classification datamodules."""
    return {
        'dataset_config_path': dataset_config_path,
        'splitting_strategy': TimeSeriesClassificationDatasetSplittingStrategy.MANUAL,
        'batch_size': batch_size,
        **_get_global_datamodule_defaults(num_workers),
    }


def _get_forecasting_datamodule_params(
    *, num_workers: int = 0, batch_size: int = 1
) -> dict[str, Any]:
    """Build parameter dict for forecasting datamodules."""
    return {'batch_size': batch_size, **_get_global_datamodule_defaults(num_workers)}


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _get_dataset_folder_path(dataset_name: str, data_root: Path) -> Path:
    """Resolve the filesystem path for a dataset.

    Conventions:
    - UCR:  data_root / ucr_classification_univariate / {name}
    - UEA:  data_root / uea_classification_multivariate / {name}
    - ETT:  data_root / ett
    - Other: data_root / {name}

    Args:
        dataset_name: Registered dataset name.
        data_root: Root data directory.

    Returns:
        Path to the dataset directory or file.
    """
    meta = get_dataset_metadata(dataset_name=dataset_name)
    family = meta.family

    if family == 'ucr':
        return data_root / 'ucr_classification_univariate' / dataset_name
    if family == 'uea':
        return data_root / 'uea_classification_multivariate' / dataset_name
    if family == 'ett':
        return data_root / 'ett'
    # electricity, weather, exchange, traffic, illness
    return data_root / family


def _get_datamodule_path_params(dataset_name: str, data_root: Path) -> dict[str, Path]:
    """Build the path-specific kwargs for a datamodule constructor.

    Classification datamodules expect `dataset_folder_path`.
    Forecasting datamodules expect `dataset_file_path`.

    Args:
        dataset_name: Registered dataset name.
        data_root: Root data directory.

    Returns:
        Dictionary with the appropriate path key.
    """
    meta = get_dataset_metadata(dataset_name=dataset_name)
    family = meta.family

    if family in CLASSIFICATION_FAMILIES:
        return {'dataset_folder_path': _get_dataset_folder_path(dataset_name, data_root)}

    folder = _get_dataset_folder_path(dataset_name, data_root)

    # File name conventions
    if family == 'ett':
        return {'dataset_file_path': folder / f'{dataset_name}.csv'}
    if family == 'electricity':
        return {'dataset_file_path': folder / 'LD2011_2014.txt'}
    if family == 'weather':
        return {'dataset_file_path': folder / f'{dataset_name}.csv'}
    # exchange, traffic, illness — generic CSV naming
    return {'dataset_file_path': folder / f'{dataset_name}.csv'}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def get_datamodule_with_downstream_tasks(
    dataset_name: str,
    data_root: Path,
    num_workers: int = 0,
    extra_params: dict[str, Any] | None = None,
) -> Any:
    """Factory: create a datamodule for any registered dataset.

    Resolves the dataset family, assembles parameters, and returns
    a (data_module, downstream_tasks) named tuple.

    Args:
        dataset_name: Registered dataset name.
        data_root: Root data directory on disk.
        num_workers: DataLoader worker count.
        extra_params: Override dict (e.g. {'forecasting_mode': 'univariate'}).

    Returns:
        DataModuleWithDownstreamTasks named tuple.

    Raises:
        KeyError: If dataset_name is not in the registry.
        NotImplementedError: If the family has no datamodule yet.
    """
    meta = get_dataset_metadata(dataset_name=dataset_name)
    family = meta.family
    extra_params = extra_params or {}

    if family == 'ucr':
        config_path = (
            data_root / 'configurations' / 'datasets' / 'ucr_classification_univariate_config.json'
        )
        module_params = _get_classification_datamodule_params(
            dataset_config_path=config_path, num_workers=num_workers
        )
        path_params = _get_datamodule_path_params(dataset_name, data_root)
        module_params.update(path_params)

        return get_ucr_classification_univariate_datamodule_with_tasks(
            dataset_name=dataset_name, data_module_params=module_params
        )

    if family == 'uea':
        config_path = (
            data_root
            / 'configurations'
            / 'datasets'
            / 'uea_classification_multivariate_config.json'
        )
        module_params = _get_classification_datamodule_params(
            dataset_config_path=config_path, num_workers=num_workers
        )
        path_params = _get_datamodule_path_params(dataset_name, data_root)
        module_params.update(path_params)

        return get_uea_classification_multivariate_datamodule_with_tasks(
            dataset_name=dataset_name, data_module_params=module_params
        )

    if family in FORECASTING_FAMILIES:
        module_params = _get_forecasting_datamodule_params(num_workers=num_workers)
        path_params = _get_datamodule_path_params(dataset_name, data_root)
        module_params.update(path_params)

        # Process forecasting mode override
        if extra_params.get('forecasting_mode'):
            mode_str = extra_params['forecasting_mode']
            module_params['mode'] = ForecastingTimeSeriesDatasetMode(mode_str)

        logging.debug('datamodule_params: %s', module_params)

        if family == 'ett':
            return get_ett_datamodule_with_tasks(
                dataset_name=dataset_name, data_module_params=module_params
            )

        raise NotImplementedError(
            f"Datamodule for family '{family}' (dataset '{dataset_name}') "
            'not yet implemented. You can add it by importing the appropriate '
            'partial from rbspaper.data.preparation.'
        )

    raise NotImplementedError(f"No datamodule for family '{family}' (dataset '{dataset_name}')")
