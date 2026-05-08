"""Factory functions for creating LightningDataModules with downstream tasks.

Provides generic instantiation helpers and partial-function shortcuts
for each dataset family.
"""

from __future__ import annotations

from collections import namedtuple
from functools import partial
from typing import TypeVar

from src.rbspaper.data.modules import (
    ElectricityLoadDataModule,
    ETTDataModule,
    UCRTimeSeriesClassificationUnivariateDataModule,
    UEATimeSeriesClassificationMultivariateDataModule,
    WeatherDataModule,
)
from src.rbspaper.data.modules.abstract import BaseTimeSeriesDataModule

__all__ = [
    'DataModuleWithDownstreamTasks',
    'get_dataset_lightning_module',
    'get_datasets_lightning_module_with_downstream_tasks',
    'get_electricity_load_datamodule_with_tasks',
    'get_ett_datamodule_with_tasks',
    # UCR partials
    'get_ucr_classification_univariate_datamodule_with_tasks',
    'get_uea_classification_multivariate_datamodule_with_tasks',
    'get_weather_datamodule_with_tasks',
]

DataModuleWithDownstreamTasks = namedtuple(
    'DataModuleWithDownstreamTasks', ['data_module', 'downstream_tasks']
)

DataModuleType = TypeVar('DataModuleType', bound=BaseTimeSeriesDataModule)


def get_dataset_lightning_module(
    *,
    dataset_name: str,
    data_module_class: type[DataModuleType],
    data_module_params: dict,
    execute_setup: bool = True,
) -> DataModuleType:
    """Instantiate a single LightningDataModule.

    Args:
        dataset_name: Dataset identifier (for logging).
        data_module_class: DataModule class to instantiate.
        data_module_params: Keyword arguments for the class constructor.
        execute_setup: Whether to call setup() after prepare_data().

    Returns:
        Configured LightningDataModule instance.
    """
    data_module = data_module_class(**data_module_params)
    data_module.prepare_data()

    if execute_setup:
        data_module.setup()

    return data_module


def get_datasets_lightning_module_with_downstream_tasks(
    *,
    dataset_name: str,
    data_module_class: type[DataModuleType],
    data_module_params: dict,
    downstream_tasks: str | list[str] | tuple[str, ...],
    execute_setup: bool = True,
) -> DataModuleWithDownstreamTasks:
    """Create a datamodule paired with its downstream task identifiers.

    Args:
        dataset_name: Dataset identifier.
        data_module_class: DataModule class to instantiate.
        data_module_params: Constructor kwargs.
        downstream_tasks: One or more task names (e.g. 'classification').
        execute_setup: Whether to call setup().

    Returns:
        Named tuple (data_module, downstream_tasks).
    """
    if isinstance(downstream_tasks, str):
        downstream_tasks = (downstream_tasks,)
    if isinstance(downstream_tasks, list):
        downstream_tasks = tuple(downstream_tasks)

    module = get_dataset_lightning_module(
        dataset_name=dataset_name,
        data_module_class=data_module_class,
        data_module_params=data_module_params,
        execute_setup=execute_setup,
    )
    return DataModuleWithDownstreamTasks(data_module=module, downstream_tasks=downstream_tasks)


# -- Partial-function shortcuts for each dataset family --

get_ucr_classification_univariate_datamodule_with_tasks = partial(
    get_datasets_lightning_module_with_downstream_tasks,
    data_module_class=UCRTimeSeriesClassificationUnivariateDataModule,
    downstream_tasks='classification',
)

get_uea_classification_multivariate_datamodule_with_tasks = partial(
    get_datasets_lightning_module_with_downstream_tasks,
    data_module_class=UEATimeSeriesClassificationMultivariateDataModule,
    downstream_tasks='classification',
)

get_electricity_load_datamodule_with_tasks = partial(
    get_datasets_lightning_module_with_downstream_tasks,
    data_module_class=ElectricityLoadDataModule,
    downstream_tasks='forecasting',
)

get_ett_datamodule_with_tasks = partial(
    get_datasets_lightning_module_with_downstream_tasks,
    data_module_class=ETTDataModule,
    downstream_tasks='forecasting',
)

get_weather_datamodule_with_tasks = partial(
    get_datasets_lightning_module_with_downstream_tasks,
    data_module_class=WeatherDataModule,
    downstream_tasks='forecasting',
)
