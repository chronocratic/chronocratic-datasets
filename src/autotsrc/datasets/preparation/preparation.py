__all__ = [
    'get_electricity_load_dataset_lightning_module',
    'get_electricity_load_dataset_lightning_module_with_downstream_tasks',
    'get_ett_dataset_lightning_module',
    'get_ett_dataset_lightning_module_with_downstream_tasks',
    'get_ucr_classification_univariate_datasets_lightning_modules',
    'get_ucr_classification_univariate_datasets_lightning_modules_with_downstream_tasks',
    'get_uea_classification_multivariate_datasets_lightning_modules_with_downstream_tasks',
    'get_weather_dataset_lightning_module',
    'get_weather_dataset_lightning_module_with_downstream_tasks',
]

from functools import partial
import logging
from typing import NamedTuple

from src.autotsrc.datasets.modules import (
    ElectricityLoadDataModule,
    ETTDataModule,
    UCRTimeSeriesClassificationUnivariateDataModule,
    UEATimeSeriesClassificationMultivariateDataModule,
    WeatherDataModule,
)
from src.autotsrc.datasets.modules.abstract import BaseTimeSeriesDataModule

logger = logging.getLogger(__name__)


class DataModuleWithDownstreamTasks(NamedTuple):
    """A data module bundled with the downstream tasks it should serve."""

    data_module: BaseTimeSeriesDataModule
    downstream_tasks: tuple[str, ...]


def get_dataset_lightning_module[DataModuleType: BaseTimeSeriesDataModule](
    dataset_name: str,
    data_module_class: type[DataModuleType],
    data_module_params: dict,
    *,
    execute_setup: bool = True,
) -> DataModuleType:
    """
    Get initialized LightningDataModules for datasets.

    Args:
        dataset_name: the dataset to get LightningDataModules for
        data_module_class: the LightningDataModule class to use
        data_module_params: dictionary with parameters for the LightningDataModules
        execute_setup: whether to execute the setup method of the LightningDataModules

    Returns:
        tuple of LightningDataModules
    """
    logger.debug('Getting LightningDataModule for dataset %s', dataset_name)
    dataset_module = data_module_class(**data_module_params)

    dataset_module.prepare_data()

    if execute_setup:
        dataset_module.setup()

    return dataset_module


def get_datasets_lightning_module_with_downstream_tasks[DataModuleType: BaseTimeSeriesDataModule](
    dataset_name: str,
    data_module_class: type[DataModuleType],
    data_module_params: dict,
    downstream_tasks: str | list[str] | tuple[str, ...],
    *,
    execute_setup: bool = True,
) -> DataModuleWithDownstreamTasks:
    """
    Get initialized LightningDataModules for datasets with corresponding downstream tasks.

    Args:
        dataset_name: the datasets to get LightningDataModules for
        data_module_class: the LightningDataModule class to use
        data_module_params: dictionary with parameters for the LightningDataModules
        downstream_tasks: the downstream tasks to associate with the datasets
        execute_setup: whether to execute the setup method of the LightningDataModules

    Returns:
        list of LightningDataModules
    """
    if isinstance(downstream_tasks, str):
        downstream_tasks = (downstream_tasks,)
    if isinstance(downstream_tasks, list):
        downstream_tasks = tuple(downstream_tasks)

    lightning_data_module = get_dataset_lightning_module(
        dataset_name=dataset_name,
        data_module_class=data_module_class,
        data_module_params=data_module_params,
        execute_setup=execute_setup,
    )

    return DataModuleWithDownstreamTasks(
        data_module=lightning_data_module, downstream_tasks=downstream_tasks
    )


# ------- Partial functions ------- #

get_ucr_classification_univariate_datasets_lightning_modules = partial(
    get_dataset_lightning_module, data_module_class=UCRTimeSeriesClassificationUnivariateDataModule
)

get_electricity_load_dataset_lightning_module = partial(
    get_dataset_lightning_module, data_module_class=ElectricityLoadDataModule
)

get_ett_dataset_lightning_module = partial(
    get_dataset_lightning_module, data_module_class=ETTDataModule
)

get_weather_dataset_lightning_module = partial(
    get_dataset_lightning_module, data_module_class=WeatherDataModule
)

get_ucr_classification_univariate_datasets_lightning_modules_with_downstream_tasks = partial(
    get_datasets_lightning_module_with_downstream_tasks,
    data_module_class=UCRTimeSeriesClassificationUnivariateDataModule,
    downstream_tasks='classification',
)

get_uea_classification_multivariate_datasets_lightning_modules_with_downstream_tasks = partial(
    get_datasets_lightning_module_with_downstream_tasks,
    data_module_class=UEATimeSeriesClassificationMultivariateDataModule,
    downstream_tasks='classification',
)

get_electricity_load_dataset_lightning_module_with_downstream_tasks = partial(
    get_datasets_lightning_module_with_downstream_tasks,
    data_module_class=ElectricityLoadDataModule,
    downstream_tasks='forecasting',
)

get_ett_dataset_lightning_module_with_downstream_tasks = partial(
    get_datasets_lightning_module_with_downstream_tasks,
    data_module_class=ETTDataModule,
    downstream_tasks='forecasting',
)

get_weather_dataset_lightning_module_with_downstream_tasks = partial(
    get_datasets_lightning_module_with_downstream_tasks,
    data_module_class=WeatherDataModule,
    downstream_tasks='forecasting',
)
