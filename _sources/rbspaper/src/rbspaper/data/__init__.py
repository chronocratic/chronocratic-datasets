"""Data infrastructure for RBSPaper.

Submodules:
- data.registry: Static dataset metadata.
- data.datasets: PyTorch Dataset classes.
- data.modules: LightningDataModule classes.
- data.utils: Shared data utilities.
- data.preparation: Factory functions.
- data.data_setup: Dataset-to-datamodule resolution.
"""

from src.rbspaper.data.data_setup import (
    get_all_datasets,
    get_classification_datasets,
    get_datamodule_with_downstream_tasks,
    get_datasets_names,
    get_forecasting_datasets,
)
from src.rbspaper.data.preparation import (
    DataModuleWithDownstreamTasks,
    get_dataset_lightning_module,
    get_datasets_lightning_module_with_downstream_tasks,
)
from src.rbspaper.data.registry import (
    DATASET_REGISTRY,
    DatasetMetadata,
    get_dataset_metadata,
    list_dataset_names,
    list_datasets_for_families,
)

__all__ = [
    # registry
    'DATASET_REGISTRY',
    # preparation
    'DataModuleWithDownstreamTasks',
    'DatasetMetadata',
    'get_all_datasets',
    'get_classification_datasets',
    # data_setup
    'get_datamodule_with_downstream_tasks',
    'get_dataset_lightning_module',
    'get_dataset_metadata',
    'get_datasets_lightning_module_with_downstream_tasks',
    'get_datasets_names',
    'get_forecasting_datasets',
    'list_dataset_names',
    'list_datasets_for_families',
]
