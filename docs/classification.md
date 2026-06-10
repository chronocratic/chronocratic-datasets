# Classification Datasets

The classification module provides data loaders for the UCR/UEA Time Series
Classification Archive, a standard benchmark collection.

## Available Datasets

### UCR (Univariate)

The UCR archive contains univariate time series classification datasets.

```python
from pathlib import Path

from chronocratic.datasets import UCRClassificationDataModule

module = UCRClassificationDataModule(
    dataset_folder_path=Path("data/FogiDataset1"),
    target_column_name="class",
    scale_data=True,
    batch_size=32,
)

module.prepare_data()
module.setup()
```

### UEA (Multivariate)

The UEA archive contains multivariate time series classification datasets.

```python
from pathlib import Path

from chronocratic.datasets import UEAClassificationDataModule

module = UEAClassificationDataModule(
    dataset_folder_path=Path("data/ArrowHead"),
    target_column_name="class",
    scale_data=True,
    batch_size=32,
)

module.prepare_data()
module.setup()
```

## ClassificationLoaderMode

Controls how classification samples are constructed:

- **SAMPLE_ONLY** — Returns only the input sample tensor (no labels)
- **SAMPLE_LABEL** — Returns the input sample tensor and its label

```python
from chronocratic.datasets import ClassificationLoaderMode

ClassificationLoaderMode.SAMPLE_ONLY
ClassificationLoaderMode.SAMPLE_LABEL
```

## Parameters

All classification data modules accept these common parameters:

| Parameter               | Type                              | Description                                               |
| ----------------------- | --------------------------------- | --------------------------------------------------------- |
| `dataset_folder_path`   | `Path`                            | Path to the dataset ARFF directory (required)             |
| `target_column_name`    | `str`                             | Name of the target/label column in the ARFF files (required) |
| `batch_size`            | `int`                             | Batch size for dataloaders (default: 32)                  |
| `valid_size`            | `float`                           | Validation fraction from training data (default: 0.1)     |
| `shuffle`               | `bool`                            | Whether to shuffle training data (default: False)         |
| `scale_data`            | `bool`                            | Whether to apply data normalization (default: True)       |
| `data_scaling_method`   | `ScalingMethod`                   | Scaling algorithm: `NONE`, `MINMAX`, `STANDARD`           |
| `data_scaling_range`    | `tuple[float, float]`             | Target min-max range (default: `(0, 1)`)                  |
| `splitting_strategy`    | `ClassificationSplitMode`         | `AS_DEFINED` (use provided splits) or `MANUAL`            |
| `test_size`             | `float`                           | Test set fraction for `MANUAL` splitting (default: 0.5)   |
| `num_workers`           | `int`                             | Number of DataLoader workers (default: 0)                 |

## Dataset Classes

Under the hood, the data modules use these dataset classes:

- {py:class}`~chronocratic.datasets.datatypes.UCRClassificationUnivariateDataset`
- {py:class}`~chronocratic.datasets.datatypes.UEAClassificationMultivariateDataset`

See the {doc}`api/datatypes` reference for full class documentation.

## Data Splitting

The package provides flexible data splitting via {py:class}`~chronocratic.datasets.enums.ClassificationSplitMode`
and {py:class}`~chronocratic.datasets.enums.DataPartition`.

## Next Steps

- See the {doc}`forecasting` guide for forecasting datasets.
- See the {doc}`api/modules` reference for the full API of classification data modules.
