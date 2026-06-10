# Classification Datasets

The classification module provides data loaders for the UCR/UEA Time Series
Classification Archive, a standard benchmark collection.

## Available Datasets

### UCR (Univariate)

The UCR archive contains univariate time series classification datasets.

```python
from chronocratic.datasets import UCRClassificationDataModule

module = UCRClassificationDataModule(
    dataset_name="FogiDataset1",
    scale=True,
    batch_size=32,
)

module.prepare_data()
module.setup()
```

### UEA (Multivariate)

The UEA archive contains multivariate time series classification datasets.

```python
from chronocratic.datasets import UEAClassificationDataModule

module = UEAClassificationDataModule(
    dataset_name="ArrowHead",
    scale=True,
    batch_size=32,
)

module.prepare_data()
module.setup()
```

## ClassificationLoaderMode

Controls how classification samples are constructed:

- **FULL_SERIES** — Each sample is the entire time series
- **SLIDING_WINDOW** — Samples are extracted using a sliding window over the series

```python
from chronocratic.datasets import ClassificationLoaderMode

ClassificationLoaderMode.FULL_SERIES
ClassificationLoaderMode.SLIDING_WINDOW
```

## Parameters

All classification data modules accept these common parameters:

| Parameter       | Type     | Description                              |
| --------------- | -------- | ---------------------------------------- |
| `dataset_name`  | `str`    | Name of the dataset from the UCR/UEA archive |
| `scale`         | `bool`   | Whether to apply data normalization      |
| `batch_size`    | `int`    | Batch size for dataloaders (default: 32) |
| `shuffle`       | `bool`   | Whether to shuffle training data         |
| `num_workers`   | `int`    | Number of DataLoader workers             |
| `train_split`   | `float`  | Fraction of data for training            |
| `val_split`     | `float`  | Fraction of data for validation          |
| `test_split`    | `float`  | Fraction of data for testing             |

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
