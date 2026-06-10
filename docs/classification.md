# Classification Datasets

The classification module provides data loaders for the UCR/UEA Time Series
Classification Archive, a standard benchmark collection for time series
classification research.

For full parameter reference, see the {doc}`api/modules` API documentation for
{py:mod}`chronocratic.datasets.modules`.

## UCR Classification Data Module

The UCR archive contains **univariate, equal-length** time series classification
datasets stored in ARFF format. Each dataset directory provides `TRAIN.arff`
and `TEST.arff` files with feature columns and a target label column.

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
train_loader = module.train_dataloader()
```

**Key details:**

- `dataset_folder_path` points to the directory containing the `.arff` files.
  The module auto-discovers `{dataset_name}_TRAIN.arff` and `{dataset_name}_TEST.arff`.
- `target_column_name` specifies the label column name in the ARFF files.
- Sequence length is derived from the number of feature columns.
- Handles variable-length series automatically via padding.

## UEA Classification Data Module

The UEA archive contains **multivariate and/or variable-length** time series
classification datasets stored in nested ARFF format. These datasets have
multiple dimensions per timestep and may have different sequence lengths per sample.

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
train_loader = module.train_dataloader()
```

**Key details:**

- Uses `scipy.io.arff.loadarff` directly for reading nested ARFF format.
- Automatically encodes string labels via `sklearn.preprocessing.LabelEncoder`.
- Data form is `NESTED`, meaning each sample may have variable length and multiple dimensions.
- Sequence length and feature count are derived from the data at load time.

## ClassificationLoaderMode

Controls how classification samples are constructed by DataLoaders:

- **SAMPLE_ONLY** -- Returns only the input sample tensor (no labels)
- **SAMPLE_LABEL** -- Returns the input sample tensor and its label

Set this on the `train_dataloader()`, `val_dataloader()`, and `test_dataloader()`
calls via the `mode` keyword argument. The default is `SAMPLE_LABEL`.

```python
from chronocratic.datasets import ClassificationLoaderMode

# With labels (default)
train_loader = module.train_dataloader()

# Without labels
train_loader = module.train_dataloader(mode=ClassificationLoaderMode.SAMPLE_ONLY)
```

## Dataset Classes

Under the hood, the data modules use these PyTorch Dataset classes:

- {py:class}`~chronocratic.datasets.datatypes.UCRClassificationUnivariateDataset`
- {py:class}`~chronocratic.datasets.datatypes.UEAClassificationMultivariateDataset`

See the {doc}`api/datatypes` reference for full class documentation.

## Data Splitting

The package provides flexible data splitting via
{py:class}`~chronocratic.datasets.enums.ClassificationSplitMode` and
{py:class}`~chronocratic.datasets.enums.DataPartition`.

Use `splitting_strategy=ClassificationSplitMode.AS_DEFINED` to keep the
original train/test split from the archive, or
`splitting_strategy=ClassificationSplitMode.MANUAL` to re-split the combined
data with a custom `test_size` fraction.

## Next Steps

- See the {doc}`forecasting` guide for forecasting datasets.
- See the {doc}`api/modules` reference for the full API of classification data modules.
