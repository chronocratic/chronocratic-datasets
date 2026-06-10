# Classification Datasets

The classification module provides data loaders for the UCR/UEA Time Series
Classification Archive, a standard benchmark collection for time series
classification research.

For the full API of all classification data modules, see the
{doc}`api/modules` reference.

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

See the {py:class}`~chronocratic.datasets.modules.UCRClassificationDataModule` API
reference for all constructor parameters.

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

See the {py:class}`~chronocratic.datasets.modules.UEAClassificationDataModule` API
reference for all constructor parameters.

## Dataset Classes

Under the hood, the data modules use these PyTorch Dataset classes:

- {py:class}`~chronocratic.datasets.datatypes.UCRClassificationUnivariateDataset`
- {py:class}`~chronocratic.datasets.datatypes.UEAClassificationMultivariateDataset`

See the {doc}`api/datatypes` reference for full class documentation.

## Loader Mode

Classification modules support multiple loader modes via
{py:class}`~chronocratic.datasets.enums.ClassificationLoaderMode`:

- **SAMPLE_ONLY** -- Returns only the input sample tensor (no labels)
- **SAMPLE_LABEL** -- Returns the input sample tensor and its label (default)

Set this on the `train_dataloader()`, `val_dataloader()`, and `test_dataloader()`
calls via the `mode` keyword argument.

```python
# Without labels
train_loader = module.train_dataloader(mode=ClassificationLoaderMode.SAMPLE_ONLY)
```

## Splitting Strategy

Control how the archive's train/test split is handled via
{py:class}`~chronocratic.datasets.enums.ClassificationSplitMode`:

- **AS_DEFINED** -- Keep the original train/test split from the archive
- **MANUAL** -- Re-split the combined data with a custom `test_size` fraction

Set this on the module constructor via the `splitting_strategy` keyword argument.

## Scaling

Data scaling is configured via {py:class}`~chronocratic.datasets.enums.ScalingMethod`:

- **NONE** -- No scaling applied
- **MINMAX** -- Scales to a specified range (default 0-1)
- **STANDARD** -- Standardizes to zero mean and unit variance

## Next Steps

- See the {doc}`forecasting` guide for forecasting datasets.
- See the {doc}`api/modules` reference for the full API of classification data modules.
- See the {doc}`api/enums` reference for all enum options.
