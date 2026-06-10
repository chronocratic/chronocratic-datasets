# Forecasting Datasets

The forecasting module provides data loaders for popular multivariate time series
forecasting benchmarks, built on top of PyTorch Lightning's `LightningDataModule`.

## Available Datasets

### ETT (Electricity Transformer Temperature)

The ETT dataset contains two sub-datasets:

- **ETTm1** — Temperature data at 15-minute intervals
- **ETTm2** — Temperature data at 15-minute intervals (different sensor)
- **ETTh1** — Temperature data at hourly intervals
- **ETTh2** — Temperature data at hourly intervals (different sensor)

```python
from chronocratic.datasets import ForecastingMode, ETTDataModule

module = ETTDataModule(
    mode=ForecastingMode.UNIVARIATE,
    dataset_name="ETTm1",
    seq_len=96,
    label_len=48,
    pred_len=96,
    scale=True,
    batch_size=32,
)

module.prepare_data()
module.setup()
```

### Weather

The Weather dataset contains 21 features recorded every 10 minutes over 7 years.

```python
from chronocratic.datasets import ForecastingMode, WeatherModule

module = WeatherModule(
    mode=ForecastingMode.MULTIVARIATE,
    seq_len=24,
    label_len=12,
    pred_len=168,
    scale=True,
    batch_size=32,
)

module.prepare_data()
module.setup()
```

### Electricity

The Electricity dataset contains hourly power consumption data for 321 customers
over 4 years.

```python
from chronocratic.datasets import ForecastingMode, ElectricityLoadModule

module = ElectricityLoadModule(
    mode=ForecastingMode.MULTIVARIATE,
    seq_len=96,
    label_len=48,
    pred_len=96,
    scale=True,
    batch_size=32,
)

module.prepare_data()
module.setup()
```

## ForecastingMode

Controls how the input sequence is constructed:

- **UNIVARIATE** — Use a single target variable per sample
- **MULTIVARIATE** — Use all available variables per sample

```python
from chronocratic.datasets import ForecastingMode

ForecastingMode.UNIVARIATE
ForecastingMode.MULTIVARIATE
```

## ForecastingLoaderMode

Controls how samples are returned from the dataset:

- **SAMPLE_ONLY** — Returns only the input sample tensor
- **SAMPLE_LABEL** — Returns the input sample and the label tensor
- **INPUT_OUTPUT** — Returns input and output tensors for supervised learning

## Parameters

All forecasting data modules accept these common parameters:

| Parameter    | Type     | Description                              |
| ------------ | -------- | ---------------------------------------- |
| `mode`       | `ForecastingMode` | Input mode (UNIVARIATE/MULTIVARIATE) |
| `seq_len`    | `int`    | Sequence length for the input window     |
| `label_len`  | `int`    | Length of the label window               |
| `pred_len`   | `int`    | Prediction horizon length                |
| `scale`      | `bool`   | Whether to apply data normalization      |
| `batch_size` | `int`    | Batch size for dataloaders (default: 32) |
| `shuffle`    | `bool`   | Whether to shuffle training data         |
| `num_workers`| `int`    | Number of DataLoader workers             |

## Dataset Classes

Under the hood, the data modules use these dataset classes:

- {py:class}`~chronocratic.datasets.datatypes.ETTDataset`
- {py:class}`~chronocratic.datasets.datatypes.WeatherDataset`
- {py:class}`~chronocratic.datasets.datatypes.ElectricityDataset`

See the {doc}`api/datatypes` reference for full class documentation.

## Next Steps

- See the {doc}`classification` guide for time series classification datasets.
- See the {doc}`api/modules` reference for the full API of forecasting data modules.
