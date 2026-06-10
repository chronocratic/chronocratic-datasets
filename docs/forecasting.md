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
from pathlib import Path

from chronocratic.datasets import ETTDataModule, ForecastingMode

module = ETTDataModule(
    dataset_file_path=Path("data/ETTm1.csv"),
    variant="ETTm1",
    mode=ForecastingMode.UNIVARIATE,
    seq_len=96,
    forecast_horizon=96,
    scale_data=True,
    batch_size=32,
)

module.prepare_data()
module.setup()
```

### Weather

The Weather dataset contains 22 features recorded hourly over 7 years.

```python
from pathlib import Path

from chronocratic.datasets import ForecastingMode, WeatherModule

module = WeatherModule(
    dataset_file_path=Path("data/weather.csv"),
    mode=ForecastingMode.MULTIVARIATE,
    seq_len=24,
    forecast_horizon=168,
    scale_data=True,
    batch_size=32,
)

module.prepare_data()
module.setup()
```

### Electricity

The Electricity dataset contains hourly power consumption data for 321 customers
over 4 years.

```python
from pathlib import Path

from chronocratic.datasets import ElectricityLoadModule, ForecastingMode

module = ElectricityLoadModule(
    dataset_file_path=Path("data/electricity.csv"),
    mode=ForecastingMode.MULTIVARIATE,
    seq_len=96,
    forecast_horizon=24,
    scale_data=True,
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

- **RAW_SERIES** — Returns the full raw time series
- **INPUT_TARGET** — Returns input and target tensors for supervised learning
- **INPUT_ONLY** — Returns only the input tensor without targets

## Parameters

All forecasting data modules accept these common parameters:

| Parameter               | Type                    | Description                                               |
| ----------------------- | ----------------------- | --------------------------------------------------------- |
| `dataset_file_path`     | `Path`                  | Path to the dataset file (required)                       |
| `variant`               | `str`                   | ETT variant: `"ETTh1"`, `"ETTh2"`, `"ETTm1"`, `"ETTm2"`   |
| `mode`                  | `ForecastingMode`       | Input mode (`UNIVARIATE` or `MULTIVARIATE`)               |
| `seq_len`               | `int`                   | Sequence length for the input window (default: 128)       |
| `forecast_horizon`      | `int`                   | Prediction horizon length (default: 96; 24 for Electricity) |
| `scale_data`            | `bool`                  | Whether to apply data normalization (default: True)       |
| `data_scaling_method`   | `ScalingMethod`         | Scaling algorithm: `NONE`, `MINMAX`, `STANDARD`           |
| `data_scaling_range`    | `tuple[float, float]`   | Target min-max range (default: `(0, 1)`)                  |
| `batch_size`            | `int`                   | Batch size for dataloaders (default: 32)                  |
| `valid_size`            | `float`                 | Validation fraction (default: 0.1)                        |
| `test_size`             | `float`                 | Test fraction (default: 0.5)                              |
| `shuffle`               | `bool`                  | Whether to shuffle training data (default: False)         |
| `num_workers`           | `int`                   | Number of DataLoader workers (default: 0)                 |
| `step`                  | `int \| None`           | Stride between consecutive windows (default: None)        |

## Dataset Classes

Under the hood, the data modules use these dataset classes:

- {py:class}`~chronocratic.datasets.datatypes.ETTDataset`
- {py:class}`~chronocratic.datasets.datatypes.WeatherDataset`
- {py:class}`~chronocratic.datasets.datatypes.ElectricityDataset`

See the {doc}`api/datatypes` reference for full class documentation.

## Next Steps

- See the {doc}`classification` guide for time series classification datasets.
- See the {doc}`api/modules` reference for the full API of forecasting data modules.
