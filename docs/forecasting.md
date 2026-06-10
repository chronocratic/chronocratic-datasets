# Forecasting Modules

The forecasting module provides data loaders for popular multivariate time series
forecasting benchmarks, built on top of PyTorch Lightning's `LightningDataModule`.
Each module handles data loading, splitting, caching, and provides ready-to-use
DataLoaders for training, validation, and testing.

For the full API of all forecasting data modules, see the
{doc}`api/modules` reference.

## Weather Module

The Weather dataset contains 22 meteorological features recorded hourly over 7 years
(2012-2017). It uses fixed 60/20/20 temporal splits (train/validation/test).

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
train_loader = module.train_dataloader()
```

In **univariate mode**, only the last column (`WetBulbCelsius`) is retained as the target
variable. In **multivariate mode**, all 22 features are used.

See the {py:class}`~chronocratic.datasets.modules.WeatherModule` API reference for all
constructor parameters.

## ETT Data Module

The ETT (Electricity Transformer Temperature) dataset contains two frequency variants:

- **ETTh1, ETTh2** -- Hourly temperature data (17,420 timesteps)
- **ETTm1, ETTm2** -- 15-minute temperature data (69,680 timesteps)

The module requires an explicit `variant` parameter to determine the correct
16-month / 4-month / 4-month train/validation/test split boundaries.

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

In **univariate mode**, only the `OT` (outer temperature) column is used.

See the {py:class}`~chronocratic.datasets.modules.ETTDataModule` API reference for all
constructor parameters.

## Electricity Load Module

The Electricity dataset contains hourly power consumption data for 370 independent
customers over 3 years (2012-2014). Each customer is treated as a separate time
series (not as a feature), producing shape `(370, T, 1)` after transformation.
It uses fixed 60/20/20 temporal splits.

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

In **univariate mode**, only customer `MT_001` is retained.

See the {py:class}`~chronocratic.datasets.modules.ElectricityLoadModule` API reference
for all constructor parameters.

## Dataset Classes

Under the hood, the data modules use these PyTorch Dataset classes:

- {py:class}`~chronocratic.datasets.datatypes.ETTDataset`
- {py:class}`~chronocratic.datasets.datatypes.WeatherDataset`
- {py:class}`~chronocratic.datasets.datatypes.ElectricityDataset`

See the {doc}`api/datatypes` reference for full class documentation.

## Loader Mode

Forecasting modules support multiple loader modes via
{py:class}`~chronocratic.datasets.enums.ForecastingLoaderMode`:

- **RAW_SERIES** -- Returns the full raw time series (default)
- **INPUT_TARGET** -- Returns input and target tensors for supervised learning
- **INPUT_ONLY** -- Returns only the input tensor without targets

Set this on the `train_dataloader()`, `val_dataloader()`, and `test_dataloader()`
calls via the `loader_mode` keyword argument.

```python
# Supervised learning format: (input_window, target_window)
train_loader = module.train_dataloader(loader_mode=ForecastingLoaderMode.INPUT_TARGET)
```

## Forecasting Mode

Control variable selection with
{py:class}`~chronocratic.datasets.enums.ForecastingMode`:

- **UNIVARIATE** -- Use a single target variable per sample
- **MULTIVARIATE** -- Use all available variables per sample

Set this on the module constructor via the `mode` keyword argument.

## Scaling

Data scaling is configured via {py:class}`~chronocratic.datasets.enums.ScalingMethod`:

- **NONE** -- No scaling applied
- **MINMAX** -- Scales to a specified range (default 0-1)
- **STANDARD** -- Standardizes to zero mean and unit variance

## Next Steps

- See the {doc}`classification` guide for time series classification datasets.
- See the {doc}`api/modules` reference for the full API of forecasting data modules.
- See the {doc}`api/enums` reference for all enum options.
