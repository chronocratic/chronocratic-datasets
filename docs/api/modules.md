# DataModule API Reference

DataModule classes are defined in {py:mod}`chronocratic.datasets.modules` and
re-exported from the package root. They provide PyTorch Lightning
`LightningDataModule` implementations for time series datasets.

## Forecasting Modules

Forecasting modules share a common interface centered around windowing (`seq_len`,
`forecast_horizon`), mode selection, and data scaling.

- {py:class}`WeatherModule` — Weather dataset. Parameters: `dataset_file_path`,
  `seq_len` (128), `mode`, `forecast_horizon` (96), `step`.
- {py:class}`ETTDataModule` — ETT dataset (ETTm1, ETTm2). Parameters:
  `dataset_file_path`, `variant` (required, e.g., `"ETTm1"`), `seq_len` (128),
  `mode`, `forecast_horizon` (96), `step`.
- {py:class}`ElectricityLoadModule` — Hourly electricity load data. Parameters:
  `dataset_file_path`, `seq_len` (128), `mode`, `forecast_horizon` (24), `step`.

**Common parameters:** `batch_size`, `valid_size`, `test_size`, `shuffle`,
`scale_data`, `data_scaling_method`, `data_scaling_range`, `num_workers`.

## Classification Modules

Classification modules deal with labeled time series (univariate or multivariate).
They require a dataset folder path and the name of the target column, plus
a splitting strategy.

- {py:class}`UCRClassificationDataModule` — UCR archive (univariate, equal-length
  series). Parameters: `dataset_folder_path`, `target_column_name`.
- {py:class}`UEAClassificationDataModule` — UEA archive (multivariate and/or
  variable-length series). Parameters: `dataset_folder_path`, `target_column_name`.

**Common parameters:** `batch_size`, `valid_size`, `shuffle`, `scale_data`,
`data_scaling_method`, `data_scaling_range`, `splitting_strategy`, `test_size`,
`num_workers`.

## Base Modules

Base classes provide shared functionality and are not meant to be instantiated
directly.

- {py:class}`BaseTimeSeriesDataModule` — Common foundation for all data modules.
  Handles caching, DDP-safe data loading, and standard Lightning lifecycle hooks.
- {py:class}`BaseForecastingTimeSeriesDataModule` — Extends the base with
  sliding-window configuration, scaling pipelines, and forecast-specific splits.
- {py:class}`BaseClassificationTimeSeriesDataModule` — Extends the base with
  label handling and classification-specific data transforms.

```{eval-rst}
.. automodule:: chronocratic.datasets.modules
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
```
