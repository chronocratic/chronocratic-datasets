# Dataset API Reference

Dataset classes are defined in {py:mod}`chronocratic.datasets.datatypes` and
re-exported from the package root. They provide PyTorch `Dataset` implementations
for time series data.

## Base Datasets

Abstract and mixin classes that define the core interface for time series datasets.

- {py:class}`TimeSeriesDataset` — Root base class. Defines the standard `__getitem__`
  and `__len__` contract for all time series datasets.
- {py:class}`FixedTimeSeriesDatasetUnivariate` — For univariate series with a fixed
  length. Handles single-channel data with consistent indexing.
- {py:class}`FixedTimeSeriesDatasetMultivariate` — For multivariate series with
  fixed length. Supports per-variable scaling.
- {py:class}`FlexibleTimeSeriesDatasetSingleFile` — For datasets stored in a single
  file with variable window sizes.
- {py:class}`FlexibleTimeSeriesDatasetSingleFileMultipleSeries` — Extension for
  multi-series files where each series may have different lengths.

## Forecasting Datasets

Concrete dataset implementations for forecasting benchmarks.

- {py:class}`ETTDataset` — ETT dataset loader. Supports both ETTm1 and ETTm2 variants.
- {py:class}`WeatherDataset` — Weather dataset loader.
- {py:class}`ElectricityDataset` — Hourly electricity load dataset loader.

## Classification Datasets

Concrete dataset implementations for classification benchmarks.

- {py:class}`UCRClassificationUnivariateDataset` — UCR archive loader for univariate,
  fixed-length classification.
- {py:class}`UEAClassificationMultivariateDataset` — UEA archive loader for
  multivariate and variable-length classification.

```{eval-rst}
.. automodule:: chronocratic.datasets.datatypes
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
```
