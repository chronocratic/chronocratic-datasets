# DataModule API Reference

DataModule classes are defined in {py:mod}`chronocratic.datasets.modules` and
re-exported from the package root. They provide PyTorch Lightning
`LightningDataModule` implementations for time series datasets.

## Forecasting Modules

Forecasting modules share a common interface centered around windowing
(`seq_len`, `forecast_horizon`), mode selection, and data scaling.

## Classification Modules

Classification modules deal with labeled time series (univariate or multivariate).
They require a dataset folder path and the name of the target column, plus
a splitting strategy.

## Base Modules

Base classes provide shared functionality and are not meant to be instantiated
directly.

```{eval-rst}
.. automodule:: chronocratic.datasets.modules
   :members:
   :imported-members:
   :undoc-members:
   :show-inheritance:
```
