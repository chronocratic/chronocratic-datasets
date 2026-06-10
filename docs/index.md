# chronocratic-datasets

Ready-to-use time series datasets for PyTorch Lightning.

**chronocratic-datasets** provides pre-built, Lightning-compatible data modules for popular
time series benchmarks. It covers forecasting (ETT, Weather, Electricity) and classification
(UCR, UEA) with caching, scaling, and distributed-data-parallel (DDP) compliance out of the box.

## Features

- **Forecasting datasets:** ETT, Weather, Electricity with sliding-window and raw-series modes
- **Classification datasets:** UCR (univariate) and UEA (multivariate) benchmarks
- **Lightning integration:** Ready-to-use `LightningDataModule` subclasses
- **Caching:** Automatic dataset caching for fast reloads
- **Scaling:** Configurable data normalization via scikit-learn scalers
- **DDP compliant:** Works seamlessly with `DataParallel` and `Strategy("ddp")`

## Quick Install

```bash
pip install chronocratic-datasets
```

## Documentation Contents

```{toctree}
:maxdepth: 3
:caption: Guides

quickstart
forecasting
classification
```

```{toctree}
:maxdepth: 3
:caption: API Reference

api/enums
api/datatypes
api/modules
api/utils
```

```{toctree}
:maxdepth: 1
:caption: Project

changelog
contributing
```
