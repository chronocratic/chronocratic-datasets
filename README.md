# chronocratic-datasets

Ready-to-use time series datasets for PyTorch Lightning.

[![PyPI version](https://img.shields.io/pypi/v/chronocratic-datasets.svg)](https://pypi.org/project/chronocratic-datasets/)
[![Python 3.12+](https://img.shields.io/pypi/pyversions/chronocratic-datasets.svg)](https://pypi.org/project/chronocratic-datasets/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Build and Test](https://github.com/chronocratic/datasets/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/chronocratic/datasets/actions/workflows/build-and-test.yml)
[![Documentation Status](https://readthedocs.org/projects/chronocratic-datasets/badge/?version=stable)](https://chronocratic-datasets.readthedocs.io/en/stable/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/chronocratic-datasets)](https://pypi.org/project/chronocratic-datasets/)
[![GitHub Stars](https://img.shields.io/github/stars/chronocratic/datasets)](https://github.com/chronocratic/datasets)

## Installation

Install the package via pip:

```bash
pip install chronocratic-datasets
```

> **Note:** The PyPI package name uses a hyphen (`chronocratic-datasets`), but the import uses the `chronocratic.datasets` namespace:
> ```python
> from chronocratic.datasets import ...
> ```

## Quick Start

```python
from chronocratic.datasets import ForecastingMode, WeatherModule

weather = WeatherModule(mode=ForecastingMode.UNIVARIATE)
weather.prepare_data()
train_loader = weather.train_dataloader()
```

## Datasets

### Forecasting

- **ETT** (Electricity Transformer Temperature): ETTh1, ETTh2, ETTm1, ETTm2 — transformer temperature data at hourly and 15-minute intervals
- **Weather**: Weather and meteorological features from 2012 to 2017
- **Electricity**: Hourly electricity load data

### Classification

- **UCR** (Univariate): Archive of univariate time series classification datasets
- **UEA** (Multivariate): Archive of multivariate time series classification datasets

## Features

- **PyTorch Lightning DataModules** — Drop-in `LightningDataModule` implementations for seamless integration with Lightning training loops
- **Automatic caching with atomic writes** — Downloaded and processed data is cached locally with atomic file operations to prevent corruption
- **DDP-compliant data loading** — Workers share cached data correctly under Distributed Data Parallel training
- **Multiple forecasting modes** — Switch between `UNIVARIATE` and `MULTIVARIATE` forecasting configurations
- **Built-in scaling** — MinMax and Standard scalers applied automatically per dataset conventions
- **Type-safe API** — Full type hints and Google-style docstrings for IDE autocomplete and static analysis

## Documentation

Comprehensive documentation, including API reference, quickstart guides, and contributing instructions, is available at [chronocratic-datasets.readthedocs.io](https://chronocratic-datasets.readthedocs.io/en/stable/).

## Citation

If you use this package in your research, please cite it using the metadata in [CITATION.cff](CITATION.cff):

```yaml
title: "chronocratic-datasets"
authors:
  - name: "The Chronocratic Developers"
version: "0.1.0"
repository-code: "https://github.com/chronocratic/datasets"
```

## License

BSD 3-Clause — see [LICENSE](LICENSE) for the full text.
