<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/header/surface.svg?title=chronocratic-datasets&subtitle=Ready-to-use+time+series+datasets+and+DataModules+for+PyTorch+Lightning.&align=center&font=jetbrains-mono&width=1200&height=600&bg=0E2A44&titleColor=F3F1EC&subtitleColor=A8D5D1&mode=dark&overlay=0&logo=https://raw.githubusercontent.com/chronocratic/chronocratic-datasets/main/.github/assets/icon-dark.png&image=https://raw.githubusercontent.com/chronocratic/chronocratic-datasets/main/.github/assets/header-bg-dark.png&v=1" />
    <img alt="chronocratic-datasets" src="https://shieldcn.dev/header/surface.svg?title=chronocratic-datasets&subtitle=Ready-to-use+time+series+datasets+and+DataModules+for+PyTorch+Lightning.&align=center&font=jetbrains-mono&width=1200&height=600&bg=F3F1EC&titleColor=0E2A44&subtitleColor=2A6A7C&mode=light&overlay=0&logo=https://raw.githubusercontent.com/chronocratic/chronocratic-datasets/main/.github/assets/icon-light.png&image=https://raw.githubusercontent.com/chronocratic/chronocratic-datasets/main/.github/assets/header-bg-light.png&v=1" />
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/chronocratic-datasets/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/pypi/chronocratic-datasets.svg?variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="PyPI version" src="https://shieldcn.dev/pypi/chronocratic-datasets.svg?variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
  <a href="https://pepy.tech/projects/chronocratic-datasets"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/badge/dynamic/json.svg?url=https%3A%2F%2Fimg.shields.io%2Fpepy%2Fdt%2Fchronocratic-datasets.json&query=%24.message&label=downloads&variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="Total downloads" src="https://shieldcn.dev/badge/dynamic/json.svg?url=https%3A%2F%2Fimg.shields.io%2Fpepy%2Fdt%2Fchronocratic-datasets.json&query=%24.message&label=downloads&variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
  <a href="https://github.com/chronocratic/chronocratic-datasets/actions/workflows/build-and-test.yml"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/github/ci/chronocratic/chronocratic-datasets.svg?workflow=build-and-test.yml&branch=main&variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="CI status" src="https://shieldcn.dev/github/ci/chronocratic/chronocratic-datasets.svg?workflow=build-and-test.yml&branch=main&variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
  <a href="https://github.com/chronocratic/chronocratic-datasets/blob/main/LICENSE"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/github/license/chronocratic/chronocratic-datasets.svg?variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="License" src="https://shieldcn.dev/github/license/chronocratic/chronocratic-datasets.svg?variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
  <a href="https://github.com/chronocratic/chronocratic-datasets/stargazers"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/github/stars/chronocratic/chronocratic-datasets.svg?variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="GitHub stars" src="https://shieldcn.dev/github/stars/chronocratic/chronocratic-datasets.svg?variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
  <a href="https://github.com/astral-sh/ruff"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/badge/code_style-ruff-261230.svg?logo=ruff&variant=secondary&font=jetbrains-mono&mode=dark" /><img alt="Code style: ruff" src="https://shieldcn.dev/badge/code_style-ruff-261230.svg?logo=ruff&variant=secondary&font=jetbrains-mono&mode=light" /></picture></a>
</p>

<div align="center">

| Python | Docs |
| :---: | :---: |
| [![Python versions](https://img.shields.io/pypi/pyversions/chronocratic-datasets.svg)](https://pypi.org/project/chronocratic-datasets/) | [![Documentation status](https://readthedocs.org/projects/chronocratic-datasets/badge/?version=latest)](https://chronocratic-datasets.readthedocs.io/en/latest/) |

</div>

> ⭐ **Found this useful? Give it a star.** chronocratic-datasets is free and open source. A star takes one click, tells us the work is worth continuing, and helps other time series researchers find the project.

## Installation

With pip:

```bash
pip install chronocratic-datasets
```

With uv:

```bash
uv add chronocratic-datasets
```

> **Note:** The PyPI package name uses a hyphen (`chronocratic-datasets`), but the import uses the `chronocratic.datasets` namespace:
> ```python
> from chronocratic.datasets import ...
> ```

## Quick Start

```python
from pathlib import Path

from chronocratic.datasets import ForecastingMode, WeatherDataModule

weather = WeatherDataModule(
    dataset_file_path=Path("data/weather.csv"),
    mode=ForecastingMode.UNIVARIATE,
)
weather.prepare_data()
weather.setup()
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

Comprehensive documentation, including API reference, quickstart guides, and contributing instructions, is available at [chronocratic-datasets.readthedocs.io](https://chronocratic-datasets.readthedocs.io/).

## License

BSD 3-Clause — see [LICENSE](LICENSE) for the full text.

The Chronocratic name and logo are not covered by the license. All rights reserved.
