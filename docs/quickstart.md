# Quick Start

This guide shows you how to install and use **chronocratic-datasets** with PyTorch Lightning.

## Installation

Install the package from PyPI:

```bash
pip install chronocratic-datasets
```

For development (with docs tooling):

```bash
pip install -e ".[docs]"
```

## Basic Workflow

The package provides `LightningDataModule` subclasses for each dataset family.
A typical workflow looks like this:

### Forecasting Example

```python
from pathlib import Path

from chronocratic.datasets import ForecastingMode, WeatherDataModule

weather = WeatherDataModule(
    dataset_file_path=Path("data/weather.csv"),
    mode=ForecastingMode.UNIVARIATE,
    seq_len=24,
    forecast_horizon=168,
)

weather.prepare_data()
weather.setup()
train_loader = weather.train_dataloader()
```

### Classification Example

```python
from pathlib import Path

from chronocratic.datasets import UCRClassificationDataModule

module = UCRClassificationDataModule(
    dataset_folder_path=Path("data/FogiDataset1"),
    target_column_name="class",
    scale_data=True,
)

module.prepare_data()
module.setup()
train_loader = module.train_dataloader()
val_loader = module.val_dataloader()
test_loader = module.test_dataloader()
```

## Next Steps

- See the {doc}`forecasting` guide for in-depth forecasting dataset usage and
  additional datasets (ETT, Electricity).
- See the {doc}`classification` guide for UCR and UEA classification benchmarks.
- Browse the {doc}`api/modules` reference for full parameter documentation.
- Browse the {doc}`api/enums` reference for all enum options.
