# Quick Start

This guide shows you how to install and use **chronocratic-datasets** with PyTorch Lightning.

## Installation

Install the package from PyPI:

```bash
pip install chronocratic-datasets
```

For development (with docs tooling):

```bash
pip install -e "chronocratic-datasets[docs]"
```

## Basic Workflow

The package provides `LightningDataModule` subclasses for each dataset family.
A typical workflow looks like this:

### Forecasting Example

```python
from chronocratic.datasets import ForecastingMode, WeatherModule

# Create the data module
weather = WeatherModule(mode=ForecastingMode.UNIVARIATE, seq_len=24, pred_len=168)

# Download and prepare data (runs once, cached after that)
weather.prepare_data()
weather.setup()

# Get a dataloader
train_loader = weather.train_dataloader()

# Use with Lightning
# trainer = Trainer(max_epochs=10)
# trainer.fit(model, datamodule=weather)
```

### Classification Example

```python
from chronocratic.datasets import UCRClassificationDataModule

# Create the data module for a UCR dataset
module = UCRClassificationDataModule(dataset_name="FogiDataset1", scale=True)

# Prepare and setup
module.prepare_data()
module.setup()

# Get dataloaders
train_loader = module.train_dataloader()
val_loader = module.val_dataloader()
test_loader = module.test_dataloader()
```

## Key Concepts

- **ForecastingMode** — Choose between `UNIVARIATE` and `MULTIVARIATE` input modes.
- **ForecastingLoaderMode** — Choose how samples are constructed: `SAMPLE_ONLY`,
  `SAMPLE_LABEL`, or `INPUT_OUTPUT`.
- **ClassificationLoaderMode** — Choose between full-series and sliding-window loading
  for classification tasks.
- **Caching** — All datasets are cached locally after first download to speed up repeated runs.
- **Scaling** — Enable or disable data normalization via the `scale` parameter.

## Next Steps

- See the {doc}`forecasting` guide for in-depth forecasting dataset usage.
- See the {doc}`classification` guide for classification benchmarks.
- Browse the {doc}`api/enums` reference for all enum options.
