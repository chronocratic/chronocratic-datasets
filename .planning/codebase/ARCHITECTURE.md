# Architecture

Last mapped: 2026-05-08

## Overview

The repository is a **monorepo with two source sub-projects** living under `_sources/` and a PEP 420 namespace package `src/tscollection/datasets/`. The sub-projects share a common domain (time series ML) but serve different purposes:

- **autotsrc** — Dataset and data module building blocks for time series tasks
- **rbspaper** — Full robustness benchmark pipeline built on top of similar data patterns

## Repository Layout

```
tsdatasets/                  # repo name (unchanged)
├── src/tscollection/        # PEP 420 implicit namespace
│   └── datasets/            # Unified time series dataset package
├── _sources/
│   ├── autotsrc/            # Dataset library (reference only)
│   └── rbspaper/            # Robustness benchmark pipeline (reference only)
├── pyproject.toml           # Root: tscollection-datasets package config
├── ruff.toml                # Root: linting config (mirrors rbspaper)
└── .gitignore               # Python standard ignores
```

## autotsrc Architecture

**Pattern:** Strategy-based dataset abstraction

```
autotsrc/
└── src/autotsrc/
    ├── datasets/
    │   ├── classes/          # PyTorch Dataset implementations
    │   │   └── abstract/     # Base classes + strategies
    │   ├── modules/          # LightningDataModule implementations
    │   │   └── abstract/     # Base data module classes
    │   ├── preparation/      # Data preparation utilities
    │   └── utils/            # Collation, features, general helpers
    ├── enums/                # Shared enums (modes, metrics, strategies)
    └── utils/                # Transformations, ARFF parsing, scaling, validation
```

**Key design patterns:**

1. **Template Method** — `TimeSeriesDataset` defines `__getitem__` flow; subclasses implement `_go_to_idx`, `_get_current_data`, `_get_current_label`
2. **Strategy** — `SequenceHandlingStrategy` abstracts how sliding windows produce labels. Concrete strategies: `ForecastingStrategySingleFile`, `ClassificationStrategySingleFile`, `ClassificationStrategyMultipleFiles`
3. **Composition** — Transforms are chained via `compose()` functional utility; `expand_dims_axis` is appended as a partial

**Dataset class hierarchy:**
- `TimeSeriesDataset` (ABC) → `FixedTimeSeriesDataset` (ABC) → `FixedTimeSeriesDatasetUnivariate`, `FixedTimeSeriesDatasetMultivariate`
- `TimeSeriesDataset` (ABC) → `FlexibleTimeSeriesDataset` (ABC) → `FlexibleTimeSeriesDatasetSingleFile`, `FlexibleTimeSeriesDatasetMultipleFiles`

**Data module hierarchy:**
- `BaseTimeSeriesDataModule` (ABC) → `BaseClassificationTimeSeriesDataModule`, `BaseForecastingTimeSeriesDataModule`

Each concrete dataset (ETT, Weather, Electricity, UCR, UEA) and data module inherits from these bases.

## rbspaper Architecture

**Pattern:** Pipeline-oriented with adapter layer for attacks

```
rbspaper/
├── src/rbspaper/
│   ├── adapters/             # Adapter pattern: attack + model + task adapters
│   ├── attacks/              # Attack execution, config, registry, backends
│   ├── configs/              # Config dataclasses for attacks, models, augmentations
│   ├── data/                 # Datasets, data modules, preparation, registry
│   │   ├── datasets/         # Concrete + abstract datasets
│   │   ├── modules/          # Concrete + abstract data modules
│   │   └── utils/            # ARFF, scaling, features, general utilities
│   ├── enums/                # Task enums, data enums, general enums
│   ├── evaluation/           # Downstream evaluation (classification, forecasting)
│   ├── models/               # Model implementations (TS2Vec, AutoTCL, CoST) + layers
│   │   ├── abstract/         # Encoding mixin
│   │   ├── augmentation/     # Augmentation strategies + factories
│   │   ├── autotcl/          # AutoTCL model
│   │   ├── cost/             # CoST model
│   │   ├── encoders/         # Positional encoders, masking
│   │   ├── layers/           # Dilated convolutions, general layers
│   │   └── ts2vec/           # TS2Vec model
│   └── pipeline/             # Orchestration core, config, state, loggers, analysis
├── runners/py/               # CLI entry point
└── experiment_instances/     # Experiment presets + registry
```

**Pipeline flow (`run_experiment_pipeline`):**

1. **Preflight** — Validates config (tasks, attacks, scope bindings, threat models)
2. **Setup** — Seeds RNG, prepares run directory
3. **Train** — Trains model via Lightning Trainer, copies best checkpoint
4. **Collect** — materializes train/valid/test tensors from dataloaders
5. **Attack** — Generates perturbed inputs (SHARED_INPUT or TASK_CONDITIONED scope)
6. **Encode** — Extracts representations from clean + attacked inputs per downstream task
7. **Evaluate** — Runs downstream classification/forecasting evaluation
8. **Analyze** — Computes geometry, shift, linear separability, low-dim artifacts
9. **Persist** — Saves NPZ representations + JSON metrics

**Key design patterns:**

1. **Adapter** — `AttackAdapter`, `ModelAdapter`, `TaskAdapter` normalize heterogeneous interfaces
2. **Registry** — Attack registry maps methods to backends and threat models; experiment registry maps IDs to configs
3. **Strategy** — `AttackScopePolicy.SHARED_INPUT` vs `TASK_CONDITIONED` controls attack encoding flow
4. **State Machine** — `PipelineState` tracks phase progression with checkpoint recovery
5. **Factory** — `build_model_from_parameters` constructs models from config dicts

**Entry point:** `runners/py/runner.py:main()` — CLI that resolves experiment, builds config, and runs pipeline
