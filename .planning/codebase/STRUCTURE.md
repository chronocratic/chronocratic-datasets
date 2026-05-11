# Directory Structure

Last mapped: 2026-05-08

## Top-Level Layout

```
tsdatasets/
├── .claude/                  # Claude Code configuration and GSD workflows
├── .git/                     # Git repository
├── .planning/                # GSD planning artifacts
│   └── config.json           # Workflow configuration
├── .venv/                    # Python virtual environment (uv-managed)
├── .vscode/                  # VS Code settings
├── _sources/                 # Source sub-projects (not packaged)
│   ├── autotsrc/             # Time series dataset library
│   └── rbspaper/             # Robustness benchmark pipeline
├── src/
│   └── tscollection/         # PEP 420 implicit namespace (no __init__.py)
│       └── datasets/         # Unified time series dataset package
│           ├── __init__.py   # Public API surface, __version__
│           ├── datasets/     # PyTorch Dataset classes
│           ├── modules/      # LightningDataModule classes
│           ├── download/     # Data download and caching
│           ├── config/       # Pydantic configuration
│           ├── enums/        # Typed enumerations
│           └── utils/        # Utility functions
├── pyproject.toml            # Root project config (tscollection-datasets)
├── ruff.toml                 # Root linting config
├── uv.lock                   # Locked dependencies
└── README.md                 # Project description
```

## autotsrc Structure (`_sources/autotsrc/`)

```
autotsrc/
├── pyproject.toml            # Package config (AutoTSRC)
├── ruff.toml                 # Lint/format config
├── uv.lock                   # Locked deps
├── dependencies/             # (excluded from linting)
├── experiments_output/       # (excluded from linting)
└── src/
    └── autotsrc/
        ├── datasets/
        │   ├── __init__.py
        │   ├── classes/
        │   │   ├── __init__.py
        │   │   ├── abstract/         # Base dataset ABCs
        │   │   │   ├── abstract.py   # TimeSeriesDataset hierarchy
        │   │   │   └── strategies.py # Sequence handling strategies
        │   │   ├── electricity_load_dataset.py
        │   │   ├── ett_dataset.py
        │   │   ├── ucr_classification_univariate_dataset.py
        │   │   ├── uea_classification_multivariate_dataset.py
        │   │   └── weather_dataset.py
        │   ├── modules/
        │   │   ├── __init__.py
        │   │   ├── abstract/         # Base data module ABCs
        │   │   │   └── abstract.py   # BaseTimeSeriesDataModule hierarchy
        │   │   ├── electricity_load_data_module.py
        │   │   ├── ett_data_module.py
        │   │   ├── ucr_classification_univariate_data_module.py
        │   │   ├── uea_classification_multivariate_data_module.py
        │   │   └── weather_data_module.py
        │   ├── preparation/
        │   │   └── preparation.py    # Data preparation pipeline
        │   └── utils/
        │       ├── features.py       # Time feature extraction
        │       └── general.py        # Collation, padding helpers
        ├── enums/
        │   └── data_enums.py         # DatasetMode, DistanceMetric, SplittingStrategy
        └── utils/
            ├── data/
            │   ├── arff.py           # ARFF file parser
            │   └── strategies/
            │       └── scaling.py    # Scaler factory
            ├── decorators/
            │   └── validation.py     # Validation decorators
            ├── transformations.py    # NumPy ↔ Tensor conversion, dim expansion
            └── utils.py              # compose(), load_json(), etc.
```

## rbspaper Structure (`_sources/rbspaper/`)

```
rbspaper/
├── pyproject.toml            # Package config (rbspaper)
├── ruff.toml                 # Lint/format config
├── uv.lock                   # Locked deps
├── dependencies/             # (excluded from linting)
├── experiments_output/       # (excluded from linting)
├── data/                     # (excluded from linting)
├── experiment_instances/     # Experiment presets — on pythonpath
│   ├── instances.py          # EXPERIMENTS_REGISTRY, get_experiment_instance()
│   └── data_utils.py         # build_dataset_task_profile()
├── runners/
│   └── py/
│       └── runner.py         # CLI entry point: rbspaper-run
├── test/                     # Unit tests (~18 files)
└── src/
    └── rbspaper/
        ├── adapters/         # Attack, model, task adapters
        ├── attacks/          # Attack execution, config, enums, registry
        ├── configs/          # Attack, model, augmentation config dataclasses
        ├── data/
        │   ├── datasets/     # Dataset implementations (mirror autotsrc patterns)
        │   ├── modules/      # LightningDataModule implementations
        │   ├── preparation.py
        │   ├── data_setup.py # Registry helpers
        │   ├── registry.py   # Dataset registry
        │   └── utils/        # ARFF, features, scaling, common, general
        ├── enums/            # Data enums, general enums
        ├── evaluation/       # Classification + forecasting evaluation
        ├── models/
        │   ├── abstract/     # Encoding mixin
        │   ├── augmentation/ # Augmentation config, strategies, factories
        │   ├── autotcl/      # AutoTCL self-supervised model
        │   ├── cost/         # CoST contrastive model
        │   ├── encoders/     # Positional encoders, masking
        │   ├── layers/       # Dilated conv, general layers
        │   ├── ts2vec/       # TS2Vec model
        │   ├── config.py     # Model parameter dataclasses
        │   ├── encoding.py   # encode_data() function
        │   ├── losses.py     # Loss definitions
        │   └── utils.py      # extract_features_from_batch()
        └── pipeline/
            ├── analysis.py   # Geometry, shift, separability, low-dim metrics
            ├── config.py     # Pipeline config dataclasses
            ├── core.py       # run_experiment_pipeline()
            ├── loggers.py    # create_loggers() factory
            ├── state.py      # PipelineState, checkpoint recovery
            └── setup/
                └── model.py  # build_model_from_parameters()
```

## Key File Locations

| What | Location |
|------|----------|
| Root package config | `pyproject.toml` |
| Dataset base classes | `_sources/autotsrc/src/autotsrc/datasets/classes/abstract/abstract.py` |
| Data module base classes | `_sources/autotsrc/src/autotsrc/datasets/modules/abstract/abstract.py` |
| Pipeline orchestrator | `_sources/rbspaper/src/rbspaper/pipeline/core.py` |
| CLI entry point | `_sources/rbspaper/runners/py/runner.py` |
| Attack registry | `_sources/rbspaper/src/rbspaper/attacks/registry.py` |
| Experiment presets | `_sources/rbspaper/experiment_instances/instances.py` |
| Models | `_sources/rbspaper/src/rbspaper/models/` |
| Evaluation | `_sources/rbspaper/src/rbspaper/evaluation/` |
