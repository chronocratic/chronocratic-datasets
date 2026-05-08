# Codebase Structure

**Analysis Date:** 2026-05-05

## Directory Layout

```
robust_time_series_representations/
├── pyproject.toml              # Project metadata, dependencies, scripts, tool config
├── ruff.toml                   # Linting and formatting rules
├── README.md                   # Project instructions
├── LICENSE                     # License file
├── uv.lock                     # Lockfile for deterministic builds
├── uv_lock.sh                  # Helper script for uv sync
├── .gitignore                  # Git ignore rules
│
├── src/rbspaper/               # Main package (src layout)
│   ├── __init__.py             # Package marker
│   ├── utils.py                # Shared utilities (NaN-padding)
│   │
│   ├── adapters/               # Abstract adapter interfaces (not yet wired in)
│   │   ├── __init__.py
│   │   ├── attack_adapter.py
│   │   ├── model_adapter.py
│   │   └── task_adapter.py
│   │
│   ├── attacks/                # Adversarial attacks module
│   │   ├── __init__.py         # Public API re-exports
│   │   ├── _backend.py         # Backend dispatch (ART, torchattacks)
│   │   ├── _common.py          # Shared attack utilities
│   │   ├── batch.py            # Batched/dataset attack execution
│   │   ├── config.py           # Attack parameter dataclasses
│   │   ├── enums.py            # Attack enums (method, backend, threat model)
│   │   ├── functional.py       # Function-first attack API
│   │   └── registry.py         # Attack support lookup tables
│   │
│   ├── configs/                # Config re-exports
│   │   ├── __init__.py
│   │   ├── attacks.py          # Attack config re-exports
│   │   ├── augmentation_methods.py  # Augmentation config re-exports
│   │   └── models.py           # Model parameter re-exports
│   │
│   ├── data/                   # Data infrastructure
│   │   ├── __init__.py         # Public data API re-exports
│   │   ├── data_setup.py       # Dataset-to-datamodule factory
│   │   ├── preparation.py      # Datamodule factory + partial shortcuts
│   │   ├── registry.py         # Static dataset metadata registry
│   │   ├── datasets/           # PyTorch Dataset classes
│   │   │   ├── __init__.py
│   │   │   ├── abstract.py     # Base dataset classes
│   │   │   ├── electricity_load_dataset.py
│   │   │   ├── ett_dataset.py
│   │   │   ├── strategies.py   # Sequence handling strategies
│   │   │   ├── transformations.py  # Data transforms
│   │   │   ├── ucr_dataset.py
│   │   │   ├── uea_dataset.py
│   │   │   └── weather_dataset.py
│   │   ├── modules/            # LightningDataModule classes
│   │   │   ├── __init__.py
│   │   │   ├── abstract.py     # Base datamodule classes
│   │   │   ├── electricity_load_datamodule.py
│   │   │   ├── ett_datamodule.py
│   │   │   ├── ucr_datamodule.py
│   │   │   ├── uea_datamodule.py
│   │   │   └── weather_datamodule.py
│   │   └── utils/              # Data utilities
│   │       ├── __init__.py
│   │       ├── arff.py         # ARFF file parsing
│   │       ├── common.py       # Compose, collation helpers
│   │       ├── features.py     # Feature extraction
│   │       ├── general.py      # General utilities
│   │       └── scaling.py      # Data scaling setup
│   │
│   ├── enums/                  # Shared enumeration types
│   │   ├── __init__.py
│   │   ├── data_enums.py       # Dataset mode, splitting strategy enums
│   │   └── general.py          # TimeSeriesDownstreamTask enum
│   │
│   ├── evaluation/             # Downstream evaluation
│   │   ├── __init__.py         # Public eval API (evaluate)
│   │   ├── classification.py   # Classification evaluation
│   │   ├── enums.py            # Evaluation task enum
│   │   ├── evaluation.py       # Main evaluate() dispatcher
│   │   ├── forecasting.py      # Forecasting evaluation
│   │   └── protocols.py        # SVM/Ridge classifier protocols
│   │
│   ├── models/                 # Neural network models
│   │   ├── __init__.py
│   │   ├── config.py           # ModelParameters dataclasses (TS2Vec, AutoTCL, CoST)
│   │   ├── encoding.py         # Encoding dispatch by model type + task
│   │   ├── losses.py           # Contrastive and representation losses
│   │   ├── utils.py            # Tensor utilities (pooling, slicing, windows)
│   │   │
│   │   ├── abstract/           # Shared model abstractions
│   │   │   ├── __init__.py
│   │   │   └── encoding_functionality_mixin.py  # Encoding mixin for all models
│   │   │
│   │   ├── augmentation/       # Data augmentation
│   │   │   ├── __init__.py
│   │   │   ├── config.py       # Augmentation parameter dataclasses
│   │   │   ├── enums.py        # Augmentation mode enums
│   │   │   ├── factories.py    # Augmentation method factories
│   │   │   └── strategies.py   # Augmentation implementations
│   │   │
│   │   ├── autotcl/            # AutoTCL model
│   │   │   ├── __init__.py
│   │   │   ├── model.py        # AutoTCL LightningModule
│   │   │   └── utils.py
│   │   │
│   │   ├── cost/               # CoST model
│   │   │   ├── __init__.py
│   │   │   ├── model.py        # CoST LightningModule
│   │   │   └── utils.py
│   │   │
│   │   ├── encoders/           # Encoder architectures
│   │   │   ├── __init__.py
│   │   │   ├── encoders.py     # Time series encoder nn.Modules
│   │   │   └── masking.py      # Mask generation strategies
│   │   │
│   │   ├── layers/             # Neural network layers
│   │   │   ├── __init__.py
│   │   │   ├── general.py      # Banded Fourier layer
│   │   │   └── convolutions/   # Convolution layers
│   │   │       ├── __init__.py
│   │   │       ├── dilated.py  # Dilated conv encoder
│   │   │       └── same_pad.py # Same-padding conv
│   │   │
│   │   └── ts2vec/             # TS2Vec model
│   │       ├── __init__.py
│   │       ├── model.py        # TS2Vec LightningModule
│   │       └── utils.py
│   │
│   └── pipeline/               # Experiment pipeline
│       ├── __init__.py         # Public pipeline API re-exports
│       ├── analysis.py         # Representation analysis (geometry, shift, etc.)
│       ├── config.py           # Pipeline config dataclasses
│       ├── core.py             # run_experiment_pipeline()
│       └── setup/              # Pipeline setup utilities
│           ├── __init__.py
│           └── model.py        # Model factory
│
├── experiment_instances/       # Pre-configured experiment definitions
│   ├── __init__.py
│   ├── data_utils.py           # Dataset task profile builder
│   └── instances.py            # ExperimentInstance + EXPERIMENTS_REGISTRY
│
├── runners/                    # Entry point scripts
│   └── py/
│       └── runner.py           # CLI runner (rbspaper-run)
│
├── test/                       # Test suite
│   ├── test_attacks_batch.py
│   ├── test_attacks_functional.py
│   ├── test_attacks_registry.py
│   └── test_pipeline_core.py
│
├── _sources/                   # Archived external source references
│   ├── tscar_jesse/
│   ├── tscar_thesis/
│   └── autotsaugment/
│
├── .claude/                    # Claude project configuration
├── .github/                    # GitHub configuration
├── .vscode/                    # VS Code settings
├── .planning/                  # GSD planning output
│   └── codebase/               # Codebase analysis documents
└── .venv/                      # Virtual environment (git-ignored)
```

## Directory Purposes

**`src/rbspaper/`:**
- Purpose: Main package containing all library code
- Contains: Models, attacks, evaluation, data, pipeline, adapters, configs, enums
- Key files: `utils.py` (shared helpers)

**`src/rbspaper/pipeline/`:**
- Purpose: Experiment orchestration
- Contains: Core pipeline function, typed configuration objects, representation analysis, model factory
- Key files: `core.py` (run_experiment_pipeline), `config.py` (all config dataclasses)

**`src/rbspaper/models/`:**
- Purpose: Self-supervised representation learning models and their components
- Contains: Three model implementations (TS2Vec, AutoTCL, CoST), encoders, losses, augmentation, utilities
- Key files: `config.py` (model parameters), `encoding.py` (encoding dispatch), `losses.py` (contrastive losses)

**`src/rbspaper/attacks/`:**
- Purpose: Adversarial attack execution
- Contains: Functional API, backend dispatch, config dataclasses, registry of supported attacks
- Key files: `functional.py` (execute_attack), `config.py` (attack params), `registry.py` (support lookup)

**`src/rbspaper/evaluation/`:**
- Purpose: Downstream task evaluation on representations
- Contains: Classification and forecasting evaluation, SVM/Ridge protocols
- Key files: `evaluation.py` (evaluate dispatcher), `classification.py` (classify_and_evaluate)

**`src/rbspaper/data/`:**
- Purpose: Data loading, dataset definitions, LightningDataModule infrastructure
- Contains: Dataset classes, datamodules, registry, preparation factories, utilities
- Key files: `registry.py` (dataset metadata), `data_setup.py` (datamodule factory), `preparation.py` (family-specific partials)

**`src/rbspaper/adapters/`:**
- Purpose: Abstract interfaces for model/attack/task contracts (not yet wired into pipeline)
- Contains: ModelAdapter, AttackAdapter, TaskAdapter ABCs

**`src/rbspaper/configs/`:**
- Purpose: Re-export layer for config types to avoid circular imports
- Contains: Re-exports from models.config and attacks.config

**`src/rbspaper/enums/`:**
- Purpose: Shared enumeration types
- Contains: TimeSeriesDownstreamTask, data mode enums, splitting strategy enums

**`experiment_instances/`:**
- Purpose: Pre-configured experiment definitions (model + attack combinations)
- Contains: ExperimentInstance dataclass, registry dict, helper factories
- Key files: `instances.py` (EXPERIMENTS_REGISTRY), `data_utils.py` (task profile builder)

**`runners/py/`:**
- Purpose: CLI entry points
- Contains: Main runner script with argument parsing and pipeline assembly

**`test/`:**
- Purpose: Pytest test suite
- Contains: Tests for attacks (batch, functional, registry) and pipeline core
- Key files: `test_pipeline_core.py` (smoke tests with mocked dependencies)

**`_sources/`:**
- Purpose: Archived external source code references (not imported)
- Contains: TS-CAR Jesse code, TS-CAR thesis code, AutoTSAugment code
- Generated: No
- Committed: Yes

## Key File Locations

**Entry Points:**
- `runners/py/runner.py`: CLI entry point (`rbspaper-run` script)
- `src/rbspaper/pipeline/core.py:run_experiment_pipeline()`: Programmatic pipeline entry

**Configuration:**
- `pyproject.toml`: Project metadata, dependencies, tool configuration, scripts
- `ruff.toml`: Linting/formatting rules (Google docstring convention, full lint select)

**Core Logic:**
- `src/rbspaper/pipeline/core.py`: Full pipeline orchestration
- `src/rbspaper/pipeline/config.py`: All pipeline config dataclasses
- `src/rbspaper/models/encoding.py`: Encoding strategy dispatch
- `src/rbspaper/attacks/functional.py`: Attack execution entry point
- `src/rbspaper/evaluation/evaluation.py`: Evaluation entry point

**Model Definitions:**
- `src/rbspaper/models/ts2vec/model.py`: TS2Vec LightningModule
- `src/rbspaper/models/autotcl/model.py`: AutoTCL LightningModule
- `src/rbspaper/models/cost/model.py`: CoST LightningModule
- `src/rbspaper/models/encoders/encoders.py`: Encoder nn.Module architectures

**Data Infrastructure:**
- `src/rbspaper/data/registry.py`: Static dataset registry
- `src/rbspaper/data/data_setup.py`: Dataset-to-datamodule factory
- `src/rbspaper/data/modules/abstract.py`: Base LightningDataModule classes

**Testing:**
- `test/test_pipeline_core.py`: Pipeline smoke tests with mocks
- `test/test_attacks_functional.py`: Attack functional API tests
- `test/test_attacks_batch.py`: Batch attack tests
- `test/test_attacks_registry.py`: Attack registry tests

## Naming Conventions

**Files:**
- `snake_case.py` for all Python modules
- `__init__.py` for package markers with `__all__` exports
- `test_*.py` for test files (matching convention for the module under test)
- `*_datamodule.py` for LightningDataModule classes
- `*_dataset.py` for PyTorch Dataset classes
- `_*.py` for private/internal modules (e.g., `_backend.py`, `_common.py`)

**Directories:**
- `snake_case/` for all directories
- Sub-packages under `models/` use model name as PascalCase-lowercase (e.g., `ts2vec/`, `autotcl/`, `cost/`)
- `layers/` for reusable neural network layers
- `convolutions/` as a sub-package of `layers/`

**Classes:**
- `PascalCase` for all classes
- `*DataModule` suffix for LightningDataModule subclasses
- `*Dataset` suffix for PyTorch Dataset subclasses
- `*Parameters` suffix for config dataclasses
- `*Config` suffix for pipeline configuration objects
- `*Adapter` suffix for abstract adapter interfaces
- `*Mixin` suffix for mixin classes

**Functions:**
- `snake_case` for all functions
- `get_*` prefix for factory/lookup functions
- `compute_*` prefix for analysis/metric functions
- `_private_*` prefix for internal helpers (single underscore)

**Variables:**
- `snake_case` for local and instance variables
- `UPPER_SNAKE_CASE` for module-level constants (e.g., `MIN_LABELED_BATCH_LENGTH`, `DATASET_REGISTRY`)
- `_private_var` for module-private variables

## Where to Add New Code

**New Model:**
- Model parameters: `src/rbspaper/models/config.py` (add new `ModelParameters` subclass)
- LightningModule: `src/rbspaper/models/newmodel/model.py` and `src/rbspaper/models/newmodel/__init__.py`
- Encoder: `src/rbspaper/models/encoders/encoders.py`
- Encoding dispatch: `src/rbspaper/models/encoding.py` (add `_newmodel_encode()` and case in `encode_data()`)
- Model factory: `src/rbspaper/pipeline/setup/model.py` (add isinstance branch)
- Config re-export: `src/rbspaper/configs/models.py`

**New Attack Method:**
- Parameters dataclass: `src/rbspaper/attacks/config.py`
- Functional implementation: `src/rbspaper/attacks/functional.py`
- Registry entries: `src/rbspaper/attacks/registry.py` (add to `ATTACK_THREAT_MODEL`, `SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK`, etc.)
- Backend implementation: `src/rbspaper/attacks/_backend.py`
- Public re-export: `src/rbspaper/attacks/__init__.py`

**New Dataset Family:**
- Dataset class: `src/rbspaper/data/datasets/newfamily_dataset.py`
- DataModule class: `src/rbspaper/data/modules/newfamily_datamodule.py`
- Registry entry: `src/rbspaper/data/registry.py` (`DATASET_REGISTRY` tuple)
- Preparation partial: `src/rbspaper/data/preparation.py`
- Data setup dispatch: `src/rbspaper/data/data_setup.py` (`get_datamodule_with_downstream_tasks()`)
- Module re-exports: `src/rbspaper/data/modules/__init__.py`, `src/rbspaper/data/datasets/__init__.py`

**New Downstream Task:**
- Enum value: `src/rbspaper/enums/general.py` (`TimeSeriesDownstreamTask`)
- Evaluation logic: `src/rbspaper/evaluation/newtask.py`
- Evaluation dispatch: `src/rbspaper/evaluation/evaluation.py`
- Encoding strategy: `src/rbspaper/models/encoding.py` (add task-specific encode function)

**New Experiment Instance:**
- `experiment_instances/instances.py` (add to `EXPERIMENTS_REGISTRY`)

**New Pipeline Config Option:**
- `src/rbspaper/pipeline/config.py` (add dataclass)
- `src/rbspaper/pipeline/core.py` (use in pipeline logic)
- `src/rbspaper/pipeline/__init__.py` (re-export)

**New Test:**
- `test/test_new_module.py` (co-located in `test/` directory, named after module)

## Special Directories

**`_sources/`:**
- Purpose: Archived external source code references used during development
- Generated: No
- Committed: Yes
- Note: These are not imported by the package. They serve as reference material.

**`.planning/codebase/`:**
- Purpose: GSD analysis output (architecture, structure, conventions, etc.)
- Generated: Yes (by GSD commands)
- Committed: Yes

**`.venv/`:**
- Purpose: Python virtual environment managed by uv
- Generated: Yes (`uv sync`)
- Committed: No (in `.gitignore`)

**`src/rbspaper.egg-info/`:**
- Purpose: Setuptools build artifacts
- Generated: Yes (during `uv sync` or `pip install`)
- Committed: No (in `.gitignore`)

---

*Structure analysis: 2026-05-05*
