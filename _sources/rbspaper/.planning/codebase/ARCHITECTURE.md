<!-- refreshed: 2026-05-05 -->
# Architecture

**Analysis Date:** 2026-05-05

## System Overview

```text
+---------------------------------------------------------------------------+
|                       CLI Runner / Entry Points                            |
|                  `runners/py/runner.py`                                    |
+-----------------------------+---------------------------------------------+
                              |
                              v
+---------------------------------------------------------------------------+
|                  Experiment Instance Registry                               |
|              `experiment_instances/instances.py`                           |
|              `experiment_instances/data_utils.py`                          |
+-----------------------------+---------------------------------------------+
                              |
                              v
+---------------------------------------------------------------------------+
|                    Pipeline Orchestration Layer                             |
|         `src/rbspaper/pipeline/core.py` (run_experiment_pipeline)          |
|         `src/rbspaper/pipeline/config.py` (typed config objects)           |
|         `src/rbspaper/pipeline/analysis.py` (geometry/shift analysis)      |
|         `src/rbspaper/pipeline/setup/model.py` (model factory)             |
+-----------------------------+---------------------------------------------+
          |                    |                    |
          v                    v                    v
+--------------------+ +------------------+ +-----------------------------+
|   Models Layer     | |  Attacks Layer   | |    Evaluation Layer         |
| `src/rbspaper/`    | |`src/rbspaper/`   | |  `src/rbspaper/evaluation/` |
|   models/          | |  attacks/        | |                             |
|   (TS2Vec,         | |  (FGSM, PGD, BIM,| |  classify_and_evaluate,     |
|    AutoTCL, CoST)  | |   DeepFool, CW)  | |   forecast_and_evaluate)    |
+--------------------+ +------------------+ +-----------------------------+
          |                    |                    |
          v                    v                    v
+---------------------------------------------------------------------------+
|                        Data Infrastructure Layer                             |
|         `src/rbspaper/data/` (datasets, modules, registry, utils)          |
+-----------------------------+---------------------------------------------+
                              |
                              v
+---------------------------------------------------------------------------+
|              External: Lightning, PyTorch, scikit-learn, ART               |
+---------------------------------------------------------------------------+
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| CLI Runner | Parse args, resolve experiment + dataset, assemble pipeline config, invoke pipeline | `runners/py/runner.py` |
| Experiment Registry | Static dict of `ExperimentInstance` objects mapping ID to model+attack config | `experiment_instances/instances.py` |
| Pipeline Core | Execute full experiment: train, encode, attack, evaluate, analyze | `src/rbspaper/pipeline/core.py` |
| Pipeline Config | Frozen dataclass hierarchy for all pipeline settings | `src/rbspaper/pipeline/config.py` |
| Representation Analysis | Geometry, shift, linear separability, and low-dim projection metrics | `src/rbspaper/pipeline/analysis.py` |
| Model Factory | Build `LightningModule` from typed `ModelParameters` dataclass | `src/rbspaper/pipeline/setup/model.py` |
| Model Parameters | Abstract base + subclasses (TS2Vec, AutoTCL, CoST) config dataclasses | `src/rbspaper/models/config.py` |
| TS2Vec Model | LightningModule implementing TS2Vec self-supervised representation learning | `src/rbspaper/models/ts2vec/model.py` |
| AutoTCL Model | LightningModule implementing AutoTCL with neural-network augmentation | `src/rbspaper/models/autotcl/model.py` |
| CoST Model | LightningModule implementing CoST with trend/seasonality decomposition | `src/rbspaper/models/cost/model.py` |
| Encoding Mixin | Shared encoding functionality (pooling, sliding window, multi-scale) | `src/rbspaper/models/abstract/encoding_functionality_mixin.py` |
| Encoding Dispatch | Task-specific encoding strategy per model type | `src/rbspaper/models/encoding.py` |
| Time Series Encoders | nn.Module encoder architectures (dilated conv backbone) | `src/rbspaper/models/encoders/encoders.py` |
| Loss Functions | Contrastive and representation learning losses | `src/rbspaper/models/losses.py` |
| Model Utilities | Tensor pooling, slicing, sliding window, batch extraction helpers | `src/rbspaper/models/utils.py` |
| Augmentation Strategies | Data augmentation methods (crop-shift, neural-network, random function) | `src/rbspaper/models/augmentation/strategies.py` |
| Augmentation Factories | Factory for creating augmentation methods by mode enum | `src/rbspaper/models/augmentation/factories.py` |
| Masking | Mask generation strategies for encoder inputs | `src/rbspaper/models/encoders/masking.py` |
| Dilated Convolutions | Same-pad and dilated conv layers for encoders | `src/rbspaper/models/layers/convolutions/dilated.py` |
| Banded Fourier Layer | Fourier-domain layer for frequency-banded representations | `src/rbspaper/models/layers/general.py` |
| Attack Registry | Lookup of threat model, backend support, supervision requirements | `src/rbspaper/attacks/registry.py` |
| Attack Functional API | Function-first attack execution (execute_attack, attack_dataset) | `src/rbspaper/attacks/functional.py` |
| Attack Config | Frozen dataclasses for attack parameters and execution context | `src/rbspaper/attacks/config.py` |
| Attack Enums | AttackMethod, AttackBackend, AttackThreatModel, AttackObjective | `src/rbspaper/attacks/enums.py` |
| Attack Backend | Backend dispatch (ART, torchattacks) for attack execution | `src/rbspaper/attacks/_backend.py` |
| Attack Batch | Batched and dataset-level attack orchestration | `src/rbspaper/attacks/batch.py` |
| Model Adapter | Abstract interface for model-facing API (encode, forward_for_attack) | `src/rbspaper/adapters/model_adapter.py` |
| Attack Adapter | Abstract interface for backend-agnostic attack execution | `src/rbspaper/adapters/attack_adapter.py` |
| Task Adapter | Abstract interface for downstream task evaluation | `src/rbspaper/adapters/task_adapter.py` |
| Evaluation Entry | Dispatch to classification or forecasting evaluation | `src/rbspaper/evaluation/evaluation.py` |
| Classification Eval | Train classifier (SVM/Ridge), compute accuracy/F1/metrics | `src/rbspaper/evaluation/classification.py` |
| Forecasting Eval | Forecasting evaluation logic | `src/rbspaper/evaluation/forecasting.py` |
| Eval Protocols | SVM and Ridge classifiers with GridSearchCV tuning | `src/rbspaper/evaluation/protocols.py` |
| Eval Enums | Downstream task enum (classification, forecasting, clustering) | `src/rbspaper/evaluation/enums.py` |
| Dataset Registry | Static tuple of `DatasetMetadata` for all known datasets | `src/rbspaper/data/registry.py` |
| Data Setup | Factory resolving dataset name to configured LightningDataModule | `src/rbspaper/data/data_setup.py` |
| Data Preparation | Generic datamodule factory + partial-function shortcuts per family | `src/rbspaper/data/preparation.py` |
| Abstract DataModule | Base LightningDataModule with scaling, splitting, dataloader setup | `src/rbspaper/data/modules/abstract.py` |
| UCR DataModule | Classification datamodule for univariate UCR datasets | `src/rbspaper/data/modules/ucr_datamodule.py` |
| UEA DataModule | Classification datamodule for multivariate UEA datasets | `src/rbspaper/data/modules/uea_datamodule.py` |
| ETT DataModule | Forecasting datamodule for ETT time series | `src/rbspaper/data/modules/ett_datamodule.py` |
| Electricity DataModule | Forecasting datamodule for electricity load data | `src/rbspaper/data/modules/electricity_load_datamodule.py` |
| Weather DataModule | Forecasting datamodule for weather data | `src/rbspaper/data/modules/weather_datamodule.py` |
| Abstract Datasets | Base PyTorch Dataset classes (fixed-length, flexible/sliding) | `src/rbspaper/data/datasets/abstract.py` |
| Dataset Strategies | Sequence handling strategies for varying-length series | `src/rbspaper/data/datasets/strategies.py` |
| Dataset Transformations | NumPy-to-tensor, dimension expansion transforms | `src/rbspaper/data/datasets/transformations.py` |
| Data Utils | ARFF parsing, feature extraction, scaling, collation | `src/rbspaper/data/utils/` |
| Enums | Shared enums (TimeSeriesDownstreamTask, data enums) | `src/rbspaper/enums/` |
| Model Config Re-export | Re-exports model parameters from models.config | `src/rbspaper/configs/models.py` |
| Shared Utils | NaN-padding helper function | `src/rbspaper/utils.py` |

## Pattern Overview

**Overall:** Staged pipeline with factory-based component construction, registry-driven resolution, and configuration objects as the primary coordination mechanism.

**Key Characteristics:**
- **Frozen dataclass configuration:** All pipeline settings use `@dataclass(frozen=True)` for immutability. Config objects are validated at construction time (e.g., `__post_init__`).
- **Registry pattern:** Static lookup dicts for experiments (`EXPERIMENTS_REGISTRY`), datasets (`DATASET_REGISTRY`), and attacks (`ATTACK_THREAT_MODEL`, `SUPPORTED_BACKENDS_BY_TASK_AND_ATTACK`).
- **Factory pattern:** Model creation via `build_model_from_parameters()` dispatches on `ModelParameters` subclass type. Data modules via `get_datamodule_with_downstream_tasks()` dispatches on dataset family.
- **Mixin pattern:** `EncodingFunctionalityMixin` provides shared encoding capability to all three `LightningModule` models (TS2Vec, AutoTCL, CoST).
- **Strategy pattern:** Encoding varies by downstream task (classification: full-series pooling, forecasting: sliding windows). Attacks vary by backend (ART, torchattacks).
- **Adapter interfaces:** Abstract base classes (`ModelAdapter`, `AttackAdapter`, `TaskAdapter`) define contracts. Not yet fully wired into the pipeline; the pipeline currently calls models and attacks directly.
- **Function-first API:** Attacks expose standalone functions (`fgsm_attack`, `pgd_attack`, `execute_attack`) rather than class-based invocations.

## Layers

**CLI Runner Layer:**
- Purpose: Parse CLI args, resolve experiment by ID, assemble full pipeline config
- Location: `runners/py/runner.py`
- Contains: Argument parsing, experiment resolution, config assembly, dry-run support
- Depends on: Experiment registry, data setup, pipeline config, model factory
- Used by: External users (terminal, HPC job scripts)

**Experiment Instances Layer:**
- Purpose: Pre-configure model+attack combinations as reusable experiment units
- Location: `experiment_instances/instances.py`
- Contains: `ExperimentInstance` dataclass, `EXPERIMENTS_REGISTRY` dict, helper factories
- Depends on: Attack configs, model params, pipeline config types
- Used by: CLI runner

**Pipeline Layer:**
- Purpose: Orchestrate the full experiment workflow
- Location: `src/rbspaper/pipeline/`
- Contains: Core pipeline function, config objects, analysis utilities, model factory
- Depends on: All lower layers (models, attacks, evaluation, data)
- Used by: CLI runner

**Adapter Layer:**
- Purpose: Define abstract interfaces for models, attacks, and downstream tasks
- Location: `src/rbspaper/adapters/`
- Contains: `ModelAdapter`, `AttackAdapter`, `TaskAdapter` ABCs
- Depends on: None (pure interfaces)
- Used by: Future pipeline refactoring (currently not integrated into the core pipeline)

**Models Layer:**
- Purpose: Self-supervised time series representation learning models
- Location: `src/rbspaper/models/`
- Contains: TS2Vec, AutoTCL, CoST LightningModules; encoders; losses; augmentation; config
- Depends on: PyTorch, Lightning, einops
- Used by: Pipeline core (training, encoding)

**Attacks Layer:**
- Purpose: Adversarial attack execution with registry of supported methods and backends
- Location: `src/rbspaper/attacks/`
- Contains: Functional attack API, config dataclasses, backend dispatch, registry
- Depends on: External attack libraries (ART, torchattacks)
- Used by: Pipeline core

**Evaluation Layer:**
- Purpose: Downstream task evaluation on clean and attacked representations
- Location: `src/rbspaper/evaluation/`
- Contains: Classification and forecasting evaluation, protocol implementations
- Depends on: scikit-learn
- Used by: Pipeline core

**Data Layer:**
- Purpose: Dataset loading, LightningDataModule setup, scaling, splitting
- Location: `src/rbspaper/data/`
- Contains: Dataset classes, datamodules, registry, preparation factories, utilities
- Depends on: PyTorch, Lightning, pandas, scikit-learn
- Used by: Pipeline core, CLI runner

**Configs and Enums Layers:**
- Purpose: Shared type definitions and re-exports
- Location: `src/rbspaper/configs/`, `src/rbspaper/enums/`
- Contains: Enum classes, config re-exports
- Depends on: None
- Used by: All layers

## Data Flow

### Primary Experiment Pipeline Path

1. **CLI Entry** -- `runners/py/runner.py:main()` parses `--experiment_id`, `--dataset_name`, `--data_root`
2. **Resolve Experiment** -- `experiment_instances/instances.py:get_experiment_instance()` returns `ExperimentInstance` with model params and attack configs
3. **Resolve DataModule** -- `src/rbspaper/data/data_setup.py:get_datamodule_with_downstream_tasks()` looks up dataset in registry, builds params, creates `LightningDataModule`
4. **Resolve Input Dimensions** -- Extract `n_features` and `sequence_len` from the datamodule
5. **Build Model** -- `src/rbspaper/pipeline/setup/model.py:build_model_from_parameters()` dispatches on `ModelParameters` type to construct the `LightningModule`
6. **Assemble Pipeline Config** -- `ExperimentPipelineConfig` bundles model, data, training, encoding, attacks, downstream tasks, analysis, artifacts, seed
7. **Run Pipeline** -- `src/rbspaper/pipeline/core.py:run_experiment_pipeline()`:

### Pipeline Internal Flow

1. **Preflight Validation** (`_preflight_pipeline_config`, line 591) -- Validates downstream tasks, attack uniqueness, scope binding, attack support
2. **Train Model** (`_train_model`, line 191) -- Creates `pl.Trainer`, calls `trainer.fit()`, saves checkpoint
3. **Collect Partition Tensors** (`_collect_partition_tensors`, line 258) -- Iterates train/val/test dataloaders, concatenates inputs and labels
4. **Generate Attacked Inputs** (lines 91-100) -- In `SHARED_INPUT` mode, runs all attacks once before the task loop. In `TASK_CONDITIONED` mode, generates per-task within the loop.
5. **For Each Downstream Task** (line 106):
   a. **Extract Clean Representations** (`_extract_clean_representations`, line 290) -- Calls `encode_data()` for train/val/test, normalizes shapes
   b. **Build Attacked Representations** (`_build_attacked_reps_for_task`, line 483) -- Per-scope: encode shared inputs or generate+encode per-task
   c. **Evaluate Downstream** (`_evaluate_downstream`, line 652) -- Runs `evaluate()` for clean and each attacked representation across hyperparam grid
6. **Representation Analysis** (`_run_representation_analysis`, line 708) -- Computes geometry, shift, linear separability, and low-dim artifacts
7. **Persist Artifacts** -- Saves NPZ representations, JSON metrics, and summary

### Secondary Flow: Encoding

```
encode_data() [src/rbspaper/models/encoding.py:15]
  -> _ts2vec_encode() / _auto_tcl_encode() / _cost_encode()
    -> _generic_classification_encode() | _generic_clustering_encode() | _generic_forecasting_encode()
      -> model.encode() [EncodingFunctionalityMixin.encode(), line 187]
        -> DataLoader iteration
          -> _eval_method (pooling or feature concatenation)
            -> self._encoder() (TS2VecTimeSeriesEncoder, etc.)
```

### Attack Execution Flow

```
execute_attack() [src/rbspaper/attacks/functional.py]
  -> _resolve_context()
  -> _resolve_supervision()
  -> build_attack_model() [src/rbspaper/attacks/_common.py]
  -> run_attack_backend() [src/rbspaper/attacks/_backend.py]
    -> ART backend OR torchattacks backend
  -> maybe_return_metadata()
```

**State Management:**
- Pipeline state is purely functional: config objects flow into `run_experiment_pipeline()`, which returns `ExperimentPipelineResults`. No mutable global state in the pipeline.
- LightningDataModules hold mutable state during `setup()` (train/val/test splits).
- Model training state managed by Lightning internally.

## Key Abstractions

**ModelParameters Hierarchy:**
- Purpose: Typed parameter containers for model construction
- Examples: `TS2VecModelParameters`, `AutoTCLModelParameters`, `CoSTModelParameters`
- Pattern: Abstract base class with `@abstractmethod model_name` and `set_input_dims()`. Subclasses add model-specific fields.
- Location: `src/rbspaper/models/config.py`

**ExperimentInstance:**
- Purpose: Pre-configured experiment bundling model params + attack configs + training settings
- Examples: `ts2vec_fgsm`, `ts2vec_pgd`, `autotcl_multi` (7 registered)
- Pattern: Dataclass stored in `EXPERIMENTS_REGISTRY` dict. Resolved by string ID.
- Location: `experiment_instances/instances.py`

**AttackParameters Hierarchy:**
- Purpose: Typed parameter containers for adversarial attacks
- Examples: `FgsmAttackParameters`, `PgdAttackParameters`, `BimAttackParameters`, `CwAttackParameters`
- Pattern: Abstract base with `backend`, `clip_min`, `clip_max`. Subclasses add attack-specific params. Each exposes `attack_method` property.
- Location: `src/rbspaper/attacks/config.py`

**AttackScopePolicy:**
- Purpose: Controls whether attacks are shared across tasks or conditioned per-task
- Values: `TASK_CONDITIONED` (default) -- attacks run per matching task; `SHARED_INPUT` -- attacks run once, representations encoded per task
- Pattern: StrEnum used in `AttackScopeConfig`. Validated during preflight.
- Location: `src/rbspaper/pipeline/config.py`

**DatasetRegistry:**
- Purpose: Static metadata for all supported datasets
- Pattern: Frozen `DatasetMetadata` tuples in `DATASET_REGISTRY`. Lookup by name.
- Location: `src/rbspaper/data/registry.py`

**EncodingFunctionalityMixin:**
- Purpose: Shared encoding capability across all three model types
- Pattern: Python mixin class (not inherited by LightningModule, but mixed in via multiple inheritance: `class TS2Vec(pl.LightningModule, EncodingFunctionalityMixin)`).
- Location: `src/rbspaper/models/abstract/encoding_functionality_mixin.py`

**DataModuleWithDownstreamTasks:**
- Purpose: Named tuple pairing a LightningDataModule with its task names
- Pattern: `namedtuple('DataModuleWithDownstreamTasks', ['data_module', 'downstream_tasks'])`
- Location: `src/rbspaper/data/preparation.py`

## Entry Points

**CLI Runner:**
- Location: `runners/py/runner.py`
- Triggers: Command-line invocation (`uv run rbspaper-run`)
- Responsibilities: Parse args, resolve experiment instance, resolve datamodule, build pipeline config, execute pipeline
- Exposed as script: `rbspaper-run` (defined in `pyproject.toml` `[project.scripts]`)

**Pipeline Function:**
- Location: `src/rbspaper/pipeline/core.py:run_experiment_pipeline()`
- Triggers: Called by CLI runner; callable from Python code directly
- Responsibilities: Full experiment orchestration (train, attack, encode, evaluate, analyze)

## Architectural Constraints

- **Threading:** Single-threaded Python with DataLoader `num_workers` for I/O parallelism. No explicit threading in the pipeline.
- **Global state:** No module-level mutable state in the pipeline. Registry dicts are read-only at runtime. Lightning manages trainer state internally.
- **Circular imports:** Mitigated via `TYPE_CHECKING` guards and `from __future__ import annotations` in pipeline config. The `src.rbspaper.configs.models` module exists as a re-export to avoid circular imports between `models/config.py` and `pipeline/setup/model.py`.
- **Device placement:** Models run on GPU when available (Lightning default). Encoding moves tensors to model device explicitly. Attack execution happens on the same device as inputs.
- **Pipeline immutability:** All config dataclasses are `frozen=True`. Deep copies of model params prevent mutation of shared experiment instances (see `runners/py/runner.py:143`).

## Anti-Patterns

### Direct isinstance dispatch in encoding

**What happens:** `encode_data()` in `src/rbspaper/models/encoding.py` uses `isinstance(model, TS2Vec)`, `isinstance(model, AutoTCL)`, `isinstance(model, CoST)` to dispatch encoding. Each branch then delegates to task-specific functions that call nearly identical generic functions.

**Why it's wrong:** Adding a new model requires modifying this central function. The classification/clustering branches call the same generic function, making them redundant.

**Do this instead:** Use the `model.encode()` method from `EncodingFunctionalityMixin` directly with task-appropriate parameters, or register encoding strategies in a dispatch dict keyed by model name.

### Adapter interfaces not integrated into pipeline

**What happens:** `src/rbspaper/adapters/` defines `ModelAdapter`, `AttackAdapter`, and `TaskAdapter` ABCs, but the pipeline calls models and attacks directly without going through these interfaces.

**Why it's wrong:** The adapter pattern is declared but unused. New code may bypass the intended abstraction, and the adapter interfaces serve as dead documentation.

**Do this instead:** Either wire the adapters into the pipeline (models implement `ModelAdapter`, attacks implement `AttackAdapter`) or remove them if the direct approach is preferred. The current `run_experiment_pipeline()` works with `pl.LightningModule` directly, which is a valid simpler design.

### Hard-coded task string comparison

**What happens:** Throughout `src/rbspaper/pipeline/core.py` and `src/rbspaper/models/encoding.py`, downstream task names are compared using string literals like `'classification'`, `'forecasting'`, `'clustering'` instead of using the `TimeSeriesEvaluationDownstreamTaskEnum` consistently.

**Why it's wrong:** Magic strings are error-prone and not caught at type-check time.

**Do this instead:** Use `TimeSeriesEvaluationDownstreamTaskEnum` enum values for all comparisons. The enum already exists in `src/rbspaper/evaluation/enums.py` and `src/rbspaper/enums/general.py`.

## Error Handling

**Strategy:** Fail-fast validation at pipeline entry (preflight), explicit exceptions throughout.

**Patterns:**
- `ValueError` for config inconsistencies (duplicate attack names, task-profile mismatch, missing anchor_task)
- `TypeError` for unexpected data shapes or types (e.g., batch format in `_extract_labels_from_batch`)
- `RuntimeError` for state prerequisites (e.g., model must be trained before encoding)
- `KeyError` for registry lookups (unknown experiment ID, unknown dataset name)
- `NotImplementedError` for unimplemented features (unsupported datamodule family, unsupported model type)
- Pipeline-level exception handling in `main()`: catches `KeyboardInterrupt` (exit 130) and generic `Exception` (exit 1)

## Cross-Cutting Concerns

**Logging:** Python standard `logging` module. Logger instances per module (`logger = logging.getLogger(name=__name__)`). Used in evaluation, data setup, encoding. Pipeline core does not log directly (no logger instance).

**Validation:** Preflight validation in `_preflight_pipeline_config()` validates the full pipeline config before execution. Dataclass `__post_init__` methods validate individual configs (e.g., `AttackScopeConfig`, `DatasetTaskProfile`). `AttackRunConfig.validate()` checks query budget and surrogate settings.

**Reproducibility:** `pl.seed_everything()` is called at pipeline start (using `config.seed`). Evaluation sets its own seed (42). DataLoader shuffle is configurable.

**Reproducibility via deterministic operations:** PyTorch deterministic mode is not enforced. CUDA operations may be non-deterministic.

---

*Architecture analysis: 2026-05-05*
