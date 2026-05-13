# tsdatasets

## What This Is

A standalone PyTorch/Lightning package for time series dataset families (UCR, UEA, ETT, electricity, weather). Each dataset exposes a PyTorch `Dataset` and a Lightning `LightningDataModule` that loads data from provided file paths. Pass your data paths and you get working dataloaders — no boilerplate.

**v2 vision:** Auto-download + caching, Pydantic config registry, factory API for zero-config usage.

## Core Value

Working dataset classes and utilities from established source codebases, restructured with improved style (file separation, type hints, docstrings).

## Requirements

### Validated

- ✓ Existing dataset classes in `_sources/rbspaper` — clean abstract hierarchies with docstrings
- ✓ Existing module classes in `_sources/rbspaper` — LightningDataModule pattern with classification/forecasting bases
- ✓ Both sources use identical Strategy pattern for flexible datasets (sliding windows)
- ✓ PyTorch Lightning integration is already wired (dataloader methods, setup/prepare_data)
- ✓ Phase 1 (package foundation) and Phase 2 (dataset classes) implemented, all tests pass

### Active (v1)

- [ ] Port utility modules from `_sources/` with better file separation (arff, scaling, features, general)
- [ ] Extract and restructure module code from `_sources/rbspaper` into `tscollection.datasets.modules`
- [ ] Convert `src.rbspaper` imports to `tscollection.datasets` relative imports
- [ ] Modules accept file paths as explicit parameters (no JSON configs, no auto-download)
- [ ] Forecasting modules use proper dataset classes instead of raw TensorDataset
- [ ] Classification: `seq_len` is computed from data (not user-passed), exposed as read-only property
- [ ] Forecasting: `seq_len` is user-configurable with sensible default
- [ ] Package ready for pip deployment
- [ ] Unit tests for dataset loading, module properties, and utility functions

### Active (v2 — Deferred)

Full implementation archived on `archive/v2-full-implementation` branch:

- [ ] Pydantic-based registry — one config class per family, instances per dataset
- [ ] Auto-download + caching in `~/.cache/tscollection/` with SHA256 validation
- [ ] Factory API: `get_module("Coffee")`, `get_dataset("Coffee")`, `list_modules(family="ucr")`
- [ ] Family-prefixed imports: `from tscollection.datasets.modules import UCRCoffeeModule`
- [ ] Modules auto-download data in `prepare_data()` — no file paths required

### Out of Scope

- Model architectures — this is data only
- Attack/robustness pipeline — rbspaper attack code is not included
- Training runners — rbspaper pipeline code is not included
- Conda recipe distribution — pip/PyPI first (conda can be v3)

## Context

**Source code lives in `_sources/`:**
- `_sources/rbspaper/src/rbspaper/data/` — primary base. Better docstrings, defensive error handling, registry, proper type annotations. Has `abstract.py` for both datasets and modules, family-specific implementations (ucr, uea, ett, weather, electricity), strategies, and ARFF utilities.
- `_sources/autotsrc/src/autotsrc/datasets/` — secondary reference. Same hierarchy but minimal docstrings and some import issues. Has useful generic-type pattern (`SequenceHandlingStrategy[DataT]`).

**What rbspaper does well:**
- `FixedTimeSeriesDataset` hierarchy — clean for classification (no sliding window)
- `FlexibleTimeSeriesDataset` hierarchy — sliding windows with strategy pattern for forecasting
- `BaseClassificationTimeSeriesDataModule` — handles ARFF parsing, label extraction, variable-length padding
- `BaseForecastingTimeSeriesDataModule` — handles CSV loading, scaling, time-feature extraction, train/val/test slices
- `registry.py` — static dataset metadata (name, family, tasks)
- Forecasting modules use hardcoded splits (Weather 60/20/20, ETT 16/4/4 months) — these are intrinsic dataset facts

**What needs to change for v1:**
- `src.rbspaper` absolute imports — convert to `tscollection.datasets` relative imports
- Forecasting modules return raw `TensorDataset` from dataloaders — should use proper dataset classes
- `data_form` ('regular' vs 'nested') hardcoded in module constructors — pass as explicit parameter
- Bundled utility code — split into separate files (arff.py, scaling.py, features.py, general.py)
- Modules require `dataset_config_path` — replace with explicit parameters (target_col_name, etc.)

**What's deferred to v2:**
- Auto-download + caching (currently implemented on `archive/v2-full-implementation`)
- Pydantic registry (currently implemented on `archive/v2-full-implementation`)
- Factory API (planned, not yet implemented)

**Package structure (v1, PEP 420 namespace):**

```
tscollection/            (no __init__.py — implicit namespace)
  datasets/
    __init__.py          # Public API surface, __version__
    datasets/
      classes/           # Abstract base Dataset classes
        __init__.py
        fixed.py         # FixedTimeSeriesDataset (univariate/multivariate)
        flexible.py      # FlexibleTimeSeriesDataset (single/multi-file)
      ucr.py             # UCRClassificationUnivariateDataset
      uea.py             # UEAClassificationMultivariateDataset
      ett.py             # ETTDataset
    modules/
      classes/           # Abstract base LightningDataModule classes
        __init__.py
        classification.py # BaseClassificationTimeSeriesDataModule
        forecasting.py    # BaseForecastingTimeSeriesDataModule
      ucr.py             # UCRClassificationDataModule (generic loader, takes file path)
      uea.py             # UEAClassificationDataModule
      ett.py             # ETTDataModule
      electricity.py     # ElectricityLoadModule
      weather.py         # WeatherModule
    enums/
      __init__.py
      data.py            # TimeSeriesDatasetMode, ScalingMethod, SplittingStrategy, etc.
    utils/
      __init__.py
      arff.py            # ARFF reading and dtype processing
      scaling.py         # create_data_scaler, MinMax/Standard scaling
      features.py        # extract_time_features, custom_collate_fn
      general.py         # compose, load_json, process_varying_lengths
```

Import patterns (v1):
- `import tscollection.datasets` — root package with enums and __version__
- `from tscollection.datasets.datasets import FixedTimeSeriesDatasetUnivariate` — dataset classes
- `from tscollection.datasets.modules import UCRClassificationDataModule` — modules
- `from tscollection.datasets.utils import create_data_scaler` — utilities

## Constraints

- **Tech stack**: Python 3.12, PyTorch, Lightning, numpy, pandas, scipy, scikit-learn
- **v1: File paths from user**: Modules accept explicit file paths, no auto-download
- **v1: No JSON configs**: Module parameters are explicit, not loaded from config files
- **v1: No Pydantic**: Config validation deferred to v2
- **Classification seq_len**: Computed from data, exposed as property (not user-passed)
- **Forecasting seq_len**: User-configurable with sensible default

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| rbspaper as primary source | Better docstrings, defensive code, registry | Done, Phase 2 |
| v1 = minimal, no pydantic/download | Ship working foundation first | Done, 2026-05-13 |
| Archive full impl to branch | Preserve work for v2 reintegration | Done, 2026-05-13 |
| Keep style improvements | Better file separation, type hints, docstrings | Done, 2026-05-13 |
| Classification seq_len from data | Intrinsic property, not configurable | Pending |
| Modules take file paths (v1) | No download dependency, user controls data | Pending |

---
*Last updated: 2026-05-13 — v1 scope revised*

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state
