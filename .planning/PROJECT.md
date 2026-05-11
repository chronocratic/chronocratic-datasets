# tsdatasets

## What This Is

A standalone PyTorch/Lightning package for time series dataset families (UCR, UEA, ETT, electricity, weather). Each dataset exposes both a PyTorch `Dataset` and a Lightning `LightningDataModule` that downloads, caches, and loads data out-of-the-box. Import `UCRCoffeeModule` from `tscollection.datasets` and it just works — no file paths, no JSON configs, no boilerplate.

## Core Value

Zero-config, ready-to-use time series datasets that plug directly into Lightning Trainer — user imports the module and trains.

## Requirements

### Validated

- ✓ Existing dataset classes in `_sources/rbspaper` — clean abstract hierarchies with docstrings
- ✓ Existing module classes in `_sources/rbspaper` — LightningDataModule pattern with classification/forecasting bases
- ✓ Both sources use identical Strategy pattern for flexible datasets (sliding windows)
- ✓ PyTorch Lightning integration is already wired (dataloader methods, setup/prepare_data)

### Active

- [ ] Extract and restructure dataset code from `_sources/` into `tscollection.datasets` package (PEP 420 namespace)
- [ ] Use rbspaper `data/` as primary source (better docstrings, defensive code, registry)
- [ ] Convert `src.rbspaper` imports to `tscollection.datasets` relative imports
- [ ] Pydantic-based registry — one config class per family, instances per dataset
- [ ] Auto-download + caching in `~/.cache/tsdatasets/` with SHA256 validation (torchtime pattern)
- [ ] Modules auto-download data in `prepare_data()` — no file paths required from user
- [ ] Factory API: `tscollection.datasets.get_module("Coffee")`, `get_dataset("Coffee")`, `list_modules(family="ucr")`
- [ ] Family-prefixed imports: `from tscollection.datasets.modules import UCRCoffeeModule`
- [ ] Classification: `seq_len` is computed from data (not user-passed), exposed as read-only property
- [ ] Forecasting: `seq_len` is user-configurable with registry default
- [ ] Enums for typed params: `ScalingMethod`, `SplittingStrategy`, `ForecastingMode`
- [ ] Package ready for pip + conda deployment
- [ ] Unit tests for dataset loading, module properties, and factory resolution

### Out of Scope

- Model architectures — this is data only
- Attack/robustness pipeline — rbspaper attack code is not included
- Training runners — rbspaper pipeline code is not included
- Conda recipe distribution — pip/PyPI first (conda can be v2)

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

**What needs to change:**
- Modules require `dataset_folder_path` + `dataset_config_path` — replace with auto-download
- JSON config files with `file_name_patterns` and `target_col_name` — replace with Pydantic registry
- `src.rbspaper` absolute imports — convert to `tscollection.datasets` relative imports
- Forecasting modules return raw `TensorDataset` from dataloaders — should use proper dataset classes
- No `~/.cache/` logic — needs download + cache like torchtime
- `data_form` ('regular' vs 'nested') hardcoded in module constructors — move to registry
- `seq_len` on classification modules computed from data — keep this, just expose as property

**Package structure (PEP 420 namespace):**

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
      electricity.py     # ElectricityLoadDataset
      weather.py         # WeatherDataset
    modules/
      classes/           # Abstract base LightningDataModule classes
        __init__.py
        classification.py # BaseClassificationTimeSeriesDataModule
        forecasting.py    # BaseForecastingTimeSeriesDataModule
      ucr.py             # UCRClassificationDataModule (generic loader)
      uea.py             # UEAClassificationDataModule
      ett.py             # ETTDataModule
      electricity.py     # ElectricityLoadModule
      weather.py         # WeatherModule
    download/
      __init__.py
      ucr_uea.py         # Fetch from timeseriesclassification.com
      forecasting.py     # Fetch ETT, electricity, weather
      cache.py           # ~/.cache/tscollection/ with SHA256
    config/
      __init__.py
      ucr.py             # UCRConfig + UCR_COFFEE, UCR_ECG200, ...
      uea.py             # UEAConfig + UEA_BASIC_MOTIONS, ...
      ett.py             # ETTConfig + ETT_H1, ...
      electricity.py     # ElectricityConfig + ELECTRICITY_LOAD
      weather.py         # WeatherConfig + WEATHER
      factory.py         # Registry lookup + dynamic module exposure
    enums/
      __init__.py
      data.py            # TimeSeriesDatasetMode, ScalingMethod, SplittingStrategy, etc.
    utils/
      __init__.py
      arff.py            # ARFF reading and dtype processing
      scaling.py         # create_data_scaler, MinMax/Standard scaling
      features.py        # extract_time_features, custom_collate_fn
      general.py         # compose, load_json, process_varying_lengths
    factory.py           # get_module(), get_dataset(), list_modules()
```

Import patterns:
- `import tscollection.datasets` — root package with enums and __version__
- `from tscollection.datasets.modules import UCRCoffeeModule` — family-prefixed
- `from tscollection.datasets import get_module, get_dataset` — factory API

## Constraints

- **Tech stack**: Python 3.12, PyTorch, Lightning, Pydantic v2, numpy, pandas, scipy
- **No JSON configs**: All dataset metadata in Pydantic config instances
- **No file paths from user**: Modules handle download + caching automatically
- **Download caching**: Raw data cached in `~/.cache/tscollection/` — SHA256 validated
- **Enums for typed params**: `ScalingMethod`, `SplittingStrategy`, etc. — not raw strings
- **Classification seq_len**: Computed from data, exposed as property (not user-passed)
- **Forecasting seq_len**: User-configurable with registry default

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| rbspaper as primary source | Better docstrings, defensive code, registry | — Pending |
| Pydantic for config | Typed, validated, no extra deps for ML projects | — Pending |
| Auto-download in prepare_data() | torchtime pattern — user provides no paths | — Pending |
| Family-prefixed imports (UCRCoffeeModule) | Disambiguates across families, IDE-friendly | — Pending |
| Factory accepts bare name + family | Flexible lookup, resolves ambiguity | — Pending |
| Classification seq_len from data | Intrinsic property, not configurable | — Pending |
| Forecasting splits from registry | 60/20/20, 16/4/4 months — dataset facts | — Pending |

---
*Last updated: 2026-05-08 after initialization*

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
