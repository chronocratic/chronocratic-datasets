# Phase 3: Pydantic Registry - Context

**Gathered:** 2026-05-11
**Status:** Ready for planning

## Phase Boundary

Typed configuration — one Pydantic class per family, frozen instances per dataset, with enums for all parameters. Replaces rbspaper's simple `DatasetMetadata` dataclass and eliminates JSON config files. Configs carry everything Phase 4 (download) and Phase 5 (modules) need.

## Implementation Decisions

### Config Class Hierarchy (D-01)
- **Layered inheritance:** Abstract `DatasetConfig` base with common fields (name, family, url, sha256, num_classes, data_form). Family-specific subclasses (UCRConfig, ETTConfig, etc.) add their own fields (file_patterns, target_col, split_bounds).
- **Why:** Each family has distinct needs — UCR needs ARFF file patterns, ETT needs split indices. Layered inheritance keeps configs type-safe and avoids sparse `None` defaults.

### Instance Exposure (D-02)
- **Both module-level constants + registry dict:** Define `UCR_COFFEE = UCRConfig(...)` for direct imports, and auto-collect into a registry dict in `config/__init__.py` for `get_config("Coffee")` lookup.
- **Why:** Best developer experience — IDE auto-complete for direct imports, iterable registry for the factory API.

### Forecasting Splits (D-03)
- **Dual-mode:** `SplitMode` enum (`INDEXED` vs `FRACTIONAL`). ETT configs use `INDEXED` with explicit train/valid/test end-indices. Electricity/weather configs use `FRACTIONAL` with train/valid/test fractions.
- **Why:** ETT's splits are time-intrinsic (16/4/4 months), not proportional. Electricity uses 60/20/20 fractions. One representation can't serve both cleanly.

### Validation Strategy (D-04)
- **Both `@field_validator` and `@model_validator`:** Field validators for single-field checks (url format, num_classes >= 1). Model validator (`mode="after"`) for cross-field constraints (classification configs must have num_classes, forecasting must have seq_len default).
- **Why:** Field validators give precise error messages for simple issues. Model validator catches incomplete configs at creation time, not module-instantiation time.

### Config Immutability (D-05)
- **Class-level `frozen=True` + `model_copy()`:** Set `model_config = ConfigDict(frozen=True)` on base class. All subclasses inherit. Use `model_copy(update={...})` for runtime variants (e.g., overriding seq_len).
- **Why:** Most idiomatic Pydantic v2. Prevents accidental mutation of shared config instances.

### Config Scope (D-06)
- **Complete:** Include all fields modules need — url, sha256, num_classes, data_form, target_col_name, file_name_patterns (UCR/UEA), split_mode, split_bounds (forecasting), default_seq_len. No separate JSON configs.
- **Why:** Enables `get_module("Coffee")` to instantiate a fully-configured module from the config alone. Makes the factory API (Phase 6) work without external files.

### Claude's Discretion
- Exact field names for split boundaries (e.g., `train_end_index` vs `train_slice_end`) — pick clear, descriptive names.
- How to auto-register configs in `config/__init__.py` — use `__all__` iteration or a decorator.
- Whether to add computed properties (e.g., `cache_key` derived from url + sha256) — add if it simplifies Phase 4 code.

## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Code (primary — rbspaper)
- `_sources/rbspaper/src/rbspaper/data/registry.py` — Current `DatasetMetadata` dataclass (name, family, tasks) and `DATASET_REGISTRY` tuple. Starting point for expansion.
- `_sources/rbspaper/src/rbspaper/data/modules/ucr_datamodule.py` — UCR module reads `dataset_config['main_config']['file_name_patterns']` and `target_column_name`. Config must supply these.
- `_sources/rbspaper/src/rbspaper/data/modules/ett_datamodule.py` — ETT module hardcodes split indices (12*30*24, 16*30*24, etc.). Config must supply these.
- `_sources/rbspaper/src/rbspaper/data/modules/uea_datamodule.py` — UEA module uses `data_form='nested'` (vs UCR `'regular'`). Config must encode this.
- `_sources/rbspaper/src/rbspaper/data/modules/abstract.py` — Base module classes show all config fields consumed (target_col_name, file_name_patterns, data_form).

### Existing Package (Phase 1 output)
- `src/tscollection/datasets/enums/data.py` — Existing enums: `TimeSeriesDatasetMode`, `ScalingMethod`, `SplittingStrategy`, `ForecastingMode`, `DistanceMetric`. Add `DatasetFamily`, `SplitMode`.
- `src/tscollection/datasets/__init__.py` — Current public API with enum exports.
- `src/tscollection/datasets/config/__init__.py` — Empty stub with `__all__ = []`, ready for Phase 3.

### Existing Package (Phase 2 output)
- `src/tscollection/datasets/datasets/__init__.py` — Dataset class exports.
- `src/tscollection/datasets/datasets/ucr.py` — UCR wrapper for reference (shows data_form='regular').
- `src/tscollection/datasets/datasets/uea.py` — UEA wrapper (shows data_form='nested').
- `src/tscollection/datasets/datasets/ett.py` — ETT wrapper (shows forecasting strategy).

### Planning Documents
- `.planning/ROADMAP.md` — Phase 3 deliverables, success criteria, requirement mapping.
- `.planning/REQUIREMENTS.md` — CFG-01 through CFG-03 requirements.
- `.planning/PROJECT.md` — Package structure, constraints, key decisions.

## Existing Code Insights

### Reusable Assets
- `TimeSeriesDatasetMode`, `ScalingMethod`, `SplittingStrategy`, `ForecastingMode` enums (Phase 1, `enums/data.py`) — ready to use in configs.
- `DatasetFamily` enum needs to be added (ucr/uea/ett/electricity/weather).
- `pyproject.toml` already includes `"pydantic>=2.10,<3.0.0"` — no dependency changes needed.
- `config/__init__.py` stub exists with `__all__ = []` — ready for population.

### Established Patterns
- **StrEnum** — All existing enums use `StrEnum` for serializability. New enums should follow.
- **Keyword-only APIs** — All module constructors use `*` to enforce kwargs. Config factory should match.
- **Frozen data structures** — rbspaper uses `@dataclass(frozen=True)`. Pydantic equivalent: `model_config = ConfigDict(frozen=True)`.

### Integration Points
- `config/factory.py` — `get_config(name)`, `list_configs(family)` functions for lookup.
- `config/__init__.py` — Import and re-export all config constants.
- Phase 4 (Download) — will read `url`, `sha256` from config.
- Phase 5 (Modules) — will read `target_col_name`, `file_name_patterns`, `data_form`, `split_bounds` from config.

## Specific Ideas

- UCRConfig fields: `target_col_name`, `file_name_patterns` (dict with train/test arff patterns), `data_form='regular'`
- ETTConfig fields: `split_mode=SplitMode.INDEXED`, `train_end_index`, `valid_end_index`, `forecast_column='OT'`, `default_seq_len=128`
- ElectricityConfig fields: `split_mode=SplitMode.FRACTIONAL`, `train_frac=0.6`, `valid_frac=0.2`, `default_seq_len`
- Use `HttpUrl` type for download URLs (from `pydantic_core`)
- Add `@model_validator(mode="after")` to verify family-specific fields are set

## Deferred Ideas

- Dynamic class generation from registry — Phase 6 (factory API)
- Conda package distribution — v2 requirement
- Full UCR archive (120+ datasets) — v2 requirement
- Multi-GPU splitting pattern — Phase 5: `setup()` vs `prepare_data()` for train/val/test splits. See `.planning/phases/05-data-modules/NOTE-splitting-pattern.md`

---

*Phase: 03-Pydantic Registry*
*Context gathered: 2026-05-11*
