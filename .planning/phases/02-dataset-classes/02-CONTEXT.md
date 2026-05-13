# Phase 2: Dataset Classes - Context

**Gathered:** 2026-05-11
**Status:** Ready for planning

## Phase Boundary

PyTorch `Dataset` hierarchy — fixed (classification) and flexible (forecasting with sliding windows) — decoupled via strategy pattern. This phase extracts the dataset ABCs, strategies, concrete dataset wrappers, and minimal utilities from rbspaper into the `tscollection.datasets` namespace.

## Implementation Decisions

### Dataset Constructor API
- **D-01:** Module-only — datasets accept pre-loaded data (`pd.DataFrame`, `np.ndarray`), never file paths. Modules (Phase 5) own data loading, ARFF/CSV parsing, and pass constructed data to datasets. Users never instantiate datasets directly in production use; they use modules or the factory API.

### Transformation Strategy
- **D-02:** Defaults + optional override. Each dataset class applies sensible default transforms (numpy→tensor, dimension expansion) but accepts `transformations_sequence` kwarg for power users who need custom pipelines.

### Utility Porting Scope
- **D-03:** Port only `transformations.py` (tensor conversion, dimension expansion) and `common.py` (`get_num_samples_from_ts`) in Phase 2. `arff.py` and `general.py` deferred to Phase 5 where modules consume them. Keeps Phase 2 focused on dataset iteration logic.

### Concrete Dataset Design
- **D-04:** Thin wrappers — UCR, UEA, ETT dataset classes set domain defaults (`expand_dims_axis=1`, `ForecastingStrategySingleFile`) and delegate to the ABC base. No new logic in wrappers; verification is straightforward.

### Testing Approach
- **D-05:** Synthetic numpy/pandas fixtures for unit tests (shapes, iteration, indexing) plus 1–2 minimal real samples in `tests/fixtures/` for format validation. No downloads required for Phase 2 tests.

### Claude's Discretion
- Internal cursor management (`_n`, `_go_to_idx`) follows rbspaper pattern verbatim.
- Type hints use `from __future__ import annotations` where rbspaper source already uses it (abstract.py, strategies.py) to avoid forward-reference issues.

## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Source Code (primary — rbspaper)
- `_sources/rbspaper/src/rbspaper/data/datasets/abstract.py` — Full dataset ABC hierarchy: `TimeSeriesDataset`, `Fixed`/`Flexible` base classes, univariate/multivariate/single-file/multi-file variants
- `_sources/rbspaper/src/rbspaper/data/datasets/strategies.py` — Strategy pattern: `SequenceHandlingStrategy` ABC + 3 concrete strategies
- `_sources/rbspaper/src/rbspaper/data/datasets/transformations.py` — Transform helpers: `convert_numpy_to_tensor`, `convert_data_to_np_array`, `expand_data_dimensionality`
- `_sources/rbspaper/src/rbspaper/data/utils/common.py` — `compose()`, `get_num_samples_from_ts()` utilities
- `_sources/rbspaper/src/rbspaper/data/datasets/ucr_dataset.py` — UCR thin wrapper example (14 lines)
- `_sources/rbspaper/src/rbspaper/data/datasets/uea_dataset.py` — UEA thin wrapper example
- `_sources/rbspaper/src/rbspaper/data/datasets/ett_dataset.py` — ETT dataset with forecasting strategy

### Source Code (secondary — autotsrc)
- `_sources/autotsrc/src/autotsrc/datasets/classes/abstract/` — Generic-type pattern (`SequenceHandlingStrategy[DataT]`) for reference

### Planning Documents
- `.planning/ROADMAP.md` — Phase 2 deliverables, success criteria, utility module table
- `.planning/REQUIREMENTS.md` — DST-01 through DST-05 requirements
- `.planning/PROJECT.md` — Package structure, constraints, key decisions

### Existing Package (Phase 1 output)
- `src/tscollection/datasets/__init__.py` — Current public API with enums
- `src/tscollection/datasets/datasets/__init__.py` — Empty stub, `__all__ = []`
- `src/tscollection/datasets/datasets/classes/__init__.py` — Empty stub
- `src/tscollection/datasets/enums/data.py` — `TimeSeriesDatasetMode` enum (already ported)

## Existing Code Insights

### Reusable Assets
- `TimeSeriesDatasetMode` enum (Phase 1, `enums/data.py`) — already has `WITH_LABELS`, `WITHOUT_LABELS`, `FORECASTING` values
- `compose()` utility in rbspaper `common.py` — functional transform chaining, port as-is
- Package `__init__.py` skeleton (Phase 1) — `datasets/__init__.py` has `__all__ = []` ready for exports

### Established Patterns
- **Template Method** — `TimeSeriesDataset` defines `__getitem__` flow; subclasses implement `_go_to_idx`, `_get_current_data`, `_get_current_label`
- **Strategy** — `SequenceHandlingStrategy` abstracts window counting and label extraction; injected into flexible datasets
- **Dict dispatch** — Mode routing via `_get_sample_fun_map = {mode: handler}` instead of if/elif chain
- **Keyword-only signatures** — Use `*` in `__init__` to enforce kwargs (from rbspaper conventions)

### Integration Points
- `src/tscollection/datasets/datasets/__init__.py` — needs `__all__` exports for all dataset classes and strategies
- `src/tscollection/datasets/datasets/classes/__init__.py` — needs ABC exports
- `src/tscollection/datasets/utils/__init__.py` — needs transformation utility exports
- Import path: `tscollection.datasets` (namespace), not `src.tscollection.datasets`

## Specific Ideas

- UCR dataset: `expand_dims_axis=1` to produce shape `(1, seq_len)` from univariate series
- ETT dataset: inject `ForecastingStrategySingleFile(forecast_horizon)` — label is the segment immediately after the input window
- Fixed datasets: `seq_len` should be exposed as a read-only property (computed from `len(data)`, not user-configurable) — per locked decision from Phase 1
- Flexible datasets: `seq_len` and `step` are constructor params — user-configurable

## Deferred Ideas

- `arff.py`, `general.py`, `scaling.py`, `features.py` utility ports — deferred to Phase 5 (modules)
- UEA multivariate dataset details — thin wrapper, but 3D array handling needs care (defer format validation to Phase 5 with real ARFF fixtures)
- Dynamic class generation from registry — Phase 6 (factory API)

---

*Phase: 02-Dataset Classes*
*Context gathered: 2026-05-11*
