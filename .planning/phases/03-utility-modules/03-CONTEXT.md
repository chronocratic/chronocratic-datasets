# Phase 3: Utility Modules - Context

**Gathered:** 2026-05-13
**Status:** Ready for planning

## Phase Boundary

Port utility functions from `_sources/rbspaper/src/rbspaper/data/utils/` into `tscollection.datasets.utils/` with improved file separation, type hints, docstrings and full CLAUDE.md style compliance.

## Implementation Decisions

### File Organization
- **D-01:** Keep separate files per concern: `arff.py`, `scaling.py`, `features.py`, `general.py`
- **D-02:** Existing `common.py` (from Phase 2) stays as-is — contains `compose`, `FunctionComposer`, `get_num_samples_from_ts`
- **D-03:** No merging files — each utility module has a clear domain boundary
- **D-04:** `flatten_list_of_np_arrays` must be added to `common.py` — `scaling.py` depends on it via import

### Enum Wiring
- **D-05:** `create_data_scaler()` uses `ScalingMethod` and `DataForm` enums from our enums package instead of strings — type-safe at call time
- **D-06:** `ScalingMethod` values stay as `'minmax'` and `'standard'` — update the ported function to compare against enum values, not rbspaper source strings (`'min_max'`/`'standardization'`)
- **D-07:** Add `DataForm` enum (REGULAR, NESTED, MULTI_FILES) to `enums/data.py` — import from there in `scaling.py`

### Porting Scope
- **D-08:** Port all 4 utility files now (arff, scaling, features, general) — Phase 4 data modules can start immediately without waiting

### Style and Typing
- **D-09:** Full CLAUDE.md compliance — keyword-only args (`*`), full type hints, clean imports, Google-style docstrings
- **D-10:** Remove `from __future__ import annotations` unless needed for circular imports
- **D-11:** Convert `src.rbspaper` imports to `tscollection.datasets` relative imports

## Canonical References

### Source Code (gitignored — local dev only)
- `_sources/rbspaper/src/rbspaper/data/utils/__init__.py` — exports all utility functions
- `_sources/rbspaper/src/rbspaper/data/utils/arff.py` — ARFF reading with dtype processing
- `_sources/rbspaper/src/rbspaper/data/utils/scaling.py` — `create_data_scaler()` with DataFormEnum
- `_sources/rbspaper/src/rbspaper/data/utils/features.py` — `extract_time_features()` from DatetimeIndex
- `_sources/rbspaper/src/rbspaper/data/utils/general.py` — `compose()`, `load_json()`, `process_varying_lengths()`
- `_sources/rbspaper/src/rbspaper/data/utils/common.py` — full rbspaper common utils (for comparison with our common.py)
- `_sources/rbspaper/src/rbspaper/data/modules/abstract.py` — consumer of these utils (shows all import paths)

### Planning Docs
- `.planning/ROADMAP.md` §Phase 3: Utility Modules — deliverables and success criteria
- `.planning/REQUIREMENTS.md` — UTI-01 through UTI-05

### Existing Code
- `src/tscollection/datasets/utils/__init__.py` — current exports (Phase 2 partial)
- `src/tscollection/datasets/utils/common.py` — already ported functions
- `src/tscollection/datasets/enums/data.py` — `ScalingMethod`, `TimeSeriesDatasetMode`, `SplittingStrategy`, etc.

## Existing Code Insights

### Reusable Assets
- `common.py` already has `compose`, `FunctionComposer`, `get_num_samples_from_ts` (clean, Phase 2)
- `flatten_list_of_np_arrays` must be added to our `common.py` — `scaling.py` depends on it
- `enums/data.py` has `ScalingMethod` — use in `scaling.py`
- `DataFormEnum` from rbspaper `scaling.py` must be added to our `enums/data.py`

### Cross-dependencies
- `scaling.py` → `common.py`: imports `flatten_list_of_np_arrays`
- `general.py` → `torch.utils.data.dataloader`: imports `default_collate`
- `features.py` → `pandas.DatetimeIndex`: self-contained, no cross-utility imports
- `arff.py` → `scipy.io.arff`: self-contained, no cross-utility imports

### Established Patterns
- Functional style with type hints and `__all__` exports
- `TYPE_CHECKING` guard for heavy imports
- Google-style docstrings with Args/Returns sections
- Keyword-only function signatures (`*` separator)

### Integration Points
- `utils/__init__.py` — public exports, must wire all 4 new modules
- Phase 4 `modules/` will import from `tscollection.datasets.utils`
- `scaling.py` depends on `flatten_list_of_np_arrays` from rbspaper common.py

## Specific Ideas

No specific requirements — port with CLAUDE.md style improvements

## Deferred Ideas

None — discussion stayed within phase scope

---

*Phase: 03-Utility Modules*
*Context gathered: 2026-05-13*
